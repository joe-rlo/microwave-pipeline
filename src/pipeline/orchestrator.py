"""Pipeline orchestrator — wires triage, search, assembly, reflection together.

This is the core cognitive loop of MicrowaveOS.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from src.config import Config
from src.llm.client import LLMClient
from src.memory.embeddings import EmbeddingClient
from src.memory.index import MemoryIndex
from src.memory.search import MemorySearcher
from src.memory.store import MemoryStore
from src.pipeline.assembly import assemble, promote_fragments
from src.pipeline.reflection import disabled_reflection, reflect, simple_hedge_check
from src.pipeline.search import search
from src.pipeline.triage import triage
from src.health.audit import HealthAuditRow, HealthAuditWriter
from src.health.disclaimers import EMPTY_RETRIEVAL_RELAXATION, load_health_channel_rules
from src.health.retrieval.base import EvidenceSource
from src.health.retrieval.medlineplus import MedlinePlusSource
from src.health.retrieval.orchestrator import RetrievalOrchestrator
from src.health.retrieval.pubmed import PubMedSource
from src.health.retrieval.query_rewrite import rewrite_query as health_rewrite_query
from src.health.router import DECLINE_PHI_MESSAGE, route as health_route
from src.projects import Project, ProjectLoader, ProjectNotFound
from src.session.engine import SessionEngine
from src.session.models import PipelineMetadata, SearchResult, Turn
from src.skills import Skill, SkillLoader, SkillNotFound
from src.tools import build_tools

log = logging.getLogger(__name__)


class Orchestrator:
    """Wires the cognitive pipeline stages together."""

    def __init__(self, config: Config):
        self.config = config
        self.llm: LLMClient | None = None
        self.memory_store: MemoryStore | None = None
        self.memory_index: MemoryIndex | None = None
        self.searcher: MemorySearcher | None = None
        self.session_engine: SessionEngine | None = None
        self._stable_mtime: float = 0.0
        self._session_id: str | None = None
        self._channel: str = "repl"
        # Skills — loader created during start(), active skill tracked
        # per-orchestrator (one active at a time in v1).
        self.skill_loader: SkillLoader | None = None
        self._active_skill: Skill | None = None

        # Projects — same shape as skills. The active project's BIBLE.md
        # joins the stable prompt; its declared skill auto-activates;
        # file output routes to its drafts/ directory.
        self.project_loader: ProjectLoader | None = None
        self._active_project: Project | None = None
        # Set when active project changes — process() reconnects on the
        # next turn so BIBLE.md updates propagate without a sync API.
        self._project_changed: bool = False
        # Set by `mark_llm_for_reset()` when an outside caller (e.g. the
        # Signal channel cancelling a pipeline mid-flight for addendum-merge)
        # has reason to believe the SDK's internal conversation history
        # diverged from what we've actually committed. The next process()
        # call rebuilds a clean LLM session before doing any LLM work.
        self._needs_llm_reset: bool = False

        # The most recent SearchResult — populated by process() after
        # search runs, consumed by `/why` so the user can see what
        # retrieval fed the last answer. Single-slot intentionally:
        # personal bot, one operator, "last across any channel" is
        # the right granularity for the use case. None when the last
        # turn skipped search (social / meta classification) or no
        # turn has run yet.
        self._last_search_result: SearchResult | None = None

        # Cross-session continuity (pipeline 3.1):
        # `_session_started_at` records when the current session was
        # opened so the close hook can write the summary's `started`
        # frontmatter. `_first_turn_pending` is True until the first
        # user turn lands; that turn force-includes recent session
        # summaries in dynamic context so a fresh `/new` doesn't
        # start cold.
        self._session_started_at: datetime | None = None
        self._first_turn_pending: bool = True

    async def start(self, channel: str = "repl") -> None:
        """Initialize all components and connect."""
        self._channel = channel
        self.config.ensure_dirs()

        # Surface workspace identity at boot so cwd/WORKSPACE_DIR drift
        # is visible loudly rather than two turns into a Signal
        # conversation. The check also catches first-run installs that
        # forgot to set up IDENTITY.md (the bot is broken without it —
        # the system prompt has no voice / persona to anchor replies).
        _verify_workspace(self.config)

        # Memory store (markdown workspace)
        self.memory_store = MemoryStore(self.config.workspace_dir)
        self.memory_store.ensure_dirs()

        # Embedding client
        embedder = EmbeddingClient(api_key=self.config.openai_api_key)

        # Memory index (SQLite + vec + FTS)
        self.memory_index = MemoryIndex(self.config.db_path, embedder)
        self.memory_index.connect()

        # Session engine (created before searcher so searcher can query turns)
        self.session_engine = SessionEngine(
            self.config.db_path,
            context_limit=self.config.context_window_limit,
            compaction_threshold=self.config.compaction_threshold,
        )
        self.session_engine.connect()

        # Breadcrumb + consolidation + user-prefs tables live alongside
        # `turns` on the same connection. Init is idempotent; cheap to
        # run on every startup.
        from src.health.user_prefs import init_tables as init_user_prefs_tables
        from src.memory.breadcrumbs import ToolCallCounter, init_tables as init_breadcrumbs_tables
        from src.memory.consolidation import init_tables as init_consolidation_tables
        init_breadcrumbs_tables(self.session_engine.conn)
        init_consolidation_tables(self.session_engine.conn)
        init_user_prefs_tables(self.session_engine.conn)
        # Per-process tool-call counter. Cleared on `new_session()`.
        self._tool_call_counter = ToolCallCounter()
        # Snapshot of cumulative tool calls in the current session — fed
        # into breadcrumbs as `tool_call_count`.
        self._session_tool_calls = 0
        # Set to True when the counter signals an auto_interval breadcrumb
        # is due. The flag is consumed at end-of-turn so the breadcrumb
        # never interrupts the user-visible stream.
        self._breadcrumb_due = False

        # Phase G.2.b: profile proposals surfaced in the most-recent
        # bot response. Used to parse the next user message as a reply
        # ("yes" / "no" / "1 2") and apply accepted proposals to the
        # profile. Cleared once consumed by the next turn.
        self._last_shown_profile_proposals: list = []

        # Memory searcher — wired to session engine so it can also pull
        # matching turns from recent conversation as a second retrieval path.
        self.searcher = MemorySearcher(self.memory_index, embedder, self.session_engine)

        # Skills — auto-created dir under workspace so new installs just work
        skills_dir = self.config.workspace_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        self.skill_loader = SkillLoader(skills_dir)

        # Projects — same auto-create pattern
        projects_dir = self.config.workspace_dir / "projects"
        projects_dir.mkdir(parents=True, exist_ok=True)
        self.project_loader = ProjectLoader(projects_dir)

        # Health module retrieval orchestrator — empty source list when
        # the module is disabled, so route() short-circuits early
        # regardless. Built once at startup; same pattern as tools.
        self.health_retrieval = RetrievalOrchestrator(
            sources=_build_health_sources(self.config),
        )

        # Health audit writer — owns its own apsw connection so audit
        # failures can't take out the session-engine's traffic. Always
        # constructed but only called on health-routed turns; a bunch
        # of dead pre-flight when the module is off, but no per-turn
        # cost.
        self.health_audit = HealthAuditWriter(self.config.db_path)
        self.health_audit.connect()

        # Tools — built once at startup. Adding a new tool means setting
        # its env var and restarting; we don't hot-reload because the
        # MCP server is wired into the SDK session at connect time.
        self.tool_bundle = build_tools(self.config)
        if self.tool_bundle.allowed_tools:
            log.info(
                "Registered %d tool(s): %s",
                len(self.tool_bundle.allowed_tools),
                ", ".join(self.tool_bundle.allowed_tools),
            )

        # LLM client — factory picks LLMClient (legacy / Agent SDK) or
        # LLMSession (Phase C provider path) based on LLM_STAGE_MAIN env.
        # Both expose the same surface; orchestrator stays oblivious.
        from src.llm.factory import build_main_llm
        self.llm = build_main_llm(self.config)

        # Assemble stable context and connect. Active project's BIBLE.md
        # (if set) joins the stable prompt — but at start() there's no
        # active project yet, so this is just identity + memory + channels.
        bible_path = self._active_bible_path()
        stable_prompt = self.memory_store.assemble_stable_context(
            channel=self._channel, bible_path=bible_path
        )
        self._stable_mtime = self.memory_store.stable_context_mtime(
            channel=self._channel, bible_path=bible_path
        )
        # Log stable-prompt size so cache-effectiveness changes from
        # context-shape work (item 2.1 daily-notes-via-retrieval, 2.2
        # MEMORY.md-via-retrieval) are visible in the startup banner.
        # ~4 chars per token is the rough Anthropic guideline; cheap
        # estimate without burning a tiktoken call here.
        log.info(
            f"[startup] stable prompt: {len(stable_prompt):,} chars "
            f"(~{len(stable_prompt) // 4:,} tokens)"
        )
        await self.llm.connect(stable_prompt)

        # Index workspace files into the memory fragment store
        await self._index_workspace()

        # Resume last session if one exists, otherwise start fresh
        last_session = self.session_engine.get_last_session_id()
        if last_session:
            self._session_id = last_session
            turns = self.session_engine.get_turns(last_session)
            if turns:
                await self._inject_history(turns)
                log.info(f"Resumed session {self._session_id} ({len(turns)} prior turns)")
                # Resumed mid-session: skip the cold-start summary injection,
                # the SDK history already carries the relevant context.
                self._first_turn_pending = False
            else:
                log.info(f"Resumed session {self._session_id} (empty)")
        else:
            self._session_id = self.session_engine.new_session_id()
            log.info(f"New session {self._session_id}")

        # Stamp the session start. On `/new` this gets re-stamped; on
        # resume we still use boot time as "started" for the eventual
        # summary, which is close enough — the prior session's actual
        # `started` was lost across the previous shutdown.
        self._session_started_at = datetime.now()

        # Phase F.3 — startup catchup for memory consolidation. If the
        # last run was >24h ago (or never), kick off a background task
        # so the bot's first-message latency isn't gated on a multi-
        # second Extract → Link → Brief sequence. Failures inside the
        # catchup are logged but don't bubble up.
        try:
            from src.memory.consolidation import run_catchup_if_due
            asyncio.create_task(
                run_catchup_if_due(
                    conn=self.session_engine.conn,
                    config=self.config,
                    interval_hours=24,
                ),
                name="consolidation-catchup",
            )
        except Exception as e:
            log.warning("Could not schedule consolidation catchup: %s", e)

    async def process(
        self,
        message: str,
        user_id: str = "default",
        channel: str = "repl",
        images: list[tuple[bytes, str]] | None = None,
    ) -> AsyncIterator[dict]:
        """Process a user message through the full cognitive pipeline.

        `images`, when provided, is a list of (bytes, content_type) tuples
        attached by the channel. They flow through to `LLMClient.send`
        as Anthropic multimodal content blocks. Triage and reflection
        stay text-only — they only see `message` (the caption) — so
        Haiku-tier vision isn't burned on routing decisions.

        Yields streaming chunks for channel delivery.
        Also yields a final metadata chunk with pipeline stats.
        """
        start = time.time()

        # If the previous pipeline was cancelled mid-LLM-call, the SDK's
        # internal history has phantom messages that don't match what's
        # committed. Resync before doing any LLM work for this turn.
        if self._needs_llm_reset:
            try:
                await self._reset_llm_after_cancel()
            except Exception as e:
                log.warning(f"LLM reset before turn failed: {e}")
            self._needs_llm_reset = False

        # Phase G.2.b: parse the user's message as a reply to any
        # profile proposals surfaced in the previous response. If it's
        # a clear yes/no/indices reply we apply the action AND let the
        # main pipeline continue (the user might also have a real
        # question in the same message; we don't short-circuit).
        try:
            profile_reply_chunks = await self._handle_profile_reply(message)
        except Exception as e:
            log.warning("Profile-reply handler failed: %s", e)
            profile_reply_chunks = []
        for chunk in profile_reply_chunks:
            yield chunk

        # User turn is built here but NOT persisted yet — we commit it as a
        # pair with the assistant turn at the end of process(), so a pipeline
        # that gets cancelled mid-flight (addendum-merge from a Signal-style
        # follow-up, etc.) never leaves an orphan user row in the session
        # engine. Triage gets the message directly via the `message` param;
        # `recent` only needs prior turns.
        user_turn = Turn(
            session_id=self._session_id,
            channel=channel,
            user_id=user_id,
            role="user",
            content=message,
        )

        # Get recent turns for triage context
        recent = self.session_engine.get_recent_turns(channel, user_id, limit=6)

        # --- Stage 1: Triage ---
        # Pass the skill catalog only when no skill is explicitly pinned.
        # If the user has set `/skill foo`, we honor that and skip triage's
        # matching work entirely — cheaper and more predictable.
        skill_catalog: list[tuple[str, str]] = []
        if self._active_skill is None and self.skill_loader is not None:
            try:
                skill_catalog = [
                    (s.name, s.description) for s in self.skill_loader.list_all()
                ]
            except Exception as e:
                log.warning(f"Could not enumerate skills for triage: {e}")

        triage_result = await triage(
            message,
            recent,
            model=self.config.model_triage,
            auth_mode=self.config.auth_mode,
            api_key=self.config.anthropic_api_key,
            cli_path=self.config.cli_path,
            skills=skill_catalog or None,
            workspace_dir=str(self.config.workspace_dir),
            health_enabled=self.config.health.enabled,
            active_project=(self._active_project.name if self._active_project else None),
        )

        # Resolve the skill in effect for THIS turn. Explicit pin always wins;
        # otherwise triage's ephemeral match is used. Neither mutates
        # self._active_skill — auto-match is per-turn, not sticky.
        turn_skill = self._active_skill
        auto_matched = False
        if turn_skill is None and triage_result.matched_skill:
            try:
                turn_skill = self.skill_loader.load(triage_result.matched_skill)
                auto_matched = True
            except Exception as e:
                # Triage returned a name that doesn't resolve on disk —
                # parse_triage_response should have filtered these out, but
                # be defensive in case the skill was removed between enum
                # and load.
                log.debug(f"Auto-matched skill {triage_result.matched_skill!r} failed to load: {e}")
                turn_skill = None

        if auto_matched and turn_skill is not None:
            # Tell the channel so it can surface the match if it wants
            # (most channels stay silent; /debug metadata shows it too).
            yield {
                "type": "skill_matched",
                "skill": turn_skill.name,
                "description": turn_skill.description,
            }

        # --- Health route ---
        # Consults triage's phi_class + the health config. When the route
        # is "skip" everything below runs unchanged; otherwise we may
        # auto-activate the health-qa skill, run evidence retrieval, and
        # splice the disclaimer block into assembly. The "decline_phi"
        # path short-circuits the LLM entirely with a safety message.
        # Phase E: pass the user's health prefs so the router can pick
        # general_private_tee when privacy_mode=private_tee. Default
        # pref (no row in user_health_prefs) keeps behavior identical
        # to pre-Phase-E.
        from src.health.user_prefs import load_pref as _load_user_pref
        _user_pref = _load_user_pref(self.session_engine.conn)
        h_route = health_route(triage_result, self.config.health, _user_pref)
        health_evidence: list = []
        health_disclaimer_text: str = ""

        if h_route.path == "decline_phi":
            log.info(
                f"Health route: decline_phi — {h_route.reason}. "
                "Returning safety message without LLM call."
            )
            yield {"type": "delta", "text": DECLINE_PHI_MESSAGE}
            yield {
                "type": "health_route",
                "path": h_route.path,
                "reason": h_route.reason,
            }
            yield {"type": "result", "text": DECLINE_PHI_MESSAGE}
            self.health_audit.write(HealthAuditRow(
                route="decline_phi",
                triage_phi_class=triage_result.phi_class,
                triage_health_topic=triage_result.health_topic,
                latency_ms=int((time.time() - start) * 1000),
                # No LLM call → provider/model/tokens are NULL by design.
            ))
            return

        if h_route.is_health:
            # Auto-activate health-qa for this turn (overrides triage's
            # match + any explicitly pinned skill — the citation rules
            # are non-negotiable for any health-routed answer).
            try:
                turn_skill = self.skill_loader.load("health-qa")
            except Exception as e:
                log.warning(
                    f"Health route fired but health-qa skill not found: {e}. "
                    "Run `python3 src/main.py health install-skill` to install."
                )
            yield {
                "type": "health_route",
                "path": h_route.path,
                "reason": h_route.reason,
                "use_baa_llm": h_route.use_baa_llm,
            }

            if h_route.enable_retrieval:
                # Rewrite the user's natural-language question into a
                # keyword-style search query before fan-out. PubMed +
                # MedlinePlus index by clinical concepts; the verbatim
                # user sentence has poor recall on side-effect /
                # mechanism / pharmacology questions. Falls back to
                # `message` verbatim on any failure path.
                try:
                    search_query = await health_rewrite_query(
                        message,
                        topic=triage_result.health_topic,
                        model=self.config.model_triage,
                        auth_mode=self.config.auth_mode,
                        api_key=self.config.anthropic_api_key,
                        cli_path=self.config.cli_path,
                        workspace_dir=str(self.config.workspace_dir),
                    )
                except Exception as e:
                    log.warning(f"Query rewrite raised; using original: {e}")
                    search_query = message

                try:
                    health_evidence = await self.health_retrieval.search(
                        search_query,
                        topic=triage_result.health_topic,
                    )
                    log.info(
                        f"Health retrieval: {len(health_evidence)} evidence "
                        f"items for topic={triage_result.health_topic!r}"
                    )
                except Exception as e:
                    # Retrieval failure shouldn't kill the turn — the
                    # empty-retrieval relaxation block (or the health-qa
                    # skill's "no evidence" path) handles it cleanly.
                    log.warning(f"Health retrieval failed: {e}")

            if h_route.require_disclaimer:
                health_disclaimer_text = load_health_channel_rules(
                    self.config.workspace_dir
                )

        # --- Stage 2: Search ---
        # Pass active project so retrieval can weight that project's
        # fragments higher and downweight other projects' (preventing
        # cross-pollination — e.g., a "Sarah" character in a novel
        # bleeding into Substack drafts about a different Sarah).
        active_project_name = (
            self._active_project.name if self._active_project else None
        )
        search_result = await search(
            message, triage_result, self.searcher,
            active_project=active_project_name,
        )
        # Cache for /why — overwritten on every turn that runs search.
        # Turns that skip search (triage `needs_memory=false`) leave
        # the cache pointing at whatever the previous searched turn
        # surfaced, which matches what the user expects from "what fed
        # the LAST substantive answer."
        self._last_search_result = search_result

        # --- Stage 3: Assembly ---
        bible_path = self._active_bible_path()
        # When the general path ran but retrieval came back empty,
        # splice the relaxation block into assembly so the model can
        # answer briefly from training on benign questions instead of
        # the prim "no sources, can't answer" refusal that surprised
        # users on simple drug-class side-effect questions.
        # PHI path doesn't get this — empty-retrieval there means the
        # decline_phi safety message already fired upstream.
        empty_retrieval_note = ""
        if (
            h_route.path == "general"
            and h_route.enable_retrieval
            and not health_evidence
        ):
            empty_retrieval_note = EMPTY_RETRIEVAL_RELAXATION
            log.info(
                "Health route general/empty-retrieval: relaxation block applied "
                "(model may answer briefly from training)"
            )

        # Cold-start recap: on the first turn of a fresh session, force-
        # include the most-recent N session summaries so the bot opens
        # the conversation with context, even when the user's message
        # has no keywords for retrieval to match against.
        session_recap = ""
        if self._first_turn_pending:
            session_recap = self._build_session_recap()
            self._first_turn_pending = False

        assembly_result = assemble(
            search_result,
            self.memory_store,
            self.memory_index,
            channel=self._channel,
            output_dir=str(self.config.output_dir),
            active_skill=turn_skill,
            complexity=triage_result.complexity,
            bible_path=bible_path,
            tool_catalog=self.tool_bundle.catalog_text if self.tool_bundle else "",
            evidence=health_evidence or None,
            health_disclaimer=health_disclaimer_text,
            health_empty_retrieval_note=empty_retrieval_note,
            session_recap=session_recap,
        )

        # Reconnect when (a) project changed since last turn, or (b) any
        # stable-context file is newer than last reconnect. Project changes
        # don't go through mtime — the user could switch from a freshly-
        # edited bible to a stale one — so we track them with a flag.
        current_mtime = self.memory_store.stable_context_mtime(
            channel=self._channel, bible_path=bible_path
        )
        if self._project_changed or current_mtime > self._stable_mtime:
            log.info(
                "Stable context changed (%s), reconnecting LLM",
                "project switch" if self._project_changed else "files on disk",
            )
            await self.llm.reconnect(assembly_result.stable_prompt)
            self._stable_mtime = current_mtime
            self._project_changed = False

        # --- LLM Generation ---
        # Snapshot output dir before generation to detect new files
        output_dir = self.config.output_dir
        pre_files = _snapshot_dir(output_dir)

        # PHI turns route through a separate BAA-covered LLM (AWS
        # Bedrock + Anthropic) built per-turn, so PHI history stays
        # isolated from the main pipeline's session. The Phase E
        # general_private_tee route does the same with NEAR Private
        # TEE open-weight models. Non-PHI / non-TEE turns use self.llm.
        alt_llm = None
        if h_route.use_baa_llm:
            from src.llm.factory import build_baa_llm
            alt_llm = build_baa_llm(self.config)
            if alt_llm is None:
                log.error(
                    "BAA LLM construction failed mid-turn; sending decline_phi"
                )
                yield {"type": "delta", "text": DECLINE_PHI_MESSAGE}
                yield {"type": "result", "text": DECLINE_PHI_MESSAGE}
                return
            await alt_llm.connect(assembly_result.stable_prompt)
        elif h_route.use_private_tee:
            from src.llm.factory import build_private_tee_llm
            alt_llm = build_private_tee_llm(
                self.config, complexity=triage_result.complexity,
            )
            if alt_llm is None:
                # NEAR_API_KEY missing → downgrade to standard general
                # path silently (the user opted into TEE but the env
                # isn't set up). Log loudly so misconfig is visible.
                log.warning(
                    "Private-TEE LLM unavailable; downgrading to standard "
                    "general-health LLM for this turn"
                )
                # alt_llm stays None → falls through to self.llm below
            else:
                await alt_llm.connect(assembly_result.stable_prompt)
        active_llm = alt_llm if alt_llm is not None else self.llm

        # Escalate to Opus+thinking for complex tasks. On the BAA path
        # the escalation target is the health-specific opus model id
        # (config.health.baa_model_escalation). The Private-TEE path
        # is already on its complex model (build_private_tee_llm
        # picked GPT OSS 120B at construction); no further swap.
        escalated = triage_result.complexity == "complex"
        if escalated:
            if h_route.use_baa_llm and alt_llm is not None:
                esc_model = self.config.health.baa_model_escalation
            elif h_route.use_private_tee and alt_llm is not None:
                # Already on the complex Private-TEE model; no escalation.
                esc_model = active_llm.model
            else:
                esc_model = self.config.model_escalation
            log.info(
                f"Escalating to {esc_model} "
                f"(effort={self.config.escalation_effort}) for complex task"
            )
            await active_llm.escalate(
                model=esc_model,
                effort=self.config.escalation_effort,
            )

        full_response = ""
        try:
            async for chunk in active_llm.send(
                message,
                memory_context=assembly_result.memory_context or None,
                images=images,
            ):
                if chunk["type"] in ("text", "delta"):
                    text = chunk.get("text") or chunk.get("chunk", "")
                    full_response += text
                    yield chunk
                elif chunk["type"] == "result":
                    full_response = chunk.get("text", full_response)
                elif chunk["type"] == "tool_use":
                    # Track every tool call for the auto-interval breadcrumb.
                    # Don't fire mid-stream — defer until the turn lands so
                    # breadcrumbs never interrupt the user-visible response.
                    self._session_tool_calls += 1
                    self._tool_call_counter.note_tool_name(
                        chunk.get("name", "")
                    )
                    if self._tool_call_counter.record_tool_call():
                        self._breadcrumb_due = True
                    yield chunk
        finally:
            if escalated:
                await active_llm.de_escalate()
            if alt_llm is not None:
                # Per-turn lifecycle: tear down so PHI / Private-TEE
                # history doesn't leak into the next turn's session.
                await alt_llm.disconnect()

        # --- Stage 4: Reflection ---
        # Route by triage complexity to avoid burning a Haiku round-trip
        # on every turn:
        #   simple   → regex-only hedge check (no model call)
        #   moderate → normal reflection prompt
        #   complex  → deep variant (adds unsupported-claim check)
        #
        # Pipeline 2.3: an active skill can override the lane via its
        # `pipeline.reflection` frontmatter (off | light | normal | deep).
        # The override wins over triage-derived routing — that's the
        # whole point (e.g. `novel-writing` sets `off` so reflection's
        # "you hedged" feedback doesn't dilute the model's voice).
        #
        # `reflection_result.path` carries the lane info downstream
        # so /debug and the audit log can show which fired.
        reflection_override = (
            turn_skill.pipeline.get("reflection") if turn_skill else None
        )
        # Validate the override here rather than at load time so a
        # typo lands as a logged warning + safe fallback (skill ships
        # without a calibrated reflection lane) instead of crashing
        # the turn. Unknown values fall through to triage-derived
        # routing — same effect as the override being absent.
        if reflection_override and reflection_override not in (
            "off", "light", "normal", "deep"
        ):
            log.warning(
                f"Skill {turn_skill.name!r}: unknown pipeline.reflection "
                f"value {reflection_override!r}; falling back to default routing"
            )
            reflection_override = None

        if reflection_override == "off":
            reflection_result = disabled_reflection(full_response)
            log.info("Reflection (off): skill override")
        elif reflection_override == "light" or (
            reflection_override is None and triage_result.complexity == "simple"
        ):
            reflection_result = simple_hedge_check(full_response)
            log.info(
                f"Reflection ({'light override' if reflection_override else 'skipped'}): "
                f"hedging={reflection_result.hedging_detected}"
            )
        else:
            if reflection_override in ("normal", "deep"):
                variant = reflection_override
            else:
                variant = "deep" if triage_result.complexity == "complex" else "normal"
            reflection_result = await reflect(
                full_response,
                context=assembly_result.memory_context,
                model=self.config.model_reflection,
                auth_mode=self.config.auth_mode,
                api_key=self.config.anthropic_api_key,
                cli_path=self.config.cli_path,
                workspace_dir=str(self.config.workspace_dir),
                variant=variant,
            )

        # Handle re-search (one retry max)
        if reflection_result.action == "re-search" and reflection_result.memory_gap:
            log.info(f"Reflection triggered re-search: {reflection_result.memory_gap}")

            # Broaden search parameters
            broader_triage = triage_result
            broader_triage.search_params["result_count"] = min(
                triage_result.search_params.get("result_count", 5) * 2, 15
            )
            broader_triage.search_params["decay_half_life"] = (
                triage_result.search_params.get("decay_half_life", 30) * 2
            )

            retry_search = await search(
                reflection_result.memory_gap, broader_triage, self.searcher,
                active_project=active_project_name,
            )

            if retry_search.fragments:
                retry_assembly = assemble(
                    retry_search, self.memory_store, self.memory_index
                )

                # Re-generate with better context (and the same image
                # content if the original turn had any).
                full_response = ""
                async for chunk in self.llm.send(
                    message,
                    memory_context=retry_assembly.memory_context or None,
                    images=images,
                ):
                    if chunk["type"] in ("text", "delta"):
                        text = chunk.get("text") or chunk.get("chunk", "")
                        full_response += text
                        yield chunk
                    elif chunk["type"] == "result":
                        full_response = chunk.get("text", full_response)

        # --- File collection ---
        # 1. Extract files from response text (tags, HTML docs, large code blocks)
        from src.pipeline.file_extract import extract_files
        cleaned_text, file_blocks = extract_files(full_response, channel=channel)
        if file_blocks:
            log.info(f"Extracted {len(file_blocks)} file(s) from response text")
            for fb in file_blocks:
                # When a project is active, drafts/code/HTML go to the
                # project's drafts/ directory on disk instead of riding
                # along as an attachment. The channel surfaces a preview
                # so the user knows where it landed.
                wrote_to = await self._write_to_project_drafts(fb.name, fb.content)
                if wrote_to is not None:
                    yield {
                        "type": "file_written",
                        "path": str(wrote_to),
                        "name": wrote_to.name,
                        "preview": _preview(fb.content),
                        "word_count": len(fb.content.split()),
                    }
                else:
                    yield {"type": "file", "name": fb.name, "content": fb.content}
            full_response = cleaned_text
            # Tell channels to replace their accumulated text with the cleaned version
            yield {"type": "text_replace", "text": cleaned_text}

        # 2. Pick up files written to output dir by SDK tool use
        new_files = _new_files(output_dir, pre_files)
        if new_files:
            log.info(f"Found {len(new_files)} file(s) written to output dir")
            for path in new_files:
                try:
                    content = path.read_text(encoding="utf-8")
                    yield {"type": "file", "name": path.name, "content": content}
                    path.unlink()  # clean up after sending
                except Exception as e:
                    log.warning(f"Could not read output file {path}: {e}")

        # Commit the (user, assistant) turn pair atomically. Writing the user
        # turn here — instead of at the top of process() — means a cancelled
        # pipeline leaves no record at all rather than an orphan user row,
        # preserving the invariant that every persisted user turn has a
        # matching assistant turn the user actually saw.
        self.session_engine.add_turn(user_turn)
        assistant_turn = Turn(
            session_id=self._session_id,
            channel=channel,
            user_id=user_id,
            role="assistant",
            content=full_response,
            metadata={
                "triage_intent": triage_result.intent,
                "triage_complexity": triage_result.complexity,
                "search_fragments": len(search_result.fragments),
                "search_time_ms": search_result.search_time_ms,
                "reflection_confidence": reflection_result.confidence,
                "reflection_action": reflection_result.action,
                "reflection_hedging_detected": reflection_result.hedging_detected,
                "reflection_path": reflection_result.path,
                "escalated": escalated,
            },
        )
        self.session_engine.add_turn(assistant_turn)

        # Phase G.2.b: run the profile extractor on health turns after
        # the response lands. Proposals append to profile.pending_updates
        # and a confirmation footer streams to the user. Failures
        # swallowed — extractor must not break the turn.
        if h_route.is_health and h_route.path != "decline_phi" and full_response:
            try:
                async for chunk in self._run_profile_extractor(
                    user_message=message,
                    assistant_response=full_response,
                ):
                    yield chunk
            except Exception as e:
                log.warning("Profile extractor hook failed: %s", e)

        # Health audit — only when the route fired. Captures what the
        # turn did and how (route, sources, model, latency) but never
        # the prompt or response content. The spec is explicit on this:
        # "you reconstruct what happened from the route and source
        # list, not from a prompt transcript."
        if h_route.is_health and h_route.path != "decline_phi":
            self.health_audit.write(HealthAuditRow(
                route=h_route.path,
                triage_phi_class=triage_result.phi_class,
                triage_health_topic=triage_result.health_topic,
                sources_queried=[s.name for s in self.health_retrieval.sources],
                sources_returned=_summarize_sources_returned(health_evidence),
                llm_provider="bedrock" if h_route.use_baa_llm else "anthropic",
                llm_model=self.config.model_main,
                latency_ms=int((time.time() - start) * 1000),
                # token counts not yet wired in this codebase; Phase 4
                # could pull them from the LLM client's usage events.
            ))

        # Write-back: promote high-retrieval fragments
        if assembly_result.promote_candidates:
            changed = promote_fragments(assembly_result.promote_candidates, self.memory_store)
            if changed:
                log.info("Write-back promoted fragments, will reconnect on next turn")
                # Re-index MEMORY.md since write-back just changed it
                try:
                    await self.memory_index.index_file(self.memory_store.memory_path, force=True)
                except Exception as e:
                    log.warning(f"Memory re-index after write-back failed: {e}")

        # Auto-interval breadcrumb — fires when the per-process counter
        # hit its threshold during this turn. Deferred from mid-stream
        # so it never interrupts the user-visible response.
        if self._breadcrumb_due:
            self._breadcrumb_due = False
            self._write_breadcrumb("auto_interval")

        # Check for compaction — yield any events it produces so the channel
        # can surface them (the user deserves to know their context was reset).
        if self.session_engine.needs_compaction(self._session_id):
            async for chunk in self._compact():
                yield chunk

        elapsed_ms = int((time.time() - start) * 1000)

        # Yield metadata
        yield {
            "type": "metadata",
            "pipeline": PipelineMetadata(
                triage=triage_result,
                search=search_result,
                reflection=reflection_result,
                escalated=escalated,
                escalated_model=self.config.model_escalation if escalated else "",
                total_time_ms=elapsed_ms,
                skill_overrides=dict(turn_skill.pipeline) if turn_skill else {},
            ),
        }

    async def _index_workspace(self) -> None:
        """Index workspace markdown files into the memory fragment store.

        Runs on every startup. Files are skipped if unchanged since last index
        (mtime comparison). Requires OPENAI_API_KEY — silently skips if absent.
        """
        if not self.config.openai_api_key:
            log.info("OpenAI API key not set — skipping memory indexing")
            return

        from datetime import date, timedelta

        files_to_index = [
            self.memory_store.identity_path,
            self.memory_store.memory_path,
        ]
        today = date.today()
        for i in range(3):  # today + 2 days back
            daily = self.memory_store.daily_path(today - timedelta(days=i))
            if daily.exists():
                files_to_index.append(daily)

        # Cross-session summaries (pipeline 3.1). Indexing the whole
        # directory keeps the topic-match retrieval path automatic —
        # "do you remember when we discussed pipeline reflection?"
        # will surface the relevant summary via normal search.
        if self.memory_store.sessions_dir.exists():
            files_to_index.extend(self.memory_store.sessions_dir.glob("*.md"))

        total = 0
        for path in files_to_index:
            if not path.exists():
                continue
            try:
                n = await self.memory_index.index_file(path)
                if n > 0:
                    log.info(f"Indexed {n} fragments from {path.name}")
                total += n
            except Exception as e:
                log.warning(f"Failed to index {path.name}: {e}")

        if total > 0:
            log.info(f"Memory index updated: {total} new/changed fragments")
        else:
            log.debug("Memory index: all workspace files up to date")

    async def _compact(self):
        """Compact old turns into a summary.

        Yields a `{"type": "compaction"}` chunk so the channel can tell the
        user their earlier context was archived — silent compaction leaves
        the user wondering why the bot "forgot" what was just discussed.

        After reconnect, re-injects the summary + retained recent turns
        into the LLM session so continuity survives the context reset.
        Also indexes the summary into the fragment store so future search
        can recover it across sessions.
        """
        from src.llm.client import SingleTurnClient

        # Pre-compaction breadcrumb — discipline paper § 5.1's strongest
        # finding (100% capture rate). Fires before any heavy work so
        # even if compaction itself fails the breadcrumb has landed.
        self._write_breadcrumb("pre_compaction")

        # Importance-aware split: substantive turns (complex triage or
        # high reflection confidence) stay verbatim past the rollup
        # boundary; only conversational chitchat gets summarized.
        to_summarize, to_keep_verbatim, recent_turns = (
            self.session_engine.get_turns_for_compaction(self._session_id)
        )
        if not to_summarize and not to_keep_verbatim:
            return

        log.info(
            f"Compacting {len(to_summarize)} turn(s); "
            f"keeping {len(to_keep_verbatim)} verbatim past rollup; "
            f"recent {len(recent_turns)} unchanged"
        )

        if not to_summarize:
            # Nothing actually rolls up — every old turn was important.
            # Skip the summarizer call entirely; the session stays as-is.
            return

        # Memory flush: extract facts before compaction
        facts_text = "\n".join(
            f"{t.role}: {t.content}" for t in to_summarize
        )
        self.memory_store.append_daily(
            f"[Session {self._session_id} compaction notes]\n"
            f"Compacted {len(to_summarize)} turns at {datetime.now().isoformat()}"
            + (f"; kept {len(to_keep_verbatim)} important turn(s) verbatim" if to_keep_verbatim else "")
        )

        # Summarize via Sonnet. The "kept verbatim" preamble tells the
        # model that some substantive turns are being preserved alongside
        # this summary, so it shouldn't try to capture everything — just
        # the conversational glue between the kept turns.
        verbatim_note = (
            "\n\nNote: some important turns from this period are being preserved "
            "verbatim outside this summary; you don't need to capture them. "
            "Focus on conversational context that needs compression."
            if to_keep_verbatim else ""
        )
        compactor = SingleTurnClient(
            model=self.config.model_compaction,
            auth_mode=self.config.auth_mode,
            api_key=self.config.anthropic_api_key,
            cli_path=self.config.cli_path,
            workspace_dir=str(self.config.workspace_dir),
        )
        summary = await compactor.query(
            "Summarize this conversation concisely, preserving key facts, decisions, and context "
            "that would be needed to continue the conversation naturally. "
            "Focus on what was discussed and any conclusions reached." + verbatim_note,
            facts_text[:8000],
        )

        # Only the to_summarize turns get replaced; to_keep_verbatim turns
        # stay in the session table untouched.
        old_ids = [t.id for t in to_summarize if t.id is not None]
        self.session_engine.replace_with_summary(self._session_id, old_ids, summary)

        # Index the summary into the fragment store so it's searchable in the
        # future — even from a different session after a restart.
        if self.memory_index and self.config.openai_api_key:
            try:
                await self.memory_index.index_text(
                    summary,
                    source=f"session:{self._session_id}:summary",
                )
            except Exception as e:
                log.warning(f"Failed to index compaction summary: {e}")

        # Reconnect with fresh stable context (compaction wrote to daily notes).
        # This wipes the LLM's live session — everything said so far is gone
        # from its working memory.
        stable = self.memory_store.assemble_stable_context(channel=self._channel)
        await self.llm.reconnect(stable)
        self._stable_mtime = self.memory_store.stable_context_mtime(channel=self._channel)

        # Restore conversational continuity: re-inject the summary + retained
        # recent turns into the new live session. Without this, the bot has
        # zero knowledge of what was just discussed and the next user message
        # lands in a cold context.
        try:
            remaining = self.session_engine.get_turns(self._session_id)
            if remaining:
                await self._inject_history(remaining)
        except Exception as e:
            log.warning(f"Post-compaction history injection failed: {e}")

        # Tell the channel so it can surface a status message to the user.
        # `turns_kept` includes both recent turns and any turns preserved
        # verbatim past the rollup (importance-aware compaction); from the
        # user's perspective these are all "still in context."
        yield {
            "type": "compaction",
            "turns_compacted": len(to_summarize),
            "turns_kept": len(recent_turns) + len(to_keep_verbatim),
            "turns_kept_important": len(to_keep_verbatim),
        }

    async def _inject_history(self, turns: list[Turn]) -> None:
        """Inject prior conversation history into the LLM session.

        Formats recent turns as a recap and sends it as a silent primer
        so the SDK session has context from before the restart.
        """
        # Build a concise recap from the last N turns (cap to avoid blowing budget)
        recent = turns[-20:]  # last 20 turns max
        lines = ["[Conversation history resumed from previous session]"]
        for t in recent:
            if t.role == "system" and t.metadata.get("type") == "compaction_summary":
                lines.append(f"[Summary] {t.content[:500]}")
            elif t.role in ("user", "assistant"):
                label = "User" if t.role == "user" else "Microwave"
                lines.append(f"{label}: {t.content[:300]}")

        recap = "\n".join(lines)

        # Send as a message and consume the response (don't show to user)
        try:
            async for chunk in self.llm.send(
                f"{recap}\n\n[You are resuming this conversation. Acknowledge silently — "
                f"do not repeat or summarize what was said. Just say \"Session resumed.\" "
                f"and nothing else.]"
            ):
                pass  # consume and discard the response
        except Exception as e:
            log.warning(f"History injection failed (non-fatal): {e}")

    def mark_llm_for_reset(self) -> None:
        """Flag the LLM session as needing a resync on the next turn.

        Called by channels (Signal in particular) when they cancel an
        in-flight pipeline for addendum-merge. Cancellation can leave
        the SDK's internal conversation history with phantom user/
        assistant turns that don't match what's been committed to the
        session engine; the next process() call resets before doing
        any LLM work so the next turn lands in a clean context.
        """
        self._needs_llm_reset = True

    async def _reset_llm_after_cancel(self) -> None:
        """Rebuild a clean LLM session matching committed history.

        Reconnects with the current stable prompt (so identity, memory,
        channel rules, active bible all stay in effect) and replays the
        committed turn history into the new SDK session. Any phantom
        messages from a cancelled `llm.send()` are discarded along with
        the old session.
        """
        if self.llm is None or self.session_engine is None:
            return
        bible_path = self._active_bible_path()
        stable_prompt = self.memory_store.assemble_stable_context(
            channel=self._channel, bible_path=bible_path
        )
        await self.llm.reconnect(stable_prompt)
        self._stable_mtime = self.memory_store.stable_context_mtime(
            channel=self._channel, bible_path=bible_path
        )
        if self._session_id:
            try:
                turns = self.session_engine.get_turns(self._session_id)
                if turns:
                    await self._inject_history(turns)
            except Exception as e:
                log.warning(f"History re-injection after cancel failed: {e}")

    # --- Project management (called by channel command handlers) ---

    def _active_bible_path(self):
        """Return the active project's BIBLE.md Path, or None."""
        if self._active_project is None:
            return None
        return self._active_project.bible_path

    async def set_active_project(self, name: str) -> Project:
        """Activate a writing project. Auto-activates its declared skill,
        marks the LLM session for reconnect on next turn, and indexes the
        project's text files into the fragment store so they're retrievable.
        """
        if not self.project_loader:
            raise RuntimeError("Orchestrator not started")
        project = self.project_loader.load(name)
        self._active_project = project
        self._project_changed = True

        # Auto-activate the project's declared skill
        if project.skill and self.skill_loader:
            try:
                self._active_skill = self.skill_loader.load(project.skill)
                log.info(f"Auto-activated skill {project.skill!r} for project {name!r}")
            except SkillNotFound:
                log.warning(
                    f"Project {name!r} declares skill {project.skill!r} "
                    f"but it's not on disk; ignoring"
                )

        # Index project files so retrieval can surface relevant chunks.
        # Best-effort — don't let an indexing failure block activation.
        try:
            await self._index_project_files(project)
        except Exception as e:
            log.warning(f"Project file indexing failed for {name!r}: {e}")

        log.info(f"Active project set: {name!r} ({project.type})")
        return project

    def clear_active_project(self) -> None:
        if self._active_project is not None:
            log.info(f"Active project cleared (was {self._active_project.name!r})")
        self._active_project = None
        self._project_changed = True
        # Don't auto-clear active_skill — user may want the skill kept on
        # for a follow-up turn even after leaving the project.

    def get_active_project(self) -> Project | None:
        return self._active_project

    def list_projects(self) -> list[Project]:
        if not self.project_loader:
            return []
        return self.project_loader.list_all()

    async def _write_to_project_drafts(self, suggested_name: str, content: str):
        """Write a file the LLM produced to the active project's drafts/.

        Returns the Path written to, or None if no project is active.
        Picks a sensible filename: novels number chapter-NN.md sequentially,
        screenplays append into the single FOUNTAIN file when it already
        exists, blogs default to draft.md (overwriting if the LLM wrote
        the same name).
        """
        project = self._active_project
        if project is None or project.drafts_dir is None:
            return None
        project.drafts_dir.mkdir(parents=True, exist_ok=True)

        target = self._pick_draft_filename(project, suggested_name)
        if project.type == "screenplay" and target.suffix == ".fountain":
            # Append additional scenes to the same file rather than
            # creating screenplay-2.fountain etc.
            existing = ""
            if target.is_file():
                existing = target.read_text(encoding="utf-8").rstrip()
            joined = (existing + "\n\n" + content.strip()) if existing else content.strip()
            target.write_text(joined + "\n", encoding="utf-8")
        else:
            target.write_text(content, encoding="utf-8")

        log.info(f"Wrote {target} ({len(content.split())} words)")
        # Re-index so the new content is searchable on the next turn
        try:
            if self.memory_index and self.config.openai_api_key:
                await self.memory_index.index_file(target, force=True)
        except Exception as e:
            log.warning(f"Failed to index new draft {target.name}: {e}")
        return target

    def _pick_draft_filename(self, project: Project, suggested: str):
        """Pick a path under project.drafts_dir for new content.

        Honors the LLM's suggested name when it's project-appropriate;
        otherwise picks a default based on project type.
        """
        from pathlib import Path
        drafts = project.drafts_dir
        suggested_path = Path(suggested).name  # strip any directory parts

        if project.type == "novel":
            # Force chapter-NN.md naming. If the LLM suggested chapter-04.md,
            # honor it; otherwise pick the next sequence number.
            import re as _re
            m = _re.match(r"chapter[-_ ]?(\d+)", suggested_path, _re.IGNORECASE)
            if m:
                return drafts / f"chapter-{int(m.group(1)):02d}.md"
            existing = sorted(drafts.glob("chapter-*.md"))
            next_n = len(existing) + 1
            return drafts / f"chapter-{next_n:02d}.md"

        if project.type == "screenplay":
            # Single FOUNTAIN file per screenplay. New scenes append.
            return drafts / "screenplay.fountain"

        # Blog and anything else: honor the suggested name, default to draft.md.
        if suggested_path and suggested_path.lower() != "response.md":
            return drafts / suggested_path
        return drafts / "draft.md"

    async def _index_project_files(self, project: Project) -> None:
        """Chunk and index the project's drafts, bible, and outline so
        memory search can surface them when the user references the work."""
        if not self.memory_index or not self.config.openai_api_key:
            return
        files: list = []
        if project.bible_path and project.bible_path.is_file():
            files.append(project.bible_path)
        if project.outline_path and project.outline_path.is_file():
            files.append(project.outline_path)
        if project.drafts_dir and project.drafts_dir.is_dir():
            for p in project.drafts_dir.iterdir():
                if p.is_file() and p.suffix in (".md", ".fountain", ".txt"):
                    files.append(p)
        for path in files:
            try:
                n = await self.memory_index.index_file(path)
                if n > 0:
                    log.info(f"Indexed {n} fragments from {path.name}")
            except Exception as e:
                log.warning(f"Failed to index project file {path.name}: {e}")

    # --- Skill management (called by channel command handlers) ---

    def set_active_skill(self, name: str) -> Skill:
        """Activate a named skill for subsequent turns. Raises SkillNotFound."""
        if not self.skill_loader:
            raise RuntimeError("Orchestrator not started")
        skill = self.skill_loader.load(name)
        self._active_skill = skill
        log.info(f"Active skill set: {name!r}")
        return skill

    def clear_active_skill(self) -> None:
        if self._active_skill is not None:
            log.info(f"Active skill cleared (was {self._active_skill.name!r})")
        self._active_skill = None

    def get_active_skill(self) -> Skill | None:
        return self._active_skill

    def list_skills(self) -> list[Skill]:
        if not self.skill_loader:
            return []
        return self.skill_loader.list_all()

    def get_pipeline_stats(self, user_id: str = "default", channel: str = "telegram", limit: int = 10) -> dict:
        """Return aggregated pipeline stats for recent assistant turns."""
        return self.session_engine.get_recent_stats(channel, user_id, limit)

    # --- Phase G.2.b: profile extractor hooks ---

    async def _handle_profile_reply(self, message: str) -> list[dict]:
        """If the previous response surfaced proposals, parse this
        message as a yes/no/indices reply and apply accepted ones.

        Returns a list of `{"type": "profile_update", ...}` chunks to
        yield BEFORE the main pipeline runs. The user's message itself
        flows through the pipeline unchanged — they may have asked a
        real question in the same turn as their confirmation reply.
        """
        if not self._last_shown_profile_proposals:
            return []
        if self.session_engine is None or self.session_engine.conn is None:
            return []

        from src.health.profile.confirmation import (
            ProposalReplyIntent,
            apply_proposal,
            mark_proposals,
            parse_user_reply,
        )
        from src.health.profile.store import (
            StaleProfileError,
            load_profile,
            save_profile,
        )

        shown = list(self._last_shown_profile_proposals)
        # Clear the shown list FIRST — even if parsing fails or the
        # reply is "no", the proposals shouldn't be re-shown on the
        # next turn. They live in pending_updates with whatever status
        # we set them to.
        self._last_shown_profile_proposals = []

        reply = parse_user_reply(message, pending_count=len(shown))
        if reply.intent == ProposalReplyIntent.NORMAL:
            return []

        key_source = self.config.health.phi_encryption_key_source
        try:
            loaded = load_profile(
                self.session_engine.conn, key_source=key_source,
            )
        except Exception as e:
            log.warning("Profile load for reply failed: %s", e)
            return []

        # Match shown proposals against the in-DB pending_updates by id.
        # If a proposal isn't in the queue anymore (manually cleared,
        # auto-expired), skip it gracefully.
        in_queue_by_id = {p.id: p for p in loaded.profile.pending_updates}

        if reply.intent == ProposalReplyIntent.YES:
            chosen = [p for p in shown if p.id in in_queue_by_id]
        elif reply.intent == ProposalReplyIntent.NO:
            chosen = []
            mark_proposals(
                loaded.profile,
                [p.id for p in shown if p.id in in_queue_by_id],
                "rejected",
            )
        elif reply.intent == ProposalReplyIntent.INDICES:
            chosen = [
                shown[i - 1] for i in reply.indices
                if shown[i - 1].id in in_queue_by_id
            ]
            # The un-chosen ones get rejected.
            chosen_ids = {p.id for p in chosen}
            un_chosen = [
                p.id for p in shown
                if p.id in in_queue_by_id and p.id not in chosen_ids
            ]
            mark_proposals(loaded.profile, un_chosen, "rejected")
        else:
            return []

        applied_chunks: list[dict] = []
        if chosen:
            for prop in chosen:
                in_queue_prop = in_queue_by_id[prop.id]
                if apply_proposal(loaded.profile, in_queue_prop):
                    applied_chunks.append({
                        "type": "profile_update",
                        "section": prop.target_section,
                        "operation": prop.operation,
                        "summary": prop.extractor_reasoning or prop.target_section,
                    })
            mark_proposals(
                loaded.profile, [p.id for p in chosen], "accepted",
            )

        try:
            save_profile(
                self.session_engine.conn,
                loaded.profile,
                expected_version=loaded.version,
                key_source=key_source,
                operation="confirm",
                section="pending_updates",
            )
        except StaleProfileError:
            log.warning("Profile-reply save raced; user may need to retry.")

        if applied_chunks:
            # Surface as text too so non-rich channels see it.
            text = "✓ Updated profile:\n" + "\n".join(
                f"  • {c['summary']}" for c in applied_chunks
            )
            applied_chunks.insert(0, {"type": "delta", "text": text + "\n\n"})
        elif reply.intent == ProposalReplyIntent.NO:
            applied_chunks.append({
                "type": "delta",
                "text": "Got it — no profile changes.\n\n",
            })
        return applied_chunks

    async def _run_profile_extractor(
        self, *, user_message: str, assistant_response: str,
    ):
        """Run the extractor on the turn that just landed.

        Yields a single delta chunk (the confirmation footer) when
        there are new proposals. Persists the proposals to
        pending_updates so they survive process restarts even if the
        user navigates away before replying.
        """
        if self.session_engine is None or self.session_engine.conn is None:
            return

        from src.health.profile.extractor import (
            persist_proposals,
            run_extractor,
            summarize_structure,
        )
        from src.health.profile.confirmation import format_confirmation_footer
        from src.health.profile.store import load_profile
        from src.llm.selector import get_stage_callable

        key_source = self.config.health.phi_encryption_key_source
        try:
            loaded = load_profile(
                self.session_engine.conn, key_source=key_source,
            )
        except Exception as e:
            log.warning("Profile load before extraction failed: %s", e)
            return

        structure = summarize_structure(loaded.profile)
        extractor_call = get_stage_callable(
            "profile_extractor",
            fallback_model=self.config.model_triage,
            auth_mode=self.config.auth_mode,
            api_key=self.config.anthropic_api_key,
            cli_path=self.config.cli_path,
            workspace_dir=str(self.config.workspace_dir),
        )

        active_project = (
            self._active_project.name if self._active_project else None
        )
        proposals = await run_extractor(
            user_message=user_message,
            assistant_response=assistant_response,
            profile_structure=structure,
            llm_call=extractor_call,
            active_project=active_project,
            source_turn_id=self._session_id or "unknown",
        )
        if not proposals:
            return

        try:
            persist_proposals(
                conn=self.session_engine.conn,
                proposals=proposals,
                key_source=key_source,
            )
        except Exception as e:
            log.warning("persist_proposals failed: %s", e)
            return

        # Track for the next-turn reply parser.
        self._last_shown_profile_proposals = list(proposals)

        footer = format_confirmation_footer(proposals)
        if footer:
            yield {"type": "delta", "text": footer}

    def _write_breadcrumb(self, trigger: str) -> None:
        """Persist a breadcrumb capturing current execution state.

        Called from three hook points:
          - top of `_compact()` for `pre_compaction`
          - top of `new_session()` for `pre_reset`
          - end-of-turn in `process()` for `auto_interval`

        Never raises. Counter / session-state issues are logged; a
        breadcrumb failure must not break the user-visible turn.
        """
        if self.session_engine is None or self.session_engine.conn is None:
            log.debug("Breadcrumb skipped (session engine not connected)")
            return
        try:
            from src.memory.breadcrumbs import write_breadcrumb
            # Inline COUNT to avoid adding a method to SessionEngine for
            # something only the breadcrumb path needs.
            turn_count = 0
            if self._session_id:
                rows = list(self.session_engine.conn.execute(
                    "SELECT COUNT(*) AS n FROM turns WHERE session_id = ?",
                    (self._session_id,),
                ))
                turn_count = int(rows[0]["n"]) if rows else 0
            write_breadcrumb(
                self.session_engine.conn,
                trigger=trigger,
                session_key=self._session_id or "unknown",
                turn_count=turn_count,
                tool_call_count=self._session_tool_calls,
                recent_tools=self._tool_call_counter.recent_tools,
                active_project=(
                    self._active_project.name if self._active_project else None
                ),
                active_skill=(
                    self._active_skill.name if self._active_skill else None
                ),
            )
        except Exception as e:
            log.warning("Breadcrumb (%s) failed: %s", trigger, e)

    async def new_session(self) -> str:
        """Start a completely fresh session. Returns the new session ID.

        Reconnects the LLM so prior conversation state is wiped from the SDK.
        History stays in SQLite — this just stops continuing from it.

        Before resetting, runs the close hook on the outgoing session so
        a summary lands in `workspace/memory/sessions/`. The summary is
        what lets the next session retrieve "what we were just doing"
        without replaying the full transcript.
        """
        # Pre-reset breadcrumb — captures execution state before the
        # context is destroyed. Runs *before* the close hook so a Sonnet
        # failure in the summary path doesn't prevent the breadcrumb.
        self._write_breadcrumb("pre_reset")
        await self._close_session()

        self._session_id = self.session_engine.new_session_id()
        stable_prompt = self.memory_store.assemble_stable_context(channel=self._channel)
        await self.llm.reconnect(stable_prompt)
        self._stable_mtime = self.memory_store.stable_context_mtime(channel=self._channel)
        # Fresh session: re-arm cold-start summary injection and re-stamp
        # the start time. Next process() call surfaces recent sessions.
        self._session_started_at = datetime.now()
        self._first_turn_pending = True
        # Reset tool-call counter so the new session doesn't inherit the
        # old one's breadcrumb cadence.
        self._tool_call_counter.reset()
        self._session_tool_calls = 0
        log.info(f"Fresh session started: {self._session_id}")
        return self._session_id

    def _build_session_recap(self, n: int = 3) -> str:
        """Format the most recent session summaries as a single block.

        Returns "" when no summaries exist yet (fresh install) — caller
        skips the block entirely so dynamic context stays clean.

        Each summary is shown with its date and topic header so the
        model knows it's reading prior-session context, not current-
        conversation content. Bodies are not truncated — they're 200
        words by design, so three of them is ~600 words / ~150 tokens
        of cold-start grounding.
        """
        try:
            entries = self.memory_store.load_recent_session_summaries(n=n)
        except Exception as e:
            log.debug(f"Could not load session summaries for recap: {e}")
            return ""
        if not entries:
            return ""
        parts = ["[Recent session summaries — for continuity, not current task]"]
        for e in entries:
            header = f"— {e.get('ended') or e.get('started') or ''} · {e.get('topic') or 'general'}"
            parts.append(f"{header}\n{e['body'].strip()}")
        return "\n\n".join(parts)

    async def _close_session(self) -> None:
        """Generate + persist a summary of the current session.

        Safe to call when no session is active or when the session is
        too short to summarize — `generate_session_summary` returns None
        and we exit quietly. Failures don't propagate because the close
        hook fires from `/new` and `stop()`; a Sonnet error must not
        block a user's reset or a clean shutdown.
        """
        if not self._session_id or not self._session_started_at:
            return
        if not self.session_engine:
            return

        try:
            turns = self.session_engine.get_turns(self._session_id)
        except Exception as e:
            log.warning(f"Close hook could not read turns: {e}")
            return

        from src.pipeline.session_summary import generate_session_summary

        try:
            summary = await generate_session_summary(
                turns,
                model=self.config.model_compaction,
                auth_mode=self.config.auth_mode,
                api_key=self.config.anthropic_api_key,
                cli_path=self.config.cli_path,
                workspace_dir=str(self.config.workspace_dir),
            )
        except Exception as e:
            log.warning(f"Session summary generation raised: {e}")
            return

        if summary is None:
            return

        ended_at = datetime.now()
        try:
            path = self.memory_store.save_session_summary(
                body=summary.body,
                started_at=self._session_started_at,
                ended_at=ended_at,
                topic_slug=summary.topic_slug,
                project=(self._active_project.name if self._active_project else None),
                turn_count=summary.turn_count,
            )
        except Exception as e:
            log.warning(f"Could not save session summary: {e}")
            return

        log.info(f"Session summary saved: {path.name} ({summary.turn_count} turns)")

        # Index immediately so it's retrievable on the next session start
        # without waiting for the next workspace-index sweep.
        if self.memory_index and self.config.openai_api_key:
            try:
                await self.memory_index.index_file(path, force=True)
            except Exception as e:
                log.debug(f"Could not index fresh session summary: {e}")

    async def stop(self) -> None:
        """Shut down all components."""
        # Close hook: summarize the live session before tearing down.
        # Wrapped so a Sonnet hiccup at shutdown can't strand the LLM
        # connection or DB handles open.
        try:
            await self._close_session()
        except Exception as e:
            log.warning(f"Close-session hook on stop() failed: {e}")
        if self.llm:
            await self.llm.disconnect()
        if self.memory_index:
            self.memory_index.close()
        if self.session_engine:
            self.session_engine.close()
        if getattr(self, "health_audit", None) is not None:
            self.health_audit.close()
        log.info("Orchestrator stopped")


def _verify_workspace(config) -> None:
    """Boot-time invariant check on the workspace.

    Logs the resolved workspace path + the process cwd so any drift
    between WORKSPACE_DIR and the bot's actual disk location is
    obvious in the startup banner (rather than surfacing as a confused
    "file not found" mid-conversation).

    `IDENTITY.md` is required — without it the system prompt has no
    voice/persona to anchor replies, and the bot defaults to a generic
    AI assistant tone that surprises users on a fresh install.
    `MEMORY.md` is recommended but not strictly required; we warn but
    don't crash on its absence so a clean install isn't blocked by it.
    """
    import os as _os
    workspace = config.workspace_dir.resolve()
    cwd = Path(_os.getcwd()).resolve()
    log.info(f"[startup] workspace = {workspace}")
    log.info(f"[startup] cwd       = {cwd}")
    if workspace != cwd and cwd not in workspace.parents:
        log.info(
            "[startup] note: bot launched from a different directory than "
            "WORKSPACE_DIR — relative file writes from the LLM will resolve "
            "under workspace/, not cwd."
        )

    identity = config.identity_path
    if not identity.is_file():
        # Fatal — IDENTITY.md is the system prompt's spine.
        raise RuntimeError(
            f"Required workspace file missing: {identity}\n"
            f"Create it with a short voice/persona description before "
            f"starting the bot — see README §Customization → Identity."
        )

    memory = config.memory_path
    if not memory.is_file():
        log.warning(
            f"[startup] {memory.name} not present at {memory.parent} — "
            "bot will run, but won't have any long-term facts to anchor on. "
            "Create it with `touch %s` or write a few starter facts.",
            memory,
        )


def _summarize_sources_returned(evidence: list) -> list[dict]:
    """Group evidence by source for the audit row.

    Stored in the audit as JSON: `[{"name": "pubmed", "count": 3}, ...]`.
    Tells us which sources came back with hits and which were noisy
    on a given query — useful for tuning retrieval and noticing
    silently-broken sources.
    """
    counts: dict[str, int] = {}
    for ev in evidence:
        counts[ev.source] = counts.get(ev.source, 0) + 1
    return [{"name": name, "count": count} for name, count in counts.items()]


def _build_health_sources(config) -> list[EvidenceSource]:
    """Pick the EvidenceSource implementations for a given config.

    Phase 1 ships PubMed and MedlinePlus only. Toggles for openFDA,
    CDC, ClinicalTrials are accepted by config but no-op until those
    sources land in Phase 3 — flipping them on logs at debug level
    instead of erroring, so .env files written for forward compat
    don't break.

    Returns an empty list when health is disabled, which makes the
    retrieval orchestrator a no-op (`search()` returns []).
    """
    h = config.health
    if not h.enabled:
        return []
    sources: list[EvidenceSource] = []
    if h.retrieval_pubmed:
        sources.append(PubMedSource(api_key=h.ncbi_api_key))
    if h.retrieval_medlineplus:
        sources.append(MedlinePlusSource())
    if h.retrieval_openfda:
        log.debug("HEALTH_RETRIEVAL_OPENFDA enabled but no source registered yet")
    if h.retrieval_cdc:
        log.debug("HEALTH_RETRIEVAL_CDC enabled but no source registered yet")
    if h.retrieval_clinicaltrials:
        log.debug("HEALTH_RETRIEVAL_CLINICALTRIALS enabled but no source registered yet")
    return sources


def _preview(content: str, max_chars: int = 240) -> str:
    """Short preview of content the LLM wrote, for the channel reply."""
    text = content.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _snapshot_dir(path: Path) -> dict[str, float]:
    """Snapshot file mtimes in a directory. Returns {name: mtime}."""
    if not path.exists():
        return {}
    return {f.name: f.stat().st_mtime for f in path.iterdir() if f.is_file()}


def _new_files(path: Path, before: dict[str, float]) -> list[Path]:
    """Find files that are new or modified since the snapshot."""
    if not path.exists():
        return []
    result = []
    for f in path.iterdir():
        if not f.is_file():
            continue
        old_mtime = before.get(f.name)
        if old_mtime is None or f.stat().st_mtime > old_mtime:
            result.append(f)
    return result
