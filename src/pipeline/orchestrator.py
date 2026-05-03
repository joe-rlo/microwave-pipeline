"""Pipeline orchestrator — wires triage, search, assembly, reflection together.

This is the core cognitive loop of MicrowaveOS.
"""

from __future__ import annotations

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
from src.pipeline.reflection import reflect
from src.pipeline.search import search
from src.pipeline.triage import triage
from src.projects import Project, ProjectLoader, ProjectNotFound
from src.session.engine import SessionEngine
from src.session.models import PipelineMetadata, Turn
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

    async def start(self, channel: str = "repl") -> None:
        """Initialize all components and connect."""
        self._channel = channel
        self.config.ensure_dirs()

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

        # LLM client — with sandboxed output directory for file creation
        # and (optionally) the MCP tool bundle wired in.
        self.llm = LLMClient(
            model=self.config.model_main,
            auth_mode=self.config.auth_mode,
            api_key=self.config.anthropic_api_key,
            cli_path=self.config.cli_path,
            output_dir=str(self.config.output_dir),
            tool_bundle=self.tool_bundle,
        )

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
            else:
                log.info(f"Resumed session {self._session_id} (empty)")
        else:
            self._session_id = self.session_engine.new_session_id()
            log.info(f"New session {self._session_id}")

    async def process(
        self,
        message: str,
        user_id: str = "default",
        channel: str = "repl",
    ) -> AsyncIterator[dict]:
        """Process a user message through the full cognitive pipeline.

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

        # --- Stage 2: Search ---
        search_result = await search(message, triage_result, self.searcher)

        # --- Stage 3: Assembly ---
        bible_path = self._active_bible_path()
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

        # Escalate to Opus+thinking for complex tasks
        escalated = triage_result.complexity == "complex"
        if escalated:
            log.info(
                f"Escalating to {self.config.model_escalation} "
                f"(effort={self.config.escalation_effort}) for complex task"
            )
            await self.llm.escalate(
                model=self.config.model_escalation,
                effort=self.config.escalation_effort,
            )

        full_response = ""
        try:
            async for chunk in self.llm.send(
                message,
                memory_context=assembly_result.memory_context or None,
            ):
                if chunk["type"] in ("text", "delta"):
                    text = chunk.get("text") or chunk.get("chunk", "")
                    full_response += text
                    yield chunk
                elif chunk["type"] == "result":
                    full_response = chunk.get("text", full_response)
        finally:
            if escalated:
                await self.llm.de_escalate()

        # --- Stage 4: Reflection ---
        reflection_result = await reflect(
            full_response,
            context=assembly_result.memory_context,
            model=self.config.model_reflection,
            auth_mode=self.config.auth_mode,
            api_key=self.config.anthropic_api_key,
            cli_path=self.config.cli_path,
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
                reflection_result.memory_gap, broader_triage, self.searcher
            )

            if retry_search.fragments:
                retry_assembly = assemble(
                    retry_search, self.memory_store, self.memory_index
                )

                # Re-generate with better context
                full_response = ""
                async for chunk in self.llm.send(
                    message,
                    memory_context=retry_assembly.memory_context or None,
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
                "escalated": escalated,
            },
        )
        self.session_engine.add_turn(assistant_turn)

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

        old_turns, recent_turns = self.session_engine.get_turns_for_compaction(self._session_id)
        if not old_turns:
            return

        log.info(f"Compacting {len(old_turns)} turns")

        # Memory flush: extract facts before compaction
        facts_text = "\n".join(
            f"{t.role}: {t.content}" for t in old_turns
        )
        self.memory_store.append_daily(
            f"[Session {self._session_id} compaction notes]\n"
            f"Compacted {len(old_turns)} turns at {datetime.now().isoformat()}"
        )

        # Summarize via Sonnet
        compactor = SingleTurnClient(
            model=self.config.model_compaction,
            auth_mode=self.config.auth_mode,
            api_key=self.config.anthropic_api_key,
            cli_path=self.config.cli_path,
        )
        summary = await compactor.query(
            "Summarize this conversation concisely, preserving key facts, decisions, and context "
            "that would be needed to continue the conversation naturally. "
            "Focus on what was discussed and any conclusions reached.",
            facts_text[:8000],
        )

        old_ids = [t.id for t in old_turns if t.id is not None]
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
        yield {
            "type": "compaction",
            "turns_compacted": len(old_turns),
            "turns_kept": len(recent_turns),
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

    async def new_session(self) -> str:
        """Start a completely fresh session. Returns the new session ID.

        Reconnects the LLM so prior conversation state is wiped from the SDK.
        History stays in SQLite — this just stops continuing from it.
        """
        self._session_id = self.session_engine.new_session_id()
        stable_prompt = self.memory_store.assemble_stable_context(channel=self._channel)
        await self.llm.reconnect(stable_prompt)
        self._stable_mtime = self.memory_store.stable_context_mtime(channel=self._channel)
        log.info(f"Fresh session started: {self._session_id}")
        return self._session_id

    async def stop(self) -> None:
        """Shut down all components."""
        if self.llm:
            await self.llm.disconnect()
        if self.memory_index:
            self.memory_index.close()
        if self.session_engine:
            self.session_engine.close()
        log.info("Orchestrator stopped")


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
