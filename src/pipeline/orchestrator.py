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
from src.session.engine import SessionEngine
from src.session.models import PipelineMetadata, Turn
from src.skills import Skill, SkillLoader, SkillNotFound

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

        # LLM client — with sandboxed output directory for file creation
        self.llm = LLMClient(
            model=self.config.model_main,
            auth_mode=self.config.auth_mode,
            api_key=self.config.anthropic_api_key,
            cli_path=self.config.cli_path,
            output_dir=str(self.config.output_dir),
        )

        # Assemble stable context and connect
        stable_prompt = self.memory_store.assemble_stable_context(channel=self._channel)
        self._stable_mtime = self.memory_store.stable_context_mtime(channel=self._channel)
        await self.llm.connect(stable_prompt)

        # Index workspace files into the memory fragment store
        self._index_workspace()

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

        # Store user turn
        user_turn = Turn(
            session_id=self._session_id,
            channel=channel,
            user_id=user_id,
            role="user",
            content=message,
        )
        self.session_engine.add_turn(user_turn)

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
        assembly_result = assemble(
            search_result,
            self.memory_store,
            self.memory_index,
            channel=self._channel,
            output_dir=str(self.config.output_dir),
            active_skill=turn_skill,
        )

        # Reconnect only if underlying files actually changed on disk
        current_mtime = self.memory_store.stable_context_mtime(channel=self._channel)
        if current_mtime > self._stable_mtime:
            log.info("Stable context files changed on disk, reconnecting LLM")
            await self.llm.reconnect(assembly_result.stable_prompt)
            self._stable_mtime = current_mtime

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

        # Store assistant turn
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
                    self.memory_index.index_file(self.memory_store.memory_path, force=True)
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

    def _index_workspace(self) -> None:
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
                n = self.memory_index.index_file(path)
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
                self.memory_index.index_text(
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
