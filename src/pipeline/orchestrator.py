"""Pipeline orchestrator — wires triage, search, assembly, reflection together.

This is the core cognitive loop of MicrowaveOS.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
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

        # Memory searcher
        self.searcher = MemorySearcher(self.memory_index, embedder)

        # Session engine
        self.session_engine = SessionEngine(
            self.config.db_path,
            context_limit=self.config.context_window_limit,
            compaction_threshold=self.config.compaction_threshold,
        )
        self.session_engine.connect()

        # LLM client
        self.llm = LLMClient(
            model=self.config.model_main,
            auth_mode=self.config.auth_mode,
            api_key=self.config.anthropic_api_key,
            cli_path=self.config.cli_path,
        )

        # Assemble stable context and connect
        stable_prompt = self.memory_store.assemble_stable_context(channel=self._channel)
        self._stable_mtime = self.memory_store.stable_context_mtime(channel=self._channel)
        await self.llm.connect(stable_prompt)

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
        triage_result = await triage(
            message,
            recent,
            model=self.config.model_triage,
            auth_mode=self.config.auth_mode,
            api_key=self.config.anthropic_api_key,
            cli_path=self.config.cli_path,
        )

        # --- Stage 2: Search ---
        search_result = await search(message, triage_result, self.searcher)

        # --- Stage 3: Assembly ---
        assembly_result = assemble(
            search_result,
            self.memory_store,
            self.memory_index,
            channel=self._channel,
        )

        # Reconnect only if underlying files actually changed on disk
        current_mtime = self.memory_store.stable_context_mtime(channel=self._channel)
        if current_mtime > self._stable_mtime:
            log.info("Stable context files changed on disk, reconnecting LLM")
            await self.llm.reconnect(assembly_result.stable_prompt)
            self._stable_mtime = current_mtime

        # --- LLM Generation ---
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

        # Store assistant turn
        assistant_turn = Turn(
            session_id=self._session_id,
            channel=channel,
            user_id=user_id,
            role="assistant",
            content=full_response,
            metadata={
                "triage_intent": triage_result.intent,
                "search_fragments": len(search_result.fragments),
                "reflection_confidence": reflection_result.confidence,
            },
        )
        self.session_engine.add_turn(assistant_turn)

        # Write-back: promote high-retrieval fragments
        if assembly_result.promote_candidates:
            changed = promote_fragments(assembly_result.promote_candidates, self.memory_store)
            if changed:
                log.info("Write-back promoted fragments, will reconnect on next turn")

        # Check for compaction
        if self.session_engine.needs_compaction(self._session_id):
            await self._compact()

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

    async def _compact(self) -> None:
        """Compact old turns into a summary."""
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

        # Reconnect with fresh stable context (compaction wrote to daily notes)
        stable = self.memory_store.assemble_stable_context(channel=self._channel)
        await self.llm.reconnect(stable)
        self._stable_mtime = self.memory_store.stable_context_mtime(channel=self._channel)

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
