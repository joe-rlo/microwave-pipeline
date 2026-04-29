"""REPL channel — stdin/stdout for development.

No streaming complexity. Pipeline metadata printed after each response for debugging.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from src.channels.base import Channel
from src.pipeline.orchestrator import Orchestrator
from src.projects.bible import handle_bible_command
from src.projects.chat import handle_project_command
from src.skills.chat import handle_skill_command

log = logging.getLogger(__name__)


class REPLChannel(Channel):
    def __init__(self, orchestrator: Orchestrator, show_metadata: bool = True):
        super().__init__(orchestrator)
        self.show_metadata = show_metadata
        self._running = False

    async def start(self) -> None:
        self._running = True
        print("MicrowaveOS REPL — type 'exit' or 'quit' to stop")
        print("---")

        while self._running:
            try:
                # Read input
                if sys.stdin.isatty():
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("\nyou: ")
                    )
                else:
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, sys.stdin.readline
                    )
                    if not line:
                        break
                    line = line.strip()

                if not line:
                    continue

                if line.lower() in ("exit", "quit", "/quit"):
                    break

                # Handle /new command — fresh session
                if line == "/new":
                    new_id = await self.orchestrator.new_session()
                    print(f"\n  Fresh session started: {new_id}")
                    continue

                # Handle debug command
                if line == "/debug" and hasattr(self, "_last_metadata"):
                    self._print_metadata(self._last_metadata)
                    continue

                # Skill / project / bible commands short-circuit the pipeline.
                for handler in (
                    handle_skill_command,
                    handle_project_command,
                    handle_bible_command,
                ):
                    reply = handler(line, self.orchestrator)
                    if reply is not None:
                        print(f"\n{reply}")
                        break
                else:
                    reply = None
                if reply is not None:
                    continue

                # Process through pipeline
                print("\nmicrowave: ", end="", flush=True)
                async for chunk in self.orchestrator.process(line, user_id="repl", channel="repl"):
                    if chunk["type"] == "delta":
                        print(chunk["text"], end="", flush=True)
                    elif chunk["type"] == "text":
                        print(chunk["chunk"], end="", flush=True)
                    elif chunk["type"] == "file":
                        # Write file to current directory
                        out_path = Path.cwd() / chunk["name"]
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(chunk["content"])
                        print(f"\n  [wrote {out_path}]", flush=True)
                    elif chunk["type"] == "file_written":
                        # Active project — orchestrator already wrote to disk.
                        print(
                            f"\n  [✓ wrote {chunk.get('path')} "
                            f"({chunk.get('word_count', 0):,} words)]",
                            flush=True,
                        )
                    elif chunk["type"] == "metadata":
                        self._last_metadata = chunk["pipeline"]
                        if self.show_metadata:
                            print()
                            self._print_metadata(chunk["pipeline"])

                print()  # newline after response

            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                log.error(f"REPL error: {e}", exc_info=True)
                print(f"\n[error: {e}]")

        print("\nGoodbye.")

    async def stop(self) -> None:
        self._running = False

    def _print_metadata(self, meta) -> None:
        """Print pipeline metadata for debugging."""
        print(f"\n  --- pipeline ---")
        if meta.triage:
            print(f"  triage: {meta.triage.intent} ({meta.triage.complexity})")
            if meta.triage.matched_skill:
                print(f"  matched skill: {meta.triage.matched_skill}")
        if meta.escalated:
            print(f"  escalated: {meta.escalated_model}")
        if meta.search:
            print(f"  search: {len(meta.search.fragments)} fragments, {meta.search.search_time_ms}ms")
        if meta.reflection:
            print(f"  reflection: confidence={meta.reflection.confidence:.2f}, action={meta.reflection.action}")
            if meta.reflection.memory_gap:
                print(f"  memory_gap: {meta.reflection.memory_gap}")
        print(f"  total: {meta.total_time_ms}ms")
        print(f"  ---")
