"""Tests for TODO.md loading + integration into stable context.

The contract is narrow:
1. Open `[ ]` items load; checked `[x]` items are skipped.
2. Anything under `## Done (recent)` (or any "Done"-flavored heading) is dropped.
3. Empty sections are not emitted (no orphan headings).
4. Missing TODO.md returns "" cleanly.
5. The file participates in stable-context mtime so edits trigger a reconnect.
6. The stable context wraps the open list in `[Open tasks — from TODO.md]` with the surfacing rules block.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path)
    s.ensure_dirs()
    return s


def _write_todo(store: MemoryStore, content: str) -> None:
    store.todo_path.write_text(content, encoding="utf-8")


class TestLoadTodoOpenItems:
    def test_missing_file_returns_empty(self, store: MemoryStore):
        assert store.load_todo_open_items() == ""

    def test_only_open_items_kept(self, store: MemoryStore):
        _write_todo(
            store,
            """# TODO

## MicrowaveOS

- [ ] First open
- [x] Already done — drop me
- [ ] Second open
""",
        )
        out = store.load_todo_open_items()
        assert "First open" in out
        assert "Second open" in out
        assert "Already done" not in out

    def test_done_section_terminates_parsing(self, store: MemoryStore):
        _write_todo(
            store,
            """## MicrowaveOS

- [ ] Live task

## Done (recent)

- [x] Shipped thing
- [ ] (this is below Done — should still be skipped)
""",
        )
        out = store.load_todo_open_items()
        assert "Live task" in out
        assert "Shipped thing" not in out
        # The bracketed `[ ]` after the Done heading must NOT leak through
        assert "below Done" not in out

    def test_done_section_variants(self, store: MemoryStore):
        # Match liberally — "Done (recent)", "Done", "Recently Done" all work
        for heading in ["## Done", "## Done (recent)", "## Recently Done"]:
            _write_todo(
                store,
                f"## A\n\n- [ ] alpha\n\n{heading}\n\n- [ ] should-be-skipped\n",
            )
            out = store.load_todo_open_items()
            assert "alpha" in out
            assert "should-be-skipped" not in out, heading

    def test_section_heading_emitted_only_when_items_follow(
        self, store: MemoryStore
    ):
        # If a section has only checked items, its heading shouldn't appear.
        # Use section names that clearly aren't "Done" buckets (the parser
        # treats anything matching the Done convention as a terminator).
        _write_todo(
            store,
            """## All Closed

- [x] one
- [x] two

## Has Open

- [ ] keep me
""",
        )
        out = store.load_todo_open_items()
        assert "## All Closed" not in out
        assert "## Has Open" in out
        assert "keep me" in out

    def test_blank_line_between_sections(self, store: MemoryStore):
        _write_todo(
            store,
            """## Section A

- [ ] alpha

## Section B

- [ ] beta
""",
        )
        out = store.load_todo_open_items()
        # Sections separated by blank line for readability
        assert "## Section A\n- [ ] alpha\n\n## Section B\n- [ ] beta" in out

    def test_prose_and_comments_skipped(self, store: MemoryStore):
        _write_todo(
            store,
            """# TODO

Some prose explaining the file.

## MicrowaveOS

This bucket has the engineering work.

- [ ] real task

> a blockquote comment
""",
        )
        out = store.load_todo_open_items()
        assert "real task" in out
        assert "Some prose" not in out
        assert "This bucket" not in out
        assert "blockquote" not in out

    def test_empty_file_returns_empty(self, store: MemoryStore):
        _write_todo(store, "")
        assert store.load_todo_open_items() == ""

    def test_only_done_section_returns_empty(self, store: MemoryStore):
        _write_todo(store, "## Done\n\n- [x] only finished things\n")
        assert store.load_todo_open_items() == ""


class TestStableContextIntegration:
    def test_open_tasks_block_appears_when_items_exist(
        self, store: MemoryStore
    ):
        _write_todo(store, "## MicrowaveOS\n\n- [ ] do the thing\n")
        prompt = store.assemble_stable_context()
        assert "[Open tasks — from TODO.md]" in prompt
        assert "do the thing" in prompt
        # Surfacing rules included so model knows when to mention them
        assert "Don't mention these unprompted" in prompt
        assert "daily briefing" in prompt.lower()

    def test_no_block_when_todo_missing(self, store: MemoryStore):
        prompt = store.assemble_stable_context()
        assert "[Open tasks" not in prompt

    def test_no_block_when_only_done_items(self, store: MemoryStore):
        _write_todo(store, "## Done\n\n- [x] shipped\n")
        prompt = store.assemble_stable_context()
        assert "[Open tasks" not in prompt


class TestStableContextMtime:
    def test_todo_changes_trigger_mtime_bump(self, store: MemoryStore):
        # Establish a baseline with no TODO.md
        baseline = store.stable_context_mtime()

        # Create TODO.md — mtime should now reflect it
        _write_todo(store, "## A\n\n- [ ] task\n")
        # Sleep just enough that mtime resolution can register the change.
        # Most filesystems have at least 10ms resolution.
        time.sleep(0.02)
        with_todo = store.stable_context_mtime()
        assert with_todo > baseline

        # Modify TODO.md — mtime should bump again
        time.sleep(0.02)
        _write_todo(store, "## A\n\n- [ ] task\n- [ ] another\n")
        after_edit = store.stable_context_mtime()
        assert after_edit >= with_todo
