"""Skill loader, assembly wiring, chat-command parser, scheduler integration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.pipeline.assembly import _format_fragments, assemble  # noqa: F401
from src.session.models import MemoryFragment, SearchResult
from src.skills import Skill, SkillLoader, SkillNotFound
from src.skills.chat import handle_skill_command


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestFrontmatter:
    def test_scalar_fields(self, tmp_path):
        _write(
            tmp_path / "alpha" / "SKILL.md",
            "---\nname: alpha\ndescription: a skill\n---\n\n# Body\ngoes here.",
        )
        s = SkillLoader(tmp_path).load("alpha")
        assert s.name == "alpha"
        assert s.description == "a skill"
        assert "Body" in s.body

    def test_folded_multiline_description(self, tmp_path):
        _write(
            tmp_path / "beta" / "SKILL.md",
            "---\n"
            "name: beta\n"
            "description: >\n"
            "  a longer description\n"
            "  that wraps across\n"
            "  multiple lines\n"
            "---\n\nbody",
        )
        s = SkillLoader(tmp_path).load("beta")
        assert s.description == "a longer description that wraps across multiple lines"

    def test_triggers_list(self, tmp_path):
        _write(
            tmp_path / "gamma" / "SKILL.md",
            "---\n"
            "name: gamma\n"
            "description: g\n"
            "triggers:\n"
            "  - first\n"
            "  - second\n"
            "  - third\n"
            "---\nbody",
        )
        s = SkillLoader(tmp_path).load("gamma")
        assert s.triggers == ["first", "second", "third"]

    def test_quoted_values_stripped(self, tmp_path):
        _write(
            tmp_path / "delta" / "SKILL.md",
            '---\nname: delta\ndescription: "has: colon"\n---\nbody',
        )
        assert SkillLoader(tmp_path).load("delta").description == "has: colon"

    def test_missing_frontmatter_uses_dir_name(self, tmp_path):
        _write(tmp_path / "epsilon" / "SKILL.md", "just a body, no frontmatter\n")
        s = SkillLoader(tmp_path).load("epsilon")
        assert s.name == "epsilon"
        assert s.description == ""
        assert s.body.startswith("just a body")

    def test_fetch_detection(self, tmp_path):
        root = tmp_path / "zeta"
        _write(root / "SKILL.md", "---\nname: zeta\ndescription: z\n---\nbody")
        assert not SkillLoader(tmp_path).load("zeta").has_fetch
        _write(root / "fetch.py", "def fetch(ctx): return 'ok'")
        assert SkillLoader(tmp_path).load("zeta").has_fetch


class TestLoaderDiscovery:
    def test_list_skips_non_skill_dirs(self, tmp_path):
        _write(tmp_path / "real" / "SKILL.md", "---\nname: real\ndescription: r\n---\nx")
        (tmp_path / "not-a-skill").mkdir()
        (tmp_path / ".hidden").mkdir()
        names = SkillLoader(tmp_path).list_names()
        assert names == ["real"]

    def test_list_all_is_defensive_against_load_failures(self, tmp_path, monkeypatch):
        _write(tmp_path / "good" / "SKILL.md", "---\nname: good\ndescription: g\n---\nx")
        _write(tmp_path / "bad" / "SKILL.md", "---\nname: bad\ndescription: b\n---\ny")
        loader = SkillLoader(tmp_path)

        # Force load('bad') to blow up so we can verify list_all eats the
        # exception and returns the remaining good skills.
        original_load = loader.load
        def flaky(name):
            if name == "bad":
                raise RuntimeError("simulated parse failure")
            return original_load(name)
        monkeypatch.setattr(loader, "load", flaky)

        names = [s.name for s in loader.list_all()]
        assert names == ["good"]

    def test_not_found_raises(self, tmp_path):
        with pytest.raises(SkillNotFound):
            SkillLoader(tmp_path).load("nope")

    def test_scaffold_creates_template(self, tmp_path):
        path = SkillLoader(tmp_path).scaffold("my-skill", description="does stuff")
        assert path.is_file()
        s = SkillLoader(tmp_path).load("my-skill")
        assert s.name == "my-skill"
        assert "does stuff" in s.description

    def test_scaffold_rejects_bad_names(self, tmp_path):
        with pytest.raises(ValueError):
            SkillLoader(tmp_path).scaffold("Has Uppercase")
        with pytest.raises(ValueError):
            SkillLoader(tmp_path).scaffold("has/slash")

    def test_scaffold_rejects_duplicate(self, tmp_path):
        loader = SkillLoader(tmp_path)
        loader.scaffold("dupe")
        with pytest.raises(FileExistsError):
            loader.scaffold("dupe")


class TestAssemblyWithSkill:
    def test_skill_block_appears_in_dynamic_context(self, tmp_path, monkeypatch):
        # We need a MemoryStore double — only assemble_stable_context is called.
        class _FakeStore:
            def assemble_stable_context(self, channel=None, bible_path=None): return "STABLE"

        class _FakeIndex:
            def get_promotion_candidates(self, min_retrievals=3): return []

        skill = Skill(
            name="substack-writer",
            description="x",
            body="BODY_LINES_HERE\nmore body",
            triggers=[],
        )
        result = assemble(
            search_result=SearchResult(fragments=[]),
            memory_store=_FakeStore(),
            memory_index=_FakeIndex(),
            channel="signal",
            active_skill=skill,
        )
        ctx = result.memory_context
        assert "[Active skill: substack-writer]" in ctx
        assert "BODY_LINES_HERE" in ctx
        assert "channel rules win" in ctx.lower()

    def test_channel_rules_appear_after_skill_block(self, tmp_path):
        """Skill must appear BEFORE the channel file-output instructions so
        channel rules have higher recency in the prompt — that's how we
        enforce 'channel rules win on conflict'."""
        class _FakeStore:
            def assemble_stable_context(self, channel=None, bible_path=None): return ""
        class _FakeIndex:
            def get_promotion_candidates(self, min_retrievals=3): return []

        skill = Skill(name="x", description="d", body="SKILL_BODY", triggers=[])
        ctx = assemble(
            SearchResult(fragments=[]),
            _FakeStore(), _FakeIndex(),
            channel="signal",
            active_skill=skill,
        ).memory_context
        assert ctx.index("SKILL_BODY") < ctx.index("[File output — Signal]")

    def test_no_skill_means_no_skill_block(self):
        class _FakeStore:
            def assemble_stable_context(self, channel=None, bible_path=None): return ""
        class _FakeIndex:
            def get_promotion_candidates(self, min_retrievals=3): return []

        ctx = assemble(
            SearchResult(fragments=[]),
            _FakeStore(), _FakeIndex(),
            channel="signal",
            active_skill=None,
        ).memory_context
        assert "[Active skill" not in ctx


class TestChatCommands:
    class _FakeOrch:
        def __init__(self, skills):
            self._skills = skills
            self._active = None

        def set_active_skill(self, name):
            for s in self._skills:
                if s.name == name:
                    self._active = s
                    return s
            raise SkillNotFound(name)

        def clear_active_skill(self):
            self._active = None

        def get_active_skill(self):
            return self._active

        def list_skills(self):
            return self._skills

    def _orch(self, *names) -> "TestChatCommands._FakeOrch":
        return self._FakeOrch([
            Skill(name=n, description=f"{n} desc", body="...", triggers=[])
            for n in names
        ])

    def test_non_command_returns_none(self):
        orch = self._orch("a")
        assert handle_skill_command("hello there", orch) is None
        assert handle_skill_command("/debug", orch) is None
        assert handle_skill_command("", orch) is None

    def test_activate(self):
        orch = self._orch("substack-writer", "github-tool")
        reply = handle_skill_command("/skill substack-writer", orch)
        assert reply and "substack-writer" in reply
        assert orch.get_active_skill().name == "substack-writer"

    def test_activate_unknown(self):
        orch = self._orch("a")
        reply = handle_skill_command("/skill nonexistent", orch)
        assert reply and "No skill named" in reply
        assert orch.get_active_skill() is None

    def test_deactivate(self):
        orch = self._orch("a")
        orch.set_active_skill("a")
        reply = handle_skill_command("/skill off", orch)
        assert "cleared" in reply.lower()
        assert orch.get_active_skill() is None

    def test_deactivate_aliases(self):
        orch = self._orch("a")
        orch.set_active_skill("a")
        handle_skill_command("/skill none", orch)
        assert orch.get_active_skill() is None
        orch.set_active_skill("a")
        handle_skill_command("/skill clear", orch)
        assert orch.get_active_skill() is None

    def test_list(self):
        orch = self._orch("alpha", "beta")
        reply = handle_skill_command("/skills", orch)
        assert "alpha" in reply
        assert "beta" in reply

    def test_list_marks_active(self):
        orch = self._orch("alpha", "beta")
        orch.set_active_skill("alpha")
        reply = handle_skill_command("/skills", orch)
        assert "→ alpha" in reply
        assert "  beta" in reply

    def test_current_when_none(self):
        orch = self._orch("a")
        reply = handle_skill_command("/skill", orch)
        assert "No active skill" in reply


class TestSchedulerPromptComposition:
    """Scheduler should wire skill.body into the SingleTurnClient system prompt
    and combine fetch output + trigger into the user message."""

    @pytest.mark.asyncio
    async def test_no_skill_uses_legacy_prompt(self, tmp_path):
        from src.config import Config
        from src.scheduler.engine import Scheduler
        from src.scheduler.store import ScheduledJob, SchedulerStore

        store = SchedulerStore(tmp_path / "db")
        store.connect()
        sch = Scheduler(
            store=store,
            channels={},
            config=Config(db_path=tmp_path / "db", workspace_dir=tmp_path),
        )
        job = ScheduledJob(
            name="j", mode="llm", prompt_or_text="Do X",
            target_channel="signal", recipient_id="+1",
        )
        sys_p, user_p = await sch._compose_prompt(job)
        assert "scheduled task" in sys_p
        assert user_p == "Do X"
        store.close()

    @pytest.mark.asyncio
    async def test_skill_body_becomes_system_prompt(self, tmp_path):
        from src.config import Config
        from src.scheduler.engine import Scheduler
        from src.scheduler.store import ScheduledJob, SchedulerStore

        # Plant a skill on disk
        skill_dir = tmp_path / "skills" / "writer"
        _write(
            skill_dir / "SKILL.md",
            "---\nname: writer\ndescription: writes\n---\nYou are a writer.",
        )

        store = SchedulerStore(tmp_path / "db")
        store.connect()
        sch = Scheduler(
            store=store,
            channels={},
            config=Config(db_path=tmp_path / "db", workspace_dir=tmp_path),
        )
        job = ScheduledJob(
            name="j", mode="llm", prompt_or_text="Generate now.",
            target_channel="signal", recipient_id="+1",
            skill_name="writer",
        )
        sys_p, user_p = await sch._compose_prompt(job)
        assert "You are a writer." in sys_p
        assert "Generate now." in user_p
        store.close()

    @pytest.mark.asyncio
    async def test_missing_skill_raises(self, tmp_path):
        from src.config import Config
        from src.scheduler.engine import Scheduler
        from src.scheduler.store import ScheduledJob, SchedulerStore

        store = SchedulerStore(tmp_path / "db")
        store.connect()
        sch = Scheduler(
            store=store, channels={},
            config=Config(db_path=tmp_path / "db", workspace_dir=tmp_path),
        )
        job = ScheduledJob(
            name="j", mode="llm", skill_name="ghost",
            target_channel="signal", recipient_id="+1",
        )
        with pytest.raises(RuntimeError, match="missing skill"):
            await sch._compose_prompt(job)
        store.close()

    @pytest.mark.asyncio
    async def test_fetch_output_prepended_to_user_message(self, tmp_path):
        """If the skill has a fetch.py, its output is prepended to the user
        message under [Pre-fetch context] so the LLM sees it first."""
        from src.config import Config
        from src.scheduler.engine import Scheduler
        from src.scheduler.store import ScheduledJob, SchedulerStore

        skill_dir = tmp_path / "skills" / "ghtool"
        _write(
            skill_dir / "SKILL.md",
            "---\nname: ghtool\ndescription: gh\n---\nSummarize the PRs.",
        )
        _write(
            skill_dir / "fetch.py",
            "def fetch(context):\n    return 'PR #1: update readme'\n",
        )

        store = SchedulerStore(tmp_path / "db")
        store.connect()
        sch = Scheduler(
            store=store, channels={},
            config=Config(db_path=tmp_path / "db", workspace_dir=tmp_path),
        )
        job = ScheduledJob(
            name="j", mode="llm", skill_name="ghtool",
            prompt_or_text="Give me the digest.",
            target_channel="signal", recipient_id="+1",
        )
        sys_p, user_p = await sch._compose_prompt(job)
        assert "Summarize the PRs." in sys_p
        assert "[Pre-fetch context]" in user_p
        assert "PR #1: update readme" in user_p
        assert "Give me the digest." in user_p
        # Pre-fetch must come BEFORE the trigger
        assert user_p.index("Pre-fetch context") < user_p.index("Give me the digest.")
        store.close()

    @pytest.mark.asyncio
    async def test_fetch_failure_doesnt_crash(self, tmp_path):
        """A broken fetch.py should leave a `[pre-fetch failed: ...]` note
        in the user message, not raise."""
        from src.config import Config
        from src.scheduler.engine import Scheduler
        from src.scheduler.store import ScheduledJob, SchedulerStore

        skill_dir = tmp_path / "skills" / "broken"
        _write(skill_dir / "SKILL.md", "---\nname: broken\ndescription: x\n---\nbody")
        _write(
            skill_dir / "fetch.py",
            "def fetch(context):\n    raise RuntimeError('nope')\n",
        )

        store = SchedulerStore(tmp_path / "db")
        store.connect()
        sch = Scheduler(
            store=store, channels={},
            config=Config(db_path=tmp_path / "db", workspace_dir=tmp_path),
        )
        job = ScheduledJob(
            name="j", mode="llm", skill_name="broken",
            prompt_or_text="Go.",
            target_channel="signal", recipient_id="+1",
        )
        _, user_p = await sch._compose_prompt(job)
        assert "pre-fetch failed" in user_p
        store.close()
