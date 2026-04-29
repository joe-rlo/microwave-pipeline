"""Project loader, scaffolding, chat commands, BIBLE flow, and pipeline wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.projects.bible import _split_name_and_description, handle_bible_command
from src.projects.chat import handle_project_command
from src.projects.loader import ProjectLoader, ProjectNotFound
from src.projects.models import PROJECT_TYPES, Project


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------- Loader ----------


class TestProjectLoader:
    def test_load_minimal(self, tmp_path):
        _write(
            tmp_path / "blog-q2" / "PROJECT.md",
            "---\nname: blog-q2\ntype: blog\n---\nVoice notes here.",
        )
        p = ProjectLoader(tmp_path).load("blog-q2")
        assert p.name == "blog-q2"
        assert p.type == "blog"
        assert p.skill == "blog-writing"  # default skill for blog
        assert "Voice notes" in p.voice_notes

    def test_explicit_skill_wins_over_default(self, tmp_path):
        _write(
            tmp_path / "novel-x" / "PROJECT.md",
            "---\nname: novel-x\ntype: novel\nskill: custom-prose\n---\n",
        )
        assert ProjectLoader(tmp_path).load("novel-x").skill == "custom-prose"

    def test_unknown_type_falls_back_to_blog(self, tmp_path):
        _write(
            tmp_path / "bad" / "PROJECT.md",
            "---\nname: bad\ntype: limerick\n---\n",
        )
        assert ProjectLoader(tmp_path).load("bad").type == "blog"

    def test_target_words_parsed(self, tmp_path):
        _write(
            tmp_path / "novel-y" / "PROJECT.md",
            "---\nname: novel-y\ntype: novel\ntarget_words: 80000\n---\n",
        )
        assert ProjectLoader(tmp_path).load("novel-y").target_words == 80000

    def test_word_count_sums_drafts(self, tmp_path):
        root = tmp_path / "novel-z"
        _write(root / "PROJECT.md", "---\nname: novel-z\ntype: novel\n---\n")
        _write(root / "drafts" / "chapter-01.md", "one two three four five")
        _write(root / "drafts" / "chapter-02.md", "six seven eight")
        # All files in drafts/ count — the heuristic doesn't filter by ext.
        p = ProjectLoader(tmp_path).load("novel-z")
        assert p.word_count() == 8

    def test_not_found_raises(self, tmp_path):
        with pytest.raises(ProjectNotFound):
            ProjectLoader(tmp_path).load("ghost")

    def test_list_skips_non_projects(self, tmp_path):
        _write(tmp_path / "real" / "PROJECT.md", "---\nname: real\ntype: blog\n---\n")
        (tmp_path / "not-a-project").mkdir()
        (tmp_path / ".archived" / "old" / "PROJECT.md").parent.mkdir(parents=True)
        (tmp_path / ".archived" / "old" / "PROJECT.md").write_text(
            "---\nname: old\ntype: blog\n---\n", encoding="utf-8"
        )
        names = ProjectLoader(tmp_path).list_names()
        assert names == ["real"]

    def test_list_includes_archived_when_asked(self, tmp_path):
        _write(tmp_path / "real" / "PROJECT.md", "---\nname: real\ntype: blog\n---\n")
        _write(
            tmp_path / ".archived" / "old" / "PROJECT.md",
            "---\nname: old\ntype: blog\n---\n",
        )
        names = ProjectLoader(tmp_path).list_names(include_archived=True)
        assert "real" in names
        assert ".archived/old" in names


# ---------- Scaffolding ----------


class TestScaffolding:
    @pytest.mark.parametrize("ptype", list(PROJECT_TYPES))
    def test_scaffold_creates_layout(self, tmp_path, ptype):
        path = ProjectLoader(tmp_path).scaffold("test-proj", ptype, description="x")
        root = path.parent
        assert (root / "PROJECT.md").is_file()
        assert (root / "drafts").is_dir()
        assert (root / "notes").is_dir()
        # Type-specific files
        if ptype in ("novel", "screenplay"):
            assert (root / "BIBLE.md").is_file()
            assert (root / "outline.md").is_file()
        if ptype == "blog":
            assert (root / "outline.md").is_file()
            assert (root / "drafts" / "draft.md").is_file()
        if ptype == "screenplay":
            assert (root / "drafts" / "screenplay.fountain").is_file()

    def test_scaffold_rejects_bad_name(self, tmp_path):
        with pytest.raises(ValueError):
            ProjectLoader(tmp_path).scaffold("Has Spaces", "blog")
        with pytest.raises(ValueError):
            ProjectLoader(tmp_path).scaffold("UPPER", "novel")

    def test_scaffold_rejects_unknown_type(self, tmp_path):
        with pytest.raises(ValueError):
            ProjectLoader(tmp_path).scaffold("ok-name", "podcast")

    def test_scaffold_rejects_duplicate(self, tmp_path):
        loader = ProjectLoader(tmp_path)
        loader.scaffold("dupe", "blog")
        with pytest.raises(FileExistsError):
            loader.scaffold("dupe", "blog")

    def test_scaffolded_project_loads_clean(self, tmp_path):
        ProjectLoader(tmp_path).scaffold("novel-it", "novel", description="d")
        p = ProjectLoader(tmp_path).load("novel-it")
        assert p.type == "novel"
        assert p.skill == "novel-writing"
        assert p.has_bible
        assert p.has_outline


class TestArchive:
    def test_archive_moves_under_dot_archived(self, tmp_path):
        loader = ProjectLoader(tmp_path)
        loader.scaffold("done", "blog")
        assert loader.archive("done") is True
        assert not (tmp_path / "done").exists()
        assert (tmp_path / ".archived" / "done" / "PROJECT.md").is_file()

    def test_archive_missing_returns_false(self, tmp_path):
        assert ProjectLoader(tmp_path).archive("ghost") is False


# ---------- Chat commands ----------


class _FakeOrch:
    def __init__(self, projects=()):
        self._projects = list(projects)
        self._active = None

    def set_active_project(self, name):
        for p in self._projects:
            if p.name == name:
                self._active = p
                return p
        raise ProjectNotFound(name)

    def clear_active_project(self):
        self._active = None

    def get_active_project(self):
        return self._active

    def list_projects(self):
        return self._projects


def _proj(name="alpha", **kw):
    defaults = dict(
        name=name,
        type=kw.get("type", "blog"),
        skill=kw.get("skill", "blog-writing"),
        status=kw.get("status", "drafting"),
        description=kw.get("description", "a thing"),
    )
    defaults.update(kw)
    return Project(**defaults)


class TestProjectChatCommands:
    def test_unknown_text_is_passthrough(self):
        orch = _FakeOrch()
        assert handle_project_command("hello world", orch) is None
        assert handle_project_command("", orch) is None
        assert handle_project_command("/skill foo", orch) is None  # other namespace

    def test_activate(self):
        orch = _FakeOrch([_proj("alpha"), _proj("beta", type="novel")])
        reply = handle_project_command("/project beta", orch)
        assert reply and "beta" in reply
        assert "novel" in reply  # type is in the activation message
        assert orch.get_active_project().name == "beta"

    def test_activate_unknown(self):
        orch = _FakeOrch([_proj("alpha")])
        reply = handle_project_command("/project ghost", orch)
        assert "No project named" in reply
        assert orch.get_active_project() is None

    def test_deactivate_aliases(self):
        orch = _FakeOrch([_proj("alpha")])
        for arg in ("off", "none", "clear"):
            orch.set_active_project("alpha")
            handle_project_command(f"/project {arg}", orch)
            assert orch.get_active_project() is None

    def test_status_when_no_project(self):
        orch = _FakeOrch()
        reply = handle_project_command("/project status", orch)
        assert "No active project" in reply

    def test_list_marks_active(self):
        orch = _FakeOrch([_proj("alpha"), _proj("beta")])
        orch.set_active_project("alpha")
        reply = handle_project_command("/projects", orch)
        assert "→ alpha" in reply


# ---------- BIBLE flow ----------


class TestBibleNameSplit:
    def test_unquoted(self):
        assert _split_name_and_description("Walsh tall and weary") == (
            "Walsh", "tall and weary"
        )

    def test_quoted(self):
        assert _split_name_and_description('"Detective Walsh" tall and weary') == (
            "Detective Walsh", "tall and weary"
        )

    def test_name_only(self):
        assert _split_name_and_description("Walsh") == ("Walsh", "")


class TestBibleCommand:
    def test_no_active_project(self, tmp_path):
        orch = _FakeOrch()
        reply = handle_bible_command("/bible add x", orch)
        assert "No active project" in reply

    def test_add_creates_facts_section(self, tmp_path):
        # Real Project with a real bible file on disk
        bible = tmp_path / "BIBLE.md"
        bible.write_text(
            "# Bible — test\n\n## Premise\nA story about a thing.\n",
            encoding="utf-8",
        )
        project = _proj("test", type="novel")
        project.bible_path = bible
        orch = _FakeOrch([project])
        orch.set_active_project("test")

        reply = handle_bible_command("/bible add Walsh tall and weary", orch)
        assert "added" in reply.lower()
        text = bible.read_text(encoding="utf-8")
        assert "## Established facts" in text
        assert "### Walsh" in text
        assert "tall and weary" in text

    def test_add_quoted_name(self, tmp_path):
        bible = tmp_path / "BIBLE.md"
        bible.write_text("# Bible\n\n## Established facts\n", encoding="utf-8")
        project = _proj("test", type="novel")
        project.bible_path = bible
        orch = _FakeOrch([project])
        orch.set_active_project("test")

        handle_bible_command(
            '/bible add "Detective Walsh" twists his ring when nervous', orch
        )
        text = bible.read_text(encoding="utf-8")
        assert "### Detective Walsh" in text
        assert "twists his ring" in text

    def test_show_returns_bible_text(self, tmp_path):
        bible = tmp_path / "BIBLE.md"
        bible.write_text("# Bible\n\nThe world.\n", encoding="utf-8")
        project = _proj("test", type="novel")
        project.bible_path = bible
        orch = _FakeOrch([project])
        orch.set_active_project("test")

        reply = handle_bible_command("/bible show", orch)
        assert "The world." in reply

    def test_unknown_subcommand_shows_usage(self, tmp_path):
        orch = _FakeOrch()
        reply = handle_bible_command("/bible weird", orch)
        assert "Usage" in reply

    def test_non_bible_text_is_passthrough(self):
        orch = _FakeOrch()
        assert handle_bible_command("hi there", orch) is None


# ---------- Memory store BIBLE inclusion ----------


class TestStableContextWithBible:
    def test_bible_appears_in_stable_context(self, tmp_path):
        from src.memory.store import MemoryStore

        store = MemoryStore(tmp_path)
        store.ensure_dirs()
        store.save_identity("# I am here")

        bible = tmp_path / "novel-x" / "BIBLE.md"
        _write(bible, "# Bible\n\nDetective Walsh, weary.\n")

        ctx = store.assemble_stable_context(channel=None, bible_path=bible)
        assert "# I am here" in ctx
        assert "[Project bible — novel-x]" in ctx
        assert "Detective Walsh" in ctx

    def test_no_bible_path_omits_block(self, tmp_path):
        from src.memory.store import MemoryStore

        store = MemoryStore(tmp_path)
        store.ensure_dirs()
        store.save_identity("# I am here")
        ctx = store.assemble_stable_context(channel=None, bible_path=None)
        assert "[Project bible" not in ctx

    def test_bible_mtime_included(self, tmp_path):
        from src.memory.store import MemoryStore

        store = MemoryStore(tmp_path)
        store.ensure_dirs()
        store.save_identity("# I am here")
        bible = tmp_path / "novel-x" / "BIBLE.md"
        _write(bible, "v1")
        m1 = store.stable_context_mtime(bible_path=bible)
        # touch the bible
        import os, time
        time.sleep(0.05)
        os.utime(bible, None)
        bible.write_text("v2", encoding="utf-8")
        m2 = store.stable_context_mtime(bible_path=bible)
        assert m2 > m1
