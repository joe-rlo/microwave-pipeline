"""Tests for the LLM-facing scheduler tool handlers.

Mirrors tests/test_scheduler.py's _store fixture style. Each handler is
fed a config-like SimpleNamespace pointing at a tmp SQLite path, so
the tests touch the real `SchedulerStore` (no mocks of the storage
layer — those would defeat the point of these handlers existing).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.scheduler.store import ScheduledJob, SchedulerStore
from src.tools import scheduler as sched_tool


def _config(tmp_path: Path, **overrides) -> SimpleNamespace:
    """Minimal config shape the handlers read.

    Handlers only ever touch:
      - config.db_path                  (always)
      - config.workspace_dir            (only when validating skill=...)
      - config.heartbeat_notify_channel (only as add-fallback)
      - config.heartbeat_notify_recipient (only as add-fallback)
    """
    defaults = dict(
        db_path=tmp_path / "test.db",
        workspace_dir=tmp_path / "workspace",
        heartbeat_notify_channel="signal",
        heartbeat_notify_recipient="+15551234567",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _seed_job(cfg: SimpleNamespace, **kw) -> ScheduledJob:
    """Insert a job directly into the store; return it with id populated."""
    s = SchedulerStore(cfg.db_path)
    s.connect()
    defaults = dict(
        name="seeded",
        cron_expr="0 9 * * *",
        mode="direct",
        prompt_or_text="hello",
        target_channel="signal",
        recipient_id="+15551234567",
    )
    defaults.update(kw)
    job = ScheduledJob(**defaults)
    job.id = s.add(job)
    s.close()
    return job


# --- list / get -----------------------------------------------------------


class TestList:
    @pytest.mark.asyncio
    async def test_empty(self, tmp_path):
        cfg = _config(tmp_path)
        out = json.loads(await sched_tool._handle_list({}, config=cfg))
        assert out == {"jobs": [], "count": 0}

    @pytest.mark.asyncio
    async def test_returns_brief_shape(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_job(cfg, name="a", cron_expr="0 7 * * *")
        _seed_job(cfg, name="b", mode="llm", prompt_or_text="ask")
        out = json.loads(await sched_tool._handle_list({}, config=cfg))
        assert out["count"] == 2
        names = {j["name"] for j in out["jobs"]}
        assert names == {"a", "b"}
        # Brief shape: prompt body NOT exposed (use scheduler_get for that)
        first = out["jobs"][0]
        assert "prompt_or_text" not in first
        assert {"name", "cron", "mode", "channel", "enabled"} <= set(first)


class TestGet:
    @pytest.mark.asyncio
    async def test_full_shape(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_job(cfg, name="briefing", prompt_or_text="the briefing prompt body")
        out = json.loads(await sched_tool._handle_get({"name": "briefing"}, config=cfg))
        # Full shape MUST include the body so the LLM can mirror format
        assert out["prompt_or_text"] == "the briefing prompt body"
        assert out["name"] == "briefing"

    @pytest.mark.asyncio
    async def test_missing_name_raises(self, tmp_path):
        cfg = _config(tmp_path)
        with pytest.raises(RuntimeError, match="requires `name`"):
            await sched_tool._handle_get({}, config=cfg)

    @pytest.mark.asyncio
    async def test_not_found_raises(self, tmp_path):
        cfg = _config(tmp_path)
        with pytest.raises(RuntimeError, match="No job named"):
            await sched_tool._handle_get({"name": "ghost"}, config=cfg)


# --- add ------------------------------------------------------------------


class TestAddDirect:
    @pytest.mark.asyncio
    async def test_minimal_direct(self, tmp_path):
        cfg = _config(tmp_path)
        out = json.loads(await sched_tool._handle_add({
            "name": "ping", "cron": "0 7 * * *", "mode": "direct", "text": "wake up",
        }, config=cfg))
        assert out["status"] == "added"
        assert out["job"]["mode"] == "direct"
        assert out["job"]["prompt_or_text"] == "wake up"
        # direct mode: card_view OFF regardless of args
        assert out["job"]["card_view"] is False

    @pytest.mark.asyncio
    async def test_direct_without_text_raises(self, tmp_path):
        cfg = _config(tmp_path)
        with pytest.raises(RuntimeError, match="mode=direct requires `text`"):
            await sched_tool._handle_add({
                "name": "x", "cron": "0 * * * *", "mode": "direct",
            }, config=cfg)


class TestAddLLM:
    @pytest.mark.asyncio
    async def test_inline_prompt(self, tmp_path):
        cfg = _config(tmp_path)
        out = json.loads(await sched_tool._handle_add({
            "name": "ask", "cron": "0 8 * * *", "mode": "llm", "prompt": "draft me X",
        }, config=cfg))
        assert out["job"]["prompt_or_text"] == "draft me X"
        assert out["job"]["skill_name"] is None
        # llm + default → card_view ON
        assert out["job"]["card_view"] is True

    @pytest.mark.asyncio
    async def test_llm_without_prompt_or_skill_raises(self, tmp_path):
        cfg = _config(tmp_path)
        with pytest.raises(RuntimeError, match="`prompt` or `skill`"):
            await sched_tool._handle_add({
                "name": "x", "cron": "0 * * * *", "mode": "llm",
            }, config=cfg)

    @pytest.mark.asyncio
    async def test_skill_requires_trigger(self, tmp_path):
        cfg = _config(tmp_path)
        with pytest.raises(RuntimeError, match="`trigger`"):
            await sched_tool._handle_add({
                "name": "x", "cron": "0 * * * *", "mode": "llm",
                "skill": "morning-briefing",
            }, config=cfg)

    @pytest.mark.asyncio
    async def test_skill_validated_before_insert(self, tmp_path):
        # workspace_dir exists but the skill doesn't → fail loudly at add
        # time rather than silently at first fire.
        (tmp_path / "workspace" / "skills").mkdir(parents=True)
        cfg = _config(tmp_path)
        with pytest.raises(RuntimeError, match="No skill named"):
            await sched_tool._handle_add({
                "name": "x", "cron": "0 * * * *", "mode": "llm",
                "skill": "doesnt-exist", "trigger": "go",
            }, config=cfg)
        # And nothing was inserted
        s = SchedulerStore(cfg.db_path); s.connect()
        assert s.list_all() == []
        s.close()

    @pytest.mark.asyncio
    async def test_skill_present_succeeds(self, tmp_path):
        skill_dir = tmp_path / "workspace" / "skills" / "real-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: real-skill\ndescription: testing\n---\n\nBody.\n"
        )
        cfg = _config(tmp_path)
        out = json.loads(await sched_tool._handle_add({
            "name": "with-skill", "cron": "0 6 * * *", "mode": "llm",
            "skill": "real-skill", "trigger": "Generate it.",
        }, config=cfg))
        assert out["job"]["skill_name"] == "real-skill"
        assert out["job"]["prompt_or_text"] == "Generate it."


class TestAddScript:
    @pytest.mark.asyncio
    async def test_minimal(self, tmp_path):
        cfg = _config(tmp_path)
        out = json.loads(await sched_tool._handle_add({
            "name": "report", "cron": "0 9 * * 1", "mode": "script",
            "command": "echo hi",
        }, config=cfg))
        assert out["job"]["prompt_or_text"] == "echo hi"
        # script + default → card_view ON (like llm)
        assert out["job"]["card_view"] is True


class TestAddValidation:
    @pytest.mark.asyncio
    async def test_missing_required(self, tmp_path):
        cfg = _config(tmp_path)
        with pytest.raises(RuntimeError, match="requires"):
            await sched_tool._handle_add({"name": "x"}, config=cfg)

    @pytest.mark.asyncio
    async def test_invalid_cron(self, tmp_path):
        cfg = _config(tmp_path)
        with pytest.raises(RuntimeError, match="Invalid cron"):
            await sched_tool._handle_add({
                "name": "x", "cron": "not-a-cron", "mode": "direct", "text": "hi",
            }, config=cfg)

    @pytest.mark.asyncio
    async def test_invalid_mode(self, tmp_path):
        cfg = _config(tmp_path)
        with pytest.raises(RuntimeError, match="mode must be"):
            await sched_tool._handle_add({
                "name": "x", "cron": "0 * * * *", "mode": "bogus",
            }, config=cfg)

    @pytest.mark.asyncio
    async def test_duplicate_name_rejected(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_job(cfg, name="dup")
        with pytest.raises(RuntimeError, match="already exists"):
            await sched_tool._handle_add({
                "name": "dup", "cron": "0 * * * *", "mode": "direct", "text": "x",
            }, config=cfg)


class TestAddDefaults:
    @pytest.mark.asyncio
    async def test_channel_recipient_default_from_config(self, tmp_path):
        # No channel/recipient in args — should fall through to heartbeat fields
        cfg = _config(tmp_path,
            heartbeat_notify_channel="telegram",
            heartbeat_notify_recipient="999",
        )
        out = json.loads(await sched_tool._handle_add({
            "name": "x", "cron": "0 * * * *", "mode": "direct", "text": "hi",
        }, config=cfg))
        assert out["job"]["channel"] == "telegram"
        assert out["job"]["recipient"] == "999"

    @pytest.mark.asyncio
    async def test_explicit_channel_wins(self, tmp_path):
        cfg = _config(tmp_path, heartbeat_notify_channel="telegram")
        out = json.loads(await sched_tool._handle_add({
            "name": "x", "cron": "0 * * * *", "mode": "direct", "text": "hi",
            "channel": "signal", "recipient": "+1",
        }, config=cfg))
        assert out["job"]["channel"] == "signal"
        assert out["job"]["recipient"] == "+1"

    @pytest.mark.asyncio
    async def test_missing_recipient_everywhere_raises(self, tmp_path):
        cfg = _config(tmp_path, heartbeat_notify_recipient="")
        with pytest.raises(RuntimeError, match="recipient missing"):
            await sched_tool._handle_add({
                "name": "x", "cron": "0 * * * *", "mode": "direct", "text": "hi",
            }, config=cfg)

    @pytest.mark.asyncio
    async def test_invalid_channel_raises(self, tmp_path):
        cfg = _config(tmp_path, heartbeat_notify_channel="")
        with pytest.raises(RuntimeError, match="channel must be"):
            await sched_tool._handle_add({
                "name": "x", "cron": "0 * * * *", "mode": "direct", "text": "hi",
                "channel": "smoke-signal", "recipient": "+1",
            }, config=cfg)


# --- remove / set_enabled -------------------------------------------------


class TestRemove:
    @pytest.mark.asyncio
    async def test_removes_existing(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_job(cfg, name="bye")
        out = json.loads(await sched_tool._handle_remove({"name": "bye"}, config=cfg))
        assert out == {"status": "removed", "name": "bye"}
        s = SchedulerStore(cfg.db_path); s.connect()
        assert s.get_by_name("bye") is None
        s.close()

    @pytest.mark.asyncio
    async def test_missing_raises(self, tmp_path):
        cfg = _config(tmp_path)
        with pytest.raises(RuntimeError, match="No job named"):
            await sched_tool._handle_remove({"name": "ghost"}, config=cfg)


class TestSetEnabled:
    @pytest.mark.asyncio
    async def test_disable(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_job(cfg, name="j")
        out = json.loads(await sched_tool._handle_set_enabled(
            {"name": "j", "enabled": False}, config=cfg
        ))
        assert out == {"status": "disabled", "name": "j"}
        s = SchedulerStore(cfg.db_path); s.connect()
        assert s.get_by_name("j").enabled is False
        s.close()

    @pytest.mark.asyncio
    async def test_enable(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_job(cfg, name="j", enabled=False)
        out = json.loads(await sched_tool._handle_set_enabled(
            {"name": "j", "enabled": True}, config=cfg
        ))
        assert out == {"status": "enabled", "name": "j"}

    @pytest.mark.asyncio
    async def test_missing_enabled_raises(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_job(cfg, name="j")
        with pytest.raises(RuntimeError, match="`enabled`"):
            await sched_tool._handle_set_enabled({"name": "j"}, config=cfg)


# --- registry wiring ------------------------------------------------------


class TestRegistryWiring:
    """The orchestrator reads tools through build_provider_tools and docs
    through build_tools.catalog_text. If either side stops listing the
    scheduler tools, the bot regresses to the original "I don't know
    where the cron lives" failure mode — pin both."""

    def test_provider_tools_register_all_five(self, tmp_path):
        from src.tools import build_provider_tools

        cfg = _config(tmp_path)
        tools = build_provider_tools(cfg)
        names = {t.definition.name for t in tools}
        assert {
            "scheduler_list",
            "scheduler_get",
            "scheduler_add",
            "scheduler_remove",
            "scheduler_set_enabled",
        } <= names

    def test_catalog_text_mentions_scheduler(self, tmp_path):
        from src.tools import build_tools

        cfg = _config(tmp_path)
        # build_tools may fail to import claude_agent_sdk in some envs —
        # in that case it returns an empty bundle. Skip the assertion then
        # rather than masking real catalog regressions on dev machines.
        bundle = build_tools(cfg)
        if not bundle.catalog_text:
            pytest.skip("SDK not available; catalog path returned empty")
        assert "scheduler_list" in bundle.catalog_text
        assert "scheduler_add" in bundle.catalog_text
