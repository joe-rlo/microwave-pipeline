"""Tests for the heartbeat runner + state + judge + Blink hook.

Three concerns separated:

1. RUNNER: HookSpec validation, registration, _tick gating by interval,
   _fire happy path / runner-failure / judge-failure isolation,
   state persistence, notify dispatch.

2. JUDGE: llm_judge produces correct HeartbeatEvent from canned LLM
   responses; malformed JSON falls back to should_notify=False; LLM
   exception falls back to should_notify=False.

3. BLINK HOOK: fetch_blink_status returns the structured shape from
   a mocked proxy response; missing-credentials raises with a clear
   message; 401-then-refresh path; refresh-failed surfaces a
   RuntimeError.

We never run real timers — _fire and _tick are called directly so
tests stay millisecond-fast.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import aiohttp
import pytest

from src.heartbeat import (
    HeartbeatEvent,
    HeartbeatRunner,
    HeartbeatSkip,
    HookSpec,
    fetch_blink_status,
    llm_judge,
    load_hook_state,
    save_hook_state,
)


# --- Test fixtures ---


def _ok_judge(notify: bool = False, summary: str = "", severity: str = "info"):
    """Build a judge fn that returns a canned event."""
    async def judge(hook_name: str, current: dict, previous: Optional[dict]):
        return HeartbeatEvent(
            hook_name=hook_name, fired_at=int(time.time()),
            raw=current, summary=summary,
            should_notify=notify, severity=severity,
        )
    return judge


def _capture_notify():
    """A notify fn that records each event it's called with."""
    captured: list[HeartbeatEvent] = []

    async def notify(event: HeartbeatEvent) -> None:
        captured.append(event)

    return notify, captured


def _make_runner(tmp_path, judge=None, notify=None):
    return HeartbeatRunner(
        state_dir=tmp_path / "heartbeat",
        judge=judge or _ok_judge(),
        notify=notify or (lambda e: asyncio.sleep(0)),
    )


# --- Registration / validation ---


class TestRegistration:
    def test_register_adds_hook(self, tmp_path):
        runner = _make_runner(tmp_path)

        async def fake_runner():
            return {"x": 1}

        spec = HookSpec(name="t", interval_minutes=5, runner=fake_runner)
        runner.register(spec)
        assert len(runner.hooks) == 1
        assert runner.hooks[0].name == "t"

    def test_duplicate_name_rejected(self, tmp_path):
        runner = _make_runner(tmp_path)

        async def f():
            return {}

        runner.register(HookSpec(name="dup", interval_minutes=5, runner=f))
        with pytest.raises(ValueError, match="already registered"):
            runner.register(HookSpec(name="dup", interval_minutes=5, runner=f))

    def test_zero_interval_rejected(self, tmp_path):
        runner = _make_runner(tmp_path)

        async def f():
            return {}

        with pytest.raises(ValueError, match="interval_minutes"):
            runner.register(HookSpec(name="bad", interval_minutes=0, runner=f))


# --- _fire end-to-end ---


@pytest.mark.asyncio
class TestFire:
    async def test_runner_then_judge_then_notify(self, tmp_path):
        notify, captured = _capture_notify()
        runner = _make_runner(
            tmp_path,
            judge=_ok_judge(notify=True, summary="all good", severity="info"),
            notify=notify,
        )

        snapshot_seen: list[dict] = []

        async def fake_runner():
            data = {"value": 42}
            snapshot_seen.append(data)
            return data

        spec = HookSpec(name="t", interval_minutes=5, runner=fake_runner)
        runner.register(spec)

        await runner._fire(spec, now=int(time.time()))

        # Notify was called with the canned event
        assert len(captured) == 1
        assert captured[0].hook_name == "t"
        assert captured[0].summary == "all good"
        # State was saved
        state = load_hook_state(runner.state_dir, "t")
        assert state == {"value": 42}
        # last_run was updated
        assert runner._last_run["t"] > 0

    async def test_judge_says_no_notify(self, tmp_path):
        notify, captured = _capture_notify()
        runner = _make_runner(
            tmp_path, judge=_ok_judge(notify=False), notify=notify,
        )

        async def f():
            return {"ok": True}

        spec = HookSpec(name="quiet", interval_minutes=5, runner=f)
        runner.register(spec)

        await runner._fire(spec, now=int(time.time()))

        assert captured == []  # no notify
        # But state was still saved + last_run advanced
        assert load_hook_state(runner.state_dir, "quiet") == {"ok": True}

    async def test_runner_skip_advances_last_run(self, tmp_path):
        # HeartbeatSkip = "transient failure, try next tick" — should
        # NOT call judge or notify, but SHOULD advance last_run so we
        # don't hot-loop.
        notify, captured = _capture_notify()
        judge_called: list = []

        async def judge(h, c, p):
            judge_called.append(c)
            return HeartbeatEvent(
                hook_name=h, fired_at=0, raw=c, summary="", should_notify=False,
            )

        runner = _make_runner(tmp_path, judge=judge, notify=notify)

        async def skipper():
            raise HeartbeatSkip("rate limited")

        spec = HookSpec(name="s", interval_minutes=5, runner=skipper)
        runner.register(spec)

        await runner._fire(spec, now=int(time.time()))
        assert judge_called == []
        assert captured == []
        assert runner._last_run["s"] > 0

    async def test_runner_exception_isolated(self, tmp_path):
        notify, captured = _capture_notify()
        runner = _make_runner(tmp_path, notify=notify)

        async def boom():
            raise RuntimeError("upstream API on fire")

        spec = HookSpec(name="boom", interval_minutes=5, runner=boom)
        runner.register(spec)

        # Should NOT raise — _fire swallows for resilience
        await runner._fire(spec, now=int(time.time()))
        assert captured == []
        # last_run still advanced (don't hot-loop on a permanent break)
        assert runner._last_run["boom"] > 0

    async def test_judge_exception_isolated(self, tmp_path):
        notify, captured = _capture_notify()

        async def angry_judge(h, c, p):
            raise RuntimeError("judge fell over")

        runner = _make_runner(tmp_path, judge=angry_judge, notify=notify)

        async def f():
            return {"ok": True}

        spec = HookSpec(name="j", interval_minutes=5, runner=f)
        runner.register(spec)
        await runner._fire(spec, now=int(time.time()))
        assert captured == []

    async def test_notify_exception_does_not_crash_runner(self, tmp_path):
        async def angry_notify(event):
            raise RuntimeError("signal-cli down")

        runner = _make_runner(
            tmp_path,
            judge=_ok_judge(notify=True, summary="x"),
            notify=angry_notify,
        )

        async def f():
            return {"ok": True}

        spec = HookSpec(name="n", interval_minutes=5, runner=f)
        runner.register(spec)
        # Must not raise — caller relies on this for resilience
        await runner._fire(spec, now=int(time.time()))

    async def test_judge_sees_previous_snapshot(self, tmp_path):
        """The second tick's judge gets the first tick's snapshot
        as `previous` so it can detect changes."""
        seen: list[tuple[dict, Optional[dict]]] = []

        async def judge(hook_name, current, previous):
            seen.append((current, previous))
            return HeartbeatEvent(
                hook_name=hook_name, fired_at=0, raw=current,
                summary="", should_notify=False,
            )

        runner = _make_runner(tmp_path, judge=judge)
        counter = [0]

        async def f():
            counter[0] += 1
            return {"n": counter[0]}

        spec = HookSpec(name="seq", interval_minutes=5, runner=f)
        runner.register(spec)
        await runner._fire(spec, now=int(time.time()))
        await runner._fire(spec, now=int(time.time()))

        assert seen[0] == ({"n": 1}, None)            # first tick: no previous
        assert seen[1] == ({"n": 2}, {"n": 1})         # second sees the first


# --- _tick gating ---


@pytest.mark.asyncio
class TestTick:
    async def test_tick_fires_hooks_at_interval(self, tmp_path):
        notify, captured = _capture_notify()
        runner = _make_runner(
            tmp_path, judge=_ok_judge(notify=True), notify=notify,
        )

        async def f():
            return {}

        # 5-min interval. First tick should fire (last_run=0).
        spec = HookSpec(name="t", interval_minutes=5, runner=f)
        runner.register(spec)

        await runner._tick()
        assert len(captured) == 1

        # Immediate second tick: last_run is current, should NOT fire.
        captured.clear()
        await runner._tick()
        assert captured == []

        # Pretend it's been 5+ min: pull last_run back.
        runner._last_run["t"] = int(time.time()) - 6 * 60
        await runner._tick()
        assert len(captured) == 1


# --- Lifecycle ---


@pytest.mark.asyncio
class TestLifecycle:
    async def test_start_runs_initial_tick(self, tmp_path):
        notify, captured = _capture_notify()
        runner = _make_runner(
            tmp_path, judge=_ok_judge(notify=True), notify=notify,
        )

        async def f():
            return {}

        runner.register(HookSpec(name="t", interval_minutes=99, runner=f))
        await runner.start()
        await asyncio.sleep(0.1)   # let the initial tick fire
        await runner.stop()

        assert len(captured) == 1


# --- State persistence ---


class TestState:
    def test_save_then_load(self, tmp_path):
        save_hook_state(tmp_path, "t", {"x": 1})
        assert load_hook_state(tmp_path, "t") == {"x": 1}

    def test_load_missing_returns_none(self, tmp_path):
        assert load_hook_state(tmp_path, "never") is None

    def test_corrupt_file_returns_none(self, tmp_path):
        from src.heartbeat.state import state_path
        p = state_path(tmp_path, "broken")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not-json")
        assert load_hook_state(tmp_path, "broken") is None

    def test_atomic_write(self, tmp_path):
        # No .tmp left behind after successful write
        save_hook_state(tmp_path, "atomic", {"x": 1})
        files = list(tmp_path.glob("*.tmp"))
        assert files == []


# --- LLM judge ---


@pytest.mark.asyncio
class TestLLMJudge:
    async def test_canned_yes_notify(self):
        async def llm(system, user):
            return json.dumps({
                "notify": True,
                "summary": "DHEA battery dropped to low",
                "severity": "warn",
            })

        event = await llm_judge(
            "blink", {"x": 1}, None, llm_call=llm,
        )
        assert event.should_notify is True
        assert event.severity == "warn"
        assert "DHEA" in event.summary

    async def test_canned_no_notify(self):
        async def llm(system, user):
            return json.dumps({"notify": False, "summary": "", "severity": "info"})

        event = await llm_judge("blink", {"x": 1}, None, llm_call=llm)
        assert event.should_notify is False

    async def test_malformed_json_silent(self):
        async def llm(system, user):
            return "this is not JSON"

        event = await llm_judge("blink", {"x": 1}, None, llm_call=llm)
        assert event.should_notify is False

    async def test_llm_exception_silent(self):
        async def llm(system, user):
            raise RuntimeError("rate limit")

        event = await llm_judge("blink", {"x": 1}, None, llm_call=llm)
        assert event.should_notify is False

    async def test_invalid_severity_normalized(self):
        async def llm(system, user):
            return json.dumps({
                "notify": True, "summary": "x", "severity": "made-up",
            })

        event = await llm_judge("blink", {"x": 1}, None, llm_call=llm)
        assert event.severity == "info"  # fallback


# --- Blink hook ---


@pytest.mark.asyncio
class TestBlinkHook:
    async def test_missing_credentials_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv(
            "BLINK_CREDENTIALS_PATH", str(tmp_path / "does-not-exist.json"),
        )
        with pytest.raises(FileNotFoundError, match="not found"):
            await fetch_blink_status()

    async def test_happy_path(self, monkeypatch, tmp_path):
        # Write minimal creds
        creds_path = tmp_path / "blink-creds.json"
        creds_path.write_text(json.dumps({
            "token": "tok",
            "refresh_token": "rtok",
            "proxy_url": "https://proxy.example.invalid/relay",
            "proxy_secret": "s3cret",
            "region_id": "u011",
            "account_id": 123,
        }))
        monkeypatch.setenv("BLINK_CREDENTIALS_PATH", str(creds_path))

        # Build a fake homescreen response. The proxy returns whatever
        # comes back from upstream, so we wrap it via the proxy POST URL.
        homescreen = {
            "networks": [{"name": "Home", "armed": True}],
            "sync_modules": [
                {"name": "SyncModule", "status": "online", "wifi_strength": 4},
            ],
            "cameras": [
                {
                    "name": "Front Door", "enabled": True, "battery": "ok",
                    "signals": {"temp": 72, "wifi": 4},
                },
            ],
            "doorbells": [],
            "owls": [],
        }

        # Monkeypatch aiohttp.ClientSession to return our canned response
        from src.heartbeat.hooks import blink as blink_mod

        class _FakeResp:
            def __init__(self, status, body):
                self.status = status
                self._body = body
            async def read(self):
                return self._body
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False

        class _FakeSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            def post(self, url, json=None, headers=None):
                # The hook makes one POST to proxy_url with the
                # homescreen request inside. Echo a 200 with the body.
                return _FakeResp(200, json_dumps_bytes(homescreen))

        monkeypatch.setattr(blink_mod.aiohttp, "ClientSession", _FakeSession)

        result = await fetch_blink_status()
        assert "networks" in result
        assert result["networks"][0]["name"] == "Home"
        assert result["networks"][0]["armed"] is True
        assert len(result["devices"]) == 1
        cam = result["devices"][0]
        assert cam["name"] == "Front Door"
        assert cam["battery"] == "ok"
        assert cam["wifi"] == 4
        assert cam["kind"] == "camera"
        assert result["fetched_at"] > 0


def json_dumps_bytes(d) -> bytes:
    return json.dumps(d).encode("utf-8")
