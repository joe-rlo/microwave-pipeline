"""Tests for the LLM-facing Blink tools.

All HTTP is faked. The hook routes through `src.channels._http.make_session`
(certifi-backed) — we monkeypatch that factory in `src.tools.blink` so
no real network call is attempted.

The proxy plumbing is bounce-routed: every request is a POST to the user's
proxy_url with the real Blink URL inside `json={"url": ..., ...}`. The
fake session inspects that inner URL and responds accordingly — so a
single fake session covers homescreen lookups + arm/disarm/snap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# --- Test fixtures: fake aiohttp session ---------------------------------


def _jb(d) -> bytes:
    return json.dumps(d).encode("utf-8")


class _FakeResp:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    async def read(self) -> bytes:
        return self._body

    async def json(self):
        return json.loads(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class _FakeSession:
    """Records inner-URL POSTs and returns canned responses.

    Pass `routes` as a dict mapping URL-substring → (status, body_dict).
    The first key that's contained in the inner URL wins; missing routes
    return 404.
    """

    def __init__(self, routes: dict[str, tuple[int, dict]]):
        self.routes = routes
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def post(self, url, json=None, headers=None):
        inner = (json or {}).get("url", "")
        self.calls.append({
            "proxy_url": url,
            "inner_url": inner,
            "inner_method": (json or {}).get("method", "GET"),
            "inner_data": (json or {}).get("data"),
        })
        for key, (status, body) in self.routes.items():
            if key in inner:
                return _FakeResp(status, _jb(body))
        return _FakeResp(404, b'{"error": "no fake route"}')


def _seed_creds(tmp_path: Path, monkeypatch) -> Path:
    """Write a minimal credentials file and point BLINK_CREDENTIALS_PATH at it."""
    creds_path = tmp_path / "blink-credentials.json"
    creds_path.write_text(json.dumps({
        "token": "tok",
        "refresh_token": "rtok",
        "region_id": "u011",
        "account_id": 123,
        "proxy_url": "https://proxy.example.invalid/blink",
        "proxy_secret": "supersecret",
    }))
    monkeypatch.setenv("BLINK_CREDENTIALS_PATH", str(creds_path))
    return creds_path


_HOMESCREEN = {
    "networks": [
        {"id": 901, "name": "Home", "armed": True},
        {"id": 902, "name": "Garage", "armed": False},
    ],
    "sync_modules": [
        {"name": "SyncModule", "status": "online", "wifi_strength": 4},
    ],
    "cameras": [
        {
            "id": 11, "network_id": 901, "name": "Front Door",
            "enabled": True, "battery": "ok",
            "signals": {"temp": 72, "wifi": 4},
        },
        {
            "id": 12, "network_id": 901, "name": "Back Yard",
            "enabled": True, "battery": "low",
            "signals": {"temp": 68, "wifi": 3},
        },
    ],
    "doorbells": [
        {
            "id": 21, "network_id": 901, "name": "Doorbell",
            "enabled": True, "signals": {"temp": 70, "wifi": 5},
        },
    ],
    "owls": [],
}


def _patch_session(monkeypatch, routes):
    session = _FakeSession(routes)
    from src.tools import blink as blink_tool
    monkeypatch.setattr(blink_tool, "make_session", lambda *a, **kw: session)
    return session


# --- credentials_available ------------------------------------------------


class TestCredentialsAvailable:
    def test_present(self, tmp_path, monkeypatch):
        _seed_creds(tmp_path, monkeypatch)
        from src.tools import blink as blink_tool
        assert blink_tool.credentials_available() is True

    def test_missing(self, tmp_path, monkeypatch):
        # Point at a path that doesn't exist (and also clear HOME-based
        # fallbacks so the test isn't sensitive to dev machine state).
        monkeypatch.setenv("BLINK_CREDENTIALS_PATH", str(tmp_path / "nope.json"))
        from src.tools import blink as blink_tool
        assert blink_tool.credentials_available() is False


# --- blink_status ---------------------------------------------------------


class TestStatus:
    @pytest.mark.asyncio
    async def test_returns_structured_snapshot(self, tmp_path, monkeypatch):
        _seed_creds(tmp_path, monkeypatch)
        _patch_session(monkeypatch, {"/homescreen": (200, _HOMESCREEN)})

        from src.tools import blink as blink_tool
        out = json.loads(await blink_tool._handle_status({}))

        assert {n["name"] for n in out["networks"]} == {"Home", "Garage"}
        assert out["networks"][0]["armed"] is True
        # All device kinds flatten into one devices list
        names = {d["name"] for d in out["devices"]}
        assert {"Front Door", "Back Yard", "Doorbell"} == names
        # Kind labeling preserved
        kinds = {d["kind"] for d in out["devices"]}
        assert kinds == {"camera", "doorbell"}

    @pytest.mark.asyncio
    async def test_propagates_http_failure(self, tmp_path, monkeypatch):
        _seed_creds(tmp_path, monkeypatch)
        _patch_session(monkeypatch, {"/homescreen": (500, {"err": "boom"})})

        from src.tools import blink as blink_tool
        with pytest.raises(RuntimeError, match="500"):
            await blink_tool._handle_status({})


# --- blink_arm / blink_disarm --------------------------------------------


class TestArmDisarm:
    @pytest.mark.asyncio
    async def test_arm_by_substring(self, tmp_path, monkeypatch):
        _seed_creds(tmp_path, monkeypatch)
        session = _patch_session(monkeypatch, {
            "/homescreen": (200, _HOMESCREEN),
            "/networks/902/state/arm": (200, {"ok": True}),
        })
        from src.tools import blink as blink_tool
        out = json.loads(await blink_tool._handle_arm({"network": "gar"}))
        assert out == {"status": "armed", "network": "Garage", "network_id": 902}
        # Confirm we hit the arm endpoint with the matched network id
        assert any("/networks/902/state/arm" in c["inner_url"] for c in session.calls)

    @pytest.mark.asyncio
    async def test_disarm_by_substring(self, tmp_path, monkeypatch):
        _seed_creds(tmp_path, monkeypatch)
        session = _patch_session(monkeypatch, {
            "/homescreen": (200, _HOMESCREEN),
            "/networks/901/state/disarm": (200, {"ok": True}),
        })
        from src.tools import blink as blink_tool
        out = json.loads(await blink_tool._handle_disarm({"network": "home"}))
        assert out["status"] == "disarmed"
        assert out["network"] == "Home"

    @pytest.mark.asyncio
    async def test_unknown_network_lists_available(self, tmp_path, monkeypatch):
        _seed_creds(tmp_path, monkeypatch)
        _patch_session(monkeypatch, {"/homescreen": (200, _HOMESCREEN)})
        from src.tools import blink as blink_tool
        with pytest.raises(RuntimeError) as exc:
            await blink_tool._handle_arm({"network": "basement"})
        msg = str(exc.value)
        # The error must surface what IS available so the LLM can self-correct
        assert "Home" in msg and "Garage" in msg

    @pytest.mark.asyncio
    async def test_missing_arg_raises(self, tmp_path, monkeypatch):
        _seed_creds(tmp_path, monkeypatch)
        from src.tools import blink as blink_tool
        with pytest.raises(RuntimeError, match="requires `network`"):
            await blink_tool._handle_disarm({})

    @pytest.mark.asyncio
    async def test_arm_failure_surfaces_status(self, tmp_path, monkeypatch):
        _seed_creds(tmp_path, monkeypatch)
        _patch_session(monkeypatch, {
            "/homescreen": (200, _HOMESCREEN),
            "/networks/901/state/arm": (500, {"err": "blink down"}),
        })
        from src.tools import blink as blink_tool
        with pytest.raises(RuntimeError, match="arm failed.*500"):
            await blink_tool._handle_arm({"network": "home"})


# --- blink_snap ----------------------------------------------------------


class TestSnap:
    @pytest.mark.asyncio
    async def test_snap_returns_command_id(self, tmp_path, monkeypatch):
        _seed_creds(tmp_path, monkeypatch)
        session = _patch_session(monkeypatch, {
            "/homescreen": (200, _HOMESCREEN),
            "/network/901/camera/11/thumbnail": (202, {"id": 999, "complete": False}),
        })
        from src.tools import blink as blink_tool
        out = json.loads(await blink_tool._handle_snap({"camera": "front"}))
        assert out["status"] == "snapshot_requested"
        assert out["camera"] == "Front Door"
        assert out["command_id"] == 999
        # No polling — we want to return immediately
        snap_calls = [c for c in session.calls if "/thumbnail" in c["inner_url"]]
        assert len(snap_calls) == 1
        assert snap_calls[0]["inner_method"] == "POST"

    @pytest.mark.asyncio
    async def test_snap_handles_doorbell(self, tmp_path, monkeypatch):
        _seed_creds(tmp_path, monkeypatch)
        _patch_session(monkeypatch, {
            "/homescreen": (200, _HOMESCREEN),
            "/network/901/camera/21/thumbnail": (200, {"id": 1234}),
        })
        from src.tools import blink as blink_tool
        out = json.loads(await blink_tool._handle_snap({"camera": "doorbell"}))
        assert out["camera"] == "Doorbell"

    @pytest.mark.asyncio
    async def test_snap_unknown_camera_lists_available(self, tmp_path, monkeypatch):
        _seed_creds(tmp_path, monkeypatch)
        _patch_session(monkeypatch, {"/homescreen": (200, _HOMESCREEN)})
        from src.tools import blink as blink_tool
        with pytest.raises(RuntimeError) as exc:
            await blink_tool._handle_snap({"camera": "kitchen"})
        msg = str(exc.value)
        assert "Front Door" in msg and "Doorbell" in msg


# --- Registry wiring ------------------------------------------------------


class TestRegistryWiring:
    """The orchestrator reads handlers via build_provider_tools and docs
    via build_tools.catalog_text. If either side stops listing Blink, the
    bot regresses to the original "no Blink integration" failure mode."""

    def test_provider_tools_register_when_creds_present(self, tmp_path, monkeypatch):
        _seed_creds(tmp_path, monkeypatch)
        # Isolate from other env-keyed tools
        monkeypatch.setenv("WEB_TOOLS_DISABLED", "1")
        monkeypatch.setenv("FILE_TOOLS_DISABLED", "1")
        monkeypatch.setenv("WEBSEARCH_DISABLED", "1")
        from types import SimpleNamespace
        from src.tools import build_provider_tools
        cfg = SimpleNamespace(
            db_path=tmp_path / "t.db",
            workspace_dir=tmp_path / "ws",
            heartbeat_notify_channel="signal",
            heartbeat_notify_recipient="+1",
            instacart_api_key="", github_token="",
        )
        names = {t.definition.name for t in build_provider_tools(cfg)}
        assert {
            "blink_status", "blink_arm", "blink_disarm", "blink_snap",
        } <= names

    def test_provider_tools_omit_blink_when_no_creds(self, tmp_path, monkeypatch):
        # Point at a non-existent file — gating must drop the tools cleanly.
        monkeypatch.setenv("BLINK_CREDENTIALS_PATH", str(tmp_path / "nope.json"))
        monkeypatch.setenv("WEB_TOOLS_DISABLED", "1")
        monkeypatch.setenv("FILE_TOOLS_DISABLED", "1")
        monkeypatch.setenv("WEBSEARCH_DISABLED", "1")
        from types import SimpleNamespace
        from src.tools import build_provider_tools
        cfg = SimpleNamespace(
            db_path=tmp_path / "t.db",
            workspace_dir=tmp_path / "ws",
            heartbeat_notify_channel="signal",
            heartbeat_notify_recipient="+1",
            instacart_api_key="", github_token="",
        )
        names = {t.definition.name for t in build_provider_tools(cfg)}
        assert not any(n.startswith("blink_") for n in names)

    def test_catalog_text_mentions_blink_when_creds_present(self, tmp_path, monkeypatch):
        _seed_creds(tmp_path, monkeypatch)
        from types import SimpleNamespace
        from src.tools import build_tools
        cfg = SimpleNamespace(
            instacart_api_key="", github_token="",
            workspace_dir=tmp_path / "ws",
        )
        bundle = build_tools(cfg)
        if not bundle.catalog_text:
            pytest.skip("SDK not available; catalog path returned empty")
        assert "blink_arm" in bundle.catalog_text
        assert "blink_disarm" in bundle.catalog_text
