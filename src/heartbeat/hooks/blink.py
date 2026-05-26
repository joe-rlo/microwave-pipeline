"""Blink camera monitoring hook (Phase: first heartbeat hook).

Adapted from the standalone CLI script in OpenClaw
(`/Users/joe/Development/Claw Files/OpenClaw/scripts/blink.py`).

What this exposes:
- `fetch_blink_status()` — async fn returning structured snapshot.
  Used as a HookSpec.runner.
- `build_blink_hook_spec(interval_minutes)` — convenience constructor.

What it does NOT do:
- No CLI commands (snap, arm, disarm, events). Those still live in
  the original script. The heartbeat hook is monitoring-only —
  read-only fetch of the homescreen.
- No 2FA / full-reauth fallback. If the refresh-token flow fails,
  the hook raises and the runner logs + moves on. User re-runs
  `blink-local-auth.py` from OpenClaw to mint fresh creds.

Credentials path resolution (in order):
1. `BLINK_CREDENTIALS_PATH` env var
2. `~/.microwaveos/blink-credentials.json`
3. The OpenClaw default at
   `~/Development/Claw Files/OpenClaw/scripts/blink-credentials.json`

The hook itself runs against the existing CF Worker proxy so the
Blink API datacenter-IP block doesn't apply.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import aiohttp

from src.channels._http import make_session

log = logging.getLogger(__name__)


# 20s total timeout — Blink's API is usually quick; if it's not,
# fail fast and let the next tick try again.
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=20)


def _credentials_path() -> Path:
    """Resolve the Blink credentials file location."""
    raw = os.environ.get("BLINK_CREDENTIALS_PATH", "").strip()
    if raw:
        return Path(raw).expanduser()

    workspace_default = Path.home() / ".microwaveos" / "blink-credentials.json"
    if workspace_default.exists():
        return workspace_default

    openclaw_default = (
        Path.home() / "Development" / "Claw Files" / "OpenClaw"
        / "scripts" / "blink-credentials.json"
    )
    return openclaw_default


def _load_creds() -> dict:
    p = _credentials_path()
    if not p.exists():
        raise FileNotFoundError(
            f"Blink credentials not found at {p}. "
            "Set BLINK_CREDENTIALS_PATH or place a credentials JSON at "
            "~/.microwaveos/blink-credentials.json."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _save_creds(creds: dict) -> None:
    p = _credentials_path()
    p.write_text(json.dumps(creds, indent=2), encoding="utf-8")


def _base_url(creds: dict) -> str:
    return f"https://rest-{creds['region_id']}.immedia-semi.com"


async def _proxy_request(
    session: aiohttp.ClientSession,
    creds: dict,
    url: str,
    method: str = "GET",
    data: Optional[dict] = None,
) -> tuple[int, bytes]:
    """Send a request through the CF Worker proxy (datacenter IP bypass).

    Mirrors the same shape as the original blink.py — same payload
    fields, same auth headers. On 401 with "Unauthorized Access", we
    try a single token refresh and replay; if refresh fails, the
    caller gets the original 401 to surface.
    """
    payload = {
        "url": url,
        "method": method,
        "headers": {
            "Authorization": f"Bearer {creds['token']}",
            "Content-Type": "application/json",
        },
    }
    if data:
        payload["data"] = data

    async with session.post(
        creds["proxy_url"],
        json=payload,
        headers={
            "Authorization": f"Bearer {creds['proxy_secret']}",
            "Content-Type": "application/json",
        },
    ) as resp:
        body = await resp.read()
        if resp.status == 401 and b"Unauthorized" in body:
            log.info("[heartbeat-blink] token expired; refreshing")
            refreshed = await _refresh_token(session, creds)
            if refreshed:
                return await _proxy_request(session, refreshed, url, method, data)
            log.warning(
                "[heartbeat-blink] token refresh failed; "
                "re-run blink-local-auth.py to mint fresh creds"
            )
        return resp.status, body


async def _refresh_token(
    session: aiohttp.ClientSession, creds: dict,
) -> Optional[dict]:
    """OAuth refresh-token flow via the proxy. Returns updated creds on
    success, None on failure (caller should not retry indefinitely).

    Mirrors blink.py's refresh_token but does NOT fall back to full
    password reauth — that path is interactive (2FA prompt) and not
    appropriate for an unattended heartbeat. If refresh fails, the
    user gets a log line and re-runs the OpenClaw auth helper.
    """
    payload = {
        "url": "https://api.oauth.blink.com/oauth/token",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/26.1 Mobile/15E148 Safari/604.1"
            ),
            "APP-BUILD": "ANDROID_28373244",
        },
        "data": {
            "grant_type": "refresh_token",
            "refresh_token": creds["refresh_token"],
            "client_id": "android",
            "scope": "client",
        },
    }
    async with session.post(
        creds["proxy_url"],
        json=payload,
        headers={
            "Authorization": f"Bearer {creds['proxy_secret']}",
            "Content-Type": "application/json",
        },
    ) as resp:
        if resp.status != 200:
            return None
        data = await resp.json()

    if "access_token" not in data:
        return None

    creds["token"] = data["access_token"]
    creds["refresh_token"] = data.get("refresh_token", creds["refresh_token"])
    creds["expiration_date"] = time.time() + data.get("expires_in", 14400)
    _save_creds(creds)
    log.info("[heartbeat-blink] OAuth token refreshed successfully")
    return creds


def _device_snapshot(entry: dict, kind: str) -> dict:
    """Flatten a homescreen camera/doorbell/owl entry into the shape
    the heartbeat judge consumes."""
    sig = entry.get("signals") or {}
    return {
        "name": entry.get("name", "(unnamed)"),
        "enabled": bool(entry.get("enabled")),
        "battery": entry.get("battery") if kind == "camera" else None,
        "temp_f": sig.get("temp"),
        "wifi": sig.get("wifi", 0),
        "kind": kind,
    }


async def fetch_blink_status() -> dict:
    """Fetch a structured snapshot of the Blink system.

    Used directly as a HookSpec.runner. Returns:

      {
        "networks": [{"name", "armed"}, ...],
        "sync_modules": [{"name", "status", "wifi"}, ...],
        "devices": [{"name", "enabled", "battery", "temp_f",
                     "wifi", "kind"}, ...],
        "fetched_at": <epoch>,
      }
    """
    creds = _load_creds()
    # Use the shared session factory — provides certifi-backed SSL trust
    # so macOS python.org installs don't fail with CERTIFICATE_VERIFY_FAILED
    # the way bare aiohttp.ClientSession does. See src/channels/_http.py.
    async with make_session(timeout=_HTTP_TIMEOUT) as session:
        url = (
            f"{_base_url(creds)}/api/v3/accounts/"
            f"{creds['account_id']}/homescreen"
        )
        status, body = await _proxy_request(session, creds, url)
        if status != 200:
            raise RuntimeError(
                f"Blink homescreen returned {status}: {body[:200]!r}"
            )

        data = json.loads(body)

    networks = [
        {"name": n["name"], "armed": bool(n.get("armed"))}
        for n in data.get("networks", [])
    ]
    sync_modules = [
        {
            "name": sm["name"],
            "status": sm.get("status", "?"),
            "wifi": sm.get("wifi_strength", 0),
        }
        for sm in data.get("sync_modules", [])
    ]
    devices = (
        [_device_snapshot(c, "camera") for c in data.get("cameras", [])]
        + [_device_snapshot(d, "doorbell") for d in data.get("doorbells", [])]
        + [_device_snapshot(o, "owl") for o in data.get("owls", [])]
    )

    return {
        "networks": networks,
        "sync_modules": sync_modules,
        "devices": devices,
        "fetched_at": int(time.time()),
    }


def build_blink_hook_spec(interval_minutes: int = 15):
    """Construct the HookSpec for the Blink monitor.

    Imported lazily by the runner setup code so that environments
    without aiohttp / Blink creds don't trip an import error.
    """
    from src.heartbeat.runner import HookSpec

    return HookSpec(
        name="blink",
        interval_minutes=interval_minutes,
        runner=fetch_blink_status,
        description="Monitor Blink cameras: battery, wifi, armed state, offline detection.",
    )
