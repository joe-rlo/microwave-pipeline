"""Shared Blink API client primitives.

Two consumers today: the monitoring heartbeat (`src/heartbeat/hooks/blink.py`)
and the LLM-facing control tools (`src/tools/blink.py`). Auth flow,
credentials lookup, proxy plumbing, and homescreen-fetch are shared
here so neither consumer drifts from the other.

What this module DOES NOT do:
- 2FA / full password reauth — out of scope for unattended callers.
  When refresh fails, the user re-runs `blink-local-auth.py` from
  OpenClaw to mint fresh credentials.
- Storage encryption — credentials sit on disk in plaintext JSON.
  Permissions are the user's responsibility; we expect ~/.microwaveos
  to be a single-user directory.

All network calls route through the user's Cloudflare Worker proxy
(`proxy_url` field in the credentials JSON) so the Blink API's
datacenter-IP block doesn't apply.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import aiohttp

log = logging.getLogger(__name__)


# 20s total — Blink's API is usually fast; if it isn't, fail and let
# the next attempt deal with it rather than blocking a turn.
DEFAULT_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=20)


# --- Credentials ---------------------------------------------------------


def credentials_path() -> Path:
    """Resolve the Blink credentials file location.

    Lookup order:
    1. `BLINK_CREDENTIALS_PATH` env var
    2. `~/.microwaveos/blink-credentials.json`
    3. The OpenClaw default at
       `~/Development/Claw Files/OpenClaw/scripts/blink-credentials.json`

    The function does NOT check existence — callers can use this to
    print a helpful "set this env" message even when the file is missing.
    """
    raw = os.environ.get("BLINK_CREDENTIALS_PATH", "").strip()
    if raw:
        return Path(raw).expanduser()

    workspace_default = Path.home() / ".microwaveos" / "blink-credentials.json"
    if workspace_default.exists():
        return workspace_default

    return (
        Path.home() / "Development" / "Claw Files" / "OpenClaw"
        / "scripts" / "blink-credentials.json"
    )


def load_creds() -> dict:
    p = credentials_path()
    if not p.exists():
        raise FileNotFoundError(
            f"Blink credentials not found at {p}. "
            "Set BLINK_CREDENTIALS_PATH or place a credentials JSON at "
            "~/.microwaveos/blink-credentials.json."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def save_creds(creds: dict) -> None:
    credentials_path().write_text(json.dumps(creds, indent=2), encoding="utf-8")


def base_url(creds: dict) -> str:
    return f"https://rest-{creds['region_id']}.immedia-semi.com"


# --- HTTP (proxy + auth) --------------------------------------------------


async def proxy_request(
    session: aiohttp.ClientSession,
    creds: dict,
    url: str,
    method: str = "GET",
    data: Optional[dict] = None,
) -> tuple[int, bytes]:
    """Send a request through the CF Worker proxy (datacenter-IP bypass).

    On a 401 "Unauthorized Access", attempts a single token refresh and
    replays the request once. If refresh fails the original 401 surfaces
    so the caller can show a useful message to the user.
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
            log.info("[blink] token expired; refreshing")
            refreshed = await refresh_token(session, creds)
            if refreshed:
                return await proxy_request(session, refreshed, url, method, data)
            log.warning(
                "[blink] token refresh failed; "
                "re-run blink-local-auth.py to mint fresh creds"
            )
        return resp.status, body


async def refresh_token(
    session: aiohttp.ClientSession, creds: dict,
) -> Optional[dict]:
    """OAuth refresh-token flow via the proxy.

    Returns updated creds on success, None on failure. Does NOT fall
    back to full password reauth — that path is interactive (2FA) and
    only appropriate for the standalone `blink-local-auth.py`.
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
    save_creds(creds)
    log.info("[blink] OAuth token refreshed successfully")
    return creds


# --- High-level operations ------------------------------------------------


async def fetch_homescreen(
    session: aiohttp.ClientSession, creds: dict,
) -> dict:
    """Fetch the raw homescreen payload (networks + cameras + sync modules)."""
    url = f"{base_url(creds)}/api/v3/accounts/{creds['account_id']}/homescreen"
    status, body = await proxy_request(session, creds, url)
    if status != 200:
        raise RuntimeError(
            f"Blink homescreen returned {status}: {body[:200]!r}"
        )
    return json.loads(body)


async def set_network_armed(
    session: aiohttp.ClientSession,
    creds: dict,
    network_id: int,
    *,
    arm: bool,
) -> dict:
    """Arm or disarm a network. Returns the parsed JSON response.

    Endpoint URL form (older v1 path, still the one the OpenClaw CLI
    uses and what's currently confirmed working in the wild):
      POST /api/v1/accounts/{account_id}/networks/{network_id}/state/{arm|disarm}
    """
    action = "arm" if arm else "disarm"
    url = (
        f"{base_url(creds)}/api/v1/accounts/{creds['account_id']}"
        f"/networks/{network_id}/state/{action}"
    )
    status, body = await proxy_request(session, creds, url, method="POST")
    if status not in (200, 202):
        raise RuntimeError(
            f"Blink {action} failed (status {status}): {body[:200]!r}"
        )
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # Some Blink endpoints return empty bodies on success; that's fine.
        return {}


async def request_snapshot(
    session: aiohttp.ClientSession,
    creds: dict,
    network_id: int,
    camera_id: int,
) -> dict:
    """Trigger a fresh thumbnail capture on a camera.

    Returns the immediate command-acceptance payload (contains the
    command id callers can poll). Does NOT block on snap completion —
    polling for "complete" is the caller's choice.
    """
    url = (
        f"{base_url(creds)}/network/{network_id}/camera/{camera_id}/thumbnail"
    )
    status, body = await proxy_request(session, creds, url, method="POST")
    if status not in (200, 202):
        raise RuntimeError(
            f"Blink snapshot request failed (status {status}): {body[:200]!r}"
        )
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {}


# --- Selectors (find-by-name) --------------------------------------------


def find_network_by_name(homescreen: dict, name: str) -> Optional[dict]:
    """Find a network by case-insensitive substring match on its name.

    Returns the raw network dict (with `id`, `name`, `armed`) or None.
    """
    needle = (name or "").strip().lower()
    if not needle:
        return None
    for n in homescreen.get("networks", []):
        if needle in n.get("name", "").lower():
            return n
    return None


def find_camera_by_name(homescreen: dict, name: str) -> Optional[dict]:
    """Find a camera/doorbell/owl by case-insensitive substring match.

    Returns the raw device dict (with `id`, `name`, `network_id`,
    `type`) or None. Searches all three device buckets.
    """
    needle = (name or "").strip().lower()
    if not needle:
        return None
    for bucket in ("cameras", "doorbells", "owls"):
        for c in homescreen.get(bucket, []):
            if needle in c.get("name", "").lower():
                return c
    return None
