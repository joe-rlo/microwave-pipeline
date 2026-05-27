"""Blink camera monitoring hook (first heartbeat hook).

Thin wrapper around `src.integrations.blink` — the heartbeat job is
purely read-only (homescreen snapshot → judge → optional notify).

Control operations (arm, disarm, snap) live in `src.tools.blink`
where the LLM can reach them.
"""

from __future__ import annotations

import logging
import time

from src.channels._http import make_session
from src.integrations import blink as blink_client

log = logging.getLogger(__name__)


_HTTP_TIMEOUT = blink_client.DEFAULT_HTTP_TIMEOUT


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
    creds = blink_client.load_creds()
    # Use the shared session factory — provides certifi-backed SSL trust
    # so macOS python.org installs don't fail with CERTIFICATE_VERIFY_FAILED
    # the way bare aiohttp.ClientSession does. See src/channels/_http.py.
    async with make_session(timeout=_HTTP_TIMEOUT) as session:
        data = await blink_client.fetch_homescreen(session, creds)

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
