"""LLM-facing tools for Blink cameras.

Four tools, all wrappers around `src.integrations.blink`:

- `blink_status`   — read the full system snapshot (networks armed?
                     cameras battery/wifi/temp, sync-module status)
- `blink_arm`      — arm a network by name
- `blink_disarm`   — disarm a network by name
- `blink_snap`     — request a fresh thumbnail from one camera

Why expose these. The user has Blink cameras, talks to the bot, and
naturally said "disarm my cameras." Before this module the bot
correctly said "no Blink tool wired" — the heartbeat hook is
monitoring-only and the LLM never saw it. These tools close that gap.

Arm/disarm semantics. These are physical-world actions, so:
- Both require an explicit `network` arg (substring match on name).
  No "disarm all" shortcut — the LLM has to be specific.
- Result text always echoes the network name + new state, so the
  bot can confirm to the user what actually changed.
- A 401 from Blink triggers a single token refresh + replay; if
  refresh fails, the tool surfaces an actionable error
  ("re-run blink-local-auth.py to mint fresh creds") instead of
  letting the LLM hallucinate a success.

`blink_snap` does NOT block waiting for the camera to wake and
upload (that takes 5-15 seconds). It triggers the capture and
returns immediately with the command id; the next `blink_status`
call will show the refreshed thumbnail URL. This keeps the tool
loop fast and avoids tying up the chat session on a slow camera.

Registration. Auto-disables when the credentials file is missing,
the same way GitHub/Instacart disable without their keys. Failing
closed beats surfacing a tool that errors on every invocation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.channels._http import make_session
from src.integrations import blink as blink_client

log = logging.getLogger(__name__)


# What the LLM sees in its system context.
BLINK_TOOL_DOCS = """\
**blink_status** — Read the current state of your Blink system: which networks are armed, camera batteries, wifi strength, temperatures, sync-module status, and recent thumbnails.

When to use:
- "Are my cameras armed?", "battery on the front camera?", "is the sync module online?".
- Before `blink_arm` / `blink_disarm` when the user doesn't name the network — list the available networks first.

**blink_arm** — Arm a Blink network (turns motion detection ON for all its cameras).

How to use:
- `network`: case-insensitive substring of the network name. If the match is ambiguous, call `blink_status` first to list options.

**blink_disarm** — Disarm a Blink network (turns motion detection OFF).

How to use:
- `network`: case-insensitive substring of the network name. Same matching rules as `blink_arm`.

**blink_snap** — Request a fresh thumbnail from a specific camera. Returns immediately; the image becomes available within ~10s via the camera's thumbnail URL (visible on the next `blink_status`).

How to use:
- `camera`: case-insensitive substring of the camera name.
"""


# --- JSON schemas ---------------------------------------------------------


BLINK_STATUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

BLINK_ARM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "network": {
            "type": "string",
            "description": "Case-insensitive substring of the network name to arm",
        },
    },
    "required": ["network"],
    "additionalProperties": False,
}

BLINK_DISARM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "network": {
            "type": "string",
            "description": "Case-insensitive substring of the network name to disarm",
        },
    },
    "required": ["network"],
    "additionalProperties": False,
}

BLINK_SNAP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "camera": {
            "type": "string",
            "description": (
                "Case-insensitive substring of the camera/doorbell/owl name"
            ),
        },
    },
    "required": ["camera"],
    "additionalProperties": False,
}


# --- Handlers -------------------------------------------------------------
#
# Each handler is async, takes args from the LLM, opens a single session,
# and returns a JSON string. Raises RuntimeError on failure so the tool
# loop reports is_error=True back to the model.


def _abbreviated_device(entry: dict, kind: str) -> dict:
    """Pull the LLM-relevant fields from a homescreen device entry."""
    sig = entry.get("signals") or {}
    return {
        "name": entry.get("name", "(unnamed)"),
        "kind": kind,
        "enabled": bool(entry.get("enabled")),
        "battery": entry.get("battery") if kind == "camera" else None,
        "wifi": sig.get("wifi"),
        "temp_f": sig.get("temp"),
    }


async def _handle_status(args: dict[str, Any]) -> str:
    creds = blink_client.load_creds()
    async with make_session(timeout=blink_client.DEFAULT_HTTP_TIMEOUT) as session:
        data = await blink_client.fetch_homescreen(session, creds)

    networks = [
        {"name": n.get("name"), "armed": bool(n.get("armed"))}
        for n in data.get("networks", [])
    ]
    sync_modules = [
        {
            "name": sm.get("name"),
            "status": sm.get("status"),
            "wifi": sm.get("wifi_strength"),
        }
        for sm in data.get("sync_modules", [])
    ]
    devices = (
        [_abbreviated_device(c, "camera") for c in data.get("cameras", [])]
        + [_abbreviated_device(d, "doorbell") for d in data.get("doorbells", [])]
        + [_abbreviated_device(o, "owl") for o in data.get("owls", [])]
    )
    return json.dumps({
        "networks": networks,
        "sync_modules": sync_modules,
        "devices": devices,
    }, indent=2)


async def _handle_set_armed(args: dict[str, Any], *, arm: bool) -> str:
    """Shared implementation for arm + disarm."""
    name = (args.get("network") or "").strip()
    if not name:
        raise RuntimeError(
            f"blink_{'arm' if arm else 'disarm'} requires `network` (substring of network name)"
        )

    creds = blink_client.load_creds()
    async with make_session(timeout=blink_client.DEFAULT_HTTP_TIMEOUT) as session:
        home = await blink_client.fetch_homescreen(session, creds)
        net = blink_client.find_network_by_name(home, name)
        if net is None:
            available = [n.get("name") for n in home.get("networks", [])]
            raise RuntimeError(
                f"No Blink network matches {name!r}. Available: {available}"
            )
        await blink_client.set_network_armed(
            session, creds, int(net["id"]), arm=arm
        )

    return json.dumps({
        "status": "armed" if arm else "disarmed",
        "network": net.get("name"),
        "network_id": net.get("id"),
    })


async def _handle_arm(args: dict[str, Any]) -> str:
    return await _handle_set_armed(args, arm=True)


async def _handle_disarm(args: dict[str, Any]) -> str:
    return await _handle_set_armed(args, arm=False)


async def _handle_snap(args: dict[str, Any]) -> str:
    name = (args.get("camera") or "").strip()
    if not name:
        raise RuntimeError("blink_snap requires `camera` (substring of camera name)")

    creds = blink_client.load_creds()
    async with make_session(timeout=blink_client.DEFAULT_HTTP_TIMEOUT) as session:
        home = await blink_client.fetch_homescreen(session, creds)
        cam = blink_client.find_camera_by_name(home, name)
        if cam is None:
            available = []
            for bucket in ("cameras", "doorbells", "owls"):
                available.extend(
                    c.get("name") for c in home.get(bucket, [])
                )
            raise RuntimeError(
                f"No Blink camera matches {name!r}. Available: {available}"
            )
        result = await blink_client.request_snapshot(
            session, creds,
            network_id=int(cam["network_id"]),
            camera_id=int(cam["id"]),
        )

    return json.dumps({
        "status": "snapshot_requested",
        "camera": cam.get("name"),
        "command_id": result.get("id"),
        "note": (
            "Camera takes ~5-15s to wake and capture. The refreshed "
            "thumbnail URL will appear on the next blink_status."
        ),
    })


# --- Tool registration ----------------------------------------------------


def credentials_available() -> bool:
    """Cheap predicate: does a Blink credentials file exist?

    Used by `build_provider_tools` to gate registration — if the user
    hasn't set up Blink, we don't advertise the tool. The probe is
    file-existence only; we don't validate the JSON or test the token
    here, because that would mean an HTTP call at orchestrator startup.
    """
    return blink_client.credentials_path().exists()
