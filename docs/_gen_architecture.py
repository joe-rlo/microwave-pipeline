"""Generate docs/architecture.excalidraw from a Python description.

Run: `python3 docs/_gen_architecture.py` from the repo root.
The output is a single Excalidraw file you can open at excalidraw.com
(File → Open) or in the VSCode Excalidraw extension.

Why generate vs hand-author: Excalidraw's JSON has a lot of required
boilerplate per element. Keeping the layout in code lets future edits
land as small diffs instead of merge-conflict-prone giant JSONs.

Shared element factories + palette live in `_excalidraw_lib.py` —
sibling generators import from there too.
"""

from __future__ import annotations

import json
from pathlib import Path

from _excalidraw_lib import (
    rect, diamond, label, arrow,
    CHANNEL_BG, PIPE_BG, ROUTER_BG, LANE_NEAR_BG, LANE_BAA_BG, LANE_TEE_BG,
    STORE_BG, LOOP_BG, TOOL_BG,
)

OUT = Path(__file__).parent / "architecture.excalidraw"

elements: list = []


def add(*items):
    for it in items:
        if isinstance(it, list):
            elements.extend(it)
        else:
            elements.append(it)


# --- Title ----------------------------------------------------------------

add(label(x=40, y=20, text="MicrowaveOS-v2 — Architecture & Request Flow",
          font_size=28))
add(label(x=40, y=60,
          text="One request from a user message → reply, plus the background loops",
          font_size=14, color="#5c5f66"))


# --- (A) Input channels (top-left column) ---------------------------------

add(label(x=40, y=110, text="① Channels (input)", font_size=18))

signal_r, signal_t, signal_id = rect(
    x=40, y=140, w=200, h=60,
    text="Signal\n(signal-cli REST + WebSocket)", bg=CHANNEL_BG,
)
tg_r, tg_t, tg_id = rect(
    x=40, y=220, w=200, h=60,
    text="Telegram\n(Bot API + polling)", bg=CHANNEL_BG,
)
repl_r, repl_t, repl_id = rect(
    x=40, y=300, w=200, h=60,
    text="REPL / CLI\n(microwaveos chat)", bg=CHANNEL_BG,
)
add(signal_r, signal_t, tg_r, tg_t, repl_r, repl_t)

# Channel-side debounce + addendum-merge note
add(label(
    x=40, y=370,
    text="Channels: debounce window, addendum-merge,\n"
         "typing pause-resume, voice transcription,\n"
         "slash-command interception (/profile, /skill, /project)",
    font_size=11, color="#495057",
))


# --- (B) Orchestrator entry (top-middle) ----------------------------------

add(label(x=320, y=110, text="② Orchestrator.process()", font_size=18))

entry_r, entry_t, entry_id = rect(
    x=320, y=140, w=240, h=80,
    text="Pre-turn hooks\n• profile-reply intercept\n• skill activation",
    bg=PIPE_BG,
)
add(entry_r, entry_t)

triage_r, triage_t, triage_id = rect(
    x=320, y=240, w=240, h=80,
    text="③ Triage (Haiku)\nintent · complexity · phi_class\nmatched_skill",
    bg=PIPE_BG,
)
add(triage_r, triage_t)

assembly_r, assembly_t, assembly_id = rect(
    x=320, y=340, w=240, h=80,
    text="④ Assembly\nstable prompt + memory_context\n+ tool catalog + evidence",
    bg=PIPE_BG,
)
add(assembly_r, assembly_t)


# --- (C) Lane router (middle) ---------------------------------------------

add(label(x=640, y=110, text="⑤ Lane router", font_size=18))

router_r, router_t, router_id = diamond(
    x=640, y=270, w=220, h=180,
    text="phi_class? + topic?",
)
add(router_r, router_t)


# --- (D) LLM lanes (right of router) --------------------------------------

add(label(x=920, y=110, text="⑥ LLM session per turn", font_size=18))

near_r, near_t, near_id = rect(
    x=920, y=170, w=280, h=120,
    text="NEAR Cloud lane (default)\nprovider=NEAR · model=Sonnet\nescalates to Opus on `complex`\n\nTools: scheduler, blink, github,\ninstacart, webfetch, websearch, read_file",
    bg=LANE_NEAR_BG, font_size=12,
)
add(near_r, near_t)

baa_r, baa_t, baa_id = rect(
    x=920, y=310, w=280, h=140,
    text="BAA lane (PHI personal)\nprovider=Bedrock · model=Sonnet\nescalates to Opus on `complex`\n\nTools: health_profile_*  (allowlist)\nExternal tools BLOCKED by design",
    bg=LANE_BAA_BG, font_size=12,
)
add(baa_r, baa_t)

tee_r, tee_t, tee_id = rect(
    x=920, y=470, w=280, h=100,
    text="Private TEE lane\n(sensitive non-PHI)\nprovider=NEAR Private TEE\nGPT-OSS-120B (no escalation)",
    bg=LANE_TEE_BG, font_size=12,
)
add(tee_r, tee_t)

decline_r, decline_t, decline_id = rect(
    x=920, y=590, w=280, h=60,
    text="decline_phi\n(BAA not configured → safe refusal)",
    bg="#ffd8d8", font_size=12,
)
add(decline_r, decline_t)


# --- (E) Tool registry (above LLM lanes) ----------------------------------

add(label(x=1240, y=110, text="Tool registry (build_provider_tools)",
          font_size=14))

tools_r, tools_t, tools_id = rect(
    x=1240, y=140, w=300, h=440,
    text="scheduler_*  (always-on)\n"
         "  list / get / add / remove /\n"
         "  set_enabled\n\n"
         "blink_*  (gated: creds file)\n"
         "  status / arm / disarm / snap\n\n"
         "health_profile_*  (gated: HEALTH_ON)\n"
         "  summary / show / audit\n"
         "  (PHI-safe, BAA allowlist)\n\n"
         "github_*  (gated: GITHUB_TOKEN)\n"
         "instacart_create_cart\n"
         "  (gated: INSTACART_API_KEY)\n\n"
         "webfetch / websearch / read_file\n"
         "  (always-on, env-disable)",
    bg=TOOL_BG, font_size=11,
)
add(tools_r, tools_t)


# --- (F) Reflection + post-turn -------------------------------------------

add(label(x=320, y=470, text="⑦ Reflection", font_size=18))

reflect_r, reflect_t, reflect_id = rect(
    x=320, y=500, w=240, h=80,
    text="off | regex | normal | deep\n(skipped on simple turns,\n deep on complex)",
    bg=PIPE_BG,
)
add(reflect_r, reflect_t)

add(label(x=320, y=600, text="⑧ Post-turn hooks", font_size=18))

post_r, post_t, post_id = rect(
    x=320, y=630, w=240, h=100,
    text="Memory extraction\n+ profile proposal extractor (PHI)\n+ contradiction queue (3.4)\n+ turn audit log",
    bg=PIPE_BG,
)
add(post_r, post_t)


# --- (G) Channel out ------------------------------------------------------

add(label(x=40, y=470, text="⑨ Channel out", font_size=18))

out_r, out_t, out_id = rect(
    x=40, y=500, w=200, h=200,
    text="Reply assembly:\n"
         "• plain text\n"
         "• HTML card-view\n"
         "• paragraph chunks\n"
         "• voice (Whisper/TTS)\n"
         "• attachments\n\n"
         "Signal: typing\nindicator + nudge",
    bg=CHANNEL_BG, font_size=12,
)
add(out_r, out_t)


# --- (H) Data stores (bottom row, all one SQLite via apsw + WAL) ----------

add(label(x=40, y=780, text="⑩ Single SQLite (~/.microwaveos/data/memory.db, WAL)",
          font_size=18))

memory_r, memory_t, memory_id = rect(
    x=40, y=820, w=220, h=110,
    text="messages · turns · session_summary\nvector index (sqlite-vec)\nuser_prefs · breadcrumbs",
    bg=STORE_BG, font_size=12,
)
profile_r, profile_t, profile_id = rect(
    x=290, y=820, w=220, h=110,
    text="health_profiles\n(per-user AES-GCM, key derived\nvia HKDF from Keychain master)\nprofile_change_log",
    bg=STORE_BG, font_size=12,
)
sched_r, sched_t, sched_id = rect(
    x=540, y=820, w=220, h=110,
    text="scheduled_jobs\n(cron / direct / script / llm-skill)\nheartbeat_state",
    bg=STORE_BG, font_size=12,
)
audit_r, audit_t, audit_id = rect(
    x=790, y=820, w=220, h=110,
    text="health_audit\n(route + phi_class + topic +\nprovider + model per turn)",
    bg=STORE_BG, font_size=12,
)
add(memory_r, memory_t, profile_r, profile_t,
    sched_r, sched_t, audit_r, audit_t)


# --- (I) Background loops (right column) ----------------------------------

add(label(x=1040, y=780, text="⑪ Background loops (asyncio tasks)",
          font_size=18))

sched_loop_r, sched_loop_t, sched_loop_id = rect(
    x=1040, y=820, w=240, h=120,
    text="Scheduler  (30s tick)\n"
         "reads scheduled_jobs\n"
         "fires llm/direct/script jobs\n"
         "delivers via channel sender\n"
         "skips stale > 5min",
    bg=LOOP_BG, font_size=12,
)
hb_r, hb_t, hb_id = rect(
    x=1300, y=820, w=240, h=120,
    text="Heartbeat  (60s tick)\n"
         "runs HookSpec.runner fns\n"
         "  • blink camera status\n"
         "judge → notify on transition\n"
         "per-hook interval + failure isolation",
    bg=LOOP_BG, font_size=12,
)
consol_r, consol_t, consol_id = rect(
    x=1560, y=820, w=240, h=120,
    text="Consolidation (daily)\n"
         "rolls turns → memories\n"
         "writes session_summary\n"
         "computes daily-notes\n"
         "drives reflection embeddings",
    bg=LOOP_BG, font_size=12,
)
add(sched_loop_r, sched_loop_t, hb_r, hb_t, consol_r, consol_t)


# --- Arrows ---------------------------------------------------------------

def link(src_id, dst_id, **kw):
    # Resolve ids → element dicts so arrow() can compute real geometry.
    src = next(e for e in elements if e["id"] == src_id)
    dst = next(e for e in elements if e["id"] == dst_id)
    el, extras = arrow(src=src, dst=dst, **kw)
    add(el)
    add(*extras)


# Channels → orchestrator entry
link(signal_id, entry_id)
link(tg_id, entry_id)
link(repl_id, entry_id)

# Pipeline stages
link(entry_id, triage_id)
link(triage_id, assembly_id)
link(assembly_id, router_id)

# Router → lanes
link(router_id, near_id, label_text="general / preference / question")
link(router_id, baa_id, label_text="phi_class=personal")
link(router_id, tee_id, label_text="sensitive non-PHI", dashed=True)
link(router_id, decline_id, label_text="BAA unconfigured", dashed=True)

# Tool registry feeds NEAR + BAA (BAA is filtered to allowlist)
link(tools_id, near_id, dashed=True, color="#868e96")
link(tools_id, baa_id, dashed=True, color="#868e96")

# Lanes → reflection
link(near_id, reflect_id, color="#5c7cfa")
link(baa_id, reflect_id, color="#5c7cfa")
link(tee_id, reflect_id, color="#5c7cfa")

# Reflection → post-turn → out
link(reflect_id, post_id)
link(post_id, out_id)

# Post-turn writes to stores
link(post_id, memory_id, dashed=True, color="#fd7e14")
link(post_id, profile_id, dashed=True, color="#fd7e14")
link(post_id, audit_id, dashed=True, color="#fd7e14")

# Stores feed assembly back
link(memory_id, assembly_id, dashed=True, color="#15aabf")

# Background loops read/write same stores
link(sched_loop_id, sched_id, dashed=True, color="#7950f2")
link(hb_id, sched_id, dashed=True, color="#7950f2")
link(consol_id, memory_id, dashed=True, color="#7950f2")

# Scheduler loop also delivers via channels (cron → user)
link(sched_loop_id, out_id, dashed=True, color="#7950f2",
     label_text="cron deliveries")


# --- Legend ---------------------------------------------------------------

add(label(x=40, y=960, text="Legend", font_size=16))
add(label(x=40, y=985,
          text="solid arrow = synchronous pipeline step",
          font_size=11, color="#495057"))
add(label(x=40, y=1005,
          text="dashed = async / cross-cutting / read-back",
          font_size=11, color="#495057"))
add(label(x=40, y=1025,
          text="orange = writes to a store · teal = reads from a store",
          font_size=11, color="#495057"))
add(label(x=40, y=1045,
          text="purple = background loop activity",
          font_size=11, color="#495057"))


# --- Wrap & write ---------------------------------------------------------

doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {
        "gridSize": None,
        "viewBackgroundColor": "#ffffff",
    },
    "files": {},
}

OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
print(f"wrote {OUT} ({len(elements)} elements)")
