"""Generate docs/architecture-request-routing.excalidraw.

A SIMPLIFIED, one-screen view of how an incoming message gets routed —
triage, then a decision tree that lands each message in a destination
chosen by sensitivity (everyday / sealed-TEE / HIPAA-BAA). Deliberately
omits the per-lane internals (evidence retrieval, reflection, audit,
profile store) — those live in the detailed architecture-health-routing
diagram. Same lib + palette so the two read as a family.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _excalidraw_lib import (  # type: ignore
    rect, diamond, label, arrow,
    CHANNEL_BG, PIPE_BG, ROUTER_BG, LANE_NEAR_BG, LANE_BAA_BG, LANE_TEE_BG,
)

OUT = Path(__file__).parent / "architecture-request-routing.excalidraw"

DECLINE_BG = "#ffd8d8"

elements: list = []
def add(*items):
    for it in items:
        if isinstance(it, list):
            elements.extend(it)
        else:
            elements.append(it)


# --- Title ---
add(label(x=40, y=20,
          text="MicrowaveOS — Request Routing (simplified)", font_size=26))
add(label(x=40, y=58,
          text="Every message is classified, then routed to a destination by "
               "sensitivity. Full detail: architecture-health-routing.excalidraw.",
          font_size=12, color="#5c5f66"))


# --- Spine: in -> triage -> router ---
in_r, in_t, in_id = rect(
    x=535, y=100, w=230, h=56,
    text="Message in\n(Signal · Telegram · REPL)", bg=CHANNEL_BG, font_size=13,
)
triage_r, triage_t, triage_id = rect(
    x=520, y=192, w=260, h=74,
    text="① Triage (fast model)\nintent · complexity · phi_class", bg=PIPE_BG,
    font_size=13,
)
router_r, router_t, router_id = diamond(
    x=535, y=300, w=230, h=120, text="② Router\nphi_class?",
)
add(in_r, in_t, triage_r, triage_t, router_r, router_t)


# --- Sub-decisions ---
priv_r, priv_t, priv_id = diamond(
    x=440, y=472, w=210, h=110, text="privacy\nmode?",
)
baa_r, baa_t, baa_id = diamond(
    x=960, y=472, w=210, h=110, text="BAA\nconfigured?",
)
add(priv_r, priv_t, baa_r, baa_t)


# --- Destination lanes (one row) ---
LANE_Y, LANE_H, LANE_W = 632, 124, 230

a1_r, a1_t, a1_id = rect(
    x=40, y=LANE_Y, w=LANE_W, h=LANE_H,
    text="Standard pipeline\n→ Everyday model\nClaude (Max) /\nNEAR Cloud (de-identified)",
    bg=PIPE_BG, font_size=12,
)
b1_r, b1_t, b1_id = rect(
    x=300, y=LANE_Y, w=LANE_W, h=LANE_H,
    text="NEAR Cloud\nAnonymised Claude\n+ cited evidence",
    bg=LANE_NEAR_BG, font_size=12,
)
b2_r, b2_t, b2_id = rect(
    x=560, y=LANE_Y, w=LANE_W, h=LANE_H,
    text="NEAR Private TEE\nsealed open-weight\n(GPT-OSS / Qwen)\nhost can't read it",
    bg=LANE_TEE_BG, font_size=12,
)
c1_r, c1_t, c1_id = rect(
    x=820, y=LANE_Y, w=LANE_W, h=LANE_H,
    text="BAA lane\nAWS Bedrock\nHIPAA-covered",
    bg=LANE_BAA_BG, font_size=12,
)
c2_r, c2_t, c2_id = rect(
    x=1080, y=LANE_Y, w=LANE_W, h=LANE_H,
    text="Decline (safe refusal)\nrephrase / enable BAA\nno LLM call",
    bg=DECLINE_BG, font_size=12,
)
add(a1_r, a1_t, b1_r, b1_t, b2_r, b2_t, c1_r, c1_t, c2_r, c2_t)


# --- Converge to user ---
ans_r, ans_t, ans_id = rect(
    x=480, y=812, w=340, h=60, text="Answer to user", bg=CHANNEL_BG, font_size=14,
)
add(ans_r, ans_t)


# --- Arrows ---
def link(src_id, dst_id, **kw):
    src = next(e for e in elements if e["id"] == src_id)
    dst = next(e for e in elements if e["id"] == dst_id)
    el, extras = arrow(src=src, dst=dst, **kw)
    add(el)
    add(*extras)


# Spine
link(in_id, triage_id, src_side="bottom", dst_side="top")
link(triage_id, router_id, src_side="bottom", dst_side="top")

# Router -> three branches
link(router_id, a1_id, label_text="none", src_side="bottom", dst_side="top")
link(router_id, priv_id, label_text="general (abstract)", src_side="bottom", dst_side="top")
link(router_id, baa_id, label_text="personal / unknown", src_side="bottom", dst_side="top")

# privacy mode? -> NEAR / TEE
link(priv_id, b1_id, label_text="standard", src_side="bottom", dst_side="top")
link(priv_id, b2_id, label_text="private", src_side="bottom", dst_side="top")

# BAA configured? -> Bedrock / decline
link(baa_id, c1_id, label_text="yes", src_side="bottom", dst_side="top")
link(baa_id, c2_id, label_text="no", dashed=True, src_side="bottom", dst_side="top")

# Lanes -> answer
link(a1_id, ans_id, src_side="bottom", dst_side="top")
link(b1_id, ans_id, src_side="bottom", dst_side="top")
link(b2_id, ans_id, src_side="bottom", dst_side="top")
link(c1_id, ans_id, src_side="bottom", dst_side="top")
link(c2_id, ans_id, dashed=True, src_side="bottom", dst_side="top")  # safety message


# --- Legend (privacy tiers) ---
add(label(x=40, y=905, text="Privacy tiers:", font_size=14))
legend = [
    (PIPE_BG, "Tier 1 — Everyday (de-identified)"),
    (LANE_TEE_BG, "Tier 2 — Sealed TEE (host can't read it)"),
    (LANE_BAA_BG, "Tier 3 — HIPAA / BAA (personal health)"),
]
lx = 40
for color, text in legend:
    sw_r, sw_t, _ = rect(x=lx, y=935, w=22, h=22, text="", bg=color)
    add(sw_r, sw_t)
    add(label(x=lx + 30, y=937, text=text, font_size=12, color="#495057"))
    lx += max(260, int(len(text) * 8))


# --- Wrap ---
doc = {
    "type": "excalidraw", "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
    "files": {},
}
OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
print(f"wrote {OUT} ({len(elements)} elements)")
