"""Generate docs/architecture-health-profile.excalidraw.

Updated version of the original `Microwave OS Health profile.excalidraw`
(May 2026). Profile READ + WRITE flows, encryption story, confirmation
surfaces, and the slash + LLM tool surfaces.

What changed since the original:
- READ flow simplified: no slice-selector (Haiku doesn't pre-pick
  sections). LLM on the BAA lane calls `health_profile_*` tools
  on demand instead. The tool wiring landed today (2026-05-28).
- /profile commands today: summary, show, audit, clear "phrase",
  export. The original showed edit / delete / undo / review —
  those are deferred (chat.py docstring lists them as future work).
- pending_updates queue still lives INSIDE the profile JSON.
- Confirmation surfaces: inline footer + auto_expire (24h) work
  today; batch `/profile review` is still deferred.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _excalidraw_lib import (  # type: ignore
    rect, diamond, label, arrow,
    CHANNEL_BG, PIPE_BG, ROUTER_BG, LANE_NEAR_BG, LANE_BAA_BG, LANE_TEE_BG,
    STORE_BG, LOOP_BG, TOOL_BG,
)

OUT = Path(__file__).parent / "architecture-health-profile.excalidraw"

elements: list = []
def add(*items):
    for it in items:
        if isinstance(it, list):
            elements.extend(it)
        else:
            elements.append(it)


# --- Title ---
add(label(x=40, y=20,
          text="MicrowaveOS — Health Profile System (current)",
          font_size=26))
add(label(x=40, y=58,
          text="User-controlled · confirmation-gated · encrypted at rest. "
               "Updates the May-2026 health-profile diagram.",
          font_size=12, color="#5c5f66"))


# === READ FLOW (top half) =================================================

add(label(x=40, y=110,
          text="READ FLOW — BAA LLM fetches profile slices on demand via tools",
          font_size=16))

# Health turn (start)
r1, t1, r1_id = rect(
    x=40, y=150, w=180, h=60,
    text="Health turn arrives\n(phi_class=personal)", bg=CHANNEL_BG,
)
# Triage (existing)
r2, t2, r2_id = rect(
    x=250, y=150, w=180, h=60,
    text="Triage\nphi_class + topic\n(existing)", bg=PIPE_BG,
)
# Assembly (no profile block anymore)
r3, t3, r3_id = rect(
    x=460, y=150, w=220, h=80,
    text="Assembly\nNO profile slice spliced —\nLLM fetches via tools",
    bg=PIPE_BG,
)
# BAA LLM
r4, t4, r4_id = rect(
    x=710, y=150, w=240, h=80,
    text="BAA LLM (Bedrock Sonnet)\ncan call health_profile_* tools\nreads slice, never writes",
    bg=LANE_BAA_BG,
)
# Cited response
r5, t5, r5_id = rect(
    x=980, y=150, w=200, h=80,
    text="Cited response\nto user\n(+ inline confirmation\nfooter if proposals)",
    bg=CHANNEL_BG,
)
add(r1, t1, r2, t2, r3, t3, r4, t4, r5, t5)

# Tool boxes under BAA LLM
tools_r, tools_t, tools_id = rect(
    x=710, y=260, w=240, h=110,
    text="health_profile_summary\nhealth_profile_show <section>\nhealth_profile_audit [limit]\n\nDestructive ops NOT exposed",
    bg=TOOL_BG, font_size=11,
)
add(tools_r, tools_t)

# Profile store
store_r, store_t, store_id = rect(
    x=980, y=260, w=200, h=110,
    text="Encrypted profile\n(SQLite blob)\nload + decrypt per call\nAES-GCM, HKDF key",
    bg=STORE_BG, font_size=11,
)
add(store_r, store_t)


# === WRITE FLOW (middle) =================================================

add(label(x=40, y=410,
          text="WRITE FLOW — extractor proposes, user confirms, profile updates",
          font_size=16))

# Turn ends
w1, w1t, w1_id = rect(
    x=40, y=450, w=160, h=60, text="Turn ends", bg=CHANNEL_BG,
)
# Project active?
w2, w2t, w2_id = diamond(
    x=230, y=440, w=180, h=80, text="Project active\nor /private?", bg=ROUTER_BG,
)
# Skip
w3, w3t, w3_id = rect(
    x=440, y=440, w=180, h=80,
    text="Skip extraction\n(fiction or /private)", bg="#ffe3e3", font_size=11,
)
# Extractor
w4, w4t, w4_id = rect(
    x=440, y=550, w=180, h=80,
    text="Profile Extractor\n(Haiku)\nsees schema + counts only\nNOT existing values",
    bg=PIPE_BG, font_size=11,
)
# confidence gate
w5, w5t, w5_id = diamond(
    x=650, y=550, w=160, h=80, text="confidence\n≥ 0.7?", bg=ROUTER_BG,
)
# Discard
w6, w6t, w6_id = rect(
    x=840, y=540, w=160, h=60, text="Discard\n(too uncertain)",
    bg="#ffe3e3", font_size=11,
)
# Pending queue
w7, w7t, w7_id = rect(
    x=840, y=620, w=240, h=80,
    text="pending_updates queue\nlives INSIDE the profile,\nnot a separate table",
    bg=STORE_BG, font_size=11,
)
add(w1, w1t, w2, w2t, w3, w3t, w4, w4t, w5, w5t, w6, w6t, w7, w7t)


# === CONFIRMATION SURFACES (lower middle) =================================

add(label(x=40, y=720,
          text="USER CONFIRMATION — silence is not consent",
          font_size=16))

c1, c1t, c1_id = rect(
    x=40, y=760, w=240, h=120,
    text="Inline (single proposal)\nappended to next response:\n"
         "“should I note metformin?\nyes / no / edit”\n"
         "format_confirmation_footer()",
    bg="#d3f9d8", font_size=11,
)
c2, c2t, c2_id = rect(
    x=300, y=760, w=240, h=120,
    text="Batch review\n/profile review\n(DEFERRED — see chat.py\ndocstring; not in cli today)",
    bg="#ffe8cc", font_size=11,
)
c3, c3t, c3_id = rect(
    x=560, y=760, w=240, h=120,
    text="Auto-expire after 24h\nstatus → auto_expired\nNOT auto_accepted\n(silence is not consent)\nauto_expire_old()",
    bg="#d3f9d8", font_size=11,
)
add(c1, c1t, c2, c2t, c3, c3t)

# User decision diamond
dec_r, dec_t, dec_id = diamond(
    x=820, y=760, w=180, h=120, text="User\ndecision", bg=ROUTER_BG,
)
add(dec_r, dec_t)

# Accept / Edit-then-accept / Reject
acc_r, acc_t, acc_id = rect(
    x=1020, y=750, w=200, h=50,
    text="ACCEPT → write profile,\nlog change", bg="#d3f9d8", font_size=11,
)
edit_r, edit_t, edit_id = rect(
    x=1020, y=810, w=200, h=50,
    text="EDIT → modify values,\nthen write", bg="#fff3bf", font_size=11,
)
rej_r, rej_t, rej_id = rect(
    x=1020, y=870, w=200, h=50,
    text="REJECT → discard,\nlog rejection", bg="#ffe3e3", font_size=11,
)
add(acc_r, acc_t, edit_r, edit_t, rej_r, rej_t)


# === ENCRYPTED STORE (right column) =======================================

add(label(x=40, y=920, text="ENCRYPTED STORE (~/.microwaveos/data/memory.db)",
          font_size=16))

s1, s1t, s1_id = rect(
    x=40, y=960, w=240, h=140,
    text="health_profiles\n\n• user_id\n• user_key_id\n• encrypted_profile (blob)\n• profile_version\n\nAES-GCM, per-user HKDF key",
    bg=STORE_BG, font_size=11,
)
s2, s2t, s2_id = rect(
    x=300, y=960, w=240, h=140,
    text="profile_change_log\n\nencrypted diffs\nop + section + trigger source\nseparate from health_audit\n\nread via /profile audit",
    bg=STORE_BG, font_size=11,
)
s3, s3t, s3_id = rect(
    x=560, y=960, w=240, h=140,
    text="deleted_recently\n\n30-day soft-delete buffer\nfor undo (LIVES INSIDE\nthe profile JSON itself)\n\n/profile undo still deferred",
    bg=STORE_BG, font_size=11,
)
s4, s4t, s4_id = rect(
    x=820, y=960, w=260, h=140,
    text="Master key\n\nKeychain (solo)\nKMS or env (multi-user / CI)\nuser-derived (highest trust)\n\nset via\nPHI_ENCRYPTION_KEY_SOURCE",
    bg="#ffd8d8", font_size=11,
)
add(s1, s1t, s2, s2t, s3, s3t, s4, s4t)


# === USER-FACING SURFACE (bottom) =========================================

add(label(x=40, y=1130, text="USER-FACING SURFACE — slash commands + LLM tools",
          font_size=16))

u1, u1t, u1_id = rect(
    x=40, y=1170, w=240, h=90,
    text="/profile (summary)\n/profile show <section>\n/profile audit [N]\n(channel-side intercept,\nno LLM call)",
    bg=CHANNEL_BG, font_size=11,
)
u2, u2t, u2_id = rect(
    x=300, y=1170, w=240, h=90,
    text="/profile clear \"clear my profile\"\n/profile export\n(typed-phrase / file write —\nexplicit human in loop)",
    bg=CHANNEL_BG, font_size=11,
)
u3, u3t, u3_id = rect(
    x=560, y=1170, w=240, h=90,
    text="LLM tools (NEW, 2026-05-28)\nhealth_profile_summary\nhealth_profile_show\nhealth_profile_audit\nBAA lane only (allowlist)",
    bg=TOOL_BG, font_size=11,
)
u4, u4t, u4_id = rect(
    x=820, y=1170, w=260, h=90,
    text="DEFERRED\n/profile edit, /profile delete,\n/profile undo, /profile review\n(see chat.py docstring)",
    bg="#ffe8cc", font_size=11,
)
add(u1, u1t, u2, u2t, u3, u3t, u4, u4t)


# === Arrows ===============================================================

def link(src_id, dst_id, **kw):
    src = next(e for e in elements if e["id"] == src_id)
    dst = next(e for e in elements if e["id"] == dst_id)
    el, extras = arrow(src=src, dst=dst, **kw)
    add(el)
    add(*extras)

# READ flow
link(r1_id, r2_id)
link(r2_id, r3_id)
link(r3_id, r4_id)
link(r4_id, r5_id)
link(r4_id, tools_id, dashed=True, color="#fd7e14")
link(tools_id, store_id, dashed=True, color="#15aabf",
     label_text="decrypt slice")

# WRITE flow
link(w1_id, w2_id)
link(w2_id, w3_id, label_text="yes")
link(w2_id, w4_id, label_text="no")
link(w4_id, w5_id)
link(w5_id, w6_id, label_text="no")
link(w5_id, w7_id, label_text="yes")

# Pending → confirmation
link(w7_id, c1_id, dashed=True, color="#7950f2")
link(c1_id, dec_id)
link(c3_id, dec_id, dashed=True, label_text="(timeout)")
link(dec_id, acc_id)
link(dec_id, edit_id)
link(dec_id, rej_id)

# Accept / edit write back to store
link(acc_id, s1_id, dashed=True, color="#fd7e14",
     label_text="save_profile + log")
link(edit_id, s1_id, dashed=True, color="#fd7e14")
link(rej_id, s2_id, dashed=True, color="#fd7e14",
     label_text="log only")


# === Design principles (right margin) =====================================

add(label(x=820, y=410, text="DESIGN PRINCIPLES (unchanged)", font_size=14))
principles = [
    "• Visibility — user can see everything stored,",
    "  with provenance, in one command.",
    "• Confirmation before storage — extractor proposes,",
    "  storage requires explicit accept.",
    "• Silence ≠ consent — pending → auto_expired,",
    "  never auto_accepted.",
    "• Encryption at rest — per-user HKDF derived",
    "  from Keychain (or KMS) master.",
    "• PHI never leaves BAA boundary —",
    "  tool allowlist enforces this on the BAA lane.",
    "• Audit separately from health_audit —",
    "  profile_change_log is encrypted, per-user.",
]
for i, p in enumerate(principles):
    add(label(x=820, y=440 + i * 18, text=p, font_size=11, color="#495057"))


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
