"""Generate docs/architecture-health-routing.excalidraw.

Updated version of the original `microwave-health-architecture.excalidraw`
(May 2026). Pulls together the routing decisions a turn goes through
when it touches health, and where each lane lands.

What changed since the original:
- BAA lane now carries tools (health_profile_* via the allowlist in
  src/llm/factory.py). Original showed BAA as tool-less.
- `decline_phi` lane (when BAA is unconfigured) — wasn't in original.
- Private TEE lane for sensitive non-PHI — wasn't in original.
- Public APIs listed are only those actually wired today: pubmed +
  medlineplus. openFDA / CDC / ClinicalTrials.gov were aspirational
  in the original and have not yet been implemented.
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

OUT = Path(__file__).parent / "architecture-health-routing.excalidraw"

elements: list = []
def add(*items):
    for it in items:
        if isinstance(it, list):
            elements.extend(it)
        else:
            elements.append(it)


# --- Title ---
add(label(x=40, y=20,
          text="MicrowaveOS — Health Module Routing (current)",
          font_size=26))
add(label(x=40, y=58,
          text="Updates the May-2026 health-architecture diagram. Marks where "
               "current code diverges from original intent.",
          font_size=12, color="#5c5f66"))


# --- Input + extended triage ---
in_r, in_t, in_id = rect(
    x=40, y=110, w=240, h=60,
    text="Message in (any channel)", bg=CHANNEL_BG,
)
triage_r, triage_t, triage_id = rect(
    x=40, y=200, w=240, h=130,
    text="① Triage (Haiku)\n"
         "intent · complexity · skill_match\n"
         "+ phi_class\n"
         "+ health_topic",
    bg=PIPE_BG,
)
add(in_r, in_t, triage_r, triage_t)

router_r, router_t, router_id = diamond(
    x=40, y=370, w=240, h=170,
    text="② Health Router\nphi_class?",
)
add(router_r, router_t)


# --- General health path (no PHI) ---
add(label(x=360, y=110, text="③ General Health path  (phi_class = general)",
          font_size=15))

ev_g_r, ev_g_t, ev_g_id = rect(
    x=360, y=140, w=280, h=80,
    text="Evidence retrieval (parallel)\n"
         "• PubMed\n"
         "• MedlinePlus Connect",
    bg="#d3f9d8", font_size=12,
)
asm_g_r, asm_g_t, asm_g_id = rect(
    x=360, y=240, w=280, h=80,
    text="Assembly\n+ evidence + health-qa skill\n+ disclaimer",
    bg="#d3f9d8", font_size=12,
)
llm_g_r, llm_g_t, llm_g_id = rect(
    x=360, y=340, w=280, h=80,
    text="NEAR Cloud lane\n(Anthropic Sonnet, can escalate to Opus)\nfull tool registry available",
    bg=LANE_NEAR_BG, font_size=12,
)
ref_g_r, ref_g_t, ref_g_id = rect(
    x=360, y=440, w=280, h=60,
    text="Reflection (citation check)",
    bg="#d3f9d8", font_size=12,
)
add(ev_g_r, ev_g_t, asm_g_r, asm_g_t, llm_g_r, llm_g_t, ref_g_r, ref_g_t)


# --- PHI health path (BAA) ---
add(label(x=680, y=110, text="④ PHI Health path  (phi_class = personal)",
          font_size=15))

ev_p_r, ev_p_t, ev_p_id = rect(
    x=680, y=140, w=280, h=80,
    text="Evidence retrieval (shared)\nsame PubMed + MedlinePlus\n(no PHI leaves to public APIs)",
    bg="#ffe3e3", font_size=12,
)
asm_p_r, asm_p_t, asm_p_id = rect(
    x=680, y=240, w=280, h=80,
    text="Assembly\nevidence + health-qa skill\n+ stronger disclaimer\n(NO profile block — tool-fetched)",
    bg="#ffe3e3", font_size=12,
)
llm_p_r, llm_p_t, llm_p_id = rect(
    x=680, y=340, w=280, h=110,
    text="BAA lane: AWS Bedrock\n(Anthropic Sonnet under BAA)\ncaching DISABLED\n"
         "Tools (allowlist):\n  health_profile_summary / show / audit",
    bg=LANE_BAA_BG, font_size=12,
)
ref_p_r, ref_p_t, ref_p_id = rect(
    x=680, y=470, w=280, h=60,
    text="Reflection (citations + safety)",
    bg="#ffe3e3", font_size=12,
)
add(ev_p_r, ev_p_t, asm_p_r, asm_p_t, llm_p_r, llm_p_t, ref_p_r, ref_p_t)


# --- Decline + TEE lanes (right column) ---
add(label(x=1000, y=110, text="⑤ Side lanes", font_size=15))

decline_r, decline_t, decline_id = rect(
    x=1000, y=140, w=280, h=90,
    text="decline_phi\nfires when phi_class=personal\nbut BAA is unconfigured\n(safe refusal, no LLM call)",
    bg="#ffd8d8", font_size=12,
)
tee_r, tee_t, tee_id = rect(
    x=1000, y=250, w=280, h=110,
    text="Private TEE lane\n(sensitive non-PHI)\nprovider=NEAR Private TEE\nGPT-OSS-120B\nno escalation, no caching",
    bg=LANE_TEE_BG, font_size=12,
)
add(decline_r, decline_t, tee_r, tee_t)

# Encrypted profile (lower right)
prof_store_r, prof_store_t, prof_store_id = rect(
    x=1000, y=380, w=280, h=130,
    text="Encrypted profile store\nhealth_profiles + profile_change_log\nAES-GCM, per-user HKDF\nmaster key in Keychain\n\nRead only via health_profile_* tools\non BAA lane (see Profile diagram)",
    bg=STORE_BG, font_size=11,
)
add(prof_store_r, prof_store_t)


# --- Output (bottom) ---
out_r, out_t, out_id = rect(
    x=460, y=580, w=320, h=70,
    text="Cited answer to user\n(with inline confirmation footer if proposals pending)",
    bg=CHANNEL_BG, font_size=12,
)
add(out_r, out_t)


# --- Audit (bottom-right) ---
add(label(x=1000, y=550, text="Audit", font_size=15))
audit_r, audit_t, audit_id = rect(
    x=1000, y=580, w=280, h=70,
    text="health_audit (one row per turn)\nroute + phi_class + topic +\nprovider + model + sources",
    bg=STORE_BG, font_size=11,
)
add(audit_r, audit_t)


# --- Arrows ---
def link(src, dst, **kw):
    el, extras = arrow(src_id=src, dst_id=dst, **kw)
    add(el)
    add(*extras)

link(in_id, triage_id)
link(triage_id, router_id)

# Router → lanes
link(router_id, ev_g_id, label_text="general")
link(router_id, ev_p_id, label_text="personal (BAA ready)")
link(router_id, decline_id, label_text="personal + no BAA", dashed=True)
link(router_id, tee_id, label_text="sensitive non-PHI", dashed=True)

# General lane chain
link(ev_g_id, asm_g_id)
link(asm_g_id, llm_g_id)
link(llm_g_id, ref_g_id)
link(ref_g_id, out_id)

# PHI lane chain
link(ev_p_id, asm_p_id)
link(asm_p_id, llm_p_id)
link(llm_p_id, ref_p_id)
link(ref_p_id, out_id)

# Profile store ← tool calls from BAA only
link(llm_p_id, prof_store_id, dashed=True, color="#fd7e14",
     label_text="health_profile_* tools")

# Decline + TEE → out
link(decline_id, out_id, dashed=True)
link(tee_id, out_id, dashed=True)

# Audit gets a row per turn (from every lane that landed an answer)
link(ref_g_id, audit_id, dashed=True, color="#7950f2")
link(ref_p_id, audit_id, dashed=True, color="#7950f2")
link(decline_id, audit_id, dashed=True, color="#7950f2")


# --- Divergence notes (right margin) ---
add(label(x=40, y=680, text="What changed since the original (May 2026)",
          font_size=14))
notes = [
    "• BAA lane now carries health_profile_* tools "
    "(allowlist in src/llm/factory.py). Original showed BAA as tool-less.",
    "• Assembly does NOT splice a [Health profile context] block. "
    "Profile reads happen on-demand via tool calls — cleaner than a "
    "pre-fetched slice-selector.",
    "• decline_phi lane (BAA unconfigured → safe refusal) — added.",
    "• Private TEE lane for sensitive non-PHI — added.",
    "• Public APIs wired today: pubmed + medlineplus only. "
    "openFDA / CDC / ClinicalTrials.gov stay aspirational.",
]
for i, n in enumerate(notes):
    add(label(x=40, y=710 + i * 24, text=n, font_size=11, color="#495057"))


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
