"""Shared Excalidraw element factories + palette.

Pulled out so each `_gen_*.py` can import without re-running the main
generator's module-level code. Stateless: every generator manages its
own `elements` list.

Coordinate system: x grows right, y grows down. Origin top-left.
"""

from __future__ import annotations

import random
import time


# Module-level RNG + ID counters. The first generator to import sets
# them up; subsequent generators reuse — fine because we only care
# about *intra-file* ID uniqueness and reproducibility.
random.seed(20260528)
_NOW = int(time.time() * 1000)
_ID_COUNTER = [0]


def _seed() -> int:
    return random.randint(1, 2**31 - 1)


def _nonce() -> int:
    return random.randint(1, 2**31 - 1)


def _id(prefix: str) -> str:
    _ID_COUNTER[0] += 1
    return f"{prefix}_{_ID_COUNTER[0]}"


def rect(
    *, x, y, w, h, text, bg="transparent", stroke="#1e1e1e",
    text_color=None, font_size=16, kind="rectangle",
    stroke_width=2, roughness=1,
):
    """Make a rounded rectangle with centered text (two linked elements)."""
    rid = _id("rect")
    tid = _id("text")
    rect_el = {
        "id": rid, "type": kind,
        "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": stroke, "backgroundColor": bg,
        "fillStyle": "solid" if bg != "transparent" else "hachure",
        "strokeWidth": stroke_width, "strokeStyle": "solid",
        "roughness": roughness, "opacity": 100,
        "groupIds": [], "frameId": None,
        "roundness": {"type": 3} if kind == "rectangle" else None,
        "seed": _seed(), "version": 1, "versionNonce": _nonce(),
        "isDeleted": False, "boundElements": [{"type": "text", "id": tid}],
        "updated": _NOW, "link": None, "locked": False,
    }
    text_el = {
        "id": tid, "type": "text",
        "x": x + 8, "y": y + 8, "width": w - 16, "height": h - 16,
        "angle": 0, "strokeColor": text_color or stroke,
        "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": None,
        "seed": _seed(), "version": 1, "versionNonce": _nonce(),
        "isDeleted": False, "boundElements": [],
        "updated": _NOW, "link": None, "locked": False,
        "fontSize": font_size, "fontFamily": 5,
        "text": text, "textAlign": "center", "verticalAlign": "middle",
        "baseline": int(font_size * 0.85),
        "containerId": rid, "originalText": text,
        "lineHeight": 1.25,
    }
    return rect_el, text_el, rid


def diamond(*, x, y, w, h, text, bg="#fff3bf"):
    return rect(x=x, y=y, w=w, h=h, text=text, bg=bg, kind="diamond")


def label(*, x, y, text, font_size=20, color="#1e1e1e"):
    tid = _id("text")
    width = max(80, int(len(text) * font_size * 0.55))
    height = int(font_size * 1.4)
    return {
        "id": tid, "type": "text",
        "x": x, "y": y, "width": width, "height": height,
        "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": None,
        "seed": _seed(), "version": 1, "versionNonce": _nonce(),
        "isDeleted": False, "boundElements": [],
        "updated": _NOW, "link": None, "locked": False,
        "fontSize": font_size, "fontFamily": 5,
        "text": text, "textAlign": "left", "verticalAlign": "top",
        "baseline": int(font_size * 0.85),
        "containerId": None, "originalText": text,
        "lineHeight": 1.25,
    }


def _edge_point(el, side: str) -> tuple[float, float]:
    """Center of one side of a rect/diamond, in absolute coords."""
    x, y, w, h = el["x"], el["y"], el["width"], el["height"]
    cx, cy = x + w / 2, y + h / 2
    if side == "right":  return (x + w, cy)
    if side == "left":   return (x, cy)
    if side == "top":    return (cx, y)
    if side == "bottom": return (cx, y + h)
    raise ValueError(side)


def _pick_sides(src, dst) -> tuple[str, str]:
    """Pick which edges of src + dst to connect.

    Heuristic: whichever axis has the larger center-to-center delta wins.
    Horizontal-dominant → right→left or left→right. Vertical-dominant →
    bottom→top or top→bottom. Same-on-an-axis falls through cleanly.
    """
    sx = src["x"] + src["width"] / 2
    sy = src["y"] + src["height"] / 2
    dx = dst["x"] + dst["width"] / 2
    dy = dst["y"] + dst["height"] / 2
    if abs(dx - sx) >= abs(dy - sy):
        return ("right", "left") if dx >= sx else ("left", "right")
    return ("bottom", "top") if dy >= sy else ("top", "bottom")


def arrow(*, src, dst, label_text=None, dashed=False, color="#1e1e1e",
          src_side=None, dst_side=None):
    """Connect two element dicts with a straight arrow.

    Excalidraw renders an arrow based on its own (x, y) + `points` —
    `startBinding` / `endBinding` only control snap-on-drag in the
    editor, not rendering geometry. So we compute real edge points
    here. Pass `src_side` / `dst_side` (one of left/right/top/bottom)
    to override the auto-picked sides when the heuristic chooses
    something ugly.
    """
    s_side, d_side = (src_side, dst_side) if src_side and dst_side else _pick_sides(src, dst)
    sx, sy = _edge_point(src, s_side)
    dx, dy = _edge_point(dst, d_side)
    aid = _id("arrow")
    el = {
        "id": aid, "type": "arrow",
        "x": sx, "y": sy,
        "width": abs(dx - sx), "height": abs(dy - sy),
        "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2,
        "strokeStyle": "dashed" if dashed else "solid",
        "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": {"type": 2},
        "seed": _seed(), "version": 1, "versionNonce": _nonce(),
        "isDeleted": False, "boundElements": [],
        "updated": _NOW, "link": None, "locked": False,
        "points": [[0, 0], [dx - sx, dy - sy]],
        "lastCommittedPoint": None,
        "startBinding": {"elementId": src["id"], "focus": 0, "gap": 1},
        "endBinding": {"elementId": dst["id"], "focus": 0, "gap": 1},
        "startArrowhead": None, "endArrowhead": "arrow",
        "elbowed": False,
    }
    extras = []
    if label_text:
        # Place the label near the arrow's midpoint, slightly above the line
        mid_x = (sx + dx) / 2
        mid_y = (sy + dy) / 2
        extras.append(label(
            x=mid_x - len(label_text) * 3, y=mid_y - 18,
            text=label_text, font_size=11, color=color,
        ))
    return el, extras


# --- Palette --------------------------------------------------------------

CHANNEL_BG = "#d0ebff"     # input channels (Signal/Telegram/REPL)
PIPE_BG = "#e7f5ff"        # pipeline stages
ROUTER_BG = "#fff3bf"      # decision points
LANE_NEAR_BG = "#d3f9d8"   # NEAR (general) lane
LANE_BAA_BG = "#ffe3e3"    # BAA Bedrock lane (PHI)
LANE_TEE_BG = "#fff0f6"    # Private TEE lane
STORE_BG = "#ffe8cc"       # data stores
LOOP_BG = "#e5dbff"        # background loops
TOOL_BG = "#fff9db"        # tool registry
