"""Weekly cognitive pipeline health report.

Queries the local memory.db and renders a self-contained HTML card-view
report covering volume, cognitive mix, search/reflection quality, and
scheduler activity over the last N days (default 7).

Intentionally stdlib-only so it can be run ad-hoc or wired into the
scheduler as a `direct`-mode job via a shell wrapper, without pulling in
the full MicrowaveOS import graph.

Usage:
    python3 scripts/weekly_pipeline_report.py                # last 7 days, HTML to stdout
    python3 scripts/weekly_pipeline_report.py --days 14
    python3 scripts/weekly_pipeline_report.py --format text  # plain-text summary
    python3 scripts/weekly_pipeline_report.py --db /path/to/memory.db

Exit codes: 0 = ok, 1 = db missing, 2 = query error.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median


DEFAULT_DB = Path.home() / ".microwaveos" / "data" / "memory.db"


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"ERROR: db not found at {db_path}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _turns_in_window(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[sqlite3.Row]:
    rows = conn.execute(
        "SELECT channel, user_id, role, timestamp, token_count, metadata "
        "FROM turns WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp ASC",
        (start.isoformat(), end.isoformat()),
    ).fetchall()
    return rows


def _scheduled_jobs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    try:
        return conn.execute(
            "SELECT name, cron_expr, mode, target_channel, enabled, "
            "last_run_at, last_error FROM scheduled_jobs ORDER BY name"
        ).fetchall()
    except sqlite3.OperationalError:
        return []  # table may not exist on older installs


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def _parse_meta(row: sqlite3.Row) -> dict:
    raw = row["metadata"]
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _pct(n: int, d: int) -> str:
    if d == 0:
        return "—"
    return f"{100 * n / d:.0f}%"


def _delta(curr: int, prior: int) -> str:
    if prior == 0:
        return "new" if curr > 0 else "—"
    pct = 100 * (curr - prior) / prior
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def compute_stats(conn: sqlite3.Connection, days: int) -> dict:
    now = datetime.now()
    start = now - timedelta(days=days)
    prior_start = start - timedelta(days=days)

    curr_rows = _turns_in_window(conn, start, now)
    prior_rows = _turns_in_window(conn, prior_start, start)

    # assistant-only rows carry the pipeline metadata
    curr_assistant = [r for r in curr_rows if r["role"] == "assistant"]

    # --- volume ---
    total = len(curr_rows)
    total_prior = len(prior_rows)
    by_channel = Counter(r["channel"] for r in curr_rows)
    by_channel_prior = Counter(r["channel"] for r in prior_rows)
    unique_sessions = len({(r["channel"], r["user_id"]) for r in curr_rows if r["user_id"]})
    total_tokens = sum(r["token_count"] or 0 for r in curr_rows)

    # --- triage mix ---
    intents = Counter()
    complexity = Counter()
    escalated = 0
    needs_memory = 0
    for r in curr_assistant:
        m = _parse_meta(r)
        if "triage_intent" in m:
            intents[m["triage_intent"]] += 1
        if "triage_complexity" in m:
            complexity[m["triage_complexity"]] += 1
        if m.get("escalated"):
            escalated += 1
        if m.get("needs_memory"):
            needs_memory += 1

    # --- search ---
    frag_counts = []
    search_times = []
    for r in curr_assistant:
        m = _parse_meta(r)
        if (f := m.get("search_fragments")) is not None:
            frag_counts.append(f)
        if (t := m.get("search_time_ms")) is not None:
            search_times.append(t)

    # --- reflection ---
    confidences = []
    hedging = 0
    actions = Counter()
    for r in curr_assistant:
        m = _parse_meta(r)
        if (c := m.get("reflection_confidence")) is not None:
            confidences.append(float(c))
        if m.get("reflection_hedging_detected"):
            hedging += 1
        if (a := m.get("reflection_action")):
            actions[a] += 1

    # --- scheduler ---
    jobs = _scheduled_jobs(conn)
    jobs_fired_recently = 0
    jobs_with_errors = []
    for j in jobs:
        lr = j["last_run_at"]
        if lr:
            try:
                ts = datetime.fromisoformat(lr)
                if ts >= start:
                    jobs_fired_recently += 1
            except ValueError:
                pass
        if j["last_error"]:
            jobs_with_errors.append((j["name"], j["last_error"]))

    return {
        "window": {
            "start": start,
            "end": now,
            "days": days,
        },
        "volume": {
            "total": total,
            "total_prior": total_prior,
            "delta": _delta(total, total_prior),
            "by_channel": dict(by_channel.most_common()),
            "by_channel_prior": dict(by_channel_prior),
            "unique_sessions": unique_sessions,
            "total_tokens": total_tokens,
        },
        "triage": {
            "assistant_turns": len(curr_assistant),
            "intents": dict(intents.most_common(5)),
            "complexity": dict(complexity.most_common()),
            "escalation_rate": _pct(escalated, len(curr_assistant)),
            "escalated_count": escalated,
            "needs_memory_rate": _pct(needs_memory, len(curr_assistant)),
        },
        "search": {
            "sample_size": len(frag_counts),
            "avg_fragments": f"{mean(frag_counts):.1f}" if frag_counts else "—",
            "median_time_ms": f"{median(search_times):.0f}" if search_times else "—",
            "p95_time_ms": (
                f"{sorted(search_times)[int(len(search_times) * 0.95) - 1]:.0f}"
                if len(search_times) >= 20 else "—"
            ),
        },
        "reflection": {
            "sample_size": len(confidences),
            "avg_confidence": f"{mean(confidences):.2f}" if confidences else "—",
            "hedging_rate": _pct(hedging, len(curr_assistant)),
            "actions": dict(actions.most_common()),
            "re_search_rate": _pct(actions.get("re-search", 0), len(curr_assistant)),
        },
        "scheduler": {
            "total_jobs": len(jobs),
            "fired_this_week": jobs_fired_recently,
            "errors": jobs_with_errors,
            "job_list": [
                {
                    "name": j["name"],
                    "cron": j["cron_expr"],
                    "mode": j["mode"],
                    "channel": j["target_channel"],
                    "enabled": bool(j["enabled"]),
                    "last_run": j["last_run_at"],
                    "last_error": j["last_error"],
                }
                for j in jobs
            ],
        },
        "blind_spots": [
            "API cost per turn / per job — not persisted to DB",
            "Per-phase (triage/search/reflection) token cost — not attributed",
            "Full scheduler run history — only last_run_at + last_error retained",
            "Signal / Telegram send failures — stderr only, not in DB",
        ],
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _headline_take(s: dict) -> str:
    """Rule-based one-liner interpretation."""
    v = s["volume"]
    t = s["triage"]
    r = s["reflection"]
    parts = []

    if v["total"] == 0:
        return "No activity this week. Either the pipeline's quiet or something upstream broke."

    if v["delta"] != "—" and v["delta"].startswith("+") and int(v["delta"].replace("%", "").lstrip("+")) > 50:
        parts.append(f"Volume up {v['delta']} wk-over-wk.")
    elif v["delta"] != "—" and v["delta"].startswith("-") and int(v["delta"].replace("%", "").lstrip("-")) > 30:
        parts.append(f"Volume down {v['delta']} wk-over-wk — worth a look.")

    try:
        conf = float(r["avg_confidence"])
        if conf < 0.6:
            parts.append(f"Reflection confidence trending low ({conf:.2f}) — re-search or prompt tuning needed.")
    except (ValueError, TypeError):
        pass

    esc_raw = t["escalation_rate"].rstrip("%")
    try:
        if esc_raw != "—" and int(esc_raw) > 30:
            parts.append(f"Escalation rate at {t['escalation_rate']} — Opus is pulling a lot of weight.")
    except ValueError:
        pass

    if s["scheduler"]["errors"]:
        parts.append(f"{len(s['scheduler']['errors'])} scheduled job(s) carrying errors.")

    if not parts:
        return "Pipeline looks boring. In this business boring is great."

    return " ".join(parts)


def render_html(stats: dict) -> str:
    w = stats["window"]
    v = stats["volume"]
    t = stats["triage"]
    sr = stats["search"]
    rf = stats["reflection"]
    sch = stats["scheduler"]
    take = _headline_take(stats)

    def esc(x) -> str:
        return html.escape(str(x))

    def kv_block(d: dict) -> str:
        if not d:
            return '<div class="muted">—</div>'
        return "".join(f'<div class="kv"><span>{esc(k)}</span><span>{esc(v)}</span></div>' for k, v in d.items())

    cards = []

    # Pulse
    channel_rows = kv_block({k: f"{v}" for k, v in v["by_channel"].items()}) if v["by_channel"] else '<div class="muted">no turns</div>'
    cards.append({
        "title": "Pulse",
        "body": f"""
          <div class="stat-big">{v['total']} <span class="delta">{esc(v['delta'])}</span></div>
          <div class="stat-label">turns vs {v['total_prior']} prior week</div>
          <div class="section-label">By channel</div>
          {channel_rows}
          <div class="kv"><span>Unique sessions</span><span>{v['unique_sessions']}</span></div>
          <div class="kv"><span>Total tokens</span><span>{v['total_tokens']:,}</span></div>
        """,
    })

    # Triage mix
    intents_block = kv_block(t["intents"])
    complexity_block = kv_block(t["complexity"])
    cards.append({
        "title": "Cognitive mix",
        "body": f"""
          <div class="section-label">Intent (top 5)</div>
          {intents_block}
          <div class="section-label">Complexity</div>
          {complexity_block}
          <div class="kv"><span>Escalation rate</span><span>{esc(t['escalation_rate'])} ({t['escalated_count']} turns)</span></div>
          <div class="kv"><span>Needed memory</span><span>{esc(t['needs_memory_rate'])}</span></div>
        """,
    })

    # Search health
    cards.append({
        "title": "Search health",
        "body": f"""
          <div class="kv"><span>Avg fragments / turn</span><span>{esc(sr['avg_fragments'])}</span></div>
          <div class="kv"><span>Median latency</span><span>{esc(sr['median_time_ms'])} ms</span></div>
          <div class="kv"><span>p95 latency</span><span>{esc(sr['p95_time_ms'])} ms</span></div>
          <div class="kv muted"><span>Sample</span><span>{sr['sample_size']} searches</span></div>
        """,
    })

    # Reflection
    actions_block = kv_block(rf["actions"])
    cards.append({
        "title": "Reflection quality",
        "body": f"""
          <div class="kv"><span>Avg confidence</span><span>{esc(rf['avg_confidence'])}</span></div>
          <div class="kv"><span>Hedging detected</span><span>{esc(rf['hedging_rate'])}</span></div>
          <div class="kv"><span>Re-search triggered</span><span>{esc(rf['re_search_rate'])}</span></div>
          <div class="section-label">Action distribution</div>
          {actions_block}
        """,
    })

    # Scheduler
    job_rows_html = ""
    if sch["job_list"]:
        for j in sch["job_list"]:
            status_dot = "🟢" if j["enabled"] else "⚪"
            err_note = f'<div class="err">error: {esc(j["last_error"][:80])}</div>' if j["last_error"] else ""
            last_run = j["last_run"] or "never"
            job_rows_html += f"""
              <div class="job">
                <div class="job-name">{status_dot} {esc(j['name'])} <span class="muted">({esc(j['cron'])})</span></div>
                <div class="muted">{esc(j['mode'])} → {esc(j['channel'])} · last: {esc(last_run)}</div>
                {err_note}
              </div>
            """
    else:
        job_rows_html = '<div class="muted">No scheduled jobs.</div>'

    cards.append({
        "title": "Scheduler",
        "body": f"""
          <div class="kv"><span>Jobs configured</span><span>{sch['total_jobs']}</span></div>
          <div class="kv"><span>Fired this week</span><span>{sch['fired_this_week']}</span></div>
          <div class="kv"><span>Carrying errors</span><span>{len(sch['errors'])}</span></div>
          <div class="section-label">Jobs</div>
          {job_rows_html}
        """,
    })

    # Take
    cards.append({
        "title": "Take",
        "body": f'<div class="take">{esc(take)}</div>',
    })

    # Blind spots
    blind_items = "".join(f"<li>{esc(b)}</li>" for b in stats["blind_spots"])
    cards.append({
        "title": "Blind spots",
        "body": f'<ul class="blind">{blind_items}</ul><div class="muted" style="margin-top:8px">Add instrumentation if any of these matter for ops.</div>',
    })

    # Build HTML
    cards_html = ""
    for i, c in enumerate(cards):
        cards_html += f"""
          <div class="card" id="card-{i}">
            <div class="card-head">
              <h2>{esc(c['title'])}</h2>
              <button class="copy-btn" onclick="copyCard({i})">Copy</button>
            </div>
            <div class="card-body" data-copy-src="{i}">{c['body']}</div>
          </div>
        """

    start_s = w["start"].strftime("%b %d")
    end_s = w["end"].strftime("%b %d, %Y")
    title = f"Pipeline Report — {start_s} → {end_s}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{esc(title)}</title>
<style>
  :root {{
    --bg: #f7f7f5;
    --card-bg: #ffffff;
    --text: #111;
    --muted: #6b6b6b;
    --accent: #0a7;
    --border: #e5e5e2;
    --err: #c33;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0f1115;
      --card-bg: #181b22;
      --text: #e8e8e8;
      --muted: #8a8f99;
      --accent: #4ade80;
      --border: #2a2e38;
      --err: #f87171;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 16px;
    line-height: 1.45;
    font-size: 15px;
  }}
  header {{ margin: 0 0 12px; }}
  h1 {{ font-size: 17px; margin: 0 0 4px; font-weight: 600; }}
  .window {{ color: var(--muted); font-size: 13px; }}
  .copy-all {{
    display: inline-block;
    margin: 8px 0 16px;
    padding: 6px 12px;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    cursor: pointer;
  }}
  .card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 14px 12px;
    margin-bottom: 12px;
  }}
  .card-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }}
  h2 {{ font-size: 14px; margin: 0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }}
  .copy-btn {{
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
  }}
  .copy-btn:active {{ background: var(--border); }}
  .stat-big {{ font-size: 28px; font-weight: 600; }}
  .stat-big .delta {{ font-size: 14px; color: var(--accent); margin-left: 6px; }}
  .stat-label {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
  .section-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; margin: 10px 0 4px; }}
  .kv {{ display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px dashed var(--border); font-size: 14px; }}
  .kv:last-child {{ border-bottom: none; }}
  .kv.muted span {{ color: var(--muted); font-size: 13px; }}
  .muted {{ color: var(--muted); font-size: 13px; }}
  .job {{ padding: 8px 0; border-bottom: 1px dashed var(--border); }}
  .job:last-child {{ border-bottom: none; }}
  .job-name {{ font-weight: 500; }}
  .err {{ color: var(--err); font-size: 12px; margin-top: 2px; }}
  .take {{ font-size: 15px; line-height: 1.5; }}
  ul.blind {{ margin: 0; padding-left: 18px; }}
  ul.blind li {{ margin-bottom: 4px; color: var(--muted); font-size: 13px; }}
</style>
</head>
<body>
<header>
  <h1>{esc(title)}</h1>
  <div class="window">{w['days']}-day window · {esc(start_s)} → {esc(end_s)}</div>
  <button class="copy-all" onclick="copyAll()">Copy all</button>
</header>

{cards_html}

<script>
  function textOfCard(i) {{
    const el = document.querySelector('[data-copy-src="' + i + '"]');
    if (!el) return '';
    return (el.innerText || el.textContent || '').trim();
  }}
  function writeClip(t) {{
    if (navigator.clipboard && navigator.clipboard.writeText) {{
      return navigator.clipboard.writeText(t).catch(fallback.bind(null, t));
    }}
    return fallback(t);
  }}
  function fallback(t) {{
    const ta = document.createElement('textarea');
    ta.value = t; document.body.appendChild(ta); ta.select();
    try {{ document.execCommand('copy'); }} catch(e) {{}}
    document.body.removeChild(ta);
  }}
  function copyCard(i) {{
    const btn = document.querySelector('#card-' + i + ' .copy-btn');
    writeClip(textOfCard(i));
    if (btn) {{ const o = btn.textContent; btn.textContent = 'Copied'; setTimeout(()=>btn.textContent=o, 1200); }}
  }}
  function copyAll() {{
    const all = Array.from(document.querySelectorAll('.card')).map(c => {{
      const h = c.querySelector('h2').innerText;
      const b = c.querySelector('.card-body').innerText;
      return h + '\\n' + b;
    }}).join('\\n\\n---\\n\\n');
    writeClip(all);
  }}
</script>
</body>
</html>"""


def render_text(stats: dict) -> str:
    """Plain-text summary — used as fallback body or for stdout inspection."""
    w = stats["window"]
    v = stats["volume"]
    t = stats["triage"]
    sr = stats["search"]
    rf = stats["reflection"]
    sch = stats["scheduler"]

    lines = []
    lines.append(f"PIPELINE REPORT — {w['days']}d ({w['start'].strftime('%b %d')} → {w['end'].strftime('%b %d')})")
    lines.append("")
    lines.append(f"Volume:      {v['total']} turns ({v['delta']} vs prior), {v['unique_sessions']} sessions, {v['total_tokens']:,} tokens")
    lines.append(f"By channel:  {', '.join(f'{k}={n}' for k, n in v['by_channel'].items()) or '—'}")
    lines.append("")
    lines.append(f"Triage:      intents={t['intents']}")
    lines.append(f"             complexity={t['complexity']}  escalated={t['escalation_rate']}")
    lines.append("")
    lines.append(f"Search:      avg={sr['avg_fragments']} frags  p50={sr['median_time_ms']}ms  p95={sr['p95_time_ms']}ms")
    lines.append(f"Reflection:  conf={rf['avg_confidence']}  hedging={rf['hedging_rate']}  re-search={rf['re_search_rate']}")
    lines.append("")
    lines.append(f"Scheduler:   {sch['total_jobs']} jobs, {sch['fired_this_week']} fired this week, {len(sch['errors'])} with errors")
    if sch["errors"]:
        for name, err in sch["errors"]:
            lines.append(f"  ! {name}: {err[:100]}")
    lines.append("")
    lines.append(f"Take: {_headline_take(stats)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Weekly cognitive pipeline report")
    p.add_argument("--db", type=Path, default=Path(os.getenv("DATA_DIR", str(DEFAULT_DB.parent))) / "memory.db")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--format", choices=["html", "text", "both"], default="html")
    p.add_argument("--out", type=Path, help="Write HTML to this path (default: stdout)")
    args = p.parse_args(argv)

    conn = _connect(args.db)
    try:
        stats = compute_stats(conn, days=args.days)
    except sqlite3.Error as e:
        print(f"ERROR: db query failed: {e}", file=sys.stderr)
        return 2
    finally:
        conn.close()

    if args.format in ("text", "both"):
        print(render_text(stats))
        if args.format == "both":
            print()

    if args.format in ("html", "both"):
        html_doc = render_html(stats)
        if args.out:
            args.out.write_text(html_doc)
            print(f"wrote {args.out}", file=sys.stderr)
        else:
            print(html_doc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
