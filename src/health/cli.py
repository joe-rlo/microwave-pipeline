"""`microwaveos health ...` subcommand.

Mirrors the scheduler / skills / projects CLI shape. Four actions:

- `status`         — show module state, source toggles, recent audit count
- `retrieve <q>`   — exercise retrieval against `q`, print results, no LLM
- `install-skill`  — copy seed_skills/health-qa into the user's workspace
- `audit list`     — show recent health_audit rows (general path only)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

from src.config import load_config


def health_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="microwaveos health",
        description="Health module — privacy-aware health Q&A pipeline.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("status", help="Show module state and source toggles.")

    p_retrieve = sub.add_parser(
        "retrieve",
        help="Run retrieval against a query without invoking the LLM.",
    )
    p_retrieve.add_argument("query", help="Search query")
    p_retrieve.add_argument(
        "--topic", default=None,
        help="Optional health_topic tag (e.g. 'diabetes')",
    )
    p_retrieve.add_argument(
        "--max", type=int, default=8,
        help="Max results across all sources (default 8)",
    )

    sub.add_parser(
        "install-skill",
        help="Copy seed_skills/health-qa into your workspace.",
    )

    p_audit = sub.add_parser("audit", help="Inspect the health audit log.")
    audit_sub = p_audit.add_subparsers(dest="audit_action", required=True)
    p_audit_list = audit_sub.add_parser("list", help="List recent rows.")
    p_audit_list.add_argument(
        "--limit", type=int, default=20,
        help="Max rows to show (default 20)",
    )

    # Phase E: user-controlled privacy prefs
    p_prefs = sub.add_parser(
        "prefs",
        help="Show or change health-route privacy preferences.",
    )
    prefs_sub = p_prefs.add_subparsers(dest="prefs_action", required=True)
    prefs_sub.add_parser("show", help="Print current prefs.")
    p_set = prefs_sub.add_parser(
        "set",
        help="Set a pref. Currently only --privacy-mode is configurable.",
    )
    p_set.add_argument(
        "--privacy-mode",
        choices=("standard", "private_tee"),
        required=True,
        help=(
            "standard: general-health turns route to NEAR Anonymised "
            "Claude (or your configured main pipeline). "
            "private_tee: route to NEAR Private TEE open-weight models "
            "(GPT OSS / Qwen3.5) — hardware-attested isolation, "
            "different quality trade-off."
        ),
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING,  # quiet by default; CLI prints its own status
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.action == "status":
        return _cmd_status()
    if args.action == "retrieve":
        return asyncio.run(_cmd_retrieve(args.query, args.topic, args.max))
    if args.action == "install-skill":
        return _cmd_install_skill()
    if args.action == "audit" and args.audit_action == "list":
        return _cmd_audit_list(args.limit)
    if args.action == "prefs":
        if args.prefs_action == "show":
            return _cmd_prefs_show()
        if args.prefs_action == "set":
            return _cmd_prefs_set(privacy_mode=args.privacy_mode)

    return 1


def _cmd_status() -> int:
    """Plain-text dashboard: enabled? which sources? skill installed?
    is BAA configured? recent audit count? Designed to be the one
    place to look when the bot's health behavior is unexpected."""
    config = load_config()
    h = config.health

    print(f"Module enabled: {h.enabled}")
    print(f"BAA provider:   {h.baa_provider}")
    print(f"PHI path ready: {h.phi_path_available}")
    print()

    print("Retrieval sources:")
    for name, on in [
        ("pubmed", h.retrieval_pubmed),
        ("medlineplus", h.retrieval_medlineplus),
        ("openfda", h.retrieval_openfda),
        ("cdc", h.retrieval_cdc),
        ("clinicaltrials", h.retrieval_clinicaltrials),
    ]:
        # Phase 1 only ships pubmed + medlineplus implementations
        impl = name in ("pubmed", "medlineplus")
        status = "on" if on else "off"
        impl_note = "" if impl else "  (no impl yet — Phase 3+)"
        print(f"  {name:<16} {status}{impl_note}")
    print()

    skill_path = config.workspace_dir / "skills" / "health-qa" / "SKILL.md"
    if skill_path.is_file():
        print(f"health-qa skill installed at {skill_path}")
    else:
        print(f"health-qa skill NOT installed.")
        print("  Run: python3 src/main.py health install-skill")
    print()

    # Recent audit count — quick sanity that the writer is working
    try:
        from src.health.audit import HealthAuditWriter
        w = HealthAuditWriter(config.db_path)
        w.connect()
        try:
            recent = w.list_recent(limit=1000)
        finally:
            w.close()
        print(f"Audit rows on disk: {len(recent)}")
        if recent:
            ts = datetime.fromtimestamp(recent[0]["timestamp"]).isoformat(timespec="seconds")
            print(f"Most recent route:  {recent[0]['route']} at {ts}")
    except Exception as e:
        print(f"Audit DB unreachable: {e}")
    return 0


async def _cmd_retrieve(query: str, topic: str | None, max_results: int) -> int:
    """Run the retrieval orchestrator, no LLM. Prints each Evidence
    with source name, title, URL, snippet — same shape the LLM would
    see in the [Evidence context] block."""
    from src.health.retrieval.medlineplus import MedlinePlusSource
    from src.health.retrieval.orchestrator import RetrievalOrchestrator
    from src.health.retrieval.pubmed import PubMedSource

    config = load_config()
    h = config.health

    sources = []
    if h.retrieval_pubmed:
        sources.append(PubMedSource(api_key=h.ncbi_api_key))
    if h.retrieval_medlineplus:
        sources.append(MedlinePlusSource())

    if not sources:
        print("No retrieval sources enabled in config.", file=sys.stderr)
        return 1

    orch = RetrievalOrchestrator(sources=sources)
    results = await orch.search(query, topic=topic, max_results=max_results)
    if not results:
        print(f"No results for {query!r}.")
        return 0

    for i, ev in enumerate(results, 1):
        print(f"[{i}] {ev.source} — {ev.title}")
        print(f"    {ev.url}")
        if ev.published:
            print(f"    Published {ev.published.isoformat()}")
        if ev.snippet:
            snippet = ev.snippet[:200] + ("…" if len(ev.snippet) > 200 else "")
            print(f"    {snippet}")
        print()
    return 0


def _cmd_install_skill() -> int:
    """Copy seed_skills/health-qa/ into the user's workspace.

    Idempotent: refuses to overwrite an existing skill so user edits
    aren't clobbered. To force a refresh, the user removes the
    workspace copy first."""
    config = load_config()
    src_dir = Path(__file__).resolve().parents[2] / "seed_skills" / "health-qa"
    if not src_dir.is_dir():
        print(f"Seed skill not found at {src_dir}.", file=sys.stderr)
        return 1

    target = config.workspace_dir / "skills" / "health-qa"
    if target.exists():
        print(f"health-qa skill already installed at {target}.")
        print(f"To reinstall: rm -rf {target} && python3 src/main.py health install-skill")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, target)
    print(f"Installed health-qa skill to {target}")
    return 0


def _cmd_audit_list(limit: int) -> int:
    """Dump recent audit rows as one row per line.

    Phase 1 shows everything in `health_audit`. Phase 2 will add
    `phi_audit` which requires a per-user key unlock — those rows
    won't appear here."""
    from src.health.audit import HealthAuditWriter

    config = load_config()
    w = HealthAuditWriter(config.db_path)
    w.connect()
    try:
        rows = w.list_recent(limit=limit)
    finally:
        w.close()

    if not rows:
        print("No audit rows yet.")
        return 0

    for r in rows:
        ts = datetime.fromtimestamp(r["timestamp"]).isoformat(timespec="seconds")
        topic = r.get("triage_health_topic") or "-"
        latency = r.get("latency_ms") or "-"
        sources = r.get("sources_returned")
        try:
            sources_summary = (
                ", ".join(f"{s['name']}:{s['count']}" for s in json.loads(sources))
                if sources else "-"
            )
        except (TypeError, ValueError):
            sources_summary = "-"
        print(
            f"{ts}  route={r['route']:<12} phi={r.get('triage_phi_class') or '-':<8} "
            f"topic={topic:<14} latency={latency}ms  sources=[{sources_summary}]"
        )
    return 0


# --- Phase E: prefs ---


def _cmd_prefs_show() -> int:
    """Print current health privacy prefs."""
    from datetime import datetime
    from src.health.user_prefs import init_tables, load_pref
    from src.session.engine import SessionEngine

    config = load_config()
    engine = SessionEngine(config.db_path)
    engine.connect()
    init_tables(engine.conn)
    pref = load_pref(engine.conn)

    print(f"Privacy mode:                  {pref.privacy_mode}")
    print(
        f"Consent to Anonymised general: "
        f"{'yes' if pref.consent_anonymised_general else 'no (default)'}"
    )
    if pref.last_updated:
        when = datetime.fromtimestamp(pref.last_updated).isoformat(timespec="seconds")
        print(f"Last updated:                  {when}")
    else:
        print("Last updated:                  never (using defaults)")
    print()
    print("Notes:")
    print(" - standard: general health routes to your main pipeline LLM")
    print("   (NEAR Anonymised Claude). PII metadata is stripped before")
    print("   forwarding, but the upstream provider sees the de-identified prompt.")
    print(" - private_tee: general health routes to NEAR Private TEE")
    print("   open-weight models (GPT OSS 120B / Qwen3.5 122B). Hardware-")
    print("   attested isolation; NEAR cannot read the prompt. Quality vs.")
    print("   Anonymised Claude is unmeasured — flag for your own bake-off.")
    print(" - Personal/PHI turns ignore this setting and route to BAA when")
    print("   configured, decline_phi otherwise.")
    return 0


def _cmd_prefs_set(*, privacy_mode: str) -> int:
    """Update health prefs."""
    from src.health.user_prefs import init_tables, save_pref
    from src.session.engine import SessionEngine

    config = load_config()
    engine = SessionEngine(config.db_path)
    engine.connect()
    init_tables(engine.conn)
    pref = save_pref(engine.conn, privacy_mode=privacy_mode)  # type: ignore[arg-type]
    print(f"✓ Privacy mode set to {pref.privacy_mode!r}.")
    if pref.privacy_mode == "private_tee":
        import os
        if not os.environ.get("NEAR_API_KEY", "").strip():
            print(
                "\n⚠ NEAR_API_KEY is not set in your environment. "
                "General-health turns will silently fall back to the standard "
                "main pipeline (the orchestrator logs a warning). "
                "Set NEAR_API_KEY in .env to actually use the Private-TEE path."
            )
    return 0
