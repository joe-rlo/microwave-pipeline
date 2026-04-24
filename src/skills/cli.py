"""`microwaveos skills ...` subcommand.

Mirrors the scheduler CLI shape. Kept in its own module so main.py stays
a thin router.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from src.config import load_config
from src.skills.loader import SkillLoader, SkillNotFound


def skills_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="microwaveos skills",
        description="Manage skills — reusable instruction bundles for the pipeline.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("list", help="Show all skills")

    p_show = sub.add_parser("show", help="Print a skill's full content")
    p_show.add_argument("name")

    p_new = sub.add_parser("new", help="Scaffold a new skill directory")
    p_new.add_argument("name")
    p_new.add_argument("--description", default="", help="One-line description")

    p_rm = sub.add_parser("remove", help="Delete a skill (directory + files)")
    p_rm.add_argument("name")
    p_rm.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    p_edit = sub.add_parser("edit", help="Open the skill's SKILL.md in $EDITOR")
    p_edit.add_argument("name")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config()
    skills_dir = config.workspace_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    loader = SkillLoader(skills_dir)

    if args.action == "list":
        return _cmd_list(loader)
    if args.action == "show":
        return _cmd_show(loader, args.name)
    if args.action == "new":
        return _cmd_new(loader, args.name, args.description)
    if args.action == "remove":
        return _cmd_remove(loader, args.name, args.force)
    if args.action == "edit":
        return _cmd_edit(loader, args.name)
    return 1


def _cmd_list(loader: SkillLoader) -> int:
    skills = loader.list_all()
    if not skills:
        print(f"(no skills in {loader.skills_dir})")
        return 0
    # Fixed-width layout. Description gets truncated to keep lines scannable.
    headers = ("NAME", "FETCH", "DESCRIPTION")
    rows = []
    for s in skills:
        desc = s.description or "—"
        if len(desc) > 70:
            desc = desc[:67] + "…"
        rows.append((s.name, "yes" if s.has_fetch else "—", desc))
    widths = [max(len(r[i]) for r in (headers,) + tuple(rows)) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))
    return 0


def _cmd_show(loader: SkillLoader, name: str) -> int:
    try:
        skill = loader.load(name)
    except SkillNotFound:
        print(f"No skill named {name!r}", file=sys.stderr)
        return 1
    print(f"# Skill: {skill.name}")
    print(f"# Location: {skill.directory}")
    print(f"# Description: {skill.description or '—'}")
    if skill.triggers:
        print(f"# Triggers: {', '.join(skill.triggers)}")
    if skill.has_fetch:
        print("# Pre-fetch: yes (scheduler-only in v1)")
    print()
    print(skill.body)
    return 0


def _cmd_new(loader: SkillLoader, name: str, description: str) -> int:
    try:
        path = loader.scaffold(name, description)
    except (ValueError, FileExistsError) as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"Created skill at {path}")
    print(f"Edit with: microwaveos skills edit {name}")
    return 0


def _cmd_remove(loader: SkillLoader, name: str, force: bool) -> int:
    target = loader.skills_dir / name
    if not target.is_dir():
        print(f"No skill named {name!r}", file=sys.stderr)
        return 1
    if not force:
        resp = input(f"Delete skill {name!r} at {target}? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Cancelled.")
            return 0
    loader.remove(name)
    print(f"Removed skill {name!r}")
    return 0


def _cmd_edit(loader: SkillLoader, name: str) -> int:
    path = loader.skills_dir / name / "SKILL.md"
    if not path.is_file():
        print(f"No skill named {name!r}", file=sys.stderr)
        return 1
    editor = os.environ.get("EDITOR", "vi")
    try:
        subprocess.run([editor, str(path)], check=False)
    except FileNotFoundError:
        print(f"Editor {editor!r} not found. Set $EDITOR.", file=sys.stderr)
        return 1
    return 0
