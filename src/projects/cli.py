"""`microwaveos projects ...` subcommand."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

from src.config import load_config
from src.projects.loader import ProjectLoader, ProjectNotFound
from src.projects.models import PROJECT_TYPES


def projects_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="microwaveos projects",
        description="Manage writing projects (blogs, novels, screenplays).",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p_list = sub.add_parser("list", help="Show all projects")
    p_list.add_argument(
        "--archived", action="store_true", help="Include archived projects"
    )

    p_show = sub.add_parser("show", help="Print project metadata + status")
    p_show.add_argument("name")

    p_new = sub.add_parser("new", help="Scaffold a new project directory")
    p_new.add_argument("name")
    p_new.add_argument(
        "--type", required=True, choices=list(PROJECT_TYPES),
        help="Project type — picks the directory layout and default skill",
    )
    p_new.add_argument("--description", default="", help="One-line description")

    p_arch = sub.add_parser("archive", help="Move a project to .archived/")
    p_arch.add_argument("name")

    p_rm = sub.add_parser("remove", help="Delete a project (irreversible)")
    p_rm.add_argument("name")
    p_rm.add_argument("--force", action="store_true")

    p_edit = sub.add_parser("edit", help="Open the project's PROJECT.md in $EDITOR")
    p_edit.add_argument("name")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config()
    projects_dir = config.workspace_dir / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    loader = ProjectLoader(projects_dir)

    if args.action == "list":
        return _cmd_list(loader, archived=args.archived)
    if args.action == "show":
        return _cmd_show(loader, args.name)
    if args.action == "new":
        return _cmd_new(loader, args.name, args.type, args.description)
    if args.action == "archive":
        return _cmd_archive(loader, args.name)
    if args.action == "remove":
        return _cmd_remove(loader, args.name, args.force)
    if args.action == "edit":
        return _cmd_edit(loader, args.name)
    return 1


def _cmd_list(loader: ProjectLoader, archived: bool) -> int:
    projects = loader.list_all()
    if not projects:
        print(f"(no projects in {loader.projects_dir})")
        return 0
    headers = ("NAME", "TYPE", "STATUS", "WORDS", "SKILL", "DESCRIPTION")
    rows = []
    for p in projects:
        words = p.word_count()
        words_str = f"{words:,}" if words else "—"
        if p.target_words:
            words_str = f"{words_str} / {p.target_words:,}"
        desc = p.description or "—"
        if len(desc) > 50:
            desc = desc[:47] + "…"
        rows.append((p.name, p.type, p.status, words_str, p.skill or "—", desc))
    widths = [max(len(r[i]) for r in (headers,) + tuple(rows)) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))
    return 0


def _cmd_show(loader: ProjectLoader, name: str) -> int:
    try:
        p = loader.load(name)
    except ProjectNotFound:
        print(f"No project named {name!r}", file=sys.stderr)
        return 1
    print(f"# Project: {p.name}")
    print(f"# Type: {p.type}")
    print(f"# Skill: {p.skill or '—'}")
    print(f"# Status: {p.status}")
    print(f"# Location: {p.directory}")
    if p.created_at:
        print(f"# Created: {p.created_at.date().isoformat()}")
    words = p.word_count()
    if words:
        target = f" / {p.target_words:,}" if p.target_words else ""
        print(f"# Words: {words:,}{target}")
    print(f"# BIBLE: {'yes' if p.has_bible else 'no'}")
    print(f"# Outline: {'yes' if p.has_outline else 'no'}")
    drafts = p.list_drafts()
    if drafts:
        print(f"# Drafts ({len(drafts)}):")
        for d in drafts:
            wc = len(d.read_text(encoding="utf-8").split())
            print(f"  - {d.name} ({wc:,} words)")
    if p.description:
        print()
        print(p.description)
    if p.voice_notes:
        print()
        print(p.voice_notes)
    return 0


def _cmd_new(
    loader: ProjectLoader, name: str, project_type: str, description: str
) -> int:
    try:
        path = loader.scaffold(name, project_type, description)
    except (ValueError, FileExistsError) as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"Created {project_type} project at {path.parent}")
    print(f"Edit metadata: microwaveos projects edit {name}")
    print(f"Activate in chat: /project {name}")
    return 0


def _cmd_archive(loader: ProjectLoader, name: str) -> int:
    if loader.archive(name):
        print(f"Archived {name!r} → {loader.projects_dir}/.archived/{name}")
        return 0
    print(f"No project named {name!r}", file=sys.stderr)
    return 1


def _cmd_remove(loader: ProjectLoader, name: str, force: bool) -> int:
    target = loader.projects_dir / name
    if not target.is_dir():
        print(f"No project named {name!r}", file=sys.stderr)
        return 1
    if not force:
        resp = input(
            f"Delete project {name!r} at {target}? "
            f"This removes drafts and notes too. [y/N] "
        ).strip().lower()
        if resp not in ("y", "yes"):
            print("Cancelled. (Try `projects archive` instead — it's reversible.)")
            return 0
    loader.remove(name)
    print(f"Removed {name!r}")
    return 0


def _cmd_edit(loader: ProjectLoader, name: str) -> int:
    path = loader.projects_dir / name / "PROJECT.md"
    if not path.is_file():
        print(f"No project named {name!r}", file=sys.stderr)
        return 1
    editor = os.environ.get("EDITOR", "vi")
    try:
        subprocess.run([editor, str(path)], check=False)
    except FileNotFoundError:
        print(f"Editor {editor!r} not found. Set $EDITOR.", file=sys.stderr)
        return 1
    return 0
