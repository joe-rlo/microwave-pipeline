"""CLI for importing data from other AI tools into MicrowaveOS.

Usage:
    python3 src/import_data.py openclaw              # auto-detect, import all
    python3 src/import_data.py openclaw --path ~/.openclaw/agents/abc123
    python3 src/import_data.py hermes                # auto-detect
    python3 src/import_data.py hermes --path ~/.hermes
    python3 src/import_data.py nanoclaw              # auto-detect
    python3 src/import_data.py nanoclaw --path ~/nanoclaw

    python3 src/import_data.py openclaw --sessions-only   # skip memory
    python3 src/import_data.py openclaw --memory-only     # skip sessions
    python3 src/import_data.py openclaw --dry-run         # show what would be imported
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.memory.embeddings import EmbeddingClient
from src.memory.index import MemoryIndex
from src.memory.store import MemoryStore
from src.importers.ingest import ingest_sessions, ingest_memories, ingest_daily_notes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def import_openclaw(args) -> None:
    from src.importers.openclaw import find_openclaw_dir, list_agents, import_sessions, import_memory

    if args.path:
        agent_dir = Path(args.path)
        if not agent_dir.exists():
            print(f"Error: path does not exist: {args.path}")
            sys.exit(1)
    else:
        openclaw_dir = find_openclaw_dir()
        if not openclaw_dir:
            print("Error: Could not find OpenClaw directory (~/.openclaw)")
            print("Use --path to specify the agent directory")
            sys.exit(1)

        agents = list_agents(openclaw_dir)
        if not agents:
            print("No OpenClaw agents found")
            sys.exit(1)

        print(f"Found {len(agents)} OpenClaw agent(s):")
        for i, a in enumerate(agents):
            print(f"  [{i}] {a['id']} — {a['session_count']} sessions, "
                  f"memory: {'yes' if a['has_memory'] else 'no'}")

        if len(agents) == 1:
            agent_dir = Path(agents[0]["path"])
        else:
            choice = input("\nWhich agent to import? [0]: ").strip() or "0"
            agent_dir = Path(agents[int(choice)]["path"])

    print(f"\nImporting from: {agent_dir}")

    if args.dry_run:
        sessions = import_sessions(agent_dir)
        memory = import_memory(agent_dir)
        print(f"\nDry run results:")
        print(f"  Sessions: {len(sessions)} ({sum(len(s['turns']) for s in sessions)} turns)")
        print(f"  MEMORY.md: {'yes' if memory['memory_md'] else 'no'}")
        print(f"  Daily notes: {len(memory['daily_notes'])}")
        print(f"  Topic memories: {len(memory['topic_memories'])}")
        return

    config = load_config()
    config.ensure_dirs()
    memory_store, memory_index = _connect(config)

    try:
        total = 0
        if not args.memory_only:
            sessions = import_sessions(agent_dir)
            total += ingest_sessions(sessions, "openclaw", memory_index)

        if not args.sessions_only:
            memory = import_memory(agent_dir)
            if memory["memory_md"]:
                total += ingest_memories(
                    [{"content": memory["memory_md"], "memory_md": True}],
                    "openclaw", memory_store, memory_index,
                )
            if memory["daily_notes"]:
                total += ingest_daily_notes(
                    memory["daily_notes"], "openclaw", memory_store, memory_index,
                )
            if memory["topic_memories"]:
                total += ingest_memories(
                    memory["topic_memories"], "openclaw", memory_store, memory_index,
                    merge_to_memory_md=False,
                )

        print(f"\nDone. Indexed {total} fragments into MicrowaveOS.")
    finally:
        memory_index.close()


def import_hermes(args) -> None:
    from src.importers.hermes import find_hermes_dir, import_sessions, import_memories

    if args.path:
        hermes_dir = Path(args.path)
    else:
        hermes_dir = find_hermes_dir()
        if not hermes_dir:
            print("Error: Could not find Hermes directory (~/.hermes)")
            print("Use --path to specify the Hermes directory")
            sys.exit(1)

    print(f"Importing from: {hermes_dir}")

    if args.dry_run:
        sessions = import_sessions(hermes_dir)
        memories = import_memories(hermes_dir)
        print(f"\nDry run results:")
        print(f"  Sessions: {len(sessions)} ({sum(len(s['turns']) for s in sessions)} turns)")
        print(f"  Memory files: {len(memories)}")
        return

    config = load_config()
    config.ensure_dirs()
    memory_store, memory_index = _connect(config)

    try:
        total = 0
        if not args.memory_only:
            sessions = import_sessions(hermes_dir)
            total += ingest_sessions(sessions, "hermes", memory_index)

        if not args.sessions_only:
            memories = import_memories(hermes_dir)
            if memories:
                total += ingest_memories(
                    memories, "hermes", memory_store, memory_index,
                    merge_to_memory_md=False,
                )

        print(f"\nDone. Indexed {total} fragments into MicrowaveOS.")
    finally:
        memory_index.close()


def import_nanoclaw(args) -> None:
    from src.importers.nanoclaw import find_nanoclaw_dir, import_sessions, import_memories

    if args.path:
        nanoclaw_dir = Path(args.path)
    else:
        nanoclaw_dir = find_nanoclaw_dir()
        if not nanoclaw_dir:
            print("Error: Could not find NanoClaw directory")
            print("Use --path to specify the NanoClaw installation directory")
            sys.exit(1)

    print(f"Importing from: {nanoclaw_dir}")

    if args.dry_run:
        sessions = import_sessions(nanoclaw_dir)
        memories = import_memories(nanoclaw_dir)
        print(f"\nDry run results:")
        print(f"  Conversations: {len(sessions)} ({sum(len(s['turns']) for s in sessions)} turns)")
        print(f"  Memory entries: {len(memories)}")
        return

    config = load_config()
    config.ensure_dirs()
    memory_store, memory_index = _connect(config)

    try:
        total = 0
        if not args.memory_only:
            sessions = import_sessions(nanoclaw_dir)
            total += ingest_sessions(sessions, "nanoclaw", memory_index)

        if not args.sessions_only:
            memories = import_memories(nanoclaw_dir)
            if memories:
                total += ingest_memories(
                    memories, "nanoclaw", memory_store, memory_index,
                    merge_to_memory_md=False,
                )

        print(f"\nDone. Indexed {total} fragments into MicrowaveOS.")
    finally:
        memory_index.close()


def _connect(config):
    """Set up memory store and index."""
    memory_store = MemoryStore(config.workspace_dir)
    memory_store.ensure_dirs()

    embedder = EmbeddingClient(api_key=config.openai_api_key)
    memory_index = MemoryIndex(config.db_path, embedder)
    memory_index.connect()

    return memory_store, memory_index


def main():
    parser = argparse.ArgumentParser(
        description="Import data from other AI tools into MicrowaveOS"
    )
    subparsers = parser.add_subparsers(dest="source", required=True)

    # Shared arguments
    for name in ["openclaw", "hermes", "nanoclaw"]:
        sub = subparsers.add_parser(name, help=f"Import from {name}")
        sub.add_argument("--path", help="Path to data directory (auto-detected if omitted)")
        sub.add_argument("--sessions-only", action="store_true", help="Only import conversations")
        sub.add_argument("--memory-only", action="store_true", help="Only import memory/knowledge")
        sub.add_argument("--dry-run", action="store_true", help="Show what would be imported without doing it")

    args = parser.parse_args()

    if args.source == "openclaw":
        import_openclaw(args)
    elif args.source == "hermes":
        import_hermes(args)
    elif args.source == "nanoclaw":
        import_nanoclaw(args)


if __name__ == "__main__":
    main()
