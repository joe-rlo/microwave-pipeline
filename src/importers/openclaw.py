"""Import conversation history and memory from OpenClaw.

Data sources:
- Sessions: ~/.openclaw/agents/<agentId>/sessions/<sessionId>.jsonl
- Memory:   ~/.openclaw/agents/<agentId>/MEMORY.md + memory/ subdirs
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


def find_openclaw_dir() -> Path | None:
    """Find the OpenClaw data directory."""
    default = Path.home() / ".openclaw"
    if default.exists():
        return default
    return None


def list_agents(openclaw_dir: Path) -> list[dict]:
    """List available OpenClaw agents."""
    agents_dir = openclaw_dir / "agents"
    if not agents_dir.exists():
        return []

    agents = []
    for agent_dir in agents_dir.iterdir():
        if agent_dir.is_dir():
            sessions_dir = agent_dir / "sessions"
            session_count = 0
            if sessions_dir.exists():
                session_count = len(list(sessions_dir.glob("*.jsonl")))
            agents.append({
                "id": agent_dir.name,
                "path": str(agent_dir),
                "session_count": session_count,
                "has_memory": (agent_dir / "MEMORY.md").exists(),
            })
    return agents


def import_sessions(agent_dir: Path) -> list[dict]:
    """Import conversation sessions from an OpenClaw agent.

    Returns list of sessions, each with:
    - session_id, started_at, turns: [{role, content, timestamp}]
    """
    sessions_dir = agent_dir / "sessions"
    if not sessions_dir.exists():
        return []

    sessions = []
    for jsonl_file in sorted(sessions_dir.glob("*.jsonl")):
        session = _parse_session(jsonl_file)
        if session and session["turns"]:
            sessions.append(session)

    log.info(f"Imported {len(sessions)} sessions from OpenClaw agent {agent_dir.name}")
    return sessions


def _parse_session(jsonl_path: Path) -> dict | None:
    """Parse a single .jsonl session file."""
    turns = []
    session_id = jsonl_path.stem
    started_at = None

    for line_num, line in enumerate(jsonl_path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        # First line is often metadata
        if line_num == 0 and "version" in entry:
            session_id = entry.get("sessionId", session_id)
            started_at = entry.get("timestamp")
            continue

        if entry.get("type") != "message":
            continue

        msg = entry.get("message", {})
        role = msg.get("role", "")

        # Skip tool results — we only want user/assistant text
        if role not in ("user", "assistant"):
            continue

        # Extract text content from content blocks
        content_blocks = msg.get("content", [])
        text_parts = []
        for block in content_blocks:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        text = "\n".join(text_parts).strip()
        if not text:
            continue

        turns.append({
            "role": role,
            "content": text,
            "timestamp": entry.get("timestamp", started_at),
        })

    return {
        "session_id": session_id,
        "started_at": started_at,
        "turns": turns,
    }


def import_memory(agent_dir: Path) -> dict:
    """Import memory files from an OpenClaw agent.

    Returns:
    - memory_md: str (MEMORY.md content)
    - daily_notes: [{date, content}]
    - topic_memories: [{path, content}]
    """
    result = {
        "memory_md": "",
        "daily_notes": [],
        "topic_memories": [],
    }

    # MEMORY.md
    memory_path = agent_dir / "MEMORY.md"
    if memory_path.exists():
        result["memory_md"] = memory_path.read_text()

    # Daily notes
    memory_dir = agent_dir / "memory"
    if memory_dir.exists():
        for day_file in sorted(memory_dir.glob("????-??-??.md")):
            result["daily_notes"].append({
                "date": day_file.stem,
                "content": day_file.read_text(),
            })

        # Topic memories (people, projects, topics, decisions)
        for subdir in ["people", "projects", "topics", "decisions"]:
            topic_dir = memory_dir / subdir
            if topic_dir.exists():
                for md_file in sorted(topic_dir.glob("*.md")):
                    result["topic_memories"].append({
                        "path": f"{subdir}/{md_file.name}",
                        "content": md_file.read_text(),
                    })

    return result
