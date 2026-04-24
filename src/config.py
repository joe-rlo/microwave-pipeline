"""Configuration for MicrowaveOS runtime."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    # Authentication
    auth_mode: str = "max"  # "max" or "api_key"
    anthropic_api_key: str = ""

    # Models
    model_main: str = "sonnet"
    model_triage: str = "haiku"
    model_reflection: str = "haiku"
    model_compaction: str = "sonnet"
    model_escalation: str = "opus"  # model for complex tasks
    escalation_effort: str = "high"  # thinking effort: "low", "medium", "high", "max"

    # Paths
    workspace_dir: Path = Path.home() / ".microwaveos" / "workspace"
    data_dir: Path = Path.home() / ".microwaveos" / "data"

    # Claude Code CLI path (for Agent SDK — uses system install with your auth session)
    cli_path: str = ""

    # Telegram
    telegram_bot_token: str = ""

    # Signal (talks to a signal-cli-rest-api daemon)
    signal_rest_url: str = ""
    signal_phone_number: str = ""  # the bot's Signal-registered number
    signal_allowed_senders: tuple[str, ...] = ()  # optional allowlist

    # Scheduler — runs as a background task inside --signal when enabled.
    # Fires cron-scheduled jobs (LLM or direct) defined via `microwaveos scheduler add`.
    scheduler_enabled: bool = False

    # OpenAI (embeddings only)
    openai_api_key: str = ""

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    embedding_version: str = "v1"

    # Session
    context_window_limit: int = 200_000  # tokens
    compaction_threshold: float = 0.8  # fraction of limit

    # Search defaults
    default_result_count: int = 5
    default_decay_half_life: float = 30.0  # days

    @property
    def identity_path(self) -> Path:
        return self.workspace_dir / "IDENTITY.md"

    @property
    def memory_path(self) -> Path:
        return self.workspace_dir / "MEMORY.md"

    @property
    def daily_dir(self) -> Path:
        return self.workspace_dir / "memory"

    @property
    def output_dir(self) -> Path:
        return self.workspace_dir / "output"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "memory.db"

    def ensure_dirs(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    """Load .env file into os.environ. No dependency required."""
    for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]:
        if candidate.is_file():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:  # don't override real env vars
                    os.environ[key] = value
            break


def _path_from_env(key: str, default: Path) -> Path:
    """Load a path from env with ~ expansion.

    Python's `Path(str)` doesn't expand `~`, so a `.env` value like
    `WORKSPACE_DIR=~/.microwaveos/workspace` would otherwise become a
    literal tilde directory inside the CWD — a subtle footgun the codebase
    hit in practice. Always route path env vars through here.
    """
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return Path(raw).expanduser()


def load_config() -> Config:
    _load_dotenv()

    # Auto-detect system Claude CLI if not explicitly set
    cli_path = os.getenv("CLAUDE_CLI_PATH", "")
    if not cli_path:
        import shutil
        system_claude = shutil.which("claude")
        if system_claude:
            cli_path = system_claude

    return Config(
        auth_mode=os.getenv("AUTH_MODE", "max"),
        cli_path=cli_path,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        model_main=os.getenv("MODEL_MAIN", "sonnet"),
        model_triage=os.getenv("MODEL_TRIAGE", "haiku"),
        model_reflection=os.getenv("MODEL_REFLECTION", "haiku"),
        model_compaction=os.getenv("MODEL_COMPACTION", "sonnet"),
        model_escalation=os.getenv("MODEL_ESCALATION", "opus"),
        escalation_effort=os.getenv("ESCALATION_EFFORT", "high"),
        workspace_dir=_path_from_env("WORKSPACE_DIR", Path.home() / ".microwaveos" / "workspace"),
        data_dir=_path_from_env("DATA_DIR", Path.home() / ".microwaveos" / "data"),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        signal_rest_url=os.getenv("SIGNAL_REST_URL", ""),
        signal_phone_number=os.getenv("SIGNAL_PHONE_NUMBER", ""),
        signal_allowed_senders=tuple(
            s.strip() for s in os.getenv("SIGNAL_ALLOWED_SENDERS", "").split(",") if s.strip()
        ),
        scheduler_enabled=os.getenv("SCHEDULER_ENABLED", "").lower() in ("1", "true", "yes", "on"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    )
