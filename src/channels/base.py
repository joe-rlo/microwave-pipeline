"""Channel protocol — all channels implement this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.pipeline.orchestrator import Orchestrator


class Channel(ABC):
    """Base channel interface. All intelligence lives in the pipeline."""

    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator

    @abstractmethod
    async def start(self) -> None:
        """Start receiving messages."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel."""
        ...

    # --- outbound API (used by the scheduler and other non-interactive callers) ---
    # Channels override these. The scheduler talks to channels through this
    # pair instead of reaching into private `_send_*` methods.

    async def send_text(self, recipient: str, text: str) -> None:
        """Deliver a plain text message to a recipient."""
        raise NotImplementedError

    async def send_attachment(
        self, recipient: str, filename: str, content: str | bytes
    ) -> None:
        """Deliver a file attachment. `content` may be text or raw bytes."""
        raise NotImplementedError
