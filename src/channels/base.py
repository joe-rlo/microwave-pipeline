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
