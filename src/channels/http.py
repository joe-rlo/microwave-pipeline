"""HTTP channel — API access via POST endpoint.

Returns JSON with response + pipeline metadata.
"""

from __future__ import annotations

import json
import logging

from aiohttp import web

from src.channels.base import Channel
from src.pipeline.orchestrator import Orchestrator

log = logging.getLogger(__name__)


class HTTPChannel(Channel):
    def __init__(self, orchestrator: Orchestrator, host: str = "127.0.0.1", port: int = 8080):
        super().__init__(orchestrator)
        self.host = host
        self.port = port
        self.app: web.Application | None = None
        self.runner: web.AppRunner | None = None

    async def start(self) -> None:
        self.app = web.Application()
        self.app.router.add_post("/chat", self._handle_chat)
        self.app.router.add_get("/health", self._handle_health)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        log.info(f"HTTP channel listening on {self.host}:{self.port}")

    async def stop(self) -> None:
        if self.runner:
            await self.runner.cleanup()

    async def _handle_chat(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        message = body.get("message", "").strip()
        if not message:
            return web.json_response({"error": "message required"}, status=400)

        user_id = body.get("user_id", "http")
        full_response = ""
        metadata = None

        async for chunk in self.orchestrator.process(message, user_id=user_id, channel="http"):
            if chunk["type"] in ("delta", "text"):
                full_response += chunk.get("text") or chunk.get("chunk", "")
            elif chunk["type"] == "metadata":
                metadata = chunk["pipeline"]

        result = {"response": full_response}
        if metadata:
            result["metadata"] = {
                "triage_intent": metadata.triage.intent if metadata.triage else None,
                "triage_complexity": metadata.triage.complexity if metadata.triage else None,
                "search_fragments": len(metadata.search.fragments) if metadata.search else 0,
                "search_time_ms": metadata.search.search_time_ms if metadata.search else 0,
                "reflection_confidence": metadata.reflection.confidence if metadata.reflection else None,
                "reflection_action": metadata.reflection.action if metadata.reflection else None,
                "total_time_ms": metadata.total_time_ms,
            }

        return web.json_response(result)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})
