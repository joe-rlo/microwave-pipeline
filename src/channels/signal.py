"""Signal channel — chat via a signal-cli-rest-api daemon.

No streaming edits (Signal's edit surface is flaky through the REST bridge);
the pipeline runs to completion and the final response is sent as one or
more messages. Files produced by the pipeline are sent as attachments.

Setup: run `signal-cli-rest-api` (docker image `bbernhard/signal-cli-rest-api`
in `json-rpc` mode), register a phone number with it, point this channel at
its URL (e.g. `http://127.0.0.1:8080`) and that phone number.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from urllib.parse import quote

import aiohttp

from src.channels.base import Channel
from src.channels.signal_format import markdown_to_signal_text
from src.pipeline.orchestrator import Orchestrator

log = logging.getLogger(__name__)

# Signal's practical max per message — the protocol allows more, but clients
# occasionally truncate above ~2000 chars. Split on paragraph/line boundaries.
MAX_MESSAGE_LENGTH = 2000
# Reconnect backoff when the receive websocket drops.
WS_RECONNECT_SECONDS = 5
# Signal's typing indicator auto-expires after ~15s; refresh inside that
# window so it stays visible while the pipeline is working.
TYPING_REFRESH_SECONDS = 10


class SignalChannel(Channel):
    def __init__(
        self,
        orchestrator: Orchestrator,
        rest_url: str,
        phone_number: str,
        allowed_senders: list[str] | None = None,
        openai_api_key: str = "",
    ):
        super().__init__(orchestrator)
        self.rest_url = rest_url.rstrip("/")
        self.phone_number = phone_number
        # Optional allowlist — without this any Signal user who knows your
        # bot number can talk to it. Keep the bot locked down.
        self.allowed_senders: set[str] = set(allowed_senders or [])
        # For Whisper transcription of voice notes. Without a key the channel
        # still works — voice messages just fall back to an error reply.
        self.openai_api_key = openai_api_key
        self._session: aiohttp.ClientSession | None = None
        self._ws_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        self._ws_task = asyncio.create_task(self._receive_loop())
        log.info(
            f"Signal channel started for {self.phone_number} via {self.rest_url}"
            + (f" (allowlist: {len(self.allowed_senders)} senders)" if self.allowed_senders else "")
        )

    async def stop(self) -> None:
        self._stop_event.set()
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._session:
            await self._session.close()

    # --- Receive loop ---

    async def _receive_loop(self) -> None:
        """Listen to the signal-cli-rest-api websocket for incoming messages.

        Reconnects on disconnect — the daemon sometimes drops idle sockets.
        """
        ws_base = self.rest_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_base}/v1/receive/{quote(self.phone_number)}"

        while not self._stop_event.is_set():
            try:
                async with self._session.ws_connect(ws_url, heartbeat=30) as ws:
                    log.info(f"Signal websocket connected: {ws_url}")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                event = json.loads(msg.data)
                            except json.JSONDecodeError:
                                log.warning(f"Non-JSON message on Signal ws: {msg.data[:200]}")
                                continue
                            asyncio.create_task(self._handle_incoming(event))
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning(f"Signal ws error ({e}); reconnecting in {WS_RECONNECT_SECONDS}s")

            if not self._stop_event.is_set():
                await asyncio.sleep(WS_RECONNECT_SECONDS)

    async def _handle_incoming(self, event: dict) -> None:
        """Dispatch one websocket event through the pipeline.

        Handles three cases:
        - plain text: pass through as-is
        - voice note (audio attachment): transcribe via Whisper, feed transcript
          into the pipeline with a [voice] marker so the LLM knows it was spoken
        - voice + text caption: combine both
        """
        envelope = event.get("envelope") or {}
        data = envelope.get("dataMessage") or {}
        text = (data.get("message") or "").strip()
        attachments = data.get("attachments") or []

        voice_attachments = [a for a in attachments if _is_voice_note(a)]

        if not text and not voice_attachments:
            return  # receipts, typing events, empty syncs — skip

        source = envelope.get("sourceNumber") or envelope.get("source") or ""
        if self.allowed_senders and source not in self.allowed_senders:
            log.info(f"Ignoring Signal message from {source} (not in allowlist)")
            return

        # Mark as read immediately — the user sees blue checkmarks right as
        # the bot picks the message up, before transcription or pipeline work
        # begins. Combined with the typing indicator: delivered → read → typing
        # → response.
        msg_ts = data.get("timestamp") or envelope.get("timestamp")
        if msg_ts:
            await self._send_read_receipt(source, int(msg_ts))

        # Transcribe any voice notes; each becomes a line in the final message
        transcripts: list[str] = []
        for att in voice_attachments:
            try:
                audio = await self._download_attachment(att["id"])
                transcript = await self._transcribe(audio, att.get("contentType", "audio/aac"))
                if transcript:
                    transcripts.append(transcript)
            except Exception as e:
                log.warning(f"Voice transcription failed: {e}")
                await self._send_text(
                    source, f"_⚠️ couldn't transcribe that voice message: {e}_"
                )
                return

        if transcripts:
            combined = "\n".join(transcripts)
            # Show the user what we heard so misrecognitions are obvious;
            # they can correct in a follow-up instead of getting a confusing reply.
            preview = f"_heard: \"{combined}\"_"
            await self._send_text(source, preview)

            if text:
                # User attached both voice and a text caption — include both,
                # with the voice tagged so the LLM treats it as speech.
                final = f"[voice] {combined}\n\n[text caption] {text}"
            else:
                final = f"[voice] {combined}"
        else:
            final = text

        try:
            await self._process_and_respond(final, source)
        except Exception as e:
            log.exception(f"Signal pipeline failed: {e}")
            await self._send_text(source, f"⚠️ internal error: {e}")

    async def _process_and_respond(self, text: str, source: str) -> None:
        # Show typing while the pipeline works so the user doesn't stare
        # at a silent thread. Refreshed periodically — Signal's indicator
        # auto-expires after ~15s.
        typing_task = asyncio.create_task(self._typing_loop(source))
        accumulated = ""
        try:
            async for chunk in self.orchestrator.process(
                text, user_id=source, channel="signal"
            ):
                t = chunk["type"]
                if t in ("delta", "text"):
                    accumulated += chunk.get("text") or chunk.get("chunk", "")
                elif t == "file":
                    await self._send_attachment(source, chunk["name"], chunk["content"])
                elif t == "text_replace":
                    accumulated = chunk["text"]
                elif t == "compaction":
                    n = chunk.get("turns_compacted", 0)
                    m = chunk.get("turns_kept", 0)
                    note = (
                        f"_↻ Archived {n} earlier turn{'s' if n != 1 else ''} "
                        f"(summary saved, last {m} kept in context)._"
                    )
                    await self._send_text(source, note)
                # metadata: nothing to show on Signal

            if accumulated:
                formatted = markdown_to_signal_text(accumulated)
                for part in _split_message(formatted):
                    await self._send_text(source, part)
        finally:
            typing_task.cancel()
            # Explicitly clear so the indicator drops immediately instead of
            # lingering until Signal's 15s auto-expire.
            await self._send_typing(source, active=False)

    # --- Read receipts ---

    async def _send_read_receipt(self, sender: str, message_timestamp: int) -> None:
        """Send a "read" receipt for one of the sender's messages.

        Signal identifies the target message by the sender's original send
        timestamp (milliseconds since epoch) — that's what pairs the receipt
        to the right message on their client.
        """
        url = f"{self.rest_url}/v1/receipts/{quote(self.phone_number)}"
        payload = {
            "receipt_type": "read",
            "recipient": sender,
            "timestamp": message_timestamp,
        }
        try:
            async with self._session.post(url, json=payload) as r:
                if r.status >= 400:
                    body = await r.text()
                    log.debug(f"Signal read-receipt {r.status}: {body}")
        except Exception as e:
            log.debug(f"Signal read-receipt failed: {e}")

    # --- Voice transcription ---

    async def _download_attachment(self, attachment_id: str) -> bytes:
        """Fetch an attachment's raw bytes from signal-cli-rest-api."""
        url = f"{self.rest_url}/v1/attachments/{quote(attachment_id)}"
        async with self._session.get(url) as r:
            if r.status >= 400:
                raise RuntimeError(f"attachment fetch {r.status}: {await r.text()}")
            return await r.read()

    async def _transcribe(self, audio: bytes, content_type: str) -> str:
        """Transcribe voice audio via OpenAI Whisper.

        Signal voice notes are typically AAC in an M4A container; Whisper
        accepts m4a/mp3/wav/etc. The filename extension hint matters — the
        API infers the format from it.
        """
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set (needed for Whisper)")

        ext = _ext_for_content_type(content_type)
        form = aiohttp.FormData()
        form.add_field(
            "file", audio,
            filename=f"voice{ext}",
            content_type=content_type,
        )
        form.add_field("model", "whisper-1")
        form.add_field("response_format", "text")

        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        async with aiohttp.ClientSession() as whisper_session:
            async with whisper_session.post(
                "https://api.openai.com/v1/audio/transcriptions",
                data=form,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as r:
                if r.status >= 400:
                    body = await r.text()
                    raise RuntimeError(f"Whisper {r.status}: {body[:200]}")
                # response_format=text returns plain text (no JSON)
                return (await r.text()).strip()

    # --- Typing indicator ---

    async def _typing_loop(self, recipient: str) -> None:
        """Keep Signal's typing indicator visible while the pipeline runs."""
        try:
            while True:
                await self._send_typing(recipient, active=True)
                await asyncio.sleep(TYPING_REFRESH_SECONDS)
        except asyncio.CancelledError:
            pass

    async def _send_typing(self, recipient: str, active: bool) -> None:
        """PUT to start the typing indicator, DELETE to clear it.

        signal-cli-rest-api's endpoint is `/v1/typing-indicator/{bot_number}`
        with `{"recipient": "<sender>"}` in the body.
        """
        url = f"{self.rest_url}/v1/typing-indicator/{quote(self.phone_number)}"
        method = self._session.put if active else self._session.delete
        try:
            async with method(url, json={"recipient": recipient}) as r:
                if r.status >= 400 and r.status != 404:
                    body = await r.text()
                    log.debug(f"Signal typing {r.status}: {body}")
        except Exception as e:
            log.debug(f"Signal typing send failed: {e}")

    # --- Send ---

    async def send_text(self, recipient: str, text: str) -> None:
        """Public send. Delegates to the private implementation so internal
        callers keep working while external callers (e.g. scheduler) stay
        off the `_` prefix."""
        await self._send_text(recipient, text)

    async def send_attachment(
        self, recipient: str, filename: str, content: str | bytes
    ) -> None:
        await self._send_attachment(recipient, filename, content)

    async def _send_text(self, recipient: str, text: str) -> None:
        if not text:
            return
        payload = {
            "number": self.phone_number,
            "recipients": [recipient],
            "message": text,
            # Signal clients don't auto-parse `**bold**` in received messages —
            # the daemon must convert markdown into style ranges server-side.
            "text_mode": "styled",
        }
        try:
            async with self._session.post(f"{self.rest_url}/v2/send", json=payload) as r:
                if r.status >= 400:
                    body = await r.text()
                    log.warning(f"Signal send {r.status}: {body}")
        except Exception as e:
            log.warning(f"Signal send failed: {e}")

    async def _send_attachment(
        self, recipient: str, filename: str, content: str | bytes
    ) -> None:
        if isinstance(content, str):
            content = content.encode("utf-8")
        mime = _guess_mime(filename)
        b64 = base64.b64encode(content).decode("ascii")
        # signal-cli-rest-api accepts data-URI style attachments with an
        # embedded filename; plain base64 also works but then Signal picks
        # a generic name.
        data_uri = f"data:{mime};filename={filename};base64,{b64}"
        payload = {
            "number": self.phone_number,
            "recipients": [recipient],
            "message": "",
            "text_mode": "styled",
            "base64_attachments": [data_uri],
        }

        # Signal's push servers occasionally time out mid-upload
        # (PushNetworkException). Retry with exponential backoff so one
        # flaky push doesn't silently swallow the attachment.
        max_attempts = 3
        last_error: str | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                async with self._session.post(f"{self.rest_url}/v2/send", json=payload) as r:
                    if r.status < 400:
                        return  # delivered
                    body = await r.text()
                    last_error = f"{r.status}: {body}"
                    # 4xx (other than timeout-style 400s) is usually not
                    # retryable — bail early to avoid burning the user's
                    # time on a doomed payload.
                    if r.status >= 400 and "timeout" not in body.lower() and "PushNetwork" not in body:
                        break
            except Exception as e:
                last_error = str(e)

            if attempt < max_attempts:
                backoff = 2 ** attempt  # 2s, 4s
                log.warning(
                    f"Signal attachment send attempt {attempt}/{max_attempts} failed "
                    f"({last_error}); retrying in {backoff}s"
                )
                await asyncio.sleep(backoff)

        # All attempts exhausted — surface the failure so the user knows
        # the file didn't arrive instead of staring at an empty thread.
        log.error(
            f"Signal attachment send failed after {max_attempts} attempts "
            f"(filename={filename}): {last_error}"
        )
        await self._send_text(
            recipient,
            f"⚠️ Couldn't deliver attachment `{filename}` — Signal push timed out. "
            f"Try asking again in a moment.",
        )


def _is_voice_note(att: dict) -> bool:
    """Signal marks voice notes with voiceNote=True; some clients omit the
    flag but still send audio/* — accept either as a voice message."""
    if att.get("voiceNote"):
        return True
    ct = (att.get("contentType") or "").lower()
    return ct.startswith("audio/")


def _ext_for_content_type(content_type: str) -> str:
    """Map Signal's content-type to a file extension Whisper understands.

    Whisper infers audio format from the filename; picking the wrong
    extension for AAC/Opus audio causes a 400 from the API.
    """
    ct = content_type.lower().split(";")[0].strip()
    return {
        "audio/aac": ".m4a",  # Signal's AAC is usually in an MP4 container
        "audio/mp4": ".m4a",
        "audio/m4a": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/ogg": ".ogg",
        "audio/opus": ".ogg",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/webm": ".webm",
        "audio/flac": ".flac",
    }.get(ct, ".m4a")


def _guess_mime(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".html") or lower.endswith(".htm"):
        return "text/html"
    if lower.endswith(".md") or lower.endswith(".txt"):
        return "text/plain"
    if lower.endswith(".json"):
        return "application/json"
    if lower.endswith(".csv"):
        return "text/csv"
    if lower.endswith(".py"):
        return "text/x-python"
    return "application/octet-stream"


def _split_message(text: str, max_len: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split long text at paragraph/line boundaries to fit Signal's limit."""
    if len(text) <= max_len:
        return [text]

    parts: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            parts.append(remaining)
            break
        chunk = remaining[:max_len]
        split_at = chunk.rfind("\n\n")
        if split_at < max_len // 4:
            split_at = chunk.rfind("\n")
        if split_at < max_len // 4:
            split_at = chunk.rfind(" ")
        if split_at < max_len // 4:
            split_at = max_len
        parts.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip("\n")
    return [p for p in parts if p.strip()]
