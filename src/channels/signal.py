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
import time
from dataclasses import dataclass, field
from urllib.parse import quote

import aiohttp

from src.channels.base import Channel
from src.channels._http import make_session
from src.channels.signal_format import markdown_to_signal_text
from src.channels.tts import TTSError, synthesize as tts_synthesize
from src.pipeline.orchestrator import Orchestrator
from src.projects.bible import handle_bible_command
from src.projects.chat import handle_project_command
from src.skills.chat import handle_skill_command

log = logging.getLogger(__name__)

# Signal's practical max per message — the protocol allows more, but clients
# occasionally truncate above ~2000 chars. Split on paragraph/line boundaries.
MAX_MESSAGE_LENGTH = 2000
# Reconnect backoff when the receive websocket drops.
WS_RECONNECT_SECONDS = 5
# Signal's typing indicator auto-expires after ~15s; refresh inside that
# window so it stays visible while the pipeline is working.
TYPING_REFRESH_SECONDS = 10
# Wait this long after the user's last activity (message OR typing-stop)
# before processing. Coalesces back-to-back messages into one pipeline run.
DEBOUNCE_SECONDS = 2.5
# Hard ceiling on how long a buffer may sit before we force-process it,
# so a stuck typing-indicator (network blip, app crash) can't strand input.
MAX_HOLD_SECONDS = 60.0
# How long after a pipeline starts a follow-up message is still treated as
# an addendum (cancel current pipeline, merge inputs). Past this window the
# follow-up becomes a queued new turn instead, so a user who gave up waiting
# on a long-running answer doesn't get their later question silently merged
# into the earlier thought. Generous on purpose — chunked-thinking pauses
# can be substantial; tune from logs.
ADDENDUM_WINDOW_SECONDS = 18.0
# Send a small "_…thinking_" message if the pipeline is still running this
# many seconds in. Signal has no streaming, so without this nudge a slow
# turn looks indistinguishable from a stalled bot.
THINKING_NUDGE_SECONDS = 4.0
# OpenAI TTS voice for voice replies. Configurable via env (TTS_VOICE)
# but a single value is opinionated by default — voice cloning / per-user
# voices is a larger feature, not a v1 concern.
DEFAULT_TTS_VOICE = "alloy"

# Inline override markers. A user can force a reply mode regardless of
# the input mode by including one of these phrases anywhere in the
# message; the marker is stripped before the message reaches the LLM
# so it doesn't pollute the prompt or session memory.
import re as _re
_REPLY_BY_VOICE_RE = _re.compile(r"\breply\s+by\s+voice\b", _re.IGNORECASE)
_REPLY_BY_TEXT_RE = _re.compile(r"\breply\s+by\s+text\b", _re.IGNORECASE)


@dataclass
class _PendingTurn:
    """Per-sender input buffer.

    Holds messages and voice-audio that arrived during a debounce window.
    Flushed to the pipeline as one combined input — that's what makes
    "user types two messages back to back" produce one coherent reply.

    The buffer's lifetime extends through the pipeline run (not just up to
    flush) so addendum-merge can cancel + re-flush without losing earlier
    content. On cancel, new input is appended to the same buffer; on
    successful pipeline completion, the buffer is dropped.
    """
    source: str
    started_at: float
    last_activity: float
    messages: list[str] = field(default_factory=list)
    # Tuples of (audio_bytes, content_type) — transcribed at flush time
    # so a user who sends voice + correction-text together still gets one turn.
    voice: list[tuple[bytes, str]] = field(default_factory=list)
    # Already-transcribed voice text. Populated by _flush_pending when it
    # consumes `voice`; persists across addendum re-flushes so a cancelled
    # turn's transcription work isn't redone.
    transcribed: list[str] = field(default_factory=list)
    # Tuples of (image_bytes, content_type) for image attachments —
    # passed through to the pipeline as multimodal content blocks.
    # Persists across addendum re-flushes so we don't have to re-download.
    images: list[tuple[bytes, str]] = field(default_factory=list)
    is_typing: bool = False
    debounce_task: asyncio.Task | None = None
    maxhold_task: asyncio.Task | None = None


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
        # Per-sender debounce buffers and currently-running pipeline tasks.
        # The lock guards both — they're touched from receive loop and
        # debounce-fire callbacks at the same time.
        self._pending: dict[str, _PendingTurn] = {}
        self._processing: dict[str, asyncio.Task] = {}
        # When a follow-up arrives past the addendum window, it parks here
        # until the active pipeline finishes — then gets promoted to _pending
        # and run as its own turn. Distinct slot so addendum-merge (cancel +
        # combine) and queued-new-turn (wait + run separately) don't collide.
        self._queued: dict[str, _PendingTurn] = {}
        # Pipeline-start timestamp per sender — used by _buffer_input to
        # decide addendum-vs-queued based on how far into the pipeline a
        # follow-up landed.
        self._pipeline_started_at: dict[str, float] = {}
        self._state_lock = asyncio.Lock()

    async def start(self) -> None:
        self._session = make_session()
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
        # Cancel any still-running pipeline tasks and debounce timers so a
        # graceful shutdown doesn't hang on dangling work.
        async with self._state_lock:
            for bucket in (self._pending, self._queued):
                for pending in bucket.values():
                    for t in (pending.debounce_task, pending.maxhold_task):
                        if t and not t.done():
                            t.cancel()
                bucket.clear()
            for task in list(self._processing.values()):
                if not task.done():
                    task.cancel()
            self._processing.clear()
            self._pipeline_started_at.clear()
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
        """Top-level event router.

        Two event shapes matter:
        - `dataMessage` — actual user content (text and/or voice). Buffers
          into `_pending[source]` and arms the debounce timer; the buffer
          flushes through the pipeline when the user falls silent.
        - `typingMessage` — heads-up that the user is composing. Pauses the
          debounce timer so we don't fire mid-sentence; restarts it when
          they stop typing.

        Anything else (receipts, sync, empty events) is ignored.
        """
        envelope = event.get("envelope") or {}
        source = envelope.get("sourceNumber") or envelope.get("source") or ""
        if not source:
            return
        if self.allowed_senders and source not in self.allowed_senders:
            return

        # Typing indicators
        typing = envelope.get("typingMessage")
        if typing:
            action = (typing.get("action") or "").upper()
            await self._handle_typing(source, action)
            return

        # Data messages (text + attachments)
        data = envelope.get("dataMessage") or {}
        text = (data.get("message") or "").strip()
        attachments = data.get("attachments") or []
        voice_attachments = [a for a in attachments if _is_voice_note(a)]
        image_attachments = [a for a in attachments if _is_image(a)]

        if not text and not voice_attachments and not image_attachments:
            return  # receipts, empty syncs, etc.

        # Swipe-to-reply context. Signal's daemon attaches a `quote` block
        # naming the message being replied to (author UUID/number + the
        # quoted text). Without this prefix the LLM has no way to know what
        # "the second one" or "scrap that" refers to.
        quote = data.get("quote")
        if quote:
            quote_prefix = _format_quote_context(quote, self.phone_number)
            if quote_prefix:
                text = f"{quote_prefix}\n\n{text}" if text else quote_prefix

        # Read receipt fires immediately on every message — gives the user
        # blue checkmarks right away, independent of debounce/pipeline timing.
        msg_ts = data.get("timestamp") or envelope.get("timestamp")
        if msg_ts:
            await self._send_read_receipt(source, int(msg_ts))

        # Skill / project / bible commands short-circuit the buffer entirely.
        # They're meant to land instantly and don't combine meaningfully
        # with other input. Skip if the user also sent a voice note OR an
        # image — treat those as content, not a command.
        if text and not voice_attachments and not image_attachments:
            for handler in (
                handle_skill_command,
                handle_project_command,
                handle_bible_command,
            ):
                reply = await handler(text, self.orchestrator)
                if reply is not None:
                    await self._send_text(source, reply)
                    return

        # Download voice payloads now (fast — just network), but defer
        # transcription to flush time so it joins the same buffer window.
        downloaded_voice: list[tuple[bytes, str]] = []
        for att in voice_attachments:
            try:
                audio = await self._download_attachment(att["id"])
                downloaded_voice.append((audio, att.get("contentType", "audio/aac")))
            except Exception as e:
                log.warning(f"Voice download failed: {e}")
                await self._send_text(
                    source, f"_⚠️ couldn't download that voice message: {e}_"
                )

        # Download image payloads now too — they're small and async, and
        # downloading at receive-time means the addendum window can merge
        # text-then-image (or vice versa) without redoing the download.
        downloaded_images: list[tuple[bytes, str]] = []
        for att in image_attachments:
            try:
                img = await self._download_attachment(att["id"])
                ct = (att.get("contentType") or "image/jpeg").lower()
                downloaded_images.append((img, ct))
            except Exception as e:
                log.warning(f"Image download failed: {e}")
                await self._send_text(
                    source, f"_⚠️ couldn't download that image: {e}_"
                )

        await self._buffer_input(
            source,
            text=text,
            voice=downloaded_voice,
            images=downloaded_images,
        )

    async def _handle_typing(self, source: str, action: str) -> None:
        """Pause/resume the debounce timer based on typing state."""
        async with self._state_lock:
            pending = self._pending.get(source)
            if pending is None:
                # No buffered input yet — nothing to pause. We don't pre-arm
                # a buffer just because the user is typing; the first
                # message creates the buffer.
                return
            running = self._processing.get(source)
            if running is not None and not running.done():
                # Pipeline is running this buffer. Typing while it works
                # doesn't change anything — debounce isn't armed during
                # pipeline, and the addendum-cancel path is driven by an
                # actual incoming message, not by typing.
                return
            now = time.monotonic()
            pending.last_activity = now
            if action == "STARTED":
                pending.is_typing = True
                # Cancel debounce so we don't fire mid-compose. Max-hold
                # timer keeps running as a safety net.
                if pending.debounce_task and not pending.debounce_task.done():
                    pending.debounce_task.cancel()
                pending.debounce_task = None
            elif action == "STOPPED":
                pending.is_typing = False
                # User stopped typing — start a fresh debounce countdown.
                # Their last keystroke counts as the "last activity."
                self._arm_debounce(pending)

    async def _buffer_input(
        self,
        source: str,
        text: str = "",
        voice: list[tuple[bytes, str]] | None = None,
        images: list[tuple[bytes, str]] | None = None,
    ) -> None:
        """Append to the per-sender buffer.

        Three modes depending on what's happening for this sender:

        - **Fresh**: no pipeline running. Standard buffer + debounce.
          Multiple messages within DEBOUNCE_SECONDS coalesce into one turn.

        - **Addendum**: pipeline running and we're still inside
          ADDENDUM_WINDOW_SECONDS. The follow-up is treated as a
          continuation of the same thought (canonical use case: user
          forgot to add something, or thinks/writes in chunks). Cancel
          the in-flight pipeline, append to the still-alive buffer, and
          let debounce fire a fresh pipeline run on the combined input.
          Already-streamed reply text stays sent — Signal can't retract
          it — but no streaming on Signal means there's nothing to retract.

        - **Queued**: pipeline running and we're past the addendum window.
          The user almost certainly thinks the previous turn is done and
          this is a new question. Park it in `_queued[source]` so the
          active pipeline finishes cleanly and writes its turn pair, then
          the queued buffer gets promoted and run as its own turn.
        """
        async with self._state_lock:
            now = time.monotonic()
            running = self._processing.get(source)
            running_alive = running is not None and not running.done()

            if running_alive:
                started_at = self._pipeline_started_at.get(source, now)
                elapsed = now - started_at

                if elapsed < ADDENDUM_WINDOW_SECONDS:
                    # Addendum: cancel and merge.
                    log.info(
                        f"Addendum (elapsed={elapsed:.1f}s): cancelling "
                        f"in-flight pipeline for {source}, merging input"
                    )
                    running.cancel()
                    target = self._pending.get(source)
                    if target is None:
                        # Defensive: pipeline was running without a live
                        # buffer (shouldn't happen with the new lifecycle,
                        # but recover gracefully if it does).
                        target = _PendingTurn(
                            source=source, started_at=now, last_activity=now
                        )
                        self._pending[source] = target
                        target.maxhold_task = asyncio.create_task(
                            self._maxhold(source, target.started_at)
                        )
                    target.last_activity = now
                    if text:
                        target.messages.append(text)
                    if voice:
                        target.voice.extend(voice)
                    if images:
                        target.images.extend(images)
                    if target.debounce_task and not target.debounce_task.done():
                        target.debounce_task.cancel()
                    if not target.is_typing:
                        self._arm_debounce(target)
                    return

                # Past addendum window: queue as a new turn.
                log.info(
                    f"Queued new turn (elapsed={elapsed:.1f}s): parking "
                    f"message for {source} until current pipeline finishes"
                )
                queued = self._queued.get(source)
                if queued is None:
                    queued = _PendingTurn(
                        source=source, started_at=now, last_activity=now
                    )
                    self._queued[source] = queued
                    # No maxhold on the queued buffer — its lifetime is
                    # bounded by the active pipeline. Promotion arms a
                    # fresh maxhold via the standard buffering path.
                else:
                    queued.last_activity = now
                if text:
                    queued.messages.append(text)
                if voice:
                    queued.voice.extend(voice)
                if images:
                    queued.images.extend(images)
                # No debounce here — promotion in _safe_process's finally
                # arms it once the active pipeline clears.
                return

            # Fresh: no pipeline running. Standard buffer + debounce.
            pending = self._pending.get(source)
            if pending is None:
                pending = _PendingTurn(
                    source=source, started_at=now, last_activity=now
                )
                self._pending[source] = pending
                pending.maxhold_task = asyncio.create_task(
                    self._maxhold(source, pending.started_at)
                )
            else:
                pending.last_activity = now

            if text:
                pending.messages.append(text)
            if voice:
                pending.voice.extend(voice)
            if images:
                pending.images.extend(images)

            # Reset debounce: cancel old timer, start a fresh DEBOUNCE_SECONDS
            # countdown. Skip if user is actively typing — they'll resume the
            # timer themselves via the STOPPED event.
            if pending.debounce_task and not pending.debounce_task.done():
                pending.debounce_task.cancel()
            if not pending.is_typing:
                self._arm_debounce(pending)

    def _arm_debounce(self, pending: _PendingTurn) -> None:
        """Spawn a fresh debounce timer that fires `_flush_pending`."""
        pending.debounce_task = asyncio.create_task(
            self._debounce_then_flush(pending.source)
        )

    async def _debounce_then_flush(self, source: str) -> None:
        try:
            await asyncio.sleep(DEBOUNCE_SECONDS)
        except asyncio.CancelledError:
            return  # newer message reset us; nothing to do
        await self._flush_pending(source)

    async def _maxhold(self, source: str, started_at: float) -> None:
        """Force-flush after MAX_HOLD_SECONDS no matter what.

        Guards against pathological cases — typing-indicator stuck on,
        network blip during STOPPED event, etc. — that would otherwise
        leave a buffer indefinitely pending.
        """
        try:
            await asyncio.sleep(MAX_HOLD_SECONDS)
        except asyncio.CancelledError:
            return
        async with self._state_lock:
            pending = self._pending.get(source)
            if pending is None or pending.started_at != started_at:
                return  # buffer already drained or replaced
            if self._processing.get(source) and not self._processing[source].done():
                # Pipeline is already running this buffer — nothing to force.
                return
            log.info(f"Max-hold timer flushing buffer for {source}")
        await self._flush_pending(source)

    async def _flush_pending(self, source: str) -> None:
        """Snapshot the buffer's content and start a pipeline run.

        The buffer is NOT popped here — it stays alive in `_pending[source]`
        through the pipeline's lifetime so addendum-merge (a follow-up
        message during the pipeline's addendum window) can cancel the
        in-flight task and re-flush the same buffer with the appended
        content. The buffer is popped by `_safe_process` only on normal
        completion.

        No-ops if a pipeline is already running for this source — that
        means the buffer is already in flight. (Can happen via spurious
        debounce fires during pipeline; the actual addendum-cancel path
        is in `_buffer_input`.)
        """
        async with self._state_lock:
            running = self._processing.get(source)
            if running is not None and not running.done():
                log.debug(f"Skipping flush for {source} — pipeline already running")
                return
            pending = self._pending.get(source)
            if pending is None:
                return
            # Cancel timer tasks. They're already firing; nothing more to do.
            for t in (pending.debounce_task, pending.maxhold_task):
                if t and not t.done() and t is not asyncio.current_task():
                    t.cancel()
            pending.debounce_task = None
            pending.maxhold_task = None

        if (
            not pending.messages
            and not pending.voice
            and not pending.transcribed
            and not pending.images
        ):
            return

        # Transcribe any voice items we haven't transcribed yet — we drain
        # `voice` into `transcribed` so a cancelled pipeline's transcription
        # work is preserved across an addendum re-flush.
        new_transcripts: list[str] = []
        if pending.voice:
            voice_to_process = list(pending.voice)
            pending.voice.clear()
            for audio, ct in voice_to_process:
                try:
                    t = await self._transcribe(audio, ct)
                    if t:
                        new_transcripts.append(t)
                except Exception as e:
                    log.warning(f"Transcription failed: {e}")
                    await self._send_text(
                        source, f"_⚠️ couldn't transcribe a voice message: {e}_"
                    )
            pending.transcribed.extend(new_transcripts)

        # Echo only newly-heard voice — re-echoing already-acknowledged
        # transcripts on a re-flush would be confusing.
        if new_transcripts:
            voice_combined = "\n".join(new_transcripts)
            await self._send_text(source, f'_heard: "{voice_combined}"_')

        # Compose the final input. Voice and text are tagged so the LLM
        # can tell them apart even when bundled together. Inline override
        # markers ("reply by voice" / "reply by text") are detected and
        # stripped before the LLM ever sees them — they're channel-layer
        # routing hints, not part of the user's message to the agent.
        all_transcripts = list(pending.transcribed)
        clean_messages, override = _extract_reply_override(pending.messages)
        parts: list[str] = []
        if all_transcripts:
            parts.append(f"[voice] {chr(10).join(all_transcripts)}")
        if clean_messages:
            joined = "\n\n".join(m for m in clean_messages if m.strip())
            if joined:
                if all_transcripts:
                    parts.append(f"[text caption] {joined}")
                else:
                    parts.append(joined)
        final = "\n\n".join(parts)
        # If only images were sent (no text, no voice), give the model a
        # generic prompt — Anthropic's API requires at least one text block
        # in the user content, and "What's in this image?" matches what the
        # user almost certainly wanted by sending a photo with no caption.
        if not final.strip() and pending.images:
            final = "What's in this image?"
        if not final.strip():
            return

        # Snapshot images for the pipeline. Like transcription, we *keep*
        # them on `pending` so an addendum-cancel re-flush doesn't lose
        # them — the buffer is the source of truth across re-flushes; only
        # the success path in _safe_process pops it.
        snap_images = list(pending.images) if pending.images else None

        # Decide voice-vs-text reply: explicit override wins; otherwise
        # match the input mode (voice in -> voice out, text in -> text out).
        if override == "voice":
            reply_as_voice = True
        elif override == "text":
            reply_as_voice = False
        else:
            reply_as_voice = bool(all_transcripts)

        # Spawn the pipeline as a tracked task so a future incoming message
        # in the addendum window can cancel and merge.
        task = asyncio.create_task(
            self._safe_process(
                final, source,
                images=snap_images,
                reply_as_voice=reply_as_voice,
            )
        )
        async with self._state_lock:
            self._processing[source] = task
            self._pipeline_started_at[source] = time.monotonic()

    async def _safe_process(
        self,
        text: str,
        source: str,
        images: list[tuple[bytes, str]] | None = None,
        reply_as_voice: bool = False,
    ) -> None:
        """Wrapper around _process_and_respond that owns the buffer lifecycle.

        On normal completion (or surfaced error reply), the buffer is
        consumed: `_pending[source]` is popped, and any `_queued[source]`
        is promoted to `_pending[source]` with a fresh debounce so the
        next turn fires after the user's debounce window expires.

        On cancellation (addendum path), `_pending[source]` is left in
        place — `_buffer_input` already appended the new input to it and
        re-armed the debounce; the next flush will run on the merged content.
        """
        completed = False  # True on normal finish OR surfaced error reply
        try:
            await self._process_and_respond(
                text, source, images=images, reply_as_voice=reply_as_voice,
            )
            completed = True
        except asyncio.CancelledError:
            log.info(f"Pipeline for {source} cancelled (addendum or shutdown)")
            # The cancelled `llm.send()` may have left phantom messages in
            # the SDK's internal history. Tell the orchestrator to resync
            # before its next turn so we don't carry the corruption forward.
            try:
                self.orchestrator.mark_llm_for_reset()
            except Exception as e:
                log.warning(f"mark_llm_for_reset failed: {e}")
            raise
        except Exception as e:
            log.exception(f"Signal pipeline failed: {e}")
            try:
                await self._send_text(source, f"⚠️ internal error: {e}")
            except Exception:
                pass
            # An LLM exception can also leave the SDK with a half-committed
            # user message; same fix as the cancel path.
            try:
                self.orchestrator.mark_llm_for_reset()
            except Exception:
                pass
            # Treat as completed: the user got a (failure) reply, so the
            # buffer's input has been "answered" — don't keep it for a
            # phantom re-flush.
            completed = True
        finally:
            async with self._state_lock:
                # Only act if we're still the current task — otherwise a
                # newer pipeline has already taken our slot.
                if self._processing.get(source) is asyncio.current_task():
                    del self._processing[source]
                    self._pipeline_started_at.pop(source, None)
                    if completed:
                        # Drain the buffer; promote any queued turn.
                        consumed = self._pending.pop(source, None)
                        if consumed is not None:
                            for t in (consumed.debounce_task, consumed.maxhold_task):
                                if t and not t.done():
                                    t.cancel()
                        queued = self._queued.pop(source, None)
                        if queued is not None:
                            queued.last_activity = time.monotonic()
                            self._pending[source] = queued
                            # Fresh maxhold for the promoted buffer.
                            queued.maxhold_task = asyncio.create_task(
                                self._maxhold(source, queued.started_at)
                            )
                            if not queued.is_typing:
                                self._arm_debounce(queued)

    async def _process_and_respond(
        self,
        text: str,
        source: str,
        images: list[tuple[bytes, str]] | None = None,
        reply_as_voice: bool = False,
    ) -> None:
        # Show typing while the pipeline works so the user doesn't stare
        # at a silent thread. Refreshed periodically — Signal's indicator
        # auto-expires after ~15s.
        typing_task = asyncio.create_task(self._typing_loop(source))
        # Belt-and-suspenders nudge for slow turns: typing alone isn't
        # enough signal for users to know whether the bot is working or
        # stalled. After THINKING_NUDGE_SECONDS we send one short message
        # so they have a clear "yes, still on it" beat. Cancelled if the
        # pipeline finishes inside the window — fast turns stay clean.
        thinking_task = asyncio.create_task(self._thinking_nudge(source))
        accumulated = ""
        try:
            async for chunk in self.orchestrator.process(
                text, user_id=source, channel="signal", images=images,
            ):
                t = chunk["type"]
                if t in ("delta", "text"):
                    accumulated += chunk.get("text") or chunk.get("chunk", "")
                elif t == "file":
                    await self._send_attachment(source, chunk["name"], chunk["content"])
                elif t == "file_written":
                    # Active-project file landed on disk. Send a small note
                    # with word count + a teaser of the opening line(s) so
                    # the user knows what arrived without opening the file.
                    name = chunk.get("name", "?")
                    words = chunk.get("word_count", 0)
                    preview = chunk.get("preview", "")
                    note = f"_✓ wrote `{name}` ({words:,} words)_"
                    if preview:
                        # Indent preview as a quote block so it's visually
                        # distinct from the rest of the reply.
                        first_line = preview.split("\n", 1)[0]
                        note += f"\n\n> {first_line}"
                    await self._send_text(source, note)
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
                # Voice reply path: synthesize first so we can attach the
                # audio + a small text transcript. If synthesis fails we
                # fall through to text-only with a brief note rather than
                # silently dropping the response.
                if reply_as_voice and accumulated.strip():
                    audio_sent = await self._send_voice_reply(source, accumulated)
                    if audio_sent:
                        # Send a short text transcript alongside so the
                        # conversation stays auditable (parallel to the
                        # `_heard:_` echo on the way in).
                        for part in _split_message(formatted):
                            await self._send_text(source, part)
                    else:
                        # Synthesis failed — already surfaced by the helper;
                        # fall through to plain-text reply so the user
                        # actually gets the answer.
                        for part in _split_message(formatted):
                            await self._send_text(source, part)
                else:
                    for part in _split_message(formatted):
                        await self._send_text(source, part)
        finally:
            typing_task.cancel()
            thinking_task.cancel()
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
        async with make_session() as whisper_session:
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

    async def _send_voice_reply(self, recipient: str, text: str) -> bool:
        """Synthesize `text` via OpenAI TTS and send as a voice attachment.

        Returns True on success, False on failure (caller falls through
        to text-only reply). Strips Markdown emphasis before synthesis
        so the spoken voice doesn't say "underscore underscore"; the
        text transcript shipped alongside keeps the formatting.
        """
        if not self.openai_api_key:
            log.warning("Voice reply requested but OPENAI_API_KEY not set")
            await self._send_text(
                recipient,
                "_⚠️ couldn't synthesize voice reply (OPENAI_API_KEY missing); "
                "here's the text:_",
            )
            return False
        try:
            # Mild cleanup so TTS doesn't read literal markdown punctuation.
            spoken = _strip_markdown_emphasis(text)
            audio = await tts_synthesize(
                spoken,
                api_key=self.openai_api_key,
                voice=DEFAULT_TTS_VOICE,
            )
        except TTSError as e:
            log.warning(f"TTS synthesis failed: {e}")
            await self._send_text(
                recipient,
                f"_⚠️ couldn't synthesize voice reply ({e}); here's the text:_",
            )
            return False
        except Exception as e:
            log.exception(f"Unexpected TTS error: {e}")
            await self._send_text(
                recipient,
                "_⚠️ voice synthesis hit an unexpected error; here's the text:_",
            )
            return False

        await self._send_attachment(recipient, "reply.aac", audio)
        return True

    async def _thinking_nudge(self, recipient: str) -> None:
        """Send a one-shot "_…thinking_" message after a quiet pause.

        Signal can't stream, so a long-running pipeline looks indistinguishable
        from a stalled bot. After THINKING_NUDGE_SECONDS we send one small
        message; if the pipeline finishes inside the window the task is
        cancelled and nothing is sent.
        """
        try:
            await asyncio.sleep(THINKING_NUDGE_SECONDS)
        except asyncio.CancelledError:
            return
        try:
            await self._send_text(recipient, "_…thinking_")
        except Exception as e:
            log.debug(f"Thinking-nudge send failed: {e}")

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


_MAX_QUOTE_CHARS = 500


def _format_quote_context(quote: dict, bot_number: str) -> str:
    """Build a `[Replying to ...]` prefix from a Signal quote payload.

    Returns an empty string when the quote has no usable text (e.g. a
    quote of a sticker or attachment-only message). The author check
    distinguishes the bot's own previous reply ("your earlier reply")
    from someone else's message ("an earlier message"), which gives the
    LLM cleaner attribution.
    """
    quoted_text = (quote.get("text") or "").strip()
    if not quoted_text:
        return ""

    if len(quoted_text) > _MAX_QUOTE_CHARS:
        # Trim from the END — the start of the quoted message is usually
        # the most identifying part (a heading, opening line, etc.).
        quoted_text = quoted_text[: _MAX_QUOTE_CHARS - 1].rstrip() + "…"

    author = (quote.get("author") or "").strip()
    is_self = bool(bot_number) and author == bot_number
    label = "your earlier reply" if is_self else "an earlier message"

    return f"[Replying to {label}]\n> {quoted_text.replace(chr(10), chr(10) + '> ')}"


def _is_voice_note(att: dict) -> bool:
    """Signal marks voice notes with voiceNote=True; some clients omit the
    flag but still send audio/* — accept either as a voice message."""
    if att.get("voiceNote"):
        return True
    ct = (att.get("contentType") or "").lower()
    return ct.startswith("audio/")


def _extract_reply_override(messages: list[str]) -> tuple[list[str], str | None]:
    """Detect inline `reply by voice` / `reply by text` markers.

    Returns `(cleaned_messages, override)` where override is "voice",
    "text", or None. Markers are stripped from the messages so the
    LLM doesn't see them — they're channel routing hints, not part
    of the user's actual content. If both markers appear (rare; user
    contradicting themselves), `voice` wins because the user
    specifically asking for voice is the more deliberate signal.
    """
    override: str | None = None
    cleaned: list[str] = []
    for m in messages:
        text = m
        if _REPLY_BY_VOICE_RE.search(text):
            override = "voice"
            text = _REPLY_BY_VOICE_RE.sub("", text).strip()
        if _REPLY_BY_TEXT_RE.search(text):
            if override != "voice":
                override = "text"
            text = _REPLY_BY_TEXT_RE.sub("", text).strip()
        # Collapse double-spaces left by the regex strip.
        text = " ".join(text.split())
        cleaned.append(text)
    return cleaned, override


def _strip_markdown_emphasis(text: str) -> str:
    """Strip *bold* / _italic_ / `code` markers so TTS doesn't read
    the punctuation literally. Headings, list bullets, and other
    markdown live structurally, not character-by-character — leaving
    them intact is fine for spoken output.
    """
    # Process in order so e.g. `**bold**` becomes `bold`, not `*bold*`.
    out = text
    out = _re.sub(r"\*\*(.+?)\*\*", r"\1", out)  # bold (markdown)
    out = _re.sub(r"\*(.+?)\*", r"\1", out)       # italic *...*
    out = _re.sub(r"_(.+?)_", r"\1", out)         # italic / Signal-style
    out = _re.sub(r"`(.+?)`", r"\1", out)         # inline code
    return out


def _is_image(att: dict) -> bool:
    """Match any image content-type. Anthropic's vision API accepts
    jpeg/png/gif/webp; we forward whatever Signal labels as image/*
    and let the API reject anything else with a clear error.
    """
    ct = (att.get("contentType") or "").lower()
    return ct.startswith("image/")


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
