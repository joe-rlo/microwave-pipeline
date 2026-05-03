"""OpenAI text-to-speech for voice replies on Signal.

Voice in via Whisper deserves voice out — symmetry, not a default.
This module wraps OpenAI's TTS endpoint and returns the audio bytes
for a channel to attach. Provider choice: OpenAI because (a) we
already have OPENAI_API_KEY for Whisper and embeddings, and (b)
Anthropic doesn't ship TTS.

The output format is `aac` so signal-cli-rest-api can attach it as a
voice note (Signal accepts m4a containers natively). If a future
backend prefers a different format, change `RESPONSE_FORMAT` and the
attachment content-type at the call site.
"""

from __future__ import annotations

import logging

import aiohttp

log = logging.getLogger(__name__)

TTS_ENDPOINT = "https://api.openai.com/v1/audio/speech"
DEFAULT_MODEL = "tts-1"
DEFAULT_VOICE = "alloy"
RESPONSE_FORMAT = "aac"
# Anthropic-style content type so the receiving channel can label the
# attachment. Signal-cli-rest-api accepts these data URIs verbatim.
RESPONSE_MIME = "audio/aac"
# Cap synthesis input. OpenAI rejects very long inputs; we trim
# defensively rather than letting the API call fail mid-reply.
MAX_INPUT_CHARS = 4000


class TTSError(RuntimeError):
    """Synthesis failed for a reason the caller should surface to the user
    (network, API error, missing key)."""


async def synthesize(
    text: str,
    *,
    api_key: str,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_MODEL,
) -> bytes:
    """Synthesize `text` to AAC audio bytes via OpenAI TTS.

    Raises TTSError on missing key, transport failure, or non-2xx
    response. Returning the failure as an exception (not silently
    falling back) lets the caller decide whether to retry, send text
    instead, or surface the problem to the user — that decision is
    channel-specific, not a TTS-layer concern.
    """
    if not api_key:
        raise TTSError("OPENAI_API_KEY not set (needed for TTS)")
    if not text or not text.strip():
        raise TTSError("Cannot synthesize empty text")

    payload = {
        "model": model,
        "voice": voice,
        "input": text[:MAX_INPUT_CHARS],
        "response_format": RESPONSE_FORMAT,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Per-call session — TTS isn't called frequently enough to justify
    # a long-lived client, and a per-call session avoids leaking state
    # across reply attempts.
    async with aiohttp.ClientSession() as session:
        async with session.post(
            TTS_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise TTSError(f"OpenAI TTS {resp.status}: {body[:200]}")
            return await resp.read()
