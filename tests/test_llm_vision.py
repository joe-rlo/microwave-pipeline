"""Vision-pass-through tests for LLMClient.

Verifies that when `images` is passed to `LLMClient.send`, the api_key
path builds a multimodal content block array (image blocks first, then
a single text block) and that the conversation history records the
same shape so subsequent turns inherit the visual context.

The Max-auth path is NOT exercised here — the Agent SDK doesn't have a
clean multimodal surface, so `send()` warns and falls through to text.
That's verified by inspecting the warning log path, not by testing the
SDK call shape.
"""

from __future__ import annotations

import pytest

from src.llm.client import LLMClient


class _FakeStreamCM:
    """Async context manager that the real Anthropic SDK returns from
    `client.messages.stream()`. Captures the kwargs and yields a stub
    that has a `.text_stream` async iterator and the trailing event
    methods the real SDK exposes (none of which we need)."""

    def __init__(self, captured_kwargs: dict, stream_text: list[str]):
        self.captured_kwargs = captured_kwargs
        self.stream_text = stream_text

    async def __aenter__(self):
        class _Stream:
            def __init__(self, parts):
                self.parts = parts

            @property
            def text_stream(self):
                async def _gen():
                    for p in self.parts:
                        yield p
                return _gen()

        return _Stream(self.stream_text)

    async def __aexit__(self, *exc):
        return False


class _FakeAnthropicClient:
    """Stub for `anthropic.AsyncAnthropic` — records the kwargs passed
    to `messages.stream(...)` so the test can inspect the content shape.
    """

    def __init__(self):
        self.captured_kwargs: dict | None = None

        class _Messages:
            def stream(_self, **kwargs):
                self.captured_kwargs = kwargs
                return _FakeStreamCM(kwargs, ["ok"])

        self.messages = _Messages()


def _make_client_with_fake() -> tuple[LLMClient, _FakeAnthropicClient]:
    """Build an LLMClient in api_key mode wired to a fake Anthropic
    client, bypassing the real connect() (which would hit the network)."""
    client = LLMClient(
        model="sonnet",
        auth_mode="api_key",
        api_key="test-key",
    )
    fake = _FakeAnthropicClient()
    client._client = fake
    client._stable_prompt = "system prompt"
    client._conversation = []
    return client, fake


class TestImagePassthroughApiKey:
    @pytest.mark.asyncio
    async def test_no_images_keeps_string_content(self):
        """Backward-compat: turn without images stays a plain string in
        conversation history. Don't touch what isn't broken."""
        client, fake = _make_client_with_fake()
        async for _ in client.send("hello", images=None):
            pass
        assert fake.captured_kwargs is not None
        msgs = fake.captured_kwargs["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_images_become_content_blocks(self):
        client, fake = _make_client_with_fake()
        img_bytes = b"\x89PNG\r\n\x1a\nfake png"
        async for _ in client.send(
            "What's in this photo?",
            images=[(img_bytes, "image/png")],
        ):
            pass
        msgs = fake.captured_kwargs["messages"]
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "image"
        assert content[0]["source"]["type"] == "base64"
        assert content[0]["source"]["media_type"] == "image/png"
        # base64-encoded data is non-empty and parseable
        import base64 as _b64
        decoded = _b64.b64decode(content[0]["source"]["data"])
        assert decoded == img_bytes
        # Text block follows the images so the model can refer to "this photo"
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "What's in this photo?"

    @pytest.mark.asyncio
    async def test_multiple_images(self):
        client, fake = _make_client_with_fake()
        async for _ in client.send(
            "compare these",
            images=[(b"a", "image/jpeg"), (b"b", "image/png")],
        ):
            pass
        content = fake.captured_kwargs["messages"][0]["content"]
        assert len(content) == 3  # two images + one text
        assert [b["type"] for b in content] == ["image", "image", "text"]
        # Image order preserved
        assert content[0]["source"]["media_type"] == "image/jpeg"
        assert content[1]["source"]["media_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_default_media_type_when_missing(self):
        """If a channel sends an image without a content type (defensive
        for malformed attachments), fall back to image/jpeg rather than
        crashing — the API will reject malformed images cleanly."""
        client, fake = _make_client_with_fake()
        async for _ in client.send("x", images=[(b"data", "")]):
            pass
        content = fake.captured_kwargs["messages"][0]["content"]
        assert content[0]["source"]["media_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_memory_context_with_images(self):
        """When both memory_context and images are present, the memory
        prefix lands in the *text* block, not as a separate content
        block. The text block remains the place for prose context."""
        client, fake = _make_client_with_fake()
        async for _ in client.send(
            "what's in this?",
            memory_context="prior fact",
            images=[(b"x", "image/jpeg")],
        ):
            pass
        content = fake.captured_kwargs["messages"][0]["content"]
        # One image block, one text block; text holds memory + message
        assert content[0]["type"] == "image"
        assert content[1]["type"] == "text"
        assert "prior fact" in content[1]["text"]
        assert "what's in this?" in content[1]["text"]

    @pytest.mark.asyncio
    async def test_conversation_history_preserves_image_shape(self):
        """A subsequent turn should see the prior image-bearing user
        content in history — not a stringified version. This is what
        keeps a follow-up question like "and the one in the
        background?" sensible to the model."""
        client, fake = _make_client_with_fake()
        async for _ in client.send("first", images=[(b"a", "image/jpeg")]):
            pass
        # Conversation now has [user(image+text), assistant("ok")]
        async for _ in client.send("second"):
            pass
        msgs = fake.captured_kwargs["messages"]
        # First message kept its multimodal shape
        assert isinstance(msgs[0]["content"], list)
        assert msgs[0]["content"][0]["type"] == "image"
        # Second turn is plain text (no images this time)
        assert msgs[2]["content"] == "second"
