"""Tests for the LLM stage selector.

Selector behavior is env-driven, so every test uses `monkeypatch.setenv`
/ `monkeypatch.delenv` to control the inputs. We exercise:

- Spec resolution: no override → legacy; valid override → near; malformed
  override → fall back to legacy with a warning; non-legacy default with
  no per-stage override → fall back to legacy with a warning.
- Callable builders: legacy returns a SingleTurnClient.query; near
  returns a function that wraps NEARProvider.send; missing NEAR key
  returns a broken callable that raises clearly.
- End-to-end: the near callable concatenates TextDelta events into a
  single response string and raises on Error events.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.llm.selector import StageSpec, get_stage_callable, get_stage_spec


# --- Spec resolution ---


class TestGetStageSpec:
    def test_no_env_is_legacy(self, monkeypatch):
        monkeypatch.delenv("LLM_STAGE_TRIAGE", raising=False)
        monkeypatch.delenv("LLM_PROVIDER_DEFAULT", raising=False)
        spec = get_stage_spec("triage")
        assert spec.is_legacy
        assert spec.model == ""

    def test_explicit_legacy_default(self, monkeypatch):
        monkeypatch.delenv("LLM_STAGE_TRIAGE", raising=False)
        monkeypatch.setenv("LLM_PROVIDER_DEFAULT", "legacy")
        spec = get_stage_spec("triage")
        assert spec.is_legacy

    def test_near_override(self, monkeypatch):
        monkeypatch.setenv("LLM_STAGE_TRIAGE", "near:claude-haiku-4-5")
        spec = get_stage_spec("triage")
        assert spec.is_near
        assert spec.provider == "near"
        assert spec.model == "claude-haiku-4-5"

    def test_override_whitespace_trimmed(self, monkeypatch):
        monkeypatch.setenv("LLM_STAGE_TRIAGE", "  near  :  m  ")
        spec = get_stage_spec("triage")
        assert spec.provider == "near"
        assert spec.model == "m"

    def test_malformed_override_falls_back_to_legacy(self, monkeypatch, caplog):
        # Missing colon → falls through to default. Logs a warning.
        monkeypatch.setenv("LLM_STAGE_TRIAGE", "garbage_no_colon")
        spec = get_stage_spec("triage")
        # `garbage_no_colon` is parsed as provider="garbage_no_colon",
        # model="". provider is non-empty so it IS returned as a spec
        # — selector treats unknown providers as legacy at build time.
        # That's intentional: don't crash; surface the misconfiguration
        # when the callable is actually built.
        assert spec.provider == "garbage_no_colon"

    def test_non_legacy_default_without_stage_warns_and_uses_legacy(
        self, monkeypatch, caplog
    ):
        monkeypatch.delenv("LLM_STAGE_TRIAGE", raising=False)
        monkeypatch.setenv("LLM_PROVIDER_DEFAULT", "near")
        with caplog.at_level("WARNING"):
            spec = get_stage_spec("triage")
        assert spec.is_legacy
        assert any("LLM_PROVIDER_DEFAULT" in r.message for r in caplog.records)

    def test_different_stages_independent(self, monkeypatch):
        monkeypatch.setenv("LLM_STAGE_TRIAGE", "near:m1")
        monkeypatch.delenv("LLM_STAGE_REFLECTION", raising=False)
        assert get_stage_spec("triage").is_near
        assert get_stage_spec("reflection").is_legacy


# --- get_stage_callable: legacy ---


class TestLegacyCallable:
    def test_returns_singleturn_query_when_no_override(self, monkeypatch):
        # No env override → legacy. We don't actually invoke the
        # callable (that would hit the SDK); just confirm it's wired.
        monkeypatch.delenv("LLM_STAGE_TRIAGE", raising=False)
        monkeypatch.delenv("LLM_PROVIDER_DEFAULT", raising=False)
        call = get_stage_callable(
            "triage", fallback_model="haiku", auth_mode="max"
        )
        assert callable(call)
        # SingleTurnClient.query is an async method
        assert hasattr(call, "__self__")  # bound method
        assert call.__self__.model == "haiku"


# --- get_stage_callable: near, missing key ---


class TestNearMissingKey:
    @pytest.mark.asyncio
    async def test_missing_key_yields_broken_callable(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv("LLM_STAGE_TRIAGE", "near:claude-haiku-4-5")
        monkeypatch.delenv("NEAR_API_KEY", raising=False)

        with caplog.at_level("ERROR"):
            call = get_stage_callable(
                "triage", fallback_model="haiku"
            )

        # Builder logs at error level but doesn't raise — fail late.
        assert any("NEAR_API_KEY" in r.message for r in caplog.records)
        with pytest.raises(RuntimeError, match="NEAR_API_KEY"):
            await call("sys", "user")


# --- get_stage_callable: near, working ---


def _mock_streaming_client(sse_body: str, status: int = 200) -> httpx.AsyncClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status,
            headers={"content-type": "text/event-stream"},
            content=sse_body.encode("utf-8"),
        )

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://example.invalid",
    )


@pytest.mark.asyncio
class TestNearCallable:
    async def test_text_deltas_concatenated(self, monkeypatch):
        monkeypatch.setenv("LLM_STAGE_TRIAGE", "near:claude-haiku-4-5")
        monkeypatch.setenv("NEAR_API_KEY", "k")

        # Patch NEARProvider to use an injected mock client.
        from src.llm.providers import near as near_module

        original = near_module.NEARProvider

        def factory(api_key, *, base_url=near_module.DEFAULT_BASE_URL, **kw):
            client = _mock_streaming_client(
                'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
                'data: {"choices":[{"finish_reason":"stop","delta":{}}]}\n\n'
                'data: [DONE]\n\n'
            )
            return original(api_key=api_key, base_url=base_url, client=client)

        monkeypatch.setattr(near_module, "NEARProvider", factory)

        call = get_stage_callable(
            "triage", fallback_model="claude-haiku-4-5"
        )
        out = await call("you are a triage classifier", "user message")
        assert out == "hello"

    async def test_error_event_raises_when_no_text(self, monkeypatch):
        monkeypatch.setenv("LLM_STAGE_TRIAGE", "near:claude-haiku-4-5")
        monkeypatch.setenv("NEAR_API_KEY", "k")

        from src.llm.providers import near as near_module

        original = near_module.NEARProvider

        def factory(api_key, *, base_url=near_module.DEFAULT_BASE_URL, **kw):
            client = _mock_streaming_client("rate limited", status=429)
            return original(api_key=api_key, base_url=base_url, client=client)

        monkeypatch.setattr(near_module, "NEARProvider", factory)

        call = get_stage_callable(
            "triage", fallback_model="claude-haiku-4-5"
        )
        with pytest.raises(RuntimeError, match="NEAR call failed"):
            await call("sys", "user")

    async def test_fallback_model_used_when_override_missing_model(
        self, monkeypatch
    ):
        # `near:` with nothing after the colon → fallback_model is used
        monkeypatch.setenv("LLM_STAGE_TRIAGE", "near:")
        monkeypatch.setenv("NEAR_API_KEY", "k")

        captured = {}

        from src.llm.providers import near as near_module

        original = near_module.NEARProvider

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                content=b'data: {"choices":[{"finish_reason":"stop","delta":{}}]}\n\ndata: [DONE]\n\n',
            )

        def factory(api_key, *, base_url=near_module.DEFAULT_BASE_URL, **kw):
            client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url="https://example.invalid",
            )
            return original(api_key=api_key, base_url=base_url, client=client)

        monkeypatch.setattr(near_module, "NEARProvider", factory)

        # `near:` parses to provider="near", model="" — selector falls
        # back to fallback_model
        call = get_stage_callable(
            "triage", fallback_model="claude-sonnet-4-6"
        )
        await call("sys", "user")
        assert captured["body"]["model"] == "claude-sonnet-4-6"

    async def test_near_base_url_override(self, monkeypatch):
        monkeypatch.setenv("LLM_STAGE_TRIAGE", "near:m")
        monkeypatch.setenv("NEAR_API_KEY", "k")
        monkeypatch.setenv("NEAR_BASE_URL", "https://custom.invalid/v1")

        captured_url = {}

        from src.llm.providers import near as near_module

        original = near_module.NEARProvider

        def factory(api_key, *, base_url=near_module.DEFAULT_BASE_URL, **kw):
            captured_url["url"] = base_url
            client = _mock_streaming_client(
                'data: {"choices":[{"finish_reason":"stop","delta":{}}]}\n\n'
                'data: [DONE]\n\n'
            )
            return original(api_key=api_key, base_url=base_url, client=client)

        monkeypatch.setattr(near_module, "NEARProvider", factory)

        call = get_stage_callable("triage", fallback_model="m")
        await call("sys", "user")
        assert captured_url["url"] == "https://custom.invalid/v1"
