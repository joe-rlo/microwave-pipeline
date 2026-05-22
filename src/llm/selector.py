"""Provider selection — picks an LLM provider per pipeline stage.

Reads environment variables directly rather than threading a Config
object through every call site. The env is already the canonical
source for these settings; mirroring them in Config and then reading
back through Config would just add noise. Tests use
`monkeypatch.setenv` to drive selection.

Env vars consumed (all optional):

  LLM_PROVIDER_DEFAULT      "legacy" (default) | "near"
    The default provider for any stage with no explicit override.
    "legacy" routes through the existing src/llm/client.py path
    (Agent SDK or direct Anthropic) — i.e., no behavior change.
    Any other value requires per-stage models to be set.

  LLM_STAGE_TRIAGE          ""       (default — use default) | "near:<model>"
  LLM_STAGE_REFLECTION      ""       (default — use default) | "near:<model>"
  LLM_STAGE_<OTHER>         (future stages added the same way)
    Per-stage overrides. Format is "<provider>:<model>", e.g.
    "near:claude-haiku-4-5". An empty string means "use the default
    provider with the caller's fallback_model".

  NEAR_API_KEY              required when any stage resolves to "near"
  NEAR_BASE_URL             default https://cloud-api.near.ai/v1

The selector returns a `(system_prompt, user_message) -> str` callable
matching `SingleTurnClient.query`'s shape. Callers — triage, reflection,
future consolidation/breadcrumb stages — pass it straight to
`query_json_with_retry`. They never see provider details.

What this module does NOT do:
- No caching of provider instances across calls. NEAR opens an
  httpx.AsyncClient per `send()` already, and triage / reflection
  fire once per turn. Caching adds lifecycle complexity (close
  on shutdown? on reconfigure?) we don't need yet.
- No "legacy" → "near" fallback when NEAR_API_KEY is missing.
  If you ask for NEAR without a key, you get a warning and the
  legacy callable. Failing closed lets the bot keep working;
  failing loudly would crash every triage call.
- No retry policy. Errors raised from inside the callable are
  caught by `query_json_with_retry`, which has its own retry shape.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Awaitable, Callable

log = logging.getLogger(__name__)


# Callable shape every stage expects: async (system_prompt, user_message) -> response_text.
StageCallable = Callable[[str, str], Awaitable[str]]


@dataclass(frozen=True)
class StageSpec:
    """Resolved choice for one stage.

    `provider` is the resolved provider name. `model` is the resolved
    model identifier (provider-specific). If the override was empty,
    `model` is "" — callers supply their fallback at callable-build time.
    """

    provider: str
    model: str

    @property
    def is_near(self) -> bool:
        return self.provider == "near"

    @property
    def is_legacy(self) -> bool:
        return self.provider == "legacy"


def get_stage_spec(stage: str) -> StageSpec:
    """Resolve a stage name to a StageSpec from env.

    `stage` is a short identifier — "triage", "reflection", etc. The env
    var name is uppercased: `LLM_STAGE_TRIAGE`.
    """
    override = os.environ.get(f"LLM_STAGE_{stage.upper()}", "").strip()
    if override:
        provider, _, model = override.partition(":")
        provider = provider.strip()
        model = model.strip()
        if provider:
            return StageSpec(provider=provider, model=model)
        # Override was malformed (no colon, or just ":model"); ignore.
        log.warning(
            "LLM_STAGE_%s=%r malformed; expected '<provider>:<model>'",
            stage.upper(), override,
        )

    default = os.environ.get("LLM_PROVIDER_DEFAULT", "legacy").strip().lower()
    if default and default != "legacy":
        # A non-legacy default with no per-stage override is unsupported —
        # we need an explicit model id for the non-legacy provider. Warn
        # and fall through to legacy so the bot doesn't crash.
        log.warning(
            "LLM_PROVIDER_DEFAULT=%r but LLM_STAGE_%s is empty; using legacy",
            default, stage.upper(),
        )
    return StageSpec(provider="legacy", model="")


def get_stage_callable(
    stage: str,
    *,
    fallback_model: str,
    auth_mode: str = "max",
    api_key: str = "",
    cli_path: str = "",
    workspace_dir: str = "",
) -> StageCallable:
    """Return the `(system, user) -> str` callable for `stage`.

    The fallback_* args mirror SingleTurnClient's constructor. They're
    used when the resolved spec is `legacy` (no override, or invalid
    override). When the spec is `near`, those args are ignored —
    NEAR uses NEAR_API_KEY from env.
    """
    spec = get_stage_spec(stage)

    if spec.is_near:
        return _build_near_callable(spec.model or fallback_model, stage)

    # Legacy path — unchanged from pre-selector behavior.
    return _build_legacy_callable(
        model=spec.model or fallback_model,
        auth_mode=auth_mode,
        api_key=api_key,
        cli_path=cli_path,
        workspace_dir=workspace_dir,
    )


# --- Builders ---------------------------------------------------------------


def _build_legacy_callable(
    *,
    model: str,
    auth_mode: str,
    api_key: str,
    cli_path: str,
    workspace_dir: str,
) -> StageCallable:
    # Imported lazily so the selector module can be imported in tests
    # without dragging in the Agent SDK transitively.
    from src.llm.client import SingleTurnClient

    client = SingleTurnClient(
        model=model,
        auth_mode=auth_mode,
        api_key=api_key,
        cli_path=cli_path,
        workspace_dir=workspace_dir,
    )
    return client.query


def _build_near_callable(model: str, stage: str) -> StageCallable:
    api_key = os.environ.get("NEAR_API_KEY", "").strip()
    if not api_key:
        # Misconfiguration: stage asked for NEAR but no key. The right
        # behavior is "don't break the bot" — log loudly, return a
        # broken callable that fails clearly the first time the stage
        # actually runs. This is better than silently routing to
        # legacy because the user explicitly opted into NEAR and a
        # quiet fallback could mask real configuration drift.
        log.error(
            "Stage %r requested NEAR but NEAR_API_KEY is not set", stage
        )

        async def broken(system: str, user: str) -> str:
            raise RuntimeError(
                f"NEAR provider requested for stage {stage!r} but NEAR_API_KEY is unset"
            )

        return broken

    base_url = (
        os.environ.get("NEAR_BASE_URL", "").strip()
        or "https://cloud-api.near.ai/v1"
    )

    # Lazy import so tests can monkey-patch without triggering httpx
    # transitively at module-load time.
    from src.llm.provider import (
        ProviderMessage,
        ProviderRequest,
        TextDelta,
        Error,
    )
    from src.llm.providers.near import NEARProvider

    provider = NEARProvider(api_key=api_key, base_url=base_url)

    async def call(system_prompt: str, user_message: str) -> str:
        req = ProviderRequest(
            model=model,
            system=system_prompt,
            messages=[ProviderMessage(role="user", content=user_message)],
            max_tokens=4096,
            stream=True,
        )
        buf: list[str] = []
        error: Error | None = None
        async for evt in provider.send(req):
            if isinstance(evt, TextDelta):
                buf.append(evt.text)
            elif isinstance(evt, Error):
                error = evt
            # Other events (Usage, Done, tool_use_*) are not relevant
            # for JSON stages — they're not in the contract callers
            # expect from SingleTurnClient.query.

        if error is not None and not buf:
            # No text came through and we have an error — surface it as
            # an exception so query_json_with_retry's outer try/except
            # catches and counts it as a client error.
            raise RuntimeError(f"NEAR call failed: {error.message}")
        return "".join(buf)

    return call
