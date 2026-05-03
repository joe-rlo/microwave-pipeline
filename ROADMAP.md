# Roadmap

A building spec for the work that's *not* in the next commit. Two kinds of entries live here:

1. **Concrete items with steps.** Pieces of work that have been thought through far enough to execute, written so a contributor (or future Claude) can pick one up and ship it without re-deciding the design.
2. **Trigger-gated items.** Work that's been designed but should *not* be built yet — it waits for a specific real-world signal so we don't over-engineer ahead of the pain.

Each item carries the same fields:

- **What** — short description.
- **Why** — what design value or coherence property it protects. The *Why* is load-bearing; if you can't connect a change to one, don't build it.
- **Build trigger** — *when* to start. "Now" / "after item N" / "on first incident of X."
- **Files to touch** — concrete paths, not generic phrases.
- **Steps** — ordered, specific actions. Reading like a checklist, not a strategy.
- **Acceptance** — how to verify the work is actually done.
- **Not in scope (yet)** — explicit guards against substrate inflation. The thing it would be tempting to also do, with a one-line reason for resisting.

---

## Build order

Recommended sequence. Each item is sized to land in one focused session.

1. **Async embedding calls** — pure infrastructure win. No deps. Unblocks responsive-event-loop assumptions every other feature relies on.
2. **Triage / reflection JSON robustness** — cheap. Provides the parse-with-retry helper that item #4 reuses.
3. **Importance-aware compaction** — keeps the agent's memory of *substantive* conversations from being rolled up at the same rate as small talk. Independent.
4. **Memory contradiction surfacing** — uses the JSON helper from #2. Independent of #3.
5. **Project search tagging** — preventive. Best done before the second active project ships.
6. **Vision (image pass-through)** — self-contained capability add. Channels + LLM client + orchestrator.
7. **TTS reply on Signal** — symmetry with voice-in. Lowest priority of the "do" items.
8. **Tool use expansion** — operating principle, not a single deliverable. Continuous.
9. **Turn semantics — journal + reflection-as-commit** — *defer.* Waits for first concrete drift incident.

A note on the deliberate omissions: single-tenant LLM session, no concurrent users, OpenAI-only embeddings, and machine-local Agent SDK auth are *not* on this list. They're load-bearing properties of the personal-agent design, not bugs. See the README's "Limitations" section, which should reframe them as "Single-tenant by design" rather than apologetic notes.

---

## 1. Async embedding calls

**Status.** Not started.

**What.** Replace synchronous `openai.embeddings.create()` calls in `EmbeddingClient` with the async OpenAI client. Today every vector query blocks the event loop — under Signal's debounce + scheduler + active conversation, the loop stutters.

**Why.** Signal's chunked-input feature depends on a responsive event loop (typing indicators refreshing on time, debounce timers firing on time, addendum cancellations propagating quickly). A blocking embedding call inside any of those windows degrades exactly the experience that feature was built for. This is the closest thing to a free fix on the list.

**Build trigger.** Now. Unblocks performance assumptions for everything else.

**Files to touch.**
- `src/memory/embeddings.py`
- Any caller of `EmbeddingClient.embed()` or `embed_batch()` (find with `grep -rn "embedder\.\|embed_batch\|\.embed(" src/`)

**Steps.**
1. Read `src/memory/embeddings.py` to confirm the current sync API: `embed(text)` and `embed_batch(texts)`, lazy client initialization in `_get_client()`.
2. Switch the import in `_get_client()` from `from openai import OpenAI` to `from openai import AsyncOpenAI`. Same constructor signature.
3. Make `embed()` and `embed_batch()` `async def`. Body becomes `await client.embeddings.create(...)`.
4. Grep callers and convert each to `await embedder.embed(...)` / `await embedder.embed_batch(...)`. Likely callers: `src/memory/index.py` (during indexing), `src/memory/search.py` (`_vector_search`), and possibly the importers.
5. Some callers may sit inside sync code paths (e.g., scheduler pre-fetch, CLI commands). For those, either (a) make the caller async too, or (b) wrap with `asyncio.run()` at the boundary. Prefer (a) when the boundary is already async-adjacent.
6. Run `pytest`. Fix any test that imported `EmbeddingClient` and exercised it synchronously.
7. Manual smoke: open REPL, send a message that triggers vector search, observe other timers continuing to fire during the call.

**Acceptance.**
- `pytest` passes.
- No `asyncio.to_thread()` wrapper around any embedding call anywhere in the codebase.
- A `time.monotonic()` measurement around an `embed()` call shows the event loop yielding to other tasks during the await (verifiable with a simple test that schedules a concurrent timer and confirms it fires within tolerance during a real or mocked embedding call).

**Not in scope (yet).**
- *Result caching at this layer.* Tempting because vector queries are repetitive, but caching belongs above `EmbeddingClient` (in `MemorySearcher` or higher), and only with a measured hit rate to justify the cache invalidation complexity.
- *Custom retry / rate-limit handling.* The `AsyncOpenAI` client handles standard retry semantics; trust it.
- *Local embedding fallback.* Different roadmap item (Tier 4 in the design plan, currently kept as a documented constraint, not a planned change).

---

## 2. Triage / reflection JSON robustness

**Status.** Not started.

**What.** Triage and reflection both call Haiku and parse the response as JSON. Failures fall back to defaults — but those defaults are sometimes *wrong* defaults (e.g., a recall question miscategorized as `social` skips memory search entirely). Add: schema validation, one-shot retry on parse failure, structured logging of every JSON parse failure for later analysis.

**Why.** Triage gates memory search. When triage silently fails and the fallback misclassifies, the agent looks stupid for reasons unrelated to model capability. Invisible-to-user fragility is the worst kind because the user can't even file a useful bug — "it just felt off." The retry is cheap; the absence of it is corrosive.

**Build trigger.** Now. Cheap; unlocks item #4 (which reuses the helper).

**Files to touch.**
- `src/pipeline/json_utils.py` — extend with the retry helper.
- `src/pipeline/triage.py` — adopt it.
- `src/pipeline/reflection.py` — adopt it.
- `tests/test_json_utils.py` — extend.
- `tests/test_triage.py` and `tests/test_reflection.py` — extend.

**Steps.**
1. Audit `src/pipeline/json_utils.py` to see what JSON parsing helpers already exist and how failures are currently handled.
2. Add a helper of approximately this shape:
   ```python
   async def parse_json_with_retry(
       client_call: Callable[[str], Awaitable[str]],
       initial_prompt: str,
       schema_hint: str,
       max_retries: int = 1,
   ) -> dict | None:
       """Call the LLM, parse JSON, retry once on failure with the bad
       response shown back to the model. Returns None after exhausting
       retries — caller is responsible for fallback behavior."""
   ```
   The retry prompt: *"Your last response was not valid JSON. Original response (between triple backticks): \`\`\`{response}\`\`\`. Return ONLY valid JSON matching this schema: {schema_hint}. No explanation, no markdown."*
3. Convert `triage.py`'s parse path to use the helper. Pass the existing schema description as `schema_hint`.
4. Convert `reflection.py`'s parse path the same way.
5. Add structured logging: every parse attempt emits one log line at `INFO` (or a dedicated structured logger) with `{model, prompt_hash, response_len, status: "parsed|retried|failed", retry_count}`. This becomes the dataset for tuning prompts later.
6. Tests:
   - Unit test on the helper: mock client returns malformed JSON once, valid the second time. Helper returns the parsed dict, log shows `status=retried`.
   - Unit test: malformed twice. Helper returns `None`, log shows `status=failed`.
   - Triage / reflection tests: parameterize one existing happy-path test to also exercise the malformed-then-valid path.

**Acceptance.**
- The malformed-twice case is logged in a way that's auditable (greppable for `status=failed`).
- An end-to-end smoke (REPL turn) shows no behavioral change on the happy path.
- A deliberate sabotage (mock Haiku to return `"not json"`) triggers the retry, recovers if the second call succeeds, and falls back to defaults if not — all without crashing the pipeline.

**Not in scope (yet).**
- *Switching triage / reflection to a larger model.* Haiku is the right speed/cost tradeoff for these stages; the JSON noise is a known cost mitigated by retry, not avoided by capability inflation.
- *Multiple retries.* If two attempts fail, the prompt is the problem, not transient noise. Adding retries past two delays the diagnosis.
- *Regex-based partial JSON rescue.* A maintenance trap. Either valid JSON or fall back; no in-between.

---

## 3. Importance-aware compaction

**Status.** Not started.

**What.** Today's compaction is blunt: roll all "old" turns into one ≤8000-char summary, keep the most recent 6 verbatim. Change: turns whose stored metadata indicates importance — `triage_complexity == "complex"` OR `reflection_confidence > THRESHOLD` (start at `0.7`) — are exempt from rollup. They stay verbatim past the compaction boundary alongside the summary.

**Why.** What makes the agent's memory feel real over time is preserving the *substantive* conversations — decisions, plans, deep discussions — not the small talk. Today's compaction inverts that: complex multi-turn reasoning gets compressed at the same rate as "what's the weather." For a system whose value-prop is memory, this is where trust erodes first.

**Build trigger.** Now (or after item #1; no hard dep). The required metadata fields already exist on turns — verified at `src/pipeline/orchestrator.py:407-410`.

**Files to touch.**
- `src/session/engine.py` — `get_turns_for_compaction()` and any helpers it uses.
- `src/pipeline/orchestrator.py` — `_compact()`, the summary prompt.
- `tests/test_pipeline_integration.py` — `test_compaction_detection` and a new test for importance preservation.

**Steps.**
1. Add a constant near `get_turns_for_compaction`:
   ```python
   COMPACTION_IMPORTANCE_THRESHOLD = 0.7  # reflection_confidence above this exempts a turn
   ```
2. Modify `get_turns_for_compaction(session_id, keep_recent=6)` to return three lists, not two:
   - `to_summarize`: old turns that are *not* important (the rollup set).
   - `to_keep_verbatim`: old turns that ARE important (`triage_complexity=="complex"` OR `reflection_confidence > THRESHOLD`).
   - `recent`: the last `keep_recent` turns regardless of importance.

   Existing callers expect a 2-tuple; either update the call site (preferred) or have the function return `(to_summarize, recent + to_keep_verbatim)` to preserve shape. Update the call site is cleaner.
3. Update `_compact()` in orchestrator:
   - Use only `to_summarize` for the Haiku summarization call.
   - Replace those turns with the summary as today.
   - `to_keep_verbatim` turns stay in the session unchanged; they bypass replacement.
   - In the summary prompt, add: *"Some important turns from this period are being preserved verbatim outside this summary; you don't need to summarize them. Focus on conversational context that needs compression."*
4. Add a `/debug compaction` view (or extend the existing `/debug` output) showing: count rolled up, count kept verbatim by importance, total old turns scanned. Gives the user visibility into whether the threshold is right.
5. Tests:
   - Extend `test_compaction_detection`: 10 simple turns + 5 complex turns, all aged out. Verify `to_summarize` has 10, `to_keep_verbatim` has 5.
   - Edge: zero complex turns (degrades to today's behavior).
   - Edge: all complex turns (`to_summarize` is empty; compaction is a no-op for that block).

**Acceptance.**
- After a long-running conversation that included substantive turns, manual `/debug compaction` after compaction shows the substantive turns kept verbatim.
- LLM session post-compaction can answer "what did we decide about X" for any X discussed in a `complex` turn (i.e., the model still has the original text, not a paraphrase).
- Existing compaction tests still pass.

**Not in scope (yet).**
- *Hierarchical roll-ups (week → month → year summaries).* Real next step but its own project; importance-aware preservation is the cheap improvement that buys the most trust per line of code.
- *Auto-tuning the threshold.* `0.7` is opinionated. Surface in `/debug` so the user can see what's getting kept; tune from logs after several compactions.
- *User-marked importance ("remember this").* Different feature. Could be added later as another exempting criterion.

---

## 4. Memory contradiction surfacing

**Status.** Not started.

**What.** A `python3 src/main.py memory health` CLI command that scans `MEMORY.md` for likely contradictions — same entity, conflicting attributes ("Dog's name is Biscuit" / "Dog's name is Max"). Surfaces them as a list for the user to resolve manually. Does **not** auto-merge or auto-delete. Human-readable markdown stays the source of truth; the command is a curation aid, not an editor.

**Why.** The append-only memory model is a real failure mode for long-running personal agents. Today, contradictions silently coexist; the agent gets confused, the user can't tell why. *Auto*-resolving would be the wrong fix because the user owns their own memory — the agent's job is to flag, not rewrite. This matches the sovereignty design value: the user's relationship with their memory file should never be black-boxed.

**Build trigger.** After item #2 (uses the JSON-with-retry helper).

**Files to touch.**
- New: `src/memory/health.py`
- `src/main.py` — register the `memory health` subcommand.
- New: `tests/test_memory_health.py`.

**Steps.**
1. New module `src/memory/health.py` with:
   ```python
   @dataclass
   class Contradiction:
       a: str          # quoted line from MEMORY.md
       b: str          # quoted line from MEMORY.md
       summary: str    # one-line description of the conflict

   async def detect_contradictions(memory_text: str, *, llm_config) -> list[Contradiction]:
       ...
   ```
2. The detection prompt (Haiku is fine):
   *"Here is a personal memory document. Identify entries that contradict each other — same entity (person, place, fact) with conflicting attributes. Each contradiction is two specific lines from the document and a one-sentence summary of the conflict. Return JSON: `[{a: "line", b: "line", summary: "string"}]`. If there are no contradictions, return `[]`. Do not flag entries that merely *update* an entity (e.g., a status change) — only entries that disagree about a stable fact."*
3. Use the JSON-with-retry helper from item #2.
4. CLI in `src/main.py`: `memory health` subcommand. Loads MEMORY.md, runs detector, prints contradictions formatted like:
   ```
   ⚠ 2 contradictions in MEMORY.md

   1. Dog's name (added 2026-04-12 vs 2026-04-29)
      A: "His dog's name is Biscuit"
      B: "His dog's name is Max"

   2. ...

   To resolve: edit workspace/MEMORY.md directly.
   ```
5. (Optional, later) Scheduler hook: a daily job that runs the detector and writes results to `workspace/memory/_health-{date}.md`. Gitignored. Surface count next time the user opens REPL/Signal.
6. Tests with fixed memory fixtures:
   - Two clearly conflicting entries → 1 contradiction returned.
   - No conflicts → empty list.
   - "Update" rather than "contradiction" (e.g., "moved from NYC" → "now lives in Berlin") → not flagged.

**Acceptance.**
- CLI runs to completion on a real `MEMORY.md` and produces sensible output.
- No automatic modification of `MEMORY.md` under any code path.
- Re-running after the user edits MEMORY.md shows fewer/zero contradictions.

**Not in scope (yet).**
- *Auto-merge / auto-delete.* Sovereignty over memory is the design value. Even "obvious" contradictions might be context-dependent (different time periods, different referents). The user resolves.
- *Cross-file contradiction detection (daily notes, project bibles).* Higher false-positive rate, much larger scan surface. Start with `MEMORY.md`, which is small and curated.
- *A structured key-value backend.* The human-readable markdown is the design. A YAML/JSON sidecar would solve the contradiction problem but cost the property that makes the file editable by hand.

---

## 5. Project search tagging

**Status.** Not started.

**What.** The `fragments.source` column already exists (verified at `src/memory/index.py:65`) — populated as the file path. Use it: at search time, when a project is active, weight that project's fragments higher and downweight other projects'. Cross-project bleed (a "Sarah" in your novel polluting a Substack draft about a different Sarah) goes away.

**Why.** Projects are first-class today (`workspace/projects/`); the indexer already labels fragments by source. The gap is just on the retrieval side. Closing it is consistency, not a new abstraction.

**Build trigger.** Before the second active project ships, OR on first cross-pollination incident — whichever comes first. Cheap to ship preemptively if you're already in the search code.

**Files to touch.**
- `src/memory/search.py` — `MemorySearcher.search()` and the scoring path.
- `src/pipeline/orchestrator.py` — pass active project name to searcher.
- `src/pipeline/search.py` — the orchestrator-side search wrapper.
- `tests/test_search.py` — new test fixtures for project scoping.

**Steps.**
1. Audit how `source` is populated for project drafts. Today it's a file path; project drafts live under `workspace/projects/<name>/drafts/`. Confirm by `grep -rn "index_file\|index_text" src/`.
2. Add a small helper to classify a source string:
   ```python
   def _project_for_source(source: str) -> str | None:
       """Return the project name if `source` is inside a project's drafts/, else None."""
       # match e.g. ".../workspace/projects/the-heist/drafts/chapter-01.md" → "the-heist"
   ```
3. Extend `MemorySearcher.search()` to accept an optional `active_project: str | None = None`. After the existing scoring + temporal-decay loop, before MMR selection, apply a multiplicative boost:
   - If `active_project` is set and `_project_for_source(frag.source) == active_project`: `score *= 1.5`
   - If `active_project` is set and `_project_for_source(frag.source)` is some *other* project: `score *= 0.5`
   - Otherwise: no change.
4. Update `src/pipeline/search.py` to thread the active project from orchestrator into the searcher.
5. In `Orchestrator`, pass `self._active_project.name if self._active_project else None` to the search call.
6. Tests:
   - Index 3 fragments: one in project A, one in project B, one global. With `active_project="A"`, search returns A's fragment first.
   - With `active_project=None`, scoring matches today's behavior (regression-protect).
   - Boost factors are constants near the top of `search.py`; assert they're reachable from the test for tuning.

**Acceptance.**
- New test cases pass.
- No change in retrieval behavior when no project is active.
- Manual: activate a project, ask a question that hits both that project and another, verify the active project's fragments rank higher in the assembled context (visible via `/debug`).

**Not in scope (yet).**
- *User-configurable boost weights.* `1.5` / `0.5` is opinionated; changing them is a one-character edit if needed.
- *Hard scoping ("only search this project").* Downweight, don't exclude. A novel question can still benefit from a tangential Substack reference. Let the model decide what's relevant after retrieval.
- *Per-project embedding spaces.* Massive complexity for marginal gain. Same fragment table is the right shape.

---

## 6. Vision — image pass-through

**Status.** Not started.

**What.** Telegram and Signal both deliver image attachments today; the orchestrator drops them. Wire them through to the LLM as multimodal content blocks so the model can actually see what was sent.

**Why.** Doesn't change the architecture — adds a content type to an existing pipeline. Single-user means no security questions about image processing. User-visible feature gap that's small to close once you decide to.

**Build trigger.** First concrete user need ("I sent a photo and Microwave couldn't see it"). Don't ship it speculatively — the work isn't free, and shipping ahead of demand means tuning a feature nobody's used.

**Files to touch.**
- `src/channels/telegram.py` — image attachment handling.
- `src/channels/signal.py` — image attachment handling (model after the voice-note path).
- `src/llm/client.py` — `send()` accepts multimodal content.
- `src/pipeline/orchestrator.py` — pass image content through to LLM.

**Steps.**
1. Audit each channel's current image handling.
   - Telegram: `_handle_photo` may already exist. If so, see what it does with the image (likely discards or acks).
   - Signal: non-voice attachments — currently ignored. Adapt the voice download path.
2. Decide the orchestrator's input shape. Options:
   - (a) Add `images: list[tuple[bytes, str]] | None = None` kwarg to `Orchestrator.process()`.
   - (b) Change the `message` parameter to accept either `str` or a structured content list.
   Recommend (a): backward compatible, smaller blast radius, doesn't touch every caller.
3. In each channel: when a message has image attachments, download the bytes (with MIME) and pass to `orchestrator.process(text=..., images=[(bytes, ct), ...])`.
4. In `LLMClient.send`, when `images` is non-empty, build the Anthropic multimodal content block array:
   ```python
   content = []
   for img_bytes, ct in images:
       content.append({
           "type": "image",
           "source": {"type": "base64", "media_type": ct, "data": base64.b64encode(img_bytes).decode()},
       })
   content.append({"type": "text", "text": text})
   ```
   Send `content` as the message instead of a plain string.
5. Triage and reflection stay text-only — they only see the text caption, not the image. Add a comment in their docstrings noting this so a future contributor doesn't accidentally try to pass images through them (Haiku-tier vision is unnecessary here and just costs latency).
6. Tests:
   - Mock `LLMClient.send` with images, verify the multimodal content block array is constructed correctly.
   - End-to-end mock: orchestrator gets `images=[fixture]`, verify the image reaches `llm.send`'s call args.

**Acceptance.**
- User sends a photo on Telegram with caption "what's in this photo?" → bot describes the actual image.
- Image-less messages: zero behavioral change.
- Triage on a captioned image gets only the caption (verifiable via `/debug`).

**Not in scope (yet).**
- *OCR / pre-extraction.* Let the model see the image; that's what multimodal models are for.
- *Image re-encoding / compression.* The Anthropic API has size limits; users hitting them get an error. Fine.
- *HTTP / REPL channels.* Webhook image upload is a different shape; stdin doesn't make sense for vision. Start with the messaging channels.

---

## 7. TTS reply on Signal

**Status.** Not started.

**What.** When the user's input was a voice note (and only then, by default), generate the bot's reply as audio via OpenAI TTS and attach it as a Signal voice note. Optional explicit override via inline marker (*"reply by voice"* or *"reply by text"*).

**Why.** Voice-in deserves voice-out. Symmetry, not a forced default. For users who prefer voice (driving, walking, accessibility), the half-implemented voice path today is a friction.

**Build trigger.** Defer until the user starts using voice-in routinely AND the asymmetry feels wrong. Optional from day one is the right framing.

**Files to touch.**
- New: `src/channels/tts.py`
- `src/channels/signal.py` — reply routing.
- `.env.example` and `src/config.py` — optional `TTS_VOICE` config.

**Steps.**
1. Decide the provider. OpenAI TTS is the obvious choice — already have `OPENAI_API_KEY` and the latency / quality is fine for messaging. (Anthropic doesn't ship TTS.)
2. New `tts.py`:
   ```python
   async def synthesize(text: str, *, voice: str = "alloy", api_key: str) -> bytes:
       """Returns AAC-encoded audio bytes ready for Signal attachment."""
   ```
   Use the OpenAI TTS endpoint with `response_format="aac"` (or `mp3` if AAC is unsupported; signal-cli accepts both).
3. In `signal.py`'s `_process_and_respond`, after the text response is composed:
   - Determine if voice reply is wanted: voice-only input AND no explicit `"reply by text"` marker, OR explicit `"reply by voice"` marker.
   - If yes: call `synthesize(text)`, send as voice attachment via the existing `_send_attachment` path (with content-type `audio/aac`). Also send a short text version (parallel to the `_heard:_` echo on the way in) so the user has a transcript.
   - If no: today's text-only behavior.
4. Strip the voice/text override markers from the reply before TTS and before sending to keep them out of the user-visible output.
5. Tests:
   - Mock `synthesize`, verify it's called when input was voice.
   - Mock `synthesize`, verify it's NOT called when input was text-only.
   - Override marker tests: voice input + "reply by text" → no TTS call; text input + "reply by voice" → TTS call.

**Acceptance.**
- Voice input gets a voice reply + text transcript.
- Text input gets a text-only reply.
- Override markers work in both directions.
- TTS failures don't crash the reply — they fall back to text-only with a small `_couldn't synthesize voice; here's the text_` note.

**Not in scope (yet).**
- *Voice calls.* signal-cli-rest-api is messaging-only; this is a different protocol entirely.
- *SSML / voice cloning / per-user voices.* One voice (configurable via `TTS_VOICE`), simple call. Tune later if it matters.
- *TTS on other channels.* Telegram could support it, but voice-out without voice-in is awkward UX. HTTP/REPL don't make sense.
- *Streaming TTS.* Signal sends complete audio attachments; chunked synthesis would be wasted complexity.

---

## 8. Tool use expansion (operating principle)

**Status.** Continuous. Active — the Instacart integration is the first deliverable in this pattern.

**What.** Not a single piece of work. A pattern for *how* to add tools: each one concrete, scoped, tied to a real user workflow, opt-in via env var, registered through `build_tools(config)`. The starting `allowed_tools=[]` was the right zero point; expansion is incremental and demand-driven, never speculative.

**Why.** Personal-agent ethos: tools should match the user's actual life, not aspire to general agentic capability. Each tool earns its place by replacing a real recurring manual workflow. Generic capability inflation ("let the agent do anything") is the opposite of sovereignty.

**Build trigger.** Per-tool. Each new tool needs a "I keep doing X manually" justification before any code is written.

**Files to touch (per tool).**
- New: `src/tools/<tool-name>.py`
- `src/tools/__init__.py` — registration in `build_tools`.
- `.env.example` and `src/config.py` — credentials/keys, gated as optional.
- `tests/test_<tool-name>.py` — happy-path + most-likely-error-mode coverage.

**Process for each new tool.**
1. Write the user workflow it replaces in one sentence. If you can't, don't ship the tool.
2. Add the env var(s) and `Config` fields. Make the tool's *registration* depend on the env var being set — missing credentials = silent skip in `build_tools`, not error. (See the Instacart implementation as the model.)
3. New module under `src/tools/<name>.py` defining the MCP tool surface. One module per tool.
4. Update `tests/test_<name>.py` with HTTP mocks. Cover the happy path and the most likely error mode.
5. Add a one-paragraph note to the README if the tool meaningfully affects user flow (e.g., requires setup beyond an env var).

**Acceptance per tool.**
- Tool is opt-in via env var; no env var = not registered = LLM doesn't see it.
- Errors during tool calls are surfaced to the LLM as readable strings, not raw exceptions.
- Tests cover happy path and most likely error mode.
- One sentence of "what user workflow this replaces" lives in the tool module's docstring.

**Anti-patterns to avoid.**
- *Generic shell-exec or filesystem-write tools.* Sovereignty + safety. The user has a shell; they don't need the agent to also be one.
- *Tools that read/write outside `output_dir`.* Sandbox boundary stays.
- *Auto-discovery / dynamic tool loading.* Tools register at startup; new ones are added explicitly. Predictability over flexibility.
- *Wide tool catalogs "in case the user wants them."* Each tool is prompt-context cost on every turn. Tools earn their place by getting used.

---

## 9. Turn semantics — journal + reflection-as-commit

**Status.** Designed. *Not* started — trigger-gated.

**What.** Give every turn an explicit start/commit/abort boundary so partial failures can't leak across turns. Two pieces:

1. **Turn journal.** Each turn writes a `started` record before any side effect and a `committed` record after reflection finishes. On startup, scan for orphans (started without committed) and surface them — either reconcile automatically or tell the user *"last turn didn't complete cleanly, here's what I tried to do."*
2. **Reflection-as-commit.** Nothing reaches `MEMORY.md`, daily notes, or project drafts until reflection commits the turn. If reflection fails, the turn is lost cleanly rather than half-remembered.

**Why.** Personal agents fail differently from platforms. There's no DBA to notice state drift — the symptom is the agent slowly getting "weird" over months. It claims to remember things it didn't durably write. Reflection partial-commits leave `MEMORY.md` inconsistent with source fragments. A tool call lies (e.g., a Substack post fails, but the response already streamed "posted ✓"). Each instance is small; cumulatively they erode the one thing a personal agent has to get right — the user trusting that the agent's memory of reality matches theirs.

**Build trigger.** *Wait for the first concrete drift incident.* Specifically: an instance of "it claimed to remember X and didn't" or "MEMORY.md and the source files disagree." Building preemptively risks designing the wrong shape; building after one real failure mode gives you a concrete template to fit the design to.

**Files to touch (when triggered).**
- `src/session/engine.py` — schema additions for the journal.
- `src/pipeline/orchestrator.py` — write `started`/`committed` rows around `process()`.
- New: `src/session/journal.py` — orphan-detection logic on startup.
- `tests/test_pipeline_integration.py` — orphan-recovery tests.

**Steps (when triggered).**
1. Add a `turn_journal` table: `(id, session_id, started_at, committed_at NULL, message_hash, status)`.
2. At the top of `Orchestrator.process()` (after the LLM-reset check, before any LLM work), write a `started` row with the message hash.
3. After the assistant turn is committed, update the journal row's `committed_at`. Same transaction as the `add_turn` call ideally.
4. On `Orchestrator.start()`, scan for journal rows where `committed_at IS NULL`. For each:
   - Surface to the user: *"Turn started 2026-04-30 at 14:22 didn't complete; here's what I tried to do: [hash → message]. Should I retry, drop, or note this?"*
   - Behavior depends on user response. For v1, "drop with note in daily notes" is the simplest correct default.
5. Move all `MEMORY.md` / daily-note / project-draft writes to happen *after* reflection completes. If reflection fails, those writes don't happen — the turn is lost cleanly rather than half-applied.
6. Tests: simulate a crash mid-pipeline (after `started`, before `committed`), restart, verify orphan is surfaced and the user-driven resolution works.

**Acceptance (when shipped).**
- Every committed turn has a journal row with non-null `committed_at`.
- A simulated mid-pipeline crash leaves an orphan journal row that's surfaced on next startup.
- Reflection failure leaves no partial writes anywhere downstream of reflection (no MEMORY.md update, no daily note, no project draft).
- The journal is small (one row per turn, low overhead).

**Not in scope (yet).**
- *A full obligations / contracts metadata system on tools* (cf. IronClaw's obligations model). Tempting, but designing a contract language before knowing which contracts matter is substrate inflation. Revisit only after the journal exists *and* we've felt the pain of two or three concrete obligation gaps.
- *Auto-retry of orphaned turns.* Surface and let the user decide. Auto-retry of a half-completed action is the kind of thing that turns "drift" into "active corruption."
- *Cross-session journal correlation.* One session = one journal scope. Multi-session reasoning is a different problem.

---

## Considered, kept as-is

These came up in the design pass but are *not* on the roadmap. They're either deliberate constraints or trade-offs where the cure is worse than the disease.

### Session-resume primer cost

When the orchestrator resumes a previous session at startup, it replays history as a single primer message and silently consumes the LLM's "Session resumed." reply. That costs one Haiku-tier roundtrip on startup.

**Why kept.** The cost is small (one call, not per-turn) and the alternative (skip the primer) means the LLM has no context until the first user turn — strictly worse UX. Document as a known minor cost; don't fix.

### Single-tenant LLM session, no concurrent users, machine-local Agent SDK auth, OpenAI-only embeddings

These are load-bearing properties of the personal-agent design, not bugs. Multi-tenant isolation is a fundamentally different architecture; concurrent-user parallelism is meaningful only in that architecture; SDK auth is inherent to upstream Claude Code; preemptive embedding-provider abstraction is substrate inflation against a hypothetical migration. The README's "Limitations" section currently treats these apologetically — reframing them as "Single-tenant by design" with brief rationale would land better.
