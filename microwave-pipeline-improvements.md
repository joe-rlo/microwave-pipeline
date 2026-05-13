# MicrowaveOS Pipeline Improvements

**Status:** Draft v0.2 (open questions resolved)
**Date:** 2026-05-13
**Owner:** Joe
**Repo:** `joe-rlo/microwave-pipeline`

## Goal

Tighten the pipeline along three axes: every-turn latency, what the LLM actually sees in context, and the user's visibility into the bot's reasoning. Changes are grouped into phases by leverage and dependency order, not by category.

## Phase 1: Latency wins

These ship first. Each removes per-turn cost from the hot path without changing the pipeline's overall shape.

### 1.1 Conditional reflection

**Problem.** Reflection runs on every turn. On `simple` classifications (greetings, acknowledgments, social), the Haiku round-trip produces no useful signal but pays the latency cost.

**Current behavior.** Reflection always runs after generation, regardless of triage output.

**Proposed behavior.** Route reflection by triage complexity:

| complexity | reflection                                                              |
|------------|-------------------------------------------------------------------------|
| simple     | skip entirely                                                           |
| moderate   | current behavior                                                        |
| complex    | run with deeper prompt (check for unsupported claims, not just hedging) |

**Touchpoints.**

- Orchestrator's reflection stage call site
- Reflection prompt: add a complex-tier variant

**Acceptance.**

- `/debug` after a simple message shows `reflection: skipped`
- `/debug` after a complex message shows `reflection: deep`
- p50 latency on simple turns drops by one Haiku round-trip

**Risks.** Hedging slips through on simple turns. Mitigation: a regex check for hedge tokens (`perhaps`, `I think`, `might be`, `it seems`) on simple-classified outputs before send. No model call, no latency cost.

### 1.2 Conversational register classification

**Problem.** Triage classifies intent and complexity but not register. Search runs unconditionally, which pollutes context on pure social or meta turns and adds latency that's already perceptible on Signal.

**Current behavior.** Triage schema: `{ intent, complexity, search_params }`. Search always runs.

**Proposed behavior.** Add a `register` field. Values: `task | recall | social | meta`.

| register | search behavior                                           |
|----------|-----------------------------------------------------------|
| task     | full pipeline                                             |
| recall   | full pipeline                                             |
| social   | skip search, generation with IDENTITY + recent turns only |
| meta     | skip search, generation with IDENTITY + recent turns only |

Distinction: `meta` is about the bot itself ("are you there", "what can you do"). `recall` is about the conversation history ("do you remember when we discussed X").

> **Implementation note (post-review).** The existing `intent` field already has `"social"`, and extending it to include `"meta"` may obviate the need for a separate `register` axis. Triage already classifies on three dimensions (intent, complexity, phi_class); a fourth dimension adds token cost on every turn. Approach the implementation by first trying to expand `intent` to cover the search-gating logic; only add a separate `register` axis if that proves insufficient.

**Touchpoints.**

- Triage prompt schema
- Orchestrator branch on register
- `/debug` output adds the register field

**Acceptance.**

- "thanks" classified `social`, search skipped
- "what can you do" classified `meta`, search skipped
- "do you remember our talk about NEAR" classified `recall`, search runs

**Risks.** Misclassification eats useful context. Mitigation: even when search is skipped, log what search *would* have surfaced so it's visible in `/debug` for tuning.

### 1.3 Structured outputs for Haiku stages

**Problem.** README documents that triage and reflection sometimes return malformed JSON. Fallback defaults fire incorrectly, which means wrong classifications, which means wrong pipeline behavior.

**Current behavior.** Haiku prompts ask for JSON. Parsed with `try/except json.loads`. We already have `query_json_with_retry` (one-shot retry + structured logging) handling ~80% of this; full schema enforcement is the remaining 20%.

**Proposed behavior.** Use the Agent SDK's tool-use schema enforcement for triage and reflection. Define each as a tool with a strict schema. Haiku invokes the tool. The SDK returns parsed args, not text.

When the direct API backend lands (separate spec), use `response_format` with JSON schema for the same effect.

**Touchpoints.**

- `src/pipeline/triage.py`
- `src/pipeline/reflection.py`
- Tool schema definitions
- `SingleTurnClient` may need restructuring (currently uses plain-text completion, not tool calls)

**Acceptance.**

- 100 consecutive turns with zero JSON parse failures
- The fallback-default code path becomes dead code (delete it)

**Risks.** Tool-use overhead on Haiku might exceed plain text by a small margin. Measure. If it adds more than 50ms, fall back to constrained text generation.

## Phase 2: Context shaping

These change what the LLM sees per turn. They depend on Phase 1 being stable.

### 2.1 Daily notes via retrieval

**Problem.** Today's and yesterday's daily notes are concatenated into the stable system prompt. Most content isn't relevant to the current turn. Result: bloated stable prefix, weaker prompt cache hits, polluted context.

**Current behavior.** Assembly reads `workspace/memory/<today>.md` and `<yesterday>.md`, concatenates into system prompt. Daily notes ARE already indexed (per the async-embeddings work) — the indexer covers them; only the assembly path concatenates them as a separate block.

**Proposed behavior.** Index daily notes the same way other memory is indexed (already done). Retrieve via search with a 24-hour half-life recency boost so recent notes still dominate when relevant. Stop concatenating into the system prompt.

**Touchpoints.**

- Memory indexer: include daily notes in the corpus (already done)
- Search ranker: add `created_at` field, apply temporal decay during scoring (already done)
- Assembly stage: remove the daily-notes concat block

**Acceptance.**

- A novel-project turn surfaces novel-related notes from today, not NEAR ticket notes
- Stable system prompt shrinks (measure token count before/after)

**Risks.** Today's notes get under-retrieved when the user message is short on keywords. Mitigation: when register is `task` or `recall`, force-include the top-1 daily-note fragment from today even if its raw score is low.

### 2.2 Stable / dynamic split tightening

**Problem.** The current stable context includes content that should be dynamic, which reduces prompt cache effectiveness and bloats the prefix.

**Current behavior.**

- Stable: IDENTITY + channel rules + MEMORY.md + daily notes
- Dynamic: retrieved fragments + recent turns

**Proposed behavior.**

- Stable: IDENTITY + active channel rules + active skill body + active project BIBLE
- Dynamic: MEMORY.md retrieval matches + daily notes retrieval matches + episodic fragments + recent turns + length hint

MEMORY.md as a whole document leaves the stable prefix. It gets indexed and retrieved like everything else. The assembler still includes the top-K matches per turn, but via ranking, not blanket inclusion.

> **Implementation note (post-review).** Parsing MEMORY.md into discrete facts is the hard part. Joe's MEMORY.md has multi-line bullets, sub-bullets, and headings — "one fact per line" assumes a structure that doesn't exist. Either build a parser that's smart enough, or ship 2.1 alone first (trim daily notes only), measure cache improvement, then decide whether 2.2's parser work is worth the marginal additional gain.

**Touchpoints.**

- Assembly stage: split function
- Memory indexer: index MEMORY.md as discrete facts (one per line or one per heading, decide during implementation)
- Search: weight MEMORY.md fragments higher than session fragments by default

**Acceptance.**

- Stable prefix token count drops by at least 30%
- Prompt cache hit rate measurable on consecutive turns where stable context didn't change

**Risks.** Critical facts get missed by retrieval. Mitigation: add a `pin: true` frontmatter option for MEMORY.md entries that forces inclusion. Use sparingly. Things like the user's name, active project, key relationships qualify. Most facts don't.

### 2.3 Skills as pipeline modifiers

**Problem.** Skills only inject instruction text today. They could also tune pipeline behavior for the task at hand.

**Current behavior.** Skill frontmatter has `name`, `description`, `triggers`. Body is instruction text.

**Proposed behavior.** Add an optional `pipeline:` block to skill frontmatter:

```yaml
---
name: novel-writing
description: ...
triggers: [...]
pipeline:
  reflection: off          # off | light | normal | deep
  search_recall: high      # low | medium | high
  max_output_tokens: 4000
  escalation_bias: high    # nudges triage toward complex
---
```

When a skill is active (pinned or auto-matched), its pipeline block overrides defaults for that turn. Defaults remain when no skill is active or the active skill omits the block.

Concrete examples:

- `novel-writing` sets `reflection: off` (reflection second-guesses creative choices)
- `github-tool` sets `max_output_tokens: 8000` (digests need room)
- `substack-writer` sets `escalation_bias: high` (good prose needs Opus)

**Touchpoints.**

- Skill loader: parse pipeline block
- Orchestrator: consult active skill's overrides before each stage
- Triage: accept an escalation_bias hint in its prompt

**Acceptance.**

- `novel-writing` active: reflection skipped regardless of complexity
- `github-tool` active: max output tokens raised
- `/debug` shows which pipeline overrides are active

**Risks.** Skills that misconfigure themselves cause silent behavior shifts. Mitigation: the `/debug` visibility above is the audit trail.

## Phase 3: Continuity and transparency

These improve user-facing behavior and the bot's character across time horizons longer than a single conversation.

### 3.1 Cross-session summary

**Problem.** `/new` wipes context. Next session starts cold. The user re-establishes what they were working on every time.

**Current behavior.** Session ends with no record. Compaction summaries get written to daily notes but aren't structured for session-start retrieval.

**Proposed behavior.** On session close (timeout, `/new`, or compaction), generate a Sonnet summary of the session: target 200 words covering what was worked on, decisions made, and what's still hanging. Store at `workspace/memory/sessions/<YYYY-MM-DD-HHMM>-<topic-slug>.md` with frontmatter:

```yaml
---
started: 2026-05-13T09:14:00-04:00
ended: 2026-05-13T10:02:00-04:00
topic: pipeline-improvements
project: microwaveos
turns: 14
---
```

On session start, retrieve the most recent 3 session summaries plus top-K by topic match against the first user message. Include in dynamic context.

**Touchpoints.**

- Session engine: close hook generates summary
- Memory indexer: index sessions/ directory
- Assembly: session-start retrieval branch

**Acceptance.**

- After `/new`, the next session's `/debug` shows session summaries in retrieval
- Summaries are coherent enough to read standalone

**Risks.** Summary generation adds cost to session close. Acceptable, since it's one Sonnet call per session, not per turn.

### 3.2 /why command

**Problem.** When the bot says something surprising, the user has no visibility into why. `/debug` shows pipeline stats (stages run, models used, latency) but not the actual retrieval content that fed the answer.

**Proposed behavior.** New command `/why` shows the last turn's retrieval results inline: top 5 fragments with source path and a 100-character snippet. Available on REPL, Telegram, Signal.

REPL output format (default):

```
why: last turn retrieved
  workspace/MEMORY.md:12              "Joe's primary project is..."
  workspace/projects/.../bible        "Sarah's character voice..."
  memory/2026-05-12.md                "Met with design team..."
  sessions/2026-05-10-...md           "Discussed pipeline reflection..."
  workspace/skills/.../SKILL          "Voice rules: direct, no..."
```

With `-v` flag (REPL) or `/why scores` (Signal/Telegram), scores are included.

**Touchpoints.**

- Orchestrator: cache last-turn retrieval results in session state
- Command handler
- Channel adapters

**Acceptance.**

- `/why` after any turn returns the retrieval that fed it
- Works across all channels
- Doesn't trigger a new pipeline run
- Scores hidden by default; revealed via verbose flag

**Risks.** Minimal. Read-only on cached state.

### 3.3 Thinking nudge with classification context

**Problem.** Signal's `*…thinking*` nudge is generic. The user can't tell if the bot is on a fast or slow path until the reply lands.

**Current behavior.** Single rotating microwave-themed nudge after 4s, picked at random from 16 phrases.

**Proposed behavior.** Defer or skip. The 16 microwave phrases are doing personality work that stage-specific text ("looking through memory") would dilute. If stage visibility matters for debugging, surface it in `/debug` instead of the conversational thinking nudge.

**Risks.** Race conditions if stage transitions between timer fire and message send (the original framing). Better: don't put stage info in the user-facing nudge at all.

### 3.4 Memory contradiction resolution

**Problem.** MEMORY.md is append-only. Contradictions accumulate. README documents this as a known limitation. (Note: `memory health` CLI was shipped earlier as a user-driven contradiction surfacer; no auto-resolution by design.)

**Proposed behavior (post-review revision).** Ship in two phases:

**Phase A — queue-only (default).** Periodic (weekly cron) or on-demand pass runs over MEMORY.md, flagging suspected contradictions via embedding similarity (cosine > 0.80) to `workspace/memory/contradictions.md`. User triages from REPL with a `/memory review` command: keep, supersede, or merge each flagged pair. Resolution is logged to `workspace/memory/supersession-log.md`.

This preserves the "sovereignty over memory" principle from the existing `memory health` CLI — the agent flags, the user resolves.

**Phase B — opt-in auto-supersession.** Behind `MEMORY_AUTO_SUPERSEDE=true` env flag, automatic supersession on add at a conservative similarity threshold (≥ 0.95). Lower-similarity matches (0.80–0.95) still go to the queue. Default off. Enable only after Phase A has produced calibration data from real corpus.

Storage format inside MEMORY.md (both phases):

```markdown
- [2026-04-12] Joe's dog is named Biscuit
- [2026-05-13] Joe's dog is named Max <!-- supersedes: 2026-04-12 -->
```

The assembly stage filters out superseded entries during retrieval.

**Touchpoints.**

- Memory promotion path: similarity check + queue write (Phase A) / auto-supersede option (Phase B)
- Scheduler: add a weekly job for review
- New REPL command: `/memory review` (only meaningful on REPL, skip on mobile channels)
- Assembly: filter `superseded: true` entries

**Acceptance.**

- Phase A: two contradicting facts added in sequence both surface until user resolves via `/memory review`
- Phase A: weekly cron flags at least one issue in a MEMORY.md seeded with deliberate contradictions
- Phase B (when enabled): only the newer of two strongly-contradicting facts surfaces in retrieval
- Audit log at `workspace/memory/supersession-log.md` records every supersession (manual or auto) with timestamp and similarity score

**Risks.** Auto-supersession misfires on subtly different facts ("Joe lives in Boston" vs "Joe lived in Boston for 10 years"). Phase A avoids this entirely. Phase B's 0.95 threshold is deliberately conservative; calibrate from queue data.

## Out of scope

The following are deliberately excluded from this round:

- Vision support. Separate effort, blocked on direct API backend. *(Note: image pass-through actually shipped earlier for Telegram + Signal via `images` content blocks on the api_key path. Max-auth path still text-only.)*
- Direct Anthropic API backend. Separate spec.
- Multi-user safety for Telegram. Orchestrator-level concern, not pipeline.
- Real in-turn tool use (`allowed_tools=["Read", ...]`). *(Note: shipped earlier via `BOT_BUILTIN_TOOLS` env var with `permission_mode=bypassPermissions`.)*
- Local embedding model. Tracked as a future infrastructure swap.

## Implementation order

Recommended sequence with rationale (revised after review):

1. **1.1 Conditional reflection.** Biggest perceived latency win, lowest risk.
2. **3.2 `/why` command.** Read-only, low risk, immediately useful for tuning everything else that follows. Promoted from Phase 3 to second in order — its tooling value pays back on every subsequent change.
3. **2.1 Daily notes retrieval.** First context-shape change, measurable via prompt cache hit rate. Most of the indexer/decay plumbing already exists.
4. **1.2 Register / intent merge.** Approach by extending `intent` first; only add a separate `register` axis if needed.
5. **3.1 Cross-session summary.** Independent, can interleave with the above once Phase 1 latency baseline is stable.
6. **1.3 Structured outputs (schema enforcement).** Bigger plumbing change than presented in the spec; defer until 1–5 land.
7. **2.2 Stable/dynamic tightening.** Validates with prompt cache metrics added in step 1.
8. **2.3 Skills as modifiers.** Needs 2.2 plumbing.
9. **3.3 Thinking nudge variants.** Likely **skip** per review; personality-vs-state-info tradeoff resolves toward keeping the microwave phrases. Surface stage info in `/debug` only.
10. **3.4 Memory contradiction resolution.** Phase A first; defer Phase B until queue data exists.

Add a **prompt cache hit-rate metric** to `/debug` as part of step 1 so steps 3, 7, 8 can be validated quantitatively.

Estimated effort: 4–5 focused weeks for a solo dev who already knows the codebase. The original 2–3 week estimate assumed everything goes cleanly; 4–5 weeks is more honest with QA loops.

## Resolved open questions

The four open questions from v0.1, with decisions:

### Q1. Should `social`/`meta` turns be indexed into the corpus?

**Decision: store in `turns`, exclude from fragment indexing.** Two-layer rule. The `turns` table stays the complete conversation log (audit, recap, recent-turns retrieval). The `fragments` table (embeddings store) filters out `social` + `meta` at index-write time — they have zero recall value and dilute vector neighborhoods. Tone calibration is handled by the recent-4-turns in dynamic context plus IDENTITY voice rules; the bot doesn't need long-term embedding recall of "thanks" to match register.

### Q2. Should cross-session summaries have a hard max age?

**Decision: temporal decay only, no hard cap.** Long memory is the point of a personal agent. If an 8-month-old session is genuinely the strongest topic match for a current question, that's correct surfacing. The existing recency-decay term in the ranker already weights newer higher when relevance is comparable. Storage isn't the bottleneck — 200-word summaries × 3 sessions/day × 365 days ≈ 220k words/year, easily handled by the indexer for years.

### Q3. Should `/why` show retrieval scores?

**Decision: `-v` flag (REPL) / `/why scores` (Signal, Telegram), default snippets only.** Score values without prior calibration on what 0.82 means in this corpus are noise — new users read them as meaningful when they aren't. The default use of `/why` is conversational ("why did you say that?") and snippets + paths answer that. Scores answer a different question (tuning), gated behind the verbose form.

### Q4. Calibrating supersession thresholds (0.92 / 0.85)?

**Decision: ship queue-only first; defer auto-supersession until calibrated.** The numerics are guesses *because no one has flagged a real corpus yet* — locking them into the design before data exists is premature. Phase A ships at a single queue threshold (~0.80) and logs every flagged pair. Phase B (auto-supersession ≥ 0.95) is opt-in behind `MEMORY_AUTO_SUPERSEDE=true` and only enabled after Phase A produces calibration data. This also preserves the "sovereignty over memory" principle from the existing `memory health` CLI work — the agent flags, the user resolves, by default.
