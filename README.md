# MicrowaveOS

A cognitive agent runtime where the pipeline — triage, search, assembly, reflection — is the architecture. Wraps Claude via the Agent SDK, authenticated through your Max subscription.

## Setup

### 1. Install Python dependencies

```bash
python3 -m pip install pydantic tiktoken apsw aiohttp anthropic openai sqlite-vec croniter
```

For Telegram (optional):
```bash
python3 -m pip install python-telegram-bot
```

The Agent SDK should already be installed if you're using Claude Code:
```bash
python3 -m pip install claude-agent-sdk
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `AUTH_MODE` | Yes | `max` (Agent SDK, no API key) or `api_key` (pay-as-you-go) |
| `ANTHROPIC_API_KEY` | If `api_key` mode | Your Anthropic API key |
| `OPENAI_API_KEY` | Yes | Embeddings + Whisper transcription (Signal voice-in) |
| `TELEGRAM_BOT_TOKEN` | For Telegram | From @BotFather |
| `SIGNAL_REST_URL` | For Signal | URL of signal-cli-rest-api daemon (e.g. `http://[::1]:8080`) |
| `SIGNAL_PHONE_NUMBER` | For Signal | Bot's Signal-registered number (e.g. `+15551234567`) |
| `SIGNAL_ALLOWED_SENDERS` | For Signal | Comma-separated allowlist; omit to accept anyone |
| `SCHEDULER_ENABLED` | No | `true` to run the scheduler alongside `--signal` |
| `CLAUDE_CLI_PATH` | No | Auto-detected. Override if needed. |

### 3. Authenticate Claude Code

If using `AUTH_MODE=max`, make sure you're logged into Claude Code on this machine:

```bash
claude login
claude whoami  # should show your username
```

The Agent SDK authenticates through your Claude Code session. No API key needed.

### 4. Import existing data (optional)

If you're coming from another AI tool, you can import your conversation history and memory into MicrowaveOS. Imported data is chunked, embedded, and indexed for vector + keyword search.

```bash
# OpenClaw — auto-detects ~/.openclaw, prompts to select agent
python3 src/import_data.py openclaw
python3 src/import_data.py openclaw --path ~/.openclaw/agents/abc123

# Hermes Agent — auto-detects ~/.hermes
python3 src/import_data.py hermes
python3 src/import_data.py hermes --path ~/.hermes

# NanoClaw — auto-detects common install locations
python3 src/import_data.py nanoclaw
python3 src/import_data.py nanoclaw --path ~/nanoclaw
```

Options:
- `--dry-run` — show what would be imported without doing it
- `--sessions-only` — import conversations, skip memory files
- `--memory-only` — import memory/knowledge, skip conversations

**What gets imported:**

| Source | Conversations | Memory | Daily notes | Topic memories |
|--------|:---:|:---:|:---:|:---:|
| OpenClaw | JSONL sessions | MEMORY.md | daily .md files | people/, projects/, topics/ |
| Hermes | SQLite sessions | ~/.hermes/memories/ | — | — |
| NanoClaw | SQLite messages | CLAUDE.md per group | — | FTS conversations |

Imported content becomes searchable immediately — the pipeline's triage and search stages will surface relevant fragments from your history.

### 5. Run

```bash
# REPL (default)
python3 src/main.py

# With pipeline metadata visible
python3 src/main.py -v

# Telegram bot
python3 src/main.py --telegram

# Signal bot (requires signal-cli-rest-api daemon)
python3 src/main.py --signal

# HTTP API server
python3 src/main.py --http
python3 src/main.py --http --port 9000

# Scheduler management (no daemon required for these)
python3 src/main.py scheduler list
python3 src/main.py scheduler add --name ... --cron ... --mode llm|direct ...
python3 src/main.py scheduler run <name>

# Skills management (no daemon required)
python3 src/main.py skills list
python3 src/main.py skills new <name> --description "..."
python3 src/main.py skills edit <name>

# Projects management (no daemon required)
python3 src/main.py projects list
python3 src/main.py projects new <name> --type blog|novel|screenplay
python3 src/main.py projects show <name>
```

## Customization

### Identity — `workspace/IDENTITY.md`

This is Microwave's soul. Everything in this file is loaded into the system prompt at startup and shapes every response. The more specific you make it, the less it feels like a generic assistant.

Write it like you're describing a person, not configuring a product.

**What to put here:**

- **Voice and tone** — Warm or dry? Formal or casual? Does it use slang? Swear? Make jokes?
- **Personality** — Is it opinionated or neutral? Does it push back or defer? Curious or focused?
- **Relationship to you** — Collaborator, advisor, research partner, friend? How does it address you?
- **Values and taste** — What does it care about? What does it find interesting or annoying?
- **Communication style** — Default to brevity or depth? Ask follow-up questions? Offer unsolicited opinions?
- **Context about you** — Your work, interests, how you think. The more it knows, the better it calibrates.

Example:

```markdown
# Identity

You are Microwave. You're sharp, opinionated, and genuinely interested in ideas.

## Voice
- Direct. No filler. Say what you mean.
- You can be funny but you're not performing. Humor is dry, never forced.
- You push back when something doesn't make sense. You're not a yes-machine.
- Brief by default. Go deep when the topic demands it or when asked.

## Your perspective
- You care about craft — in writing, in code, in thinking.
- You're skeptical of hype but curious about substance.
- You have opinions and you'll share them, but you hold them loosely.

## Your user
- Joe is a builder. He thinks in systems.
- He values directness — don't hedge, don't over-explain.
- If he asks a vague question, ask a clarifying one back.
```

Changes to IDENTITY.md take effect on the next restart (or next reconnect if memory/daily notes also change).

### Channel rules — `workspace/channels/`

Per-channel formatting and behavior rules. Each file is only loaded when that channel is active.

```
workspace/channels/
├── telegram.md    # mobile-friendly, brief, Telegram markdown
├── repl.md        # technical, detailed, full markdown
└── http.md        # structured, no filler
```

Edit these to control how Microwave communicates on each platform. Example `telegram.md`:

```markdown
# Telegram Channel Rules

## Formatting
- Keep responses concise — phone screen, not a document
- Short paragraphs. No walls of text.
- Telegram Markdown: *bold*, _italic_, `code`, ```code blocks```
- Bullet points over numbered lists
- No headers (Telegram doesn't render #)

## Behavior
- Default to brief. Expand only when asked.
- Lead with the short answer, offer to elaborate.
```

Changes take effect on next restart (or next reconnect if stable context files change).

### Long-term memory — `workspace/MEMORY.md`

Durable facts that persist across conversations. You can add things here manually, or the pipeline will promote frequently-retrieved fragments automatically over time.

```markdown
# Long-term Memory

- Joe's main project is MicrowaveOS, a cognitive agent runtime
- He prefers terse responses with no trailing summaries
- His dog's name is Biscuit
```

### Daily notes — `workspace/memory/`

Working notes organized by date (`2026-04-12.md`, etc.). Today's and yesterday's notes are included in the system prompt. The pipeline also writes compaction summaries here.

You can manually add notes for things you want Microwave to know today:

```bash
echo "Meeting with design team at 3pm about the rebrand" >> workspace/memory/2026-04-12.md
```

### Skills — `workspace/skills/`

Skills are named, reusable instruction bundles that extend the pipeline's dynamic context for a specific task or domain. Think of a skill as a scoped IDENTITY.md — voice rules, topic guardrails, output format — that you can activate when you need it and turn off when you don't.

Each skill is a directory with a `SKILL.md` file containing YAML-ish frontmatter and a markdown body:

```
workspace/skills/substack-writer/
├── SKILL.md          # frontmatter + body (required)
└── fetch.py          # optional: pre-fetch script for scheduled jobs
```

`SKILL.md` shape:

```markdown
---
name: substack-writer
description: >
  Generate Substack Notes in Joe's voice. Direct, opinionated,
  anti-LinkedIn. Use for short-form published content.
triggers:
  - substack
  - note
  - hot take
---

# Substack Writer

## Voice rules
- Direct, opinionated, a little sharp. No corporate sludge.
- ...
```

**Activation — three paths:**

1. **In chat** (REPL, Telegram, Signal): `/skill substack-writer` activates it for the next turns; `/skill off` deactivates; `/skills` lists. The skill body is spliced into the pipeline's dynamic context.
2. **In scheduled jobs**: `scheduler add --skill substack-writer --trigger "Generate 3 notes for today."` — the skill body becomes the system prompt for a one-shot LLM call.
3. **Direct invocation via CLI**: `skills show <name>` for inspection.

**Precedence:** skill instructions are additive to IDENTITY.md. When they conflict with channel rules (message length, markdown syntax, attachment behavior), channel rules win — the assembler adds an explicit note to the prompt saying so.

**Pre-fetch** (scheduler-only in v1): a skill can include a `fetch.py` with `async def fetch(context)` or `def fetch(context)`. The scheduler runs it before the LLM call and prepends the output to the prompt under `[Pre-fetch context]`. The seed `github-tool` skill uses this to shell out to `gh search prs --author=@me --state=open` before writing a digest.

**CLI:**

```bash
microwaveos skills list
microwaveos skills show <name>
microwaveos skills new <name> [--description "..."]
microwaveos skills edit <name>            # opens in $EDITOR
microwaveos skills remove <name>          # asks for confirmation
```

Seed skills shipped with v1: `substack-writer`, `blog-writing`, `github-tool`, `novel-writing`, `screenplay-writing`.

**Adaptive activation.** Triage classifies every message and *also* picks a matching skill from your catalog. If you say "draft a Substack about NEAR's Q2 metrics," `substack-writer` auto-activates for that turn — no need to type `/skill`. Auto-match is per-turn and ephemeral; explicit `/skill <name>` always wins as a sticky pin. Triage uses the description field on each skill, so write descriptions that tell the LLM when *not* to match (the seed skills include "skip when X" guards). `/debug` shows whether a skill auto-matched.

### Writing projects — `workspace/projects/`

Projects are per-assignment workspaces for ongoing writing — a blog post, a novel, a screenplay. Where skills define *how* to write (voice, format, anti-patterns), projects define *what* you're working on (characters, outline, prior chapters). Activating a project auto-loads its declared skill, joins its `BIBLE.md` to the system prompt, indexes its drafts into memory, and routes new file output to the project's `drafts/` directory.

**Three project types, each with its own scaffold:**

| Type | Layout | Default skill | File output |
|------|--------|---------------|-------------|
| `blog` | `outline.md`, `drafts/draft.md` | `blog-writing` | LLM-suggested name, `drafts/draft.md` default |
| `novel` | `BIBLE.md`, `outline.md`, `drafts/chapter-NN.md` | `novel-writing` | sequential `chapter-NN.md` |
| `screenplay` | `BIBLE.md`, `outline.md`, `drafts/screenplay.fountain` | `screenplay-writing` | scenes append into single FOUNTAIN file (Highland-compatible) |

`PROJECT.md` carries the metadata:

```markdown
---
name: the-heist
type: novel
skill: novel-writing
status: drafting          # drafting | revising | paused | done | archived
description: >
  Sarah pulls one last job before her sister's wedding.
target_words: 80000
created: 2026-04-28
---

## Voice notes
First-person, present tense. Sarah's POV throughout.
British spellings ("colour", "realise").
```

**Activation in chat** (REPL, Telegram, Signal):

```
/project the-heist           # activate; auto-loads novel-writing skill
/project status              # word count, draft count, BIBLE/outline status
/project off                 # deactivate
/projects                    # list all with status + word counts
```

**File output flow.** When the bot drafts a chapter, the orchestrator writes it to `drafts/chapter-NN.md` and the channel sends a small confirmation: `✓ wrote chapter-04.md (1,247 words)` plus the opening line as a quote-block preview. No wall of text in chat — open the file in your editor of choice (Highland for screenplays, anything for prose). The new file is also re-indexed immediately, so the next turn's retrieval can reference it.

**BIBLE flow** (novels and screenplays). The novel/screenplay skills tell the LLM to surface "Possibly new for BIBLE" at the end of any draft that introduces named entities. You commit them with:

```
/bible add Walsh tall, weary, twists his wedding ring when nervous
/bible add "Detective Walsh" same as above but multi-word names need quotes
/bible show               # print current BIBLE.md
```

Bible writes are user-approved on purpose — auto-writing risks polluting canonical state with mid-draft inventions you haven't endorsed. After a `/bible add`, the next turn picks up the change automatically (mtime-based reconnect).

**CLI:**

```bash
microwaveos projects list
microwaveos projects new the-heist --type novel --description "Last job"
microwaveos projects show the-heist          # status, words, drafts list
microwaveos projects edit the-heist          # opens PROJECT.md in $EDITOR
microwaveos projects archive blog-q1-recap   # moves under .archived/
microwaveos projects remove dead-project     # irreversible (asks for confirmation)
```

**Switching context.** Blogs can run in parallel with each other (research, drafts, multiple posts) but long-form work is one-at-a-time — switching projects forces an LLM reconnect on the next turn so the new bible cleanly replaces the old one. Skills stay attached to their project; deactivating a project doesn't auto-clear the skill (you might want to keep `novel-writing` rules for a follow-up turn even after stepping out of the project).

### Model selection

In `.env`:

```bash
MODEL_MAIN=sonnet        # Main conversation (sonnet recommended)
MODEL_TRIAGE=haiku       # Intent classification (haiku — fast + cheap)
MODEL_REFLECTION=haiku   # Quality gate (haiku — fast + cheap)
MODEL_COMPACTION=sonnet  # Session summarization (sonnet — quality matters)
MODEL_ESCALATION=opus    # Escalation model for complex tasks
ESCALATION_EFFORT=high   # Thinking budget: low, medium, high, max
```

### Embedding model

Currently pinned to OpenAI `text-embedding-3-small`. This is infrastructure — it maps text to coordinates for vector search. It never reasons or generates user-facing output. Configured in `src/memory/embeddings.py` if you ever want to swap it.

## Architecture

```
Message in → Triage (haiku) → Search (sqlite-vec + FTS5) → Assembly → LLM (sonnet/opus) → Reflection (haiku) → Message out
```

- **Triage** classifies intent, complexity, and dynamically configures search parameters
- **Search** runs hybrid vector + keyword retrieval, merged with Reciprocal Rank Fusion
- **Assembly** builds the system prompt (stable) and per-turn context (dynamic)
- **LLM Generation** — Sonnet by default, escalated to Opus with extended thinking for complex tasks
- **Reflection** runs a quality gate — catches hedging, triggers re-search if context was insufficient
- **Session engine** tracks conversation history in SQLite, handles compaction when approaching context limits

### Model escalation

Not every message needs the same model. Simple greetings and quick questions go to Sonnet. Hard problems get Opus with extended thinking — automatically.

**How it works:**

1. Triage (Haiku) classifies every message with a complexity level: `simple`, `moderate`, or `complex`
2. When complexity is `complex`, the orchestrator escalates — switches the LLM to Opus with extended thinking enabled
3. After that turn completes, it de-escalates back to Sonnet for the next message
4. No reconnect needed. For Agent SDK (Max auth), it calls `set_model()` on the live session. For API key mode, it swaps the model and adds a `thinking` parameter with a token budget

**What triggers `complex`:**
- Multi-step reasoning or analysis
- Architecture and system design
- Debugging tricky problems
- Long-form writing that requires structure
- Math proofs or formal logic
- Anything requiring careful step-by-step thought

Most messages stay on Sonnet. The escalation is automatic — you don't need to ask for it.

**Thinking budgets** (controls how long Opus thinks before responding):

| Effort | Token budget | Use case |
|--------|-------------|----------|
| `low` | 2K tokens | Light reasoning, quick sanity checks |
| `medium` | 8K tokens | Moderate analysis, short planning |
| `high` | 32K tokens | Deep reasoning, complex debugging (default) |
| `max` | 64K tokens | Exhaustive analysis, formal proofs |

Configure in `.env`:

```bash
MODEL_ESCALATION=opus    # which model to escalate to
ESCALATION_EFFORT=high   # thinking budget
```

**Visibility:** Run `/debug` after any message to see whether escalation fired. The output will show `escalated: opus` when it did.

### Channels

| Channel | Description |
|---------|-------------|
| REPL | stdin/stdout, `/debug` shows pipeline metadata |
| Telegram | Typing indicators, streaming via message edits, HTML-file attachments for tables/charts |
| Signal | Typing indicators, read receipts, voice-message transcription (Whisper), HTML card-view attachments, debounce + interrupt |
| HTTP | `POST /chat` returns JSON with response + pipeline stats |

### Conversational behavior

A few features tune how the bot reads and responds across multi-message bursts. They mostly live on the Signal channel today (where messaging is bursty by nature) but the assembly-layer changes apply everywhere.

**Debounce + coalesce (Signal).** Back-to-back messages within a 2.5s window are buffered and combined into one pipeline run. Send "tell me about NEAR" then "specifically the Q2 dev metrics" half a second later → the bot sees both as one input and replies once, coherently. Without this, each message would race the other through the pipeline against the same SDK session and produce two unrelated replies. Tunable via `DEBOUNCE_SECONDS` at the top of `src/channels/signal.py`.

**Typing-indicator awareness (Signal).** When you start typing, the debounce timer pauses — the bot doesn't fire mid-compose. When you stop typing, a fresh debounce window starts. A 60s max-hold timer guarantees a stuck "typing forever" doesn't strand input forever.

**Streaming interrupt (Signal).** A new message arriving while the bot is mid-reply cancels the in-flight LLM call and starts over with the combined input. Already-sent reply chunks stay sent (Signal's REST bridge has no clean delete API), but no further chunks from the stale run land. Closest to how humans actually talk over each other.

**Adaptive response length (all channels).** Triage's `simple` / `moderate` / `complex` classification flows into a length hint at the end of the dynamic context. Simple messages get "match the brevity, ~50 words, no preamble." Complex gets "develop fully, take the space you need." Moderate gets no hint — that's the default. Fixes the "thanks" → 200-word essay problem.

**Skill commands stay instant.** `/skill`, `/project`, `/bible` etc. bypass the debounce buffer entirely so they don't pay the 2.5s penalty. They're meant to feel like UI, not conversation.

### Signal setup

Signal has no official bot API. MicrowaveOS talks to [`signal-cli-rest-api`](https://github.com/bbernhard/signal-cli-rest-api) over HTTP + WebSocket. You need a dedicated phone number for the bot (spare SIM, Google Voice, Twilio, etc. — not your personal Signal number unless you want to take it over).

```bash
# 1. Run the daemon
docker run -d --name signal-api -p 8080:8080 \
  -v signal-data:/home/.local/share/signal-cli \
  -e MODE=json-rpc \
  bbernhard/signal-cli-rest-api:latest

# 2. Register the bot number (will prompt for a captcha token
#    from https://signalcaptchas.org/registration/generate.html)
curl -X POST http://localhost:8080/v1/register/%2B15551234567 \
  -H 'Content-Type: application/json' \
  -d '{"captcha": "<captcha-token>"}'

# 3. Verify with the 6-digit SMS code
curl -X POST http://localhost:8080/v1/register/%2B15551234567/verify/123456

# 4. Run the bot
python3 src/main.py --signal
```

**Voice messages:** send a Signal voice note to the bot — it downloads the audio, transcribes via Whisper (uses your `OPENAI_API_KEY`, ~$0.006/min), shows a `_heard: "..."_` preview so you can verify, and feeds the transcript into the pipeline tagged as `[voice]`.

**Card view:** list-shaped bot output (multiple Substack notes, PR digests, etc.) is delivered as an HTML attachment with per-card Copy buttons so you can copy cleanly from Signal's mobile client (where text selection is painful).

### Scheduler

Cron-scheduled recurring jobs — daily writing prompts, reminders, digests. Runs as a background task inside `--signal` when `SCHEDULER_ENABLED=true`. Two modes:

| Mode | Behavior | Cost |
|------|----------|------|
| `llm` | Runs the job's prompt through a one-shot LLM call, delivers as HTML card-view attachment (per-card Copy buttons) | ~1 LLM call per fire |
| `direct` | Sends a literal text message verbatim. No LLM involvement. | $0 |

LLM jobs deliberately bypass the main pipeline — no shared session with your live conversation, no contamination of recent-turn recall, no compaction triggers on your real context. Card-view output is the default for LLM jobs because Signal's mobile text selection is hostile to multi-item copying.

**CLI:**

```bash
# LLM job using a named skill (preferred) — skill body is the system
# prompt, --trigger is the short user message that kicks off the call
python3 src/main.py scheduler add \
  --name substack-notes \
  --cron "57 6 * * *" \
  --mode llm \
  --channel signal \
  --recipient "+15551234567" \
  --skill substack-writer \
  --trigger "Generate 3 Substack Notes for today."

# LLM job with an inline prompt (no skill)
python3 src/main.py scheduler add \
  --name morning-briefing \
  --cron "30 6 * * *" \
  --mode llm \
  --channel signal \
  --recipient "+15551234567" \
  --prompt "Summarize the top 3 things I should think about today..."

# Direct reminder — no LLM
python3 src/main.py scheduler add \
  --name vitamins \
  --cron "30 9 * * *" \
  --mode direct \
  --channel signal \
  --recipient "+15551234567" \
  --text "💊 Vitamine — take 'em before the day eats you."

# Manage
python3 src/main.py scheduler list
python3 src/main.py scheduler disable vitamins
python3 src/main.py scheduler enable vitamins
python3 src/main.py scheduler remove substack-notes

# Fire one right now (regardless of schedule) — useful for testing
python3 src/main.py scheduler run substack-notes
```

**Skill-driven jobs** (`--skill`) are the cleanest path for recurring LLM tasks: the skill's instructions live in one place and can be reused by interactive chat or other jobs. If the referenced skill has a `fetch.py`, the scheduler awaits it before the LLM call and prepends the output to the prompt — e.g., `github-tool`'s fetch runs `gh search prs` so the digest has fresh data.

**Missed runs:** if the daemon was offline when a job was due (laptop closed, process crashed), the scheduler does **not** fire the backlog on restart. It fast-forwards the job's baseline and waits for the next scheduled fire. A 3pm vitamin reminder for a 9am ping helps nobody.

**Persistence across reboots:** use the launchd template at `deploy/com.microwaveos.daemon.plist.template`. Replace `{{PROJECT_DIR}}` and `{{PYTHON}}` with absolute paths, copy to `~/Library/LaunchAgents/`, then `launchctl load`. It only restarts on crashes (not clean shutdowns), so `launchctl unload` actually stops it.

### Data storage

```
~/.microwaveos/
├── workspace/
│   ├── IDENTITY.md        # personality
│   ├── MEMORY.md          # long-term facts
│   ├── channels/          # per-channel rules
│   │   ├── telegram.md
│   │   ├── signal.md
│   │   ├── repl.md
│   │   └── http.md
│   ├── skills/            # reusable instruction bundles
│   │   └── <name>/SKILL.md[, fetch.py]
│   ├── projects/          # writing assignments — blog/novel/screenplay
│   │   └── <name>/
│   │       ├── PROJECT.md
│   │       ├── BIBLE.md         # novels + screenplays
│   │       ├── outline.md
│   │       ├── drafts/          # chapter-NN.md or screenplay.fountain
│   │       └── notes/
│   ├── memory/            # daily notes
│   └── output/            # files the LLM writes via SDK tool use
└── data/
    └── memory.db          # SQLite: vectors, FTS, session history, scheduled_jobs
```

The `workspace/` directory in the project root is for development. In production, files live at `~/.microwaveos/workspace/` (configurable via `WORKSPACE_DIR` in `.env`).

## Commands

In-chat commands (REPL, Telegram, and Signal all support the skill / project / bible commands):

| Command | Where | Description |
|---------|-------|-------------|
| `/debug` | REPL, Telegram | Triage/search/reflection stats for the last message (includes auto-matched skill) |
| `/new` | REPL, Telegram | Start a fresh session (wipes conversation context, keeps memory) |
| `/memory` | Telegram | Show MEMORY.md contents |
| `/status` | Telegram | Session info, auth mode, model |
| `/skill <name>` | all | Activate a skill (sticky pin — overrides auto-match) |
| `/skill off` | all | Deactivate the pinned skill (also `none`, `clear`) |
| `/skill` | all | Show which skill is currently pinned |
| `/skills` | all | List every available skill |
| `/project <name>` | all | Activate a writing project — auto-loads its skill, joins BIBLE to system prompt |
| `/project status` | all | Detailed status (word count, drafts, BIBLE) |
| `/project off` | all | Deactivate the active project |
| `/projects` | all | List every project with status |
| `/bible add <name> [description]` | all | Append an entry to the active project's BIBLE.md |
| `/bible show` | all | Print the active project's BIBLE.md |
| `exit` / `quit` | REPL | Stop the REPL |

Scheduler CLI (runs without the daemon):

| Command | Description |
|---------|-------------|
| `scheduler list` | Show all jobs with last-run status |
| `scheduler add` | Create a new job (see flags above) |
| `scheduler remove <name>` | Delete a job |
| `scheduler enable <name>` / `disable <name>` | Toggle without deleting |
| `scheduler run <name>` | Fire one job right now |

Skills CLI (runs without the daemon):

| Command | Description |
|---------|-------------|
| `skills list` | Show every skill with its description and fetch-script status |
| `skills show <name>` | Print a skill's full content (frontmatter + body) |
| `skills new <name>` | Scaffold a skill directory with a template SKILL.md |
| `skills edit <name>` | Open the skill's SKILL.md in `$EDITOR` |
| `skills remove <name>` | Delete a skill directory (asks for confirmation) |

Projects CLI (runs without the daemon):

| Command | Description |
|---------|-------------|
| `projects list` | Show every project with type, status, word count |
| `projects show <name>` | Detailed view: drafts, BIBLE/outline status, voice notes |
| `projects new <name> --type blog\|novel\|screenplay` | Scaffold a new project with type-appropriate layout |
| `projects edit <name>` | Open the project's PROJECT.md in `$EDITOR` |
| `projects archive <name>` | Move under `.archived/` (reversible by hand) |
| `projects remove <name>` | Permanently delete a project (asks for confirmation) |

### Telegram file handling

You can send files (documents, photos, text files) to Microwave in Telegram. It reads the content and includes it in the conversation:

- **Text files** (`.txt`, `.md`, `.py`, `.json`, `.csv`, etc.) — content is extracted and included with your message
- **Photos** — acknowledged but content is not visible to the model (see Limitations)
- **Captions** — if you attach a caption with the file, it's used as your message; otherwise Microwave asks what you'd like to do with it

Microwave can also send files back. If a response would be better as a file (code, data, long-form writing), it creates and sends a document rather than pasting a wall of text into chat.

## Limitations

Things this pipeline can't do yet, or does poorly.

### No vision
Images and photos are received but not seen. The pipeline is text-only — there's no multimodal path to pass images through to the LLM. Photos sent in Telegram are acknowledged but the model has no idea what's in them.

### Limited tool use
Microwave can't take actions mid-conversation — no live web search, no code execution, no runtime API calls, no file system access inside the turn. The Agent SDK supports `allowed_tools` but the pipeline currently sets it to `[]`.

The one exception is **scheduler pre-fetch**: a skill can ship a `fetch.py` that the scheduler awaits *before* the LLM call, and the result is prepended to the prompt. That's how the seed `github-tool` skill pulls fresh `gh pr list` data for its weekly digests. It's a batch-mode shortcut, not real tool use — the LLM still can't call things during a turn.

### Single user per channel
The orchestrator holds one session at a time. If multiple Telegram users message the bot, their conversations share the same LLM session and can bleed into each other. Fine for personal use, not safe for multi-user deployment. (Signal's debounce buffer is per-sender, so quick bursts coalesce correctly per user — but the underlying SDK session is still one shared instance.)

### No concurrent requests
Messages are processed sequentially. If two messages arrive at the same time, the second waits for the first to finish. The Signal channel cancels in-flight pipeline calls when the same user sends a new message (streaming interrupt), but doesn't run multiple users' turns in parallel.

### Project search isn't tagged
Active-project drafts are indexed into the same fragment table as global memory. Retrieval doesn't prefer active-project fragments over global ones, so a character named "Sarah" in your novel project might cross-pollinate with a "Sarah" mentioned in a Substack draft. Worth knowing if you have multiple projects with overlapping names.

### Memory is append-only
MEMORY.md and daily notes grow but never shrink automatically. There's no garbage collection, no forgetting, no contradiction resolution. If you tell Microwave your dog's name is Biscuit and later say it's Max, both facts persist. Manual editing is the only cleanup path.

### Triage and reflection are noisy
The Haiku calls for triage and reflection sometimes return malformed JSON or make poor classifications. The pipeline has fallback defaults, but wrong triage (e.g. classifying a recall question as social) means memory search gets skipped when it shouldn't be.

### No streaming on session resume
When the orchestrator resumes a previous session after restart, it replays history as a single primer message. The LLM's response to that primer is consumed silently — but it still costs tokens and time on startup.

### Embedding model requires OpenAI
Vector search uses OpenAI's `text-embedding-3-small`. There's no local embedding option and no Anthropic embedding model. If the OpenAI key is missing or invalid, vector search silently returns nothing and only BM25 keyword search works.

### Voice is one-way
Signal supports voice-in via Whisper transcription, but there's no text-to-speech path for replies and no voice-call support (signal-cli-rest-api is messaging-only). Other channels are text-only.

### Agent SDK auth is machine-local
Max auth works through your Claude Code CLI login session. It doesn't transfer across machines, can't be shared, and expires. If your Claude Code session expires mid-conversation, the next LLM call fails with a 401.
