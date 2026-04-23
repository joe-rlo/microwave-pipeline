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
| Signal | Typing indicators, read receipts, voice-message transcription (Whisper), HTML card-view attachments |
| HTTP | `POST /chat` returns JSON with response + pipeline stats |

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
# Create an LLM job — prompt stored in a file
python3 src/main.py scheduler add \
  --name substack-notes \
  --cron "57 6 * * *" \
  --mode llm \
  --channel signal \
  --recipient "+15551234567" \
  --prompt-file prompts/substack-notes.txt

# Create a direct reminder — no LLM
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
│   └── memory/            # daily notes
└── data/
    └── memory.db          # SQLite: vectors, FTS, session history, scheduled_jobs
```

The `workspace/` directory in the project root is for development. In production, files live at `~/.microwaveos/workspace/` (configurable via `WORKSPACE_DIR` in `.env`).

## Commands

In the REPL:

| Command | Description |
|---------|-------------|
| `/debug` | Show triage, search, and reflection stats for last message |
| `/new` | Start a fresh session (wipes conversation context, keeps memory) |
| `exit` / `quit` | Stop the REPL |

In Telegram:

| Command | Description |
|---------|-------------|
| `/new` | Start a fresh session |
| `/debug` | Pipeline stats for last message |
| `/memory` | Show MEMORY.md contents |
| `/status` | Session info, auth mode, model |

Scheduler CLI (runs without the daemon):

| Command | Description |
|---------|-------------|
| `scheduler list` | Show all jobs with last-run status |
| `scheduler add` | Create a new job (see flags above) |
| `scheduler remove <name>` | Delete a job |
| `scheduler enable <name>` / `disable <name>` | Toggle without deleting |
| `scheduler run <name>` | Fire one job right now |

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

### No tool use
Microwave can't take actions — no web search, no code execution, no API calls, no file system access. It's a conversational agent with memory, not an agentic tool-user. The Agent SDK supports `allowed_tools` but the pipeline currently sets it to `[]`.

### Single user per channel
The orchestrator holds one session at a time. If multiple Telegram users message the bot, their conversations share the same LLM session and can bleed into each other. Fine for personal use, not safe for multi-user deployment.

### No concurrent requests
Messages are processed sequentially. If two messages arrive at the same time (e.g. Telegram group chat), the second waits for the first to finish. No request queue or parallelism.

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
