# Roadmap

Non-obvious structural work worth doing eventually. Not a backlog of bugs or feature requests — those live in commits and conversation. This file is for items where the *why* needs to survive across sessions because the work is preemptive or architectural.

Each item should answer: what is it, what triggers building it, what's the smallest useful version.

---

## Future

### Turn semantics — journal + reflection-as-commit

**What.** Give every turn an explicit start/commit/abort boundary so partial failures can't leak across turns. Two pieces:

1. **Turn journal.** Each turn writes a `started` record before any side effect and a `committed` record after reflection finishes. On startup, scan for orphans (started without committed) and surface them — either reconcile automatically or tell the user *"last turn didn't complete cleanly, here's what I tried to do."*
2. **Reflection-as-commit.** Nothing reaches `MEMORY.md`, daily notes, or project drafts until reflection commits the turn. If reflection fails, the turn is lost cleanly rather than half-remembered.

**Why it matters.** Personal agents fail differently from platforms. There's no DBA to notice state drift — the symptom is the agent slowly getting "weird" over months. It claims to remember things it didn't durably write. Reflection partial-commits leave `MEMORY.md` inconsistent with source fragments. A tool call lies (Substack post fails, but the response already streamed "posted ✓"). Each instance is small; cumulatively they erode the one thing a personal agent has to get right — *the user trusting that its memory of reality matches theirs.*

**When to build.** Not preemptively. Wait until the first concrete instance of *"it claimed to do X and didn't"* or *"its memory and the source disagree."* Build the journal in response to a real bug, not a theoretical one. Doing it now risks designing the wrong shape; doing it after one drift incident gives you a real failure mode to fit the design to.

**Smallest useful version.** Journal + reflection-as-commit. A few hundred lines. Catches ~80% of the drift failure modes.

**Not in scope (yet).** A full obligations/contracts metadata system on tools (cf. IronClaw's obligations model). Tempting, but designing a contract language before knowing which contracts matter is substrate inflation. Revisit only after the journal exists and we've felt the pain of two or three concrete obligation gaps.
