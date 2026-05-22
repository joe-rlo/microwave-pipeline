---
name: github
description: >
  Walk the user's GitHub repos and report — list owned repos, summarize
  individual projects, recap recent activity. Auto-activates when the
  user asks about their projects, repos, PRs, or recent GitHub work.
  Requires GITHUB_TOKEN env var; the tools won't be registered without it.
triggers:
  - github
  - repo
  - repos
  - my projects
  - my code
  - pull request
  - pull requests
  - PR
  - PRs
  - recent activity
  - walk my repos
  - what have I been working on
---

# GitHub Walker

## When to use

The user has a GitHub PAT configured and wants you to look at their
projects. Typical phrasings:

- "Walk my repos and tell me what's interesting."
- "What have I been working on lately?"
- "Summarize my [project] repo."
- "What PRs are open across my projects?"
- "Give me a state-of-the-world on my GitHub."

## How to use the tools

You have three tools. Use them in this order for "walk my repos":

1. **`github_list_repos`** — start here. Default sort is `pushed` so the
   most recently active repos surface first. Cap `limit` at 20–30 unless
   the user asks for everything; more than that is noise, not signal.

2. **`github_repo_summary`** — drill into the 2–4 most interesting repos
   from step 1. "Interesting" usually means: recently pushed, has open
   PRs, or the user mentioned it by name. Returns README excerpt,
   recent commits, open PRs/issues, language breakdown — enough to
   write a paragraph without further calls.

3. **`github_recent_activity`** — use when the user explicitly asks
   about timeline ("what did I do this week") rather than a per-repo
   view. Don't fan out to it on every repo question — it's redundant
   with the per-repo `recent_commits` already in `github_repo_summary`.

## What to do with the results

- **Lead with what's notable.** Don't recite a list — pick out what
  matters. "Two repos are getting all your push activity (microwave-os,
  blog); three are stale (last push >6 months); one has open PRs
  waiting on you." That's a useful summary. A flat table is not.
- **Names + links, not SHAs.** Surface repo names and PR/issue URLs
  the user can click. Commit SHAs only matter if the user asks for them.
- **Read across repos.** If three repos all push around the same time
  and reference the same theme, note it. The LLM doing structural
  reading is the point of running this tool at all.

## Anti-patterns

- Don't call `github_repo_summary` on every repo from the list — that's
  10+ tool calls when 3 would do. Prioritize by recency / activity.
- Don't fabricate. If the tool errored or the field is null, say so.
  "No README found" beats inventing a description.
- Don't dump raw JSON into the reply. The tool returns structured data
  for *you* to read; the user wants prose summary + selective links.
- Don't recommend write actions ("you should close issue #42"). The
  PAT is read-only and the user didn't ask you to act, only to report.

## Privacy

- The PAT can see private repos if the user granted that scope. Don't
  paste private code or issue content into channels the user wouldn't
  expect (e.g., a public-facing channel). When in doubt, summarize
  abstractly rather than quoting verbatim.
- Issue and PR titles can contain sensitive info (security reports,
  internal project names). Mention them, but don't broadcast every
  title in the reply if there's a long list — summarize counts and
  highlight 2–3.
