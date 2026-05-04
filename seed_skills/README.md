# Seed skills

Skill bundles shipped with MicrowaveOS that aren't part of any user's
personal workspace by default. Copy any of these into your workspace
to activate it:

```bash
cp -R seed_skills/<skill-name>/ ~/.microwaveos/workspace/skills/<skill-name>/
```

Or use the per-module CLI installer when one exists. For the health
module:

```bash
python3 src/main.py health install-skill
```

The user's actual workspace lives at `~/.microwaveos/workspace/` (or
wherever `WORKSPACE_DIR` in `.env` points). Anything you copy into
`workspace/skills/<name>/` becomes activatable via `/skill <name>` and
participates in adaptive auto-matching by triage.

## Available

- `health-qa/` — citation-only health answers, used by the health
  module's general path. Auto-activates when triage classifies a
  message as health-related (`phi_class != "none"`).
