"""Channel-aware disclaimers and footer text for health-routed turns.

The orchestrator splices the disclaimer into the dynamic context when
`route.require_disclaimer` is True, so the LLM can append it to the
response naturally rather than the channel layer trying to bolt text
onto the bot's reply after the fact.

Default text lives here as a constant so a stock install works
out-of-the-box. If the user drops a `workspace/channels/health.md`
file, that overrides the default — the same opt-in/customization
pattern as channel rules elsewhere in the system.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


# Default disclaimer block. Becomes additive prompt content when a
# health route fires. The PHI-path note is conditional on
# `use_baa_llm` so the user knows when their query is going through
# the privacy-protected path vs. the standard one.
DEFAULT_HEALTH_DISCLAIMER = """\
# Health channel rules

When responding to a health-classified message:

- Append a single-line footer to every response: "Information only,
  not medical advice. For anything urgent, contact a clinician or
  call emergency services."
- For PHI-routed responses, also append: "This response was generated
  using a privacy-protected processing path."
- Never include the user's specific identifying details in the
  response unless they are essential to the answer. Refer back as
  "your situation" or "what you described" when possible.
"""


# Spliced into the dynamic context when the general path runs but
# retrieval came back empty — explicitly carves out the health-qa
# skill's "use only provided evidence" rule for benign generic
# questions, while keeping the safety floors (no diagnosis, no dose,
# emergency framing). Without this, the bot's honest "no sources
# found" reply makes it feel broken on questions every clinician
# could answer off the cuff ("does drug X cause heartburn?").
EMPTY_RETRIEVAL_RELAXATION = """\
[Empty retrieval — relaxed evidence rule for THIS turn ONLY]

The health-qa skill's "use only provided evidence" rule has a
deliberate carve-out for the case retrieval came back empty on a
benign general-path question — which is what just happened.

For THIS turn you MAY:
- Answer briefly from established medical knowledge — 1 to 3
  sentences. Don't pad.
- Open with one line acknowledging the gap, e.g. "I didn't get
  specific sources back, but here's the established read:"
- Stick to widely-known, low-stakes content: drug classes, common
  side effects, basic mechanisms, general health concepts a
  pharmacist or primary-care clinician would recite without looking
  anything up.

You MUST still:
- Refuse specific dose recommendations.
- Refuse to start, stop, or change medications for anyone.
- Lead with emergency framing for emergencies (chest pain, stroke
  symptoms, suicidal ideation, anaphylaxis, severe bleeding,
  pregnancy bleeding). Do not soften an emergency into trivia.
- Decline to engage if the question is high-stakes (specific
  contraindications for a real patient, dosing for a real case,
  definitive diagnosis, mental-health crisis). For those,
  "no specific sources retrieved; please ask your clinician" remains
  the correct response — don't extrapolate.
- Apply the channel disclaimer footer."""


def load_health_channel_rules(workspace_dir: Path) -> str:
    """Return the health-channel disclaimer block.

    If `<workspace>/channels/health.md` exists, return its contents
    (user override); otherwise the bundled default. The override path
    matches the existing per-channel customization pattern — users
    edit channel files in their workspace, no code change required.
    """
    custom = workspace_dir / "channels" / "health.md"
    if custom.is_file():
        try:
            text = custom.read_text(encoding="utf-8").strip()
            if text:
                return text
        except OSError as e:
            log.warning(f"Failed to read {custom}: {e}; using default disclaimer")
    return DEFAULT_HEALTH_DISCLAIMER
