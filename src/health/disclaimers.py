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
