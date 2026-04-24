"""Skills — named, reusable instruction bundles.

A skill is a markdown file with frontmatter that gets layered into the
pipeline's dynamic context when activated. See `src.skills.loader` for
how skills are discovered and loaded, and `src.skills.models.Skill` for
the in-memory representation.

Skills apply to every channel. When they conflict with channel
formatting rules (message length, markdown syntax), channel rules win —
the assembler adds an explicit note to that effect in the prompt.
"""

from src.skills.loader import SkillLoader, SkillNotFound
from src.skills.models import Skill

__all__ = ["Skill", "SkillLoader", "SkillNotFound"]
