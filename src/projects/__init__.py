"""Writing projects — per-assignment workspaces.

A project is a named container for one writing assignment (a blog post,
a novel, a screenplay). Holds the work, the project bible, and the
metadata that tells the pipeline how to behave when the project is
active.

See `src.projects.models.Project` for the in-memory shape and
`src.projects.loader.ProjectLoader` for disk discovery.
"""

from src.projects.loader import ProjectLoader, ProjectNotFound
from src.projects.models import Project

__all__ = ["Project", "ProjectLoader", "ProjectNotFound"]
