"""Pipeline 2.3 — skills as pipeline modifiers.

Covers:
- Frontmatter parser: nested `pipeline:` map round-trips
- SkillLoader: pipeline overrides land on Skill, malformed values logged
- Reflection routing: `pipeline.reflection` override beats triage complexity
- `disabled_reflection`: shape matches the model-call path
- PipelineMetadata: skill_overrides survive into the metadata yield
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.frontmatter import split_frontmatter
from src.pipeline.reflection import disabled_reflection, simple_hedge_check
from src.session.models import PipelineMetadata, ReflectionResult
from src.skills import SkillLoader


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestFrontmatterNestedMap:
    """The parser-level change. Skill loader and orchestrator wiring
    both rely on `pipeline:` being parsed as a dict, not a list or string."""

    def test_pipeline_block_becomes_dict(self):
        text = (
            "---\nname: x\ndescription: d\n"
            "pipeline:\n"
            "  reflection: off\n"
            "  max_output_tokens: 4000\n"
            "  escalation_bias: high\n"
            "---\nbody"
        )
        meta, _ = split_frontmatter(text)
        assert isinstance(meta["pipeline"], dict)
        assert meta["pipeline"] == {
            "reflection": "off",
            "max_output_tokens": "4000",
            "escalation_bias": "high",
        }

    def test_list_still_parses_as_list(self):
        """The peek-the-next-line trick must NOT regress list parsing —
        triggers/tags depend on this and rule every existing skill file."""
        text = (
            "---\nname: x\ndescription: d\n"
            "triggers:\n"
            "  - foo\n"
            "  - bar\n"
            "---\nbody"
        )
        meta, _ = split_frontmatter(text)
        assert meta["triggers"] == ["foo", "bar"]

    def test_empty_block_stays_empty_list(self):
        """Backwards compat: a `triggers:` line followed by nothing
        used to parse as `[]`. Must not become `{}` after the change
        or call sites that index into it would break silently."""
        text = "---\nname: x\ntriggers:\n---\nbody"
        meta, _ = split_frontmatter(text)
        assert meta["triggers"] == []

    def test_quoted_subvalues_stripped(self):
        """A user writing `reflection: "off"` to be defensive should
        get the same result as `reflection: off`."""
        text = (
            "---\nname: x\n"
            'pipeline:\n  reflection: "off"\n'
            "---\nbody"
        )
        meta, _ = split_frontmatter(text)
        assert meta["pipeline"] == {"reflection": "off"}


class TestSkillLoaderPipeline:
    def test_pipeline_lands_on_skill(self, tmp_path):
        _write(
            tmp_path / "novel-writing" / "SKILL.md",
            "---\nname: novel-writing\ndescription: d\n"
            "pipeline:\n  reflection: off\n  max_output_tokens: 4000\n"
            "---\nbody",
        )
        s = SkillLoader(tmp_path).load("novel-writing")
        assert s.pipeline == {"reflection": "off", "max_output_tokens": "4000"}

    def test_pipeline_absent_defaults_to_empty_dict(self, tmp_path):
        """Existing skills without a pipeline block must keep working —
        no AttributeError, no orchestrator branching surprises."""
        _write(
            tmp_path / "old" / "SKILL.md",
            "---\nname: old\ndescription: d\n---\nbody",
        )
        s = SkillLoader(tmp_path).load("old")
        assert s.pipeline == {}

    def test_pipeline_non_dict_ignored_with_warning(self, tmp_path, caplog):
        """If someone writes `pipeline: off` (scalar) by mistake, we log
        and ignore — failing the load would break every other skill in
        the directory via list_all()."""
        _write(
            tmp_path / "wrong" / "SKILL.md",
            "---\nname: wrong\ndescription: d\npipeline: off\n---\nbody",
        )
        import logging
        with caplog.at_level(logging.WARNING):
            s = SkillLoader(tmp_path).load("wrong")
        assert s.pipeline == {}
        assert any("pipeline" in r.message for r in caplog.records)


class TestDisabledReflection:
    def test_shape_matches_other_paths(self):
        """`off` must produce a ReflectionResult shaped exactly like the
        model-call path so downstream code (audit, /debug) doesn't have
        to branch on which path produced the result."""
        result = disabled_reflection("Some answer.")
        assert isinstance(result, ReflectionResult)
        assert result.response == "Some answer."
        assert result.action == "deliver"
        assert result.hedging_detected is False
        assert result.confidence == 1.0
        assert result.path == "off"
        assert result.memory_gap is None

    def test_off_path_distinct_from_skipped(self):
        """The /debug audit needs to distinguish `off` (skill suppressed
        it) from `skipped` (simple-tier hedge check ran). Different
        signals, different remediation."""
        off = disabled_reflection("x")
        skipped = simple_hedge_check("x")
        assert off.path != skipped.path
        assert off.path == "off"
        assert skipped.path == "skipped"


class TestPipelineMetadataFields:
    """The metadata yield is the audit trail — pipeline overrides
    must survive into it so /debug can show what shaped the turn."""

    def test_skill_overrides_default_empty(self):
        pm = PipelineMetadata()
        assert pm.skill_overrides == {}

    def test_skill_overrides_carries_dict(self):
        pm = PipelineMetadata(skill_overrides={"reflection": "off"})
        assert pm.skill_overrides == {"reflection": "off"}

    def test_each_pm_instance_owns_its_overrides(self):
        """Default-factory hygiene — two PipelineMetadata instances
        must not share the same dict, otherwise one turn's overrides
        would silently bleed into the next."""
        a = PipelineMetadata()
        b = PipelineMetadata()
        a.skill_overrides["reflection"] = "off"
        assert b.skill_overrides == {}
