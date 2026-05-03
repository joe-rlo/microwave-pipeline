from src.channels.signal import _ext_for_content_type, _is_voice_note
from src.channels.signal_format import markdown_to_signal_text


def test_preserves_native_markdown():
    # Signal renders these natively — bold/italic/code pass through
    # untouched. `~` is the exception: it's now neutralized with a
    # zero-width space (see TestStrikethroughNeutralization for why).
    s = "**bold** and *italic* and `code`"
    assert markdown_to_signal_text(s) == s


def test_underscore_italic_passes_through():
    assert markdown_to_signal_text("_also italic_") == "_also italic_"


def test_header_becomes_bold():
    assert markdown_to_signal_text("## Section") == "**Section**"
    assert markdown_to_signal_text("# Big") == "**Big**"


def test_link_flattened_to_text_and_url():
    out = markdown_to_signal_text("See [the docs](https://example.com) for more")
    assert out == "See the docs (https://example.com) for more"


def test_fenced_code_block_stripped():
    out = markdown_to_signal_text("```python\nprint('hi')\n```")
    assert "```" not in out
    assert "print('hi')" in out


def test_fenced_code_block_no_lang():
    out = markdown_to_signal_text("```\nraw\nthing\n```")
    assert "```" not in out
    assert "raw\nthing" in out


def test_html_tags_stripped():
    # If a Telegram-flavored string leaks through, strip HTML so the user
    # doesn't see literal <b>...</b>.
    out = markdown_to_signal_text("<b>bold</b> and <i>italic</i>")
    assert "<" not in out and ">" not in out
    assert "bold" in out and "italic" in out


def test_horizontal_rule_removed():
    out = markdown_to_signal_text("before\n\n---\n\nafter")
    assert "---" not in out
    assert "before" in out and "after" in out


def test_table_becomes_card_layout():
    md = """Intro.

| Model | Latency | Cost |
|-------|---------|------|
| Opus | 1.2s | $0.015 |
| Sonnet | 0.8s | $0.003 |

Outro."""
    out = markdown_to_signal_text(md)
    assert "**Model:** Opus" in out
    assert "**Latency:** 1.2s" in out
    assert "**Cost:** $0.015" in out
    assert "**Model:** Sonnet" in out
    # Raw pipe-table syntax is gone
    assert "|---" not in out and "| Model |" not in out
    # Prose around the table survives
    assert "Intro." in out
    assert "Outro." in out


def test_table_with_empty_cells_skipped():
    md = """| A | B |
|---|---|
| x |   |
|   | y |"""
    out = markdown_to_signal_text(md)
    assert "**A:** x" in out
    assert "**B:** y" in out


def test_inline_code_with_asterisks_left_alone():
    # `` `code` `` passes through; we don't try to re-parse its contents.
    out = markdown_to_signal_text("use `**not bold**` here")
    assert "`**not bold**`" in out


def test_excessive_blank_lines_collapsed():
    out = markdown_to_signal_text("a\n\n\n\n\nb")
    assert out == "a\n\nb"


class TestVoiceNoteDetection:
    def test_flagged_voice_note(self):
        assert _is_voice_note({"voiceNote": True, "contentType": "audio/aac"})

    def test_audio_without_flag_still_voice(self):
        # Some clients send audio/* without the voiceNote flag — treat as voice
        assert _is_voice_note({"contentType": "audio/aac"})
        assert _is_voice_note({"contentType": "audio/mpeg"})

    def test_non_audio_attachment(self):
        assert not _is_voice_note({"contentType": "image/jpeg"})
        assert not _is_voice_note({"contentType": "application/pdf"})
        assert not _is_voice_note({})

    def test_content_type_case_insensitive(self):
        assert _is_voice_note({"contentType": "AUDIO/AAC"})


class TestExtForContentType:
    def test_aac_to_m4a(self):
        # Signal's default voice-note format — must be .m4a for Whisper.
        assert _ext_for_content_type("audio/aac") == ".m4a"

    def test_ogg_opus(self):
        assert _ext_for_content_type("audio/opus") == ".ogg"
        assert _ext_for_content_type("audio/ogg") == ".ogg"

    def test_mp3(self):
        assert _ext_for_content_type("audio/mpeg") == ".mp3"

    def test_strips_params(self):
        assert _ext_for_content_type("audio/aac; charset=utf-8") == ".m4a"

    def test_unknown_defaults_to_m4a(self):
        # Safer default — most Signal voice notes are AAC-in-M4A.
        assert _ext_for_content_type("audio/weird-format") == ".m4a"


class TestStrikethroughNeutralization:
    """Signal's styled-mode parser pairs `~text~` as strikethrough. The LLM
    routinely uses `~` for approximation and home-dir paths and never wants
    actual strike, so we neutralize every tilde with a trailing ZWSP."""

    def test_approximation_tildes_no_longer_pair(self):
        # Classic case: two `~` in the same message would strike everything
        # between them. After neutralization, the pairing breaks.
        out = markdown_to_signal_text("spend ~5 minutes, then ~10 minutes more")
        # Both tildes still visible (with ZWSP after each — invisible in render)
        assert out.count("~") == 2
        # The pair `~5 minutes, then ~` is broken: no matching `text~text~`
        # regex can match across these now.
        import re
        assert not re.search(r"~[^~​]+~", out)

    def test_path_tildes_no_longer_pair(self):
        out = markdown_to_signal_text(
            "files at ~/.microwaveos and backups at ~/Documents/archive"
        )
        assert out.count("~") == 2
        import re
        assert not re.search(r"~[^~​]+~", out)

    def test_intentional_strike_also_defused(self):
        # Trade-off: we currently defuse intentional strike too. The bot
        # never emits this, but if a user includes `~old~` in their input
        # it won't survive the conversion. Acceptable for v1; document
        # via this test so future contributors see the intent.
        out = markdown_to_signal_text("the ~old~ approach was wrong")
        import re
        assert not re.search(r"~[^~​]+~", out)

    def test_zwsp_invisible_to_humans(self):
        # The neutralizer adds U+200B (zero-width space) — verify that's
        # what we're using and not some visible char.
        out = markdown_to_signal_text("path: ~/foo")
        assert "​" in out
        # When stripped of ZWSPs, the user-facing text is unchanged.
        visible = out.replace("​", "")
        assert visible == "path: ~/foo"

    def test_no_tildes_means_no_zwsps(self):
        # Don't pollute messages that don't need defusing.
        out = markdown_to_signal_text("plain old message, nothing to see")
        assert "​" not in out

    def test_inside_code_block_still_neutralized(self):
        # We don't try to be clever about preserving inline-code content
        # untouched — `~` inside `code` would still pair if we left it.
        # Better to neutralize uniformly.
        out = markdown_to_signal_text("use `rm ~/tmp` then `rm ~/cache`")
        import re
        # Match what's inside the backticks too — no surviving pair
        assert not re.search(r"~[^~​]+~", out)


def test_realistic_bot_response():
    md = """**What's new:**

- The pipeline now indexes summaries
- The `_compact()` method yields events

## Next steps

| Step | File |
|------|------|
| A | orchestrator.py |
| B | search.py |

See [the PR](https://github.com/x/y/pull/1) for details."""
    out = markdown_to_signal_text(md)
    # Headers → bold
    assert "**Next steps**" in out
    # Table → cards
    assert "**Step:** A" in out
    assert "**File:** orchestrator.py" in out
    # Link flattened
    assert "the PR (https://github.com/x/y/pull/1)" in out
    # Native markdown preserved
    assert "**What's new:**" in out
    assert "`_compact()`" in out
    # No HTML, no raw tables, no link markup
    assert "<" not in out
    assert "](" not in out
    assert "|---" not in out
