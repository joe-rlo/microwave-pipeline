from src.channels.telegram_format import markdown_to_telegram_html


def test_bold_double_asterisk():
    assert markdown_to_telegram_html("**What's new:**") == "<b>What's new:</b>"


def test_italic_single_asterisk():
    assert markdown_to_telegram_html("some *italic* text") == "some <i>italic</i> text"


def test_bold_beats_italic():
    # ** shouldn't also get interpreted as a pair of single *
    assert markdown_to_telegram_html("**bold**") == "<b>bold</b>"


def test_underscores_in_identifiers_not_italicized():
    out = markdown_to_telegram_html("use the my_var_name function")
    assert "<i>" not in out
    assert "my_var_name" in out


def test_inline_code():
    out = markdown_to_telegram_html("call `foo()` now")
    assert out == "call <code>foo()</code> now"


def test_code_block_with_lang():
    out = markdown_to_telegram_html("```python\nprint('hi')\n```")
    assert '<pre><code class="language-python">' in out
    assert "print(&#x27;hi&#x27;)" in out or "print('hi')" in out


def test_code_block_no_lang():
    out = markdown_to_telegram_html("```\nraw text\n```")
    assert out.startswith("<pre>")
    assert "raw text" in out


def test_escapes_html_chars_in_prose():
    out = markdown_to_telegram_html("a < b & c > d")
    assert "&lt;" in out and "&gt;" in out and "&amp;" in out


def test_does_not_escape_inside_code():
    out = markdown_to_telegram_html("`<html>`")
    assert "&lt;html&gt;" in out  # code content still escapes for HTML safety
    assert "<code>" in out


def test_header_becomes_bold():
    assert markdown_to_telegram_html("## Section title") == "<b>Section title</b>"


def test_link():
    assert (
        markdown_to_telegram_html("[docs](https://example.com)")
        == '<a href="https://example.com">docs</a>'
    )


def test_asterisks_inside_code_not_formatted():
    out = markdown_to_telegram_html("`**not bold**`")
    assert "<b>" not in out
    assert "<code>**not bold**</code>" in out


def test_table_becomes_card_layout():
    md = """Intro paragraph.

| Name | Age | City |
|------|-----|------|
| Alice | 30 | NYC |
| Bob | 25 | LA |

Outro paragraph."""
    out = markdown_to_telegram_html(md)
    # Cards for each data row, labeled by header
    assert "<b>Name:</b> Alice" in out
    assert "<b>Age:</b> 30" in out
    assert "<b>City:</b> NYC" in out
    assert "<b>Name:</b> Bob" in out
    assert "<b>Age:</b> 25" in out
    # No raw pipe-table syntax survives
    assert "|" not in out or out.count("|") < 2
    # Prose around the table is preserved
    assert "Intro paragraph" in out
    assert "Outro paragraph" in out


def test_multiple_tables_each_become_cards():
    md = """| a | b |
|---|---|
| 1 | 2 |

Middle.

| c | d |
|---|---|
| 3 | 4 |"""
    out = markdown_to_telegram_html(md)
    assert "<b>a:</b> 1" in out
    assert "<b>b:</b> 2" in out
    assert "<b>c:</b> 3" in out
    assert "<b>d:</b> 4" in out
    assert "Middle" in out


def test_pipes_in_inline_code_not_parsed_as_table():
    md = "no tables here, just `| pipes |` inline"
    out = markdown_to_telegram_html(md)
    assert "<code>| pipes |</code>" in out
    assert "<b>" not in out  # no card bolding happened


def test_full_screenshot_scenario():
    """The exact kind of output that broke before: ** bold **, `code`, dashes."""
    md = """Good read. Solid update — here's what I notice:

**What's new / different:**

- The Italian/Barese dialect context is more specific now
- The humor line about banter got sharpened: *"If I'm not giving him a little shit, something's broken."*
- NEAR Protocol context is more specific now.

**One thing that's slightly off:** The `How you work` section is nested."""
    out = markdown_to_telegram_html(md)
    assert "<b>What's new / different:</b>" in out
    assert "<b>One thing that's slightly off:</b>" in out
    assert "<i>" in out  # the italicized quote
    assert "<code>How you work</code>" in out
    assert "**" not in out  # no stray asterisks
