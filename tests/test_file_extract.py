"""Tests for file block extraction from LLM responses."""

from src.pipeline.file_extract import extract_files


class TestExtractFiles:
    def test_no_files(self):
        text = "Here's a normal response with no files."
        cleaned, files = extract_files(text)
        assert cleaned == text
        assert files == []

    def test_single_file(self):
        text = 'Here you go:\n<file name="hello.py">\nprint("hello")\n</file>\nEnjoy!'
        cleaned, files = extract_files(text)
        assert len(files) == 1
        assert files[0].name == "hello.py"
        assert files[0].content == 'print("hello")'
        assert "<file" not in cleaned
        assert "Enjoy!" in cleaned

    def test_multiple_files(self):
        text = (
            'Two files:\n'
            '<file name="a.txt">\nalpha\n</file>\n'
            '<file name="b.txt">\nbeta\n</file>\n'
            'Done.'
        )
        cleaned, files = extract_files(text)
        assert len(files) == 2
        assert files[0].name == "a.txt"
        assert files[0].content == "alpha"
        assert files[1].name == "b.txt"
        assert files[1].content == "beta"
        assert "Done." in cleaned

    def test_multiline_content(self):
        text = '<file name="script.sh">\n#!/bin/bash\necho "hello"\nexit 0\n</file>'
        cleaned, files = extract_files(text)
        assert len(files) == 1
        assert "#!/bin/bash" in files[0].content
        assert "exit 0" in files[0].content

    def test_preserves_indentation(self):
        text = '<file name="code.py">\ndef foo():\n    return 42\n</file>'
        _, files = extract_files(text)
        assert "    return 42" in files[0].content
