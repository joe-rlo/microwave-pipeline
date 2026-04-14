"""Extract file blocks from LLM responses.

The LLM can embed file content in responses using:
    <file name="example.py">
    content here
    </file>

This module extracts those blocks so the channel layer can deliver them
as actual files (Telegram documents, REPL file writes, HTTP attachments).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class FileBlock:
    name: str
    content: str


# Match <file name="...">content</file> blocks
_FILE_PATTERN = re.compile(
    r'<file\s+name="([^"]+)">\s*\n?(.*?)\n?\s*</file>',
    re.DOTALL,
)


def extract_files(text: str) -> tuple[str, list[FileBlock]]:
    """Extract file blocks from response text.

    Returns (cleaned_text, file_blocks) where cleaned_text has the
    file blocks removed and file_blocks contains the extracted files.
    """
    files = []
    for match in _FILE_PATTERN.finditer(text):
        name = match.group(1).strip()
        content = match.group(2)
        # Strip leading/trailing blank lines but preserve internal formatting
        content = content.strip("\n")
        files.append(FileBlock(name=name, content=content))

    if not files:
        return text, []

    # Remove file blocks from text, clean up extra blank lines
    cleaned = _FILE_PATTERN.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned, files
