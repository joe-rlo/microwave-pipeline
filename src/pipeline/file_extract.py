"""Extract file content from LLM responses.

Three extraction strategies, applied in order:
1. Explicit <file name="...">content</file> tags
2. Fenced code blocks with language hints (```html, ```python, etc.)
   that look like complete files (HTML documents, full scripts)
3. Files written to the output directory by SDK tool use

The pipeline layer handles extraction so we don't depend on the model
reliably using the Write tool or custom tags.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class FileBlock:
    name: str
    content: str


# Match <file name="...">content</file> blocks
_FILE_PATTERN = re.compile(
    r'<file\s+name="([^"]+)">\s*\n?(.*?)\n?\s*</file>',
    re.DOTALL,
)

# Match fenced code blocks: ```lang\ncontent\n```
_CODE_FENCE_PATTERN = re.compile(
    r'```(\w+)\s*\n(.*?)\n\s*```',
    re.DOTALL,
)

# Language hint → file extension
_LANG_EXT = {
    "html": ".html",
    "htm": ".html",
    "python": ".py",
    "py": ".py",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "json": ".json",
    "yaml": ".yaml",
    "yml": ".yaml",
    "css": ".css",
    "sql": ".sql",
    "shell": ".sh",
    "bash": ".sh",
    "sh": ".sh",
    "rust": ".rs",
    "go": ".go",
    "swift": ".swift",
    "kotlin": ".kt",
    "java": ".java",
    "ruby": ".rb",
    "xml": ".xml",
    "toml": ".toml",
    "csv": ".csv",
    "markdown": ".md",
    "md": ".md",
    "svg": ".svg",
    "mermaid": ".html",
}


def extract_files(text: str, channel: str = "") -> tuple[str, list[FileBlock]]:
    """Extract file blocks from response text.

    Returns (cleaned_text, file_blocks) where cleaned_text has the
    file blocks removed and file_blocks contains the extracted files.
    """
    files = []

    # Strategy 1: Explicit <file> tags
    for match in _FILE_PATTERN.finditer(text):
        name = match.group(1).strip()
        content = match.group(2).strip("\n")
        files.append(FileBlock(name=name, content=content))

    if files:
        cleaned = _FILE_PATTERN.sub("", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned, files

    # Strategy 2: Detect complete documents in code fences
    # Only extract as files when on Telegram (where inline rendering is bad)
    # or when the content is clearly a standalone document
    code_fences = list(_CODE_FENCE_PATTERN.finditer(text))
    log.debug(f"Found {len(code_fences)} code fence(s) in response (channel={channel})")
    extracted_spans = []
    for match in code_fences:
        lang = match.group(1).lower()
        content = match.group(2)

        if not _should_extract(lang, content, channel):
            log.debug(f"Skipping code fence: lang={lang}, len={len(content)}")
            continue
        log.debug(f"Extracting code fence as file: lang={lang}, len={len(content)}")

        ext = _LANG_EXT.get(lang, ".txt")
        name = _generate_filename(content, lang, ext, len(files))

        # For mermaid blocks, wrap in HTML
        if lang == "mermaid":
            content = _wrap_mermaid_html(content)
            name = name.replace(".html", "-diagram.html") if ".html" in name else name

        files.append(FileBlock(name=name, content=content))
        extracted_spans.append((match.start(), match.end()))

    if files and extracted_spans:
        # Remove extracted code blocks from text
        cleaned = text
        for start, end in reversed(extracted_spans):
            cleaned = cleaned[:start] + cleaned[end:]
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned, files

    return text, []


def _should_extract(lang: str, content: str, channel: str) -> bool:
    """Decide whether a code block should be extracted as a file."""
    # Always extract complete HTML documents
    if lang in ("html", "htm") and ("<!DOCTYPE" in content or "<html" in content):
        return True

    # Always extract SVG
    if lang == "svg" and "<svg" in content:
        return True

    # Always extract mermaid diagrams
    if lang == "mermaid":
        return True

    # On Telegram: extract large code blocks (>80 lines) and HTML fragments
    if channel == "telegram":
        if lang in ("html", "htm"):
            return True  # any HTML on Telegram
        if content.count("\n") > 80:
            return True  # long code blocks

    return False


def _generate_filename(content: str, lang: str, ext: str, index: int) -> str:
    """Generate a descriptive filename from content."""
    # Try to extract a title from HTML
    title_match = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
        # Sanitize: lowercase, hyphens, no special chars
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        if slug and len(slug) < 60:
            return f"{slug}{ext}"

    # Fallback
    suffix = f"-{index + 1}" if index > 0 else ""
    return f"output{suffix}{ext}"


def _wrap_mermaid_html(mermaid_code: str) -> str:
    """Wrap a Mermaid diagram in a self-contained HTML file."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Diagram</title>
<style>
  body {{
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh; margin: 0; padding: 1rem;
    background: #fff; color: #1a1a1a;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #1a1a1a; color: #e0e0e0; }}
  }}
  .mermaid {{ max-width: 100%; overflow-x: auto; }}
</style>
</head>
<body>
<div class="mermaid">
{mermaid_code}
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{ startOnLoad: true, theme: 'default' }});</script>
</body>
</html>"""
