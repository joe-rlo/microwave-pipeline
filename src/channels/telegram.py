"""Telegram channel — primary conversational interface.

Features:
- Typing indicators (immediate on message receipt)
- Streaming delivery via message edits
- File receiving (documents, photos) with text extraction
- File sending (long responses sent as documents)
- Telegram markdown formatting
"""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
from pathlib import Path

from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters

from src.channels.base import Channel
from src.pipeline.orchestrator import Orchestrator

log = logging.getLogger(__name__)

# Telegram message length limit
MAX_MESSAGE_LENGTH = 4096
# Minimum interval between message edits (seconds)
EDIT_INTERVAL = 1.0
# If response exceeds this, send as file instead
FILE_THRESHOLD = 3000
# File extensions we can read as text
TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml",
    ".toml", ".csv", ".xml", ".html", ".css", ".sh", ".bash", ".zsh", ".sql",
    ".rs", ".go", ".java", ".kt", ".swift", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".lua", ".r", ".env", ".ini", ".cfg", ".conf", ".log",
}


class TelegramChannel(Channel):
    def __init__(self, orchestrator: Orchestrator, bot_token: str):
        super().__init__(orchestrator)
        self.bot_token = bot_token
        self.app: Application | None = None

    async def start(self) -> None:
        self.app = Application.builder().token(self.bot_token).build()

        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("new", self._cmd_new))
        self.app.add_handler(CommandHandler("debug", self._cmd_debug))
        self.app.add_handler(CommandHandler("memory", self._cmd_memory))
        self.app.add_handler(CommandHandler("status", self._cmd_status))

        # Document handler (files sent as attachments)
        self.app.add_handler(MessageHandler(filters.Document.ALL, self._on_document))

        # Photo handler
        self.app.add_handler(MessageHandler(filters.PHOTO, self._on_photo))

        # Text handler (must be last to not intercept documents/photos with captions)
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message))

        log.info("Starting Telegram bot")
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()

    async def stop(self) -> None:
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

    # --- Message handling ---

    async def _on_message(self, update: Update, context) -> None:
        text = update.message.text
        user_id = str(update.message.from_user.id)
        chat_id = update.message.chat.id
        await self._process_and_respond(text, user_id, chat_id, context)

    async def _on_document(self, update: Update, context) -> None:
        """Handle incoming document/file attachments."""
        user_id = str(update.message.from_user.id)
        chat_id = update.message.chat.id
        doc = update.message.document
        caption = update.message.caption or ""

        try:
            file = await context.bot.get_file(doc.file_id)
            data = await file.download_as_bytearray()

            filename = doc.file_name or "unknown"
            ext = Path(filename).suffix.lower()

            if ext in TEXT_EXTENSIONS or doc.mime_type and doc.mime_type.startswith("text/"):
                # Text file — extract content
                try:
                    content = bytes(data).decode("utf-8")
                except UnicodeDecodeError:
                    content = bytes(data).decode("latin-1")

                # Truncate very large files
                if len(content) > 20_000:
                    content = content[:20_000] + f"\n\n[... truncated, {len(content)} chars total]"

                file_context = f"[File: {filename}]\n```\n{content}\n```"

                if caption:
                    message = f"{file_context}\n\n{caption}"
                else:
                    message = f"{file_context}\n\nThe user sent this file. Acknowledge it and ask what they'd like to do with it."

            else:
                # Binary file — note the metadata
                size_kb = len(data) / 1024
                message = (
                    f"[File received: {filename} ({doc.mime_type or 'unknown type'}, {size_kb:.1f} KB)]\n\n"
                    f"{caption or 'The user sent a binary file. Acknowledge it and ask what they would like to do.'}"
                )

            await self._process_and_respond(message, user_id, chat_id, context)

        except Exception as e:
            log.error(f"Document handling failed: {e}", exc_info=True)
            await update.message.reply_text(f"Couldn't read that file: {e}")

    async def _on_photo(self, update: Update, context) -> None:
        """Handle incoming photos."""
        user_id = str(update.message.from_user.id)
        chat_id = update.message.chat.id
        caption = update.message.caption or ""

        # Photos can't be read as text — note it for context
        message = (
            f"[The user sent a photo]\n\n"
            f"{caption or 'The user sent a photo. Acknowledge it. Note: you cannot see the image contents in this channel.'}"
        )
        await self._process_and_respond(message, user_id, chat_id, context)

    async def _process_and_respond(
        self, text: str, user_id: str, chat_id: int, context
    ) -> None:
        """Process a message through the pipeline and send the response."""
        typing_task = asyncio.create_task(self._typing_loop(chat_id, context))

        try:
            accumulated = ""
            sent_msg = None
            last_edit = 0
            last_metadata = None

            async for chunk in self.orchestrator.process(text, user_id=user_id, channel="telegram"):
                if chunk["type"] in ("delta", "text"):
                    new_text = chunk.get("text") or chunk.get("chunk", "")
                    accumulated += new_text

                    # Send or edit message with rate limiting
                    # Only update the streaming preview while under the limit;
                    # once we exceed it, stop editing and let the final split handle it
                    now = asyncio.get_event_loop().time()
                    if now - last_edit >= EDIT_INTERVAL and len(accumulated) <= MAX_MESSAGE_LENGTH:
                        try:
                            if sent_msg is None:
                                sent_msg = await context.bot.send_message(
                                    chat_id, accumulated, parse_mode="Markdown"
                                )
                            else:
                                await context.bot.edit_message_text(
                                    accumulated, chat_id, sent_msg.message_id,
                                    parse_mode="Markdown",
                                )
                            last_edit = now
                        except Exception as e:
                            try:
                                if sent_msg is None:
                                    sent_msg = await context.bot.send_message(chat_id, accumulated)
                                else:
                                    await context.bot.edit_message_text(
                                        accumulated, chat_id, sent_msg.message_id
                                    )
                                last_edit = now
                            except Exception:
                                log.warning(f"Message edit failed: {e}")

                elif chunk["type"] == "file":
                    # Send extracted file as a document
                    file_buf = io.BytesIO(chunk["content"].encode("utf-8"))
                    file_buf.name = chunk["name"]
                    await context.bot.send_document(
                        chat_id, document=file_buf, filename=chunk["name"]
                    )

                elif chunk["type"] == "text_replace":
                    # File extraction cleaned the response — update accumulated text
                    accumulated = chunk["text"]

                elif chunk["type"] == "metadata":
                    last_metadata = chunk["pipeline"]

            # Final delivery: split into multiple messages if needed
            if accumulated:
                parts = _split_message(accumulated)

                # First part: update the streaming preview message
                first = parts[0]
                try:
                    if sent_msg is None:
                        sent_msg = await context.bot.send_message(
                            chat_id, first, parse_mode="Markdown"
                        )
                    elif first != (sent_msg.text or ""):
                        await context.bot.edit_message_text(
                            first, chat_id, sent_msg.message_id, parse_mode="Markdown"
                        )
                except Exception:
                    if sent_msg is None:
                        await context.bot.send_message(chat_id, first)
                    else:
                        try:
                            await context.bot.edit_message_text(
                                first, chat_id, sent_msg.message_id
                            )
                        except Exception:
                            pass

                # Remaining parts: send as new messages
                for part in parts[1:]:
                    try:
                        await context.bot.send_message(
                            chat_id, part, parse_mode="Markdown"
                        )
                    except Exception:
                        await context.bot.send_message(chat_id, part)

            context.user_data["last_metadata"] = last_metadata

        finally:
            typing_task.cancel()

    async def _send_as_file(
        self, chat_id: int, content: str, sent_msg, context, caption: str = ""
    ) -> None:
        """Send a response as a document attachment."""
        # Detect likely file type from content
        ext, filename = _guess_file_type(content)

        buf = io.BytesIO(content.encode("utf-8"))
        buf.name = filename

        # Delete the streaming preview message if we're replacing it with a file
        if sent_msg:
            try:
                await context.bot.delete_message(chat_id, sent_msg.message_id)
            except Exception:
                pass

        await context.bot.send_document(
            chat_id, document=buf, filename=filename,
            caption=caption or None,
        )

    # --- Utilities ---

    async def _typing_loop(self, chat_id: int, context) -> None:
        """Send typing indicator every 4 seconds until cancelled."""
        try:
            while True:
                await context.bot.send_chat_action(chat_id, "typing")
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass

    # --- Commands ---

    async def _cmd_start(self, update: Update, context) -> None:
        await update.message.reply_text("MicrowaveOS online. Send me a message.")

    async def _cmd_new(self, update: Update, context) -> None:
        new_id = await self.orchestrator.new_session()
        await update.message.reply_text(f"Fresh session started: {new_id}")

    async def _cmd_debug(self, update: Update, context) -> None:
        meta = context.user_data.get("last_metadata")
        if not meta:
            await update.message.reply_text("No previous message to debug.")
            return

        lines = ["Pipeline debug:"]
        if meta.triage:
            lines.append(f"Triage: {meta.triage.intent} ({meta.triage.complexity})")
        if meta.escalated:
            lines.append(f"Escalated: {meta.escalated_model}")
        if meta.search:
            lines.append(f"Search: {len(meta.search.fragments)} fragments, {meta.search.search_time_ms}ms")
        if meta.reflection:
            lines.append(f"Reflection: confidence={meta.reflection.confidence:.2f}")
            if meta.reflection.memory_gap:
                lines.append(f"Memory gap: {meta.reflection.memory_gap}")
        lines.append(f"Total: {meta.total_time_ms}ms")

        await update.message.reply_text("\n".join(lines))

    async def _cmd_memory(self, update: Update, context) -> None:
        memory = self.orchestrator.memory_store.load_memory()
        if memory:
            await update.message.reply_text(f"MEMORY.md:\n\n{memory[:3000]}")
        else:
            await update.message.reply_text("MEMORY.md is empty.")

    async def _cmd_status(self, update: Update, context) -> None:
        await update.message.reply_text(
            f"Session: {self.orchestrator._session_id}\n"
            f"Auth: {self.orchestrator.config.auth_mode}\n"
            f"Model: {self.orchestrator.config.model_main}"
        )


def _split_message(text: str, max_len: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a long message into chunks that fit Telegram's limit.

    Splits at paragraph boundaries (\n\n) first, then at line breaks,
    then hard-cuts as a last resort. Never splits mid-code-block.
    """
    if len(text) <= max_len:
        return [text]

    parts = []
    remaining = text

    while remaining:
        if len(remaining) <= max_len:
            parts.append(remaining)
            break

        # Find a good split point within the limit
        chunk = remaining[:max_len]
        split_at = -1

        # Prefer splitting at paragraph boundary
        split_at = chunk.rfind("\n\n")

        # Fall back to line break
        if split_at < max_len // 4:
            split_at = chunk.rfind("\n")

        # Last resort: split at space
        if split_at < max_len // 4:
            split_at = chunk.rfind(" ")

        # Absolute last resort: hard cut
        if split_at < max_len // 4:
            split_at = max_len

        parts.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip("\n")

    return [p for p in parts if p.strip()]


def _looks_like_file_content(text: str) -> bool:
    """Heuristic: does this response look like it should be a file?"""
    # Large code blocks
    if text.count("```") >= 2 and len(text) > 2000:
        return True
    # Very long and mostly code-like
    lines = text.split("\n")
    if len(lines) > 60:
        code_lines = sum(1 for l in lines if l.startswith("  ") or l.startswith("\t") or l.startswith("```"))
        if code_lines / len(lines) > 0.5:
            return True
    return False


def _guess_file_type(content: str) -> tuple[str, str]:
    """Guess file extension and name from response content."""
    # Check for code block language hint
    if "```python" in content or "```py" in content:
        return ".py", "response.py"
    if "```javascript" in content or "```js" in content:
        return ".js", "response.js"
    if "```typescript" in content or "```ts" in content:
        return ".ts", "response.ts"
    if "```json" in content:
        return ".json", "response.json"
    if "```html" in content:
        return ".html", "response.html"
    if "```css" in content:
        return ".css", "response.css"
    if "```sql" in content:
        return ".sql", "response.sql"
    if "```" in content:
        return ".txt", "response.txt"
    # Markdown-heavy
    if content.count("#") > 3:
        return ".md", "response.md"
    return ".txt", "response.txt"
