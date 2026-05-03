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
import re
import tempfile
from pathlib import Path

from telegram import Update
from telegram.error import BadRequest
from telegram.ext import Application, MessageHandler, CommandHandler, filters

from src.channels.base import Channel
from src.channels.telegram_format import markdown_to_telegram_html
from src.pipeline.orchestrator import Orchestrator
from src.projects.bible import handle_bible_command
from src.projects.chat import handle_project_command
from src.skills.chat import handle_skill_command

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
        self.app.add_handler(CommandHandler("stats", self._cmd_stats))
        self.app.add_handler(CommandHandler("skill", self._cmd_skill))
        self.app.add_handler(CommandHandler("skills", self._cmd_skills))
        self.app.add_handler(CommandHandler("project", self._cmd_project))
        self.app.add_handler(CommandHandler("projects", self._cmd_projects))
        self.app.add_handler(CommandHandler("bible", self._cmd_bible))

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

    # --- Outbound API (used by scheduler and other non-interactive callers) ---

    async def send_text(self, recipient: str, text: str) -> None:
        """Send a message to a chat_id. `recipient` can be a string or int;
        Telegram accepts either for chat_id."""
        if not self.app:
            raise RuntimeError("TelegramChannel not started")
        formatted = markdown_to_telegram_html(text)
        try:
            await self.app.bot.send_message(
                chat_id=recipient, text=formatted, parse_mode="HTML"
            )
        except BadRequest:
            # HTML render failed — fall back to plain
            await self.app.bot.send_message(chat_id=recipient, text=_strip_tags(formatted))

    async def send_attachment(
        self, recipient: str, filename: str, content: str | bytes
    ) -> None:
        if not self.app:
            raise RuntimeError("TelegramChannel not started")
        data = content.encode("utf-8") if isinstance(content, str) else content
        buf = io.BytesIO(data)
        buf.name = filename
        await self.app.bot.send_document(
            chat_id=recipient, document=buf, filename=filename
        )

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
        """Handle incoming photos.

        Downloads the highest-resolution variant Telegram offers and
        passes it through the pipeline as a multimodal content block.
        Caption (if any) becomes the text portion; an empty caption
        gets a small prompt so the model has something to respond to.
        """
        user_id = str(update.message.from_user.id)
        chat_id = update.message.chat.id
        caption = update.message.caption or ""

        # Telegram exposes photos at multiple resolutions; the last
        # entry is the largest (also the most useful for vision).
        try:
            best = update.message.photo[-1]
            tg_file = await context.bot.get_file(best.file_id)
            data = await tg_file.download_as_bytearray()
            image_bytes = bytes(data)
        except Exception as e:
            log.warning(f"Failed to download photo: {e}")
            await update.message.reply_text(
                f"Couldn't download that photo: {e}"
            )
            return

        text = caption or "What's in this photo?"
        await self._process_and_respond(
            text, user_id, chat_id, context,
            images=[(image_bytes, "image/jpeg")],
        )

    async def _process_and_respond(
        self,
        text: str,
        user_id: str,
        chat_id: int,
        context,
        images: list[tuple[bytes, str]] | None = None,
    ) -> None:
        """Process a message through the pipeline and send the response."""
        typing_task = asyncio.create_task(self._typing_loop(chat_id, context))

        try:
            accumulated = ""
            sent_msg = None
            last_edit = 0
            last_rendered = ""
            last_metadata = None

            async for chunk in self.orchestrator.process(
                text, user_id=user_id, channel="telegram", images=images,
            ):
                if chunk["type"] in ("delta", "text"):
                    new_text = chunk.get("text") or chunk.get("chunk", "")
                    accumulated += new_text

                    # Send or edit message with rate limiting
                    # Only update the streaming preview while under the limit;
                    # once we exceed it, stop editing and let the final split handle it
                    now = asyncio.get_event_loop().time()
                    if now - last_edit >= EDIT_INTERVAL:
                        preview = markdown_to_telegram_html(accumulated)
                        if len(preview) <= MAX_MESSAGE_LENGTH and preview != last_rendered:
                            ok = await self._send_or_edit_html(
                                context, chat_id, sent_msg, preview, accumulated
                            )
                            if ok is not None:
                                sent_msg = ok
                                last_rendered = preview
                            last_edit = now

                elif chunk["type"] == "file":
                    # Send extracted file as a document
                    file_buf = io.BytesIO(chunk["content"].encode("utf-8"))
                    file_buf.name = chunk["name"]
                    await context.bot.send_document(
                        chat_id, document=file_buf, filename=chunk["name"]
                    )

                elif chunk["type"] == "file_written":
                    # Active-project file landed on disk — send a short note.
                    name = chunk.get("name", "?")
                    words = chunk.get("word_count", 0)
                    preview = (chunk.get("preview") or "").split("\n", 1)[0]
                    body = f"<i>✓ wrote <code>{name}</code> ({words:,} words)</i>"
                    if preview:
                        body += f"\n\n<blockquote>{preview}</blockquote>"
                    try:
                        await context.bot.send_message(chat_id, body, parse_mode="HTML")
                    except Exception as e:
                        log.debug(f"file_written note failed: {e}")

                elif chunk["type"] == "text_replace":
                    # File extraction cleaned the response — update accumulated text
                    accumulated = chunk["text"]

                elif chunk["type"] == "compaction":
                    # Surface compaction to the user — the LLM's live context
                    # just got reset, so telling them avoids the "why did you
                    # forget?" confusion.
                    n_compacted = chunk.get("turns_compacted", 0)
                    n_kept = chunk.get("turns_kept", 0)
                    note = (
                        f"<i>↻ Archived {n_compacted} earlier turn"
                        f"{'s' if n_compacted != 1 else ''} "
                        f"(summary saved, last {n_kept} kept in context).</i>"
                    )
                    try:
                        await context.bot.send_message(chat_id, note, parse_mode="HTML")
                    except Exception as e:
                        log.debug(f"Failed to send compaction note: {e}")

                elif chunk["type"] == "metadata":
                    last_metadata = chunk["pipeline"]

            # Final delivery: render as HTML (tables become card-layout
            # blocks inline), split into messages if needed.
            if accumulated:
                final_html = markdown_to_telegram_html(accumulated)
                parts = _split_message(final_html)

                # First part: update the streaming preview message (or send it
                # fresh if we never did a streaming edit). Skip if the streaming
                # preview already matches — re-editing to the same HTML would
                # raise "message is not modified" and hit the fallback.
                first = parts[0]
                if first != last_rendered:
                    ok = await self._send_or_edit_html(
                        context, chat_id, sent_msg, first, accumulated
                    )
                    if ok is not None:
                        sent_msg = ok
                        last_rendered = first

                # Remaining parts: send as new messages
                for part in parts[1:]:
                    try:
                        await context.bot.send_message(
                            chat_id, part, parse_mode="HTML"
                        )
                    except BadRequest as e:
                        log.warning(f"HTML send failed, sending as plain text: {e}")
                        await context.bot.send_message(chat_id, _strip_tags(part))

            context.user_data["last_metadata"] = last_metadata

        finally:
            typing_task.cancel()

    async def _send_or_edit_html(
        self, context, chat_id: int, sent_msg, html_text: str, raw_fallback: str
    ):
        """Send or edit a message as HTML. Returns the message object on
        success, the original `sent_msg` when Telegram reports no change,
        or None on unrecoverable failure.

        On HTML parse errors we fall back to plain text *without* the raw
        markdown — otherwise unparsed `*asterisks*` end up in the chat.
        """
        try:
            if sent_msg is None:
                return await context.bot.send_message(
                    chat_id, html_text, parse_mode="HTML"
                )
            await context.bot.edit_message_text(
                html_text, chat_id, sent_msg.message_id, parse_mode="HTML"
            )
            return sent_msg
        except BadRequest as e:
            msg = str(e).lower()
            if "not modified" in msg:
                return sent_msg  # Already shows this content, nothing to do.
            log.warning(f"HTML send/edit failed ({e}); retrying as plain text")
            plain = _strip_tags(html_text) or raw_fallback
            try:
                if sent_msg is None:
                    return await context.bot.send_message(chat_id, plain)
                await context.bot.edit_message_text(
                    plain, chat_id, sent_msg.message_id
                )
                return sent_msg
            except BadRequest as e2:
                if "not modified" in str(e2).lower():
                    return sent_msg
                log.warning(f"Plain-text fallback also failed: {e2}")
                return None

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
            if meta.triage.matched_skill:
                lines.append(f"Matched skill: {meta.triage.matched_skill}")
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

    async def _cmd_stats(self, update: Update, context) -> None:
        user_id = str(update.message.from_user.id)
        stats = self.orchestrator.get_pipeline_stats(user_id=user_id, channel="telegram")

        n = stats["turns_sampled"]
        if n == 0:
            await update.message.reply_text("No pipeline data yet.")
            return

        intent_dist = stats["intent_distribution"]
        top_intents = sorted(intent_dist.items(), key=lambda x: -x[1])
        intent_str = ", ".join(f"{k} ({v})" for k, v in top_intents) or "—"

        lines = [
            f"Pipeline stats — last {n} turns",
            f"Intents: {intent_str}",
            f"Avg search fragments: {stats['avg_search_fragments']:.1f}",
            f"Avg reflection confidence: {stats['avg_reflection_confidence']:.2f}",
        ]
        await update.message.reply_text("\n".join(lines))

    async def _cmd_skill(self, update: Update, context) -> None:
        # Telegram strips the leading `/skill`; context.args has the rest.
        arg = " ".join(context.args) if context.args else ""
        # Rebuild the full command string so the shared parser sees it
        # in the same shape as REPL/Signal.
        text = f"/skill {arg}".rstrip()
        reply = await handle_skill_command(text, self.orchestrator)
        if reply:
            await update.message.reply_text(reply)

    async def _cmd_skills(self, update: Update, context) -> None:
        reply = await handle_skill_command("/skills", self.orchestrator)
        if reply:
            await update.message.reply_text(reply)

    async def _cmd_project(self, update: Update, context) -> None:
        arg = " ".join(context.args) if context.args else ""
        text = f"/project {arg}".rstrip()
        reply = await handle_project_command(text, self.orchestrator)
        if reply:
            await update.message.reply_text(reply)

    async def _cmd_projects(self, update: Update, context) -> None:
        reply = await handle_project_command("/projects", self.orchestrator)
        if reply:
            await update.message.reply_text(reply)

    async def _cmd_bible(self, update: Update, context) -> None:
        arg = " ".join(context.args) if context.args else ""
        text = f"/bible {arg}".rstrip()
        reply = await handle_bible_command(text, self.orchestrator)
        if reply:
            await update.message.reply_text(reply)


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(html_text: str) -> str:
    """Remove HTML tags for plain-text fallback. Also decodes &amp;/&lt;/&gt;."""
    text = _TAG_RE.sub("", html_text)
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


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
