"""Scheduler tests — store CRUD, card-view rendering, fire logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from src.scheduler.card_view import (
    Card,
    plain_text_fallback,
    render_card_view,
    split_into_cards,
)
from src.scheduler.engine import MAX_CATCHUP_MINUTES, Scheduler
from src.scheduler.store import ScheduledJob, SchedulerStore


def _store(tmp_path) -> SchedulerStore:
    s = SchedulerStore(tmp_path / "test.db")
    s.connect()
    return s


def _sample_job(**kw) -> ScheduledJob:
    defaults = dict(
        name="test-job",
        cron_expr="0 9 * * *",
        mode="llm",
        prompt_or_text="Do the thing.",
        target_channel="signal",
        recipient_id="+15551234567",
        timezone="America/New_York",
    )
    defaults.update(kw)
    return ScheduledJob(**defaults)


class TestSchedulerStore:
    def test_add_and_list(self, tmp_path):
        s = _store(tmp_path)
        s.add(_sample_job(name="a"))
        s.add(_sample_job(name="b", mode="direct", prompt_or_text="hello"))
        jobs = s.list_all()
        assert [j.name for j in jobs] == ["a", "b"]
        assert jobs[1].mode == "direct"
        s.close()

    def test_duplicate_name_rejected(self, tmp_path):
        s = _store(tmp_path)
        s.add(_sample_job(name="dup"))
        with pytest.raises(Exception):
            s.add(_sample_job(name="dup"))
        s.close()

    def test_get_by_name(self, tmp_path):
        s = _store(tmp_path)
        s.add(_sample_job(name="foo"))
        assert s.get_by_name("foo").name == "foo"
        assert s.get_by_name("nonexistent") is None
        s.close()

    def test_remove_by_name(self, tmp_path):
        s = _store(tmp_path)
        s.add(_sample_job(name="kill-me"))
        assert s.remove("kill-me") is True
        assert s.get_by_name("kill-me") is None
        assert s.remove("kill-me") is False
        s.close()

    def test_enable_disable(self, tmp_path):
        s = _store(tmp_path)
        s.add(_sample_job(name="toggle"))
        s.set_enabled("toggle", False)
        assert not s.get_by_name("toggle").enabled
        assert s.list_enabled() == []
        s.set_enabled("toggle", True)
        assert s.get_by_name("toggle").enabled
        assert len(s.list_enabled()) == 1
        s.close()

    def test_mark_ran_records_status(self, tmp_path):
        s = _store(tmp_path)
        jid = s.add(_sample_job(name="runner"))
        s.mark_ran(jid, error=None)
        j = s.get_by_name("runner")
        assert j.last_run_at is not None
        assert j.last_error is None
        s.mark_ran(jid, error="boom")
        assert s.get_by_name("runner").last_error == "boom"
        s.close()

    def test_card_view_default_true_for_llm(self, tmp_path):
        s = _store(tmp_path)
        s.add(_sample_job(name="llm", mode="llm", card_view=True))
        s.add(_sample_job(name="direct", mode="direct", card_view=False))
        assert s.get_by_name("llm").card_view is True
        assert s.get_by_name("direct").card_view is False
        s.close()


class TestSplitIntoCards:
    def test_splits_on_separator(self):
        text = "Note 1 — hot take\nBody one here.\n\n---\n\nNote 2 — field note\nBody two."
        cards = split_into_cards(text)
        assert len(cards) == 2
        assert cards[0].label.startswith("Note 1")
        assert cards[1].label.startswith("Note 2")
        assert "Body one" in cards[0].body
        assert "Body two" in cards[1].body

    def test_ignores_dashes_inline(self):
        # A literal --- embedded in prose shouldn't split
        text = "A thing — with a dash — inline, not a separator."
        cards = split_into_cards(text)
        assert len(cards) == 1

    def test_custom_separator(self):
        text = "one\n\n###\n\ntwo\n\n###\n\nthree"
        cards = split_into_cards(text, separator="###")
        assert len(cards) == 3

    def test_empty_chunks_dropped(self):
        text = "first\n\n---\n\n---\n\nsecond"
        cards = split_into_cards(text)
        assert len(cards) == 2

    def test_label_fallback_when_no_header(self):
        text = "just some prose\n\n---\n\nmore prose"
        cards = split_into_cards(text)
        assert cards[0].label == "Item 1"
        assert cards[1].label == "Item 2"

    def test_word_count(self):
        c = Card(label="x", body="one two three four five")
        assert c.word_count == 5


class TestRenderCardView:
    def test_includes_cards(self):
        cards = [Card("Note 1 — hot take", "Body."), Card("Note 2 — essay", "More.")]
        html = render_card_view(cards, title="Substack")
        assert "Note 1 — hot take" in html
        assert "Body." in html
        assert "Note 2 — essay" in html
        assert "<title>Substack</title>" in html

    def test_copy_all_button_only_for_multiple_cards(self):
        # The JS handler and CSS mention 'copy-all' unconditionally — look
        # for the rendered button's <div class="toolbar"> wrapper instead.
        single = render_card_view([Card("x", "y")])
        multi = render_card_view([Card("a", "b"), Card("c", "d")])
        assert '<div class="toolbar">' not in single
        assert '<div class="toolbar">' in multi
        assert ">Copy all</button>" in multi

    def test_escapes_html_in_body(self):
        # User content must never land in the HTML unescaped.
        cards = [Card("x", "<script>evil()</script>")]
        html = render_card_view(cards)
        assert "<script>evil()</script>" not in html
        assert "&lt;script&gt;" in html

    def test_has_copy_script(self):
        html = render_card_view([Card("x", "y"), Card("a", "b")])
        assert "navigator.clipboard" in html
        assert "execCommand" in html  # fallback

    def test_dark_mode_styles(self):
        html = render_card_view([Card("x", "y")])
        assert "prefers-color-scheme: dark" in html

    def test_plain_text_fallback(self):
        cards = [Card("Note 1", "Body 1"), Card("Note 2", "Body 2")]
        plain = plain_text_fallback(cards)
        assert "Note 1" in plain and "Body 1" in plain
        assert "---" in plain
        assert "Note 2" in plain


class TestSchedulerFireLogic:
    """Tests for the schedule evaluation logic.

    We drive the _should_fire check directly with synthetic jobs and
    current times rather than spinning up the poll loop.
    """

    def _scheduler(self, tmp_path) -> Scheduler:
        from src.config import Config
        store = _store(tmp_path)
        cfg = Config(db_path=tmp_path / "test.db")
        return Scheduler(store=store, channels={}, config=cfg)

    def test_fires_when_due(self, tmp_path):
        sch = self._scheduler(tmp_path)
        # Cron: every minute. last_run_at absent, so first run fires promptly.
        job = _sample_job(name="minutely", cron_expr="* * * * *")
        job.id = 1
        job.last_run_at = None
        now = datetime.now(tz=ZoneInfo("UTC"))
        assert sch._should_fire(job, now) is True

    def test_does_not_fire_before_schedule(self, tmp_path):
        sch = self._scheduler(tmp_path)
        # Cron: 6:57am. last_run_at = today 6:57am. Now = same day, 8:00am.
        # Next fire = tomorrow 6:57am, which is in the future.
        job = _sample_job(name="morning", cron_expr="57 6 * * *")
        job.id = 1
        eastern = ZoneInfo("America/New_York")
        now_local = datetime(2026, 4, 23, 8, 0, tzinfo=eastern)
        job.last_run_at = datetime(2026, 4, 23, 6, 57, tzinfo=eastern).astimezone(
            ZoneInfo("UTC")
        )
        assert sch._should_fire(job, now_local.astimezone(ZoneInfo("UTC"))) is False

    def test_stale_run_skipped_and_baseline_set(self, tmp_path):
        """Daemon was offline for a day. On next tick, we should NOT fire the
        backlog — we should update last_run_at to now and wait for the next
        real schedule."""
        sch = self._scheduler(tmp_path)
        job_id = sch.store.add(_sample_job(name="daily", cron_expr="0 9 * * *"))
        job = sch.store.get_by_name("daily")
        job.id = job_id
        # last_run_at was 3 days ago
        job.last_run_at = datetime.now(tz=ZoneInfo("UTC")) - timedelta(days=3)

        now = datetime.now(tz=ZoneInfo("UTC"))
        result = sch._should_fire(job, now)
        assert result is False  # skipped, not fired
        # Baseline should have been written to the store
        refreshed = sch.store.get_by_name("daily")
        assert refreshed.last_run_at is not None
        assert refreshed.last_run_at.tzinfo is not None
        # Within the last minute — proves we set it to "now"
        age_sec = (now - refreshed.last_run_at).total_seconds()
        assert age_sec < 60
        sch.store.close()

    def test_recently_missed_fires(self, tmp_path):
        """If we missed a fire by less than MAX_CATCHUP_MINUTES, still fire."""
        sch = self._scheduler(tmp_path)
        job = _sample_job(name="minutely", cron_expr="* * * * *")
        job.id = 1
        # last_run_at = 2 minutes ago; missed 1 fire
        job.last_run_at = datetime.now(tz=ZoneInfo("UTC")) - timedelta(minutes=2)
        now = datetime.now(tz=ZoneInfo("UTC"))
        assert sch._should_fire(job, now) is True
        assert MAX_CATCHUP_MINUTES >= 2  # sanity on the threshold


class TestFire:
    """End-to-end fire tests with a fake channel. No real LLM calls."""

    class _FakeChannel:
        def __init__(self):
            self.texts: list[tuple[str, str]] = []
            self.attachments: list[tuple[str, str, str | bytes]] = []

        async def send_text(self, recipient: str, text: str):
            self.texts.append((recipient, text))

        async def send_attachment(self, recipient: str, filename: str, content):
            self.attachments.append((recipient, filename, content))

    @pytest.mark.asyncio
    async def test_direct_mode_sends_literal(self, tmp_path):
        from src.config import Config
        store = _store(tmp_path)
        ch = self._FakeChannel()
        sch = Scheduler(
            store=store,
            channels={"signal": ch},
            config=Config(db_path=tmp_path / "test.db"),
        )
        job = _sample_job(
            name="ping",
            mode="direct",
            prompt_or_text="take your vitamins",
        )
        job.id = store.add(job)
        await sch._fire(job)
        assert ch.texts == [("+15551234567", "take your vitamins")]
        assert ch.attachments == []

    @pytest.mark.asyncio
    async def test_llm_mode_card_view_delivers_attachment_only(self, tmp_path, monkeypatch):
        """LLM job with card_view=True and `---`-separated output should
        deliver a one-line header + HTML attachment. No plain-text
        fallback in the body — the attachment IS the content."""
        from src.config import Config
        from src.scheduler import engine as engine_mod
        store = _store(tmp_path)
        ch = self._FakeChannel()
        sch = Scheduler(
            store=store,
            channels={"signal": ch},
            config=Config(db_path=tmp_path / "test.db"),
        )

        # Short-circuit the LLM
        fake_response = (
            "Note 1 — hot take\n"
            "Body of the first.\n\n---\n\n"
            "Note 2 — essay\n"
            "Body of the second.\n\n---\n\n"
            "Note 3 — field note\n"
            "Body of the third."
        )

        async def _fake_run_llm(self, job):
            return fake_response

        monkeypatch.setattr(engine_mod.Scheduler, "_run_llm", _fake_run_llm)

        job = _sample_job(name="substack", mode="llm", card_view=True)
        job.id = store.add(job)
        await sch._fire(job)

        assert len(ch.texts) == 1
        _, body = ch.texts[0]
        assert "substack" in body  # header mentions the job name
        assert "3 items" in body  # header mentions count
        # Crucially, NO plain-text fallback content in the body. The
        # whole point of attachment-only delivery is no double-posting.
        assert "Note 1" not in body
        assert "Note 2" not in body
        assert "Body of the first" not in body

        assert len(ch.attachments) == 1
        _, filename, content = ch.attachments[0]
        assert filename.endswith(".html")
        assert "Note 1" in content
        assert "Note 2" in content
        assert "Note 3" in content
        assert "<title>substack</title>" in content

    @pytest.mark.asyncio
    async def test_llm_mode_single_chunk_sends_plain_text(self, tmp_path, monkeypatch):
        """If the LLM output doesn't split into multiple cards, we shouldn't
        waste an attachment — just send it as a plain message."""
        from src.config import Config
        from src.scheduler import engine as engine_mod
        store = _store(tmp_path)
        ch = self._FakeChannel()
        sch = Scheduler(
            store=store,
            channels={"signal": ch},
            config=Config(db_path=tmp_path / "test.db"),
        )

        async def _fake_run_llm(self, job):
            return "Just one thought, no separators."

        monkeypatch.setattr(engine_mod.Scheduler, "_run_llm", _fake_run_llm)

        job = _sample_job(name="single", mode="llm", card_view=True)
        job.id = store.add(job)
        await sch._fire(job)

        assert ch.attachments == []
        assert len(ch.texts) == 1
        assert ch.texts[0][1] == "Just one thought, no separators."

    @pytest.mark.asyncio
    async def test_unknown_channel_raises(self, tmp_path):
        from src.config import Config
        store = _store(tmp_path)
        sch = Scheduler(
            store=store, channels={}, config=Config(db_path=tmp_path / "test.db")
        )
        job = _sample_job(target_channel="discord")  # not registered
        job.id = 1
        with pytest.raises(RuntimeError, match="No channel registered"):
            await sch._fire(job)

    @pytest.mark.asyncio
    async def test_script_mode_plain_stdout_sends_text(self, tmp_path):
        """Script mode with non-HTML stdout should route through the LLM
        delivery path — single chunk → plain text, no attachment."""
        from src.config import Config
        store = _store(tmp_path)
        ch = self._FakeChannel()
        sch = Scheduler(
            store=store,
            channels={"signal": ch},
            config=Config(db_path=tmp_path / "test.db"),
        )
        job = _sample_job(
            name="echo-job",
            mode="script",
            prompt_or_text="printf 'hello from script'",
            card_view=True,
        )
        job.id = store.add(job)
        await sch._fire(job)

        assert ch.attachments == []
        assert len(ch.texts) == 1
        assert ch.texts[0][1] == "hello from script"

    @pytest.mark.asyncio
    async def test_script_mode_html_doc_ships_as_attachment(self, tmp_path):
        """Script mode whose stdout is a full HTML doc should send a short
        text body plus the HTML as a file attachment."""
        from src.config import Config
        store = _store(tmp_path)
        ch = self._FakeChannel()
        sch = Scheduler(
            store=store,
            channels={"signal": ch},
            config=Config(db_path=tmp_path / "test.db"),
        )
        # printf emits a minimal HTML document to stdout
        html_doc = "<!DOCTYPE html><html><head><title>t</title></head><body>hi</body></html>"
        job = _sample_job(
            name="html-report",
            mode="script",
            prompt_or_text=f"printf '%s' '{html_doc}'",
        )
        job.id = store.add(job)
        await sch._fire(job)

        assert len(ch.texts) == 1
        body = ch.texts[0][1]
        assert "html-report" in body
        assert "attachment" in body.lower()

        assert len(ch.attachments) == 1
        _, filename, content = ch.attachments[0]
        assert filename.endswith(".html")
        assert "<!DOCTYPE html>" in content
        assert "<title>t</title>" in content

    @pytest.mark.asyncio
    async def test_llm_mode_html_doc_ships_as_attachment_only(self, tmp_path, monkeypatch):
        """When the LLM returns a complete HTML document, deliver it as
        attachment only — no duplicate plain-text body. Skills that opt
        into HTML output get full styling control (clickable links, etc.)
        without the card-view fallback content posted alongside."""
        from src.config import Config
        from src.scheduler import engine as engine_mod
        store = _store(tmp_path)
        ch = self._FakeChannel()
        sch = Scheduler(
            store=store,
            channels={"signal": ch},
            config=Config(db_path=tmp_path / "test.db"),
        )

        html_doc = (
            "<!DOCTYPE html><html><head><title>brief</title></head>"
            "<body><h1>Hi</h1><a href='https://example.com'>link</a></body></html>"
        )

        async def _fake_run_llm(self, job):
            return html_doc

        monkeypatch.setattr(engine_mod.Scheduler, "_run_llm", _fake_run_llm)

        job = _sample_job(name="briefing", mode="llm", card_view=True)
        job.id = store.add(job)
        await sch._fire(job)

        # One short header text, NOT the rendered HTML content.
        assert len(ch.texts) == 1
        body = ch.texts[0][1]
        assert "briefing" in body
        assert "attachment" in body.lower()
        # Crucially, the raw HTML must NOT appear inside the message body.
        assert "<a href=" not in body
        assert "<h1>" not in body

        # The HTML rides on the attachment — including the clickable link.
        assert len(ch.attachments) == 1
        _, filename, content = ch.attachments[0]
        assert filename.endswith(".html")
        assert "https://example.com" in content

    @pytest.mark.asyncio
    async def test_script_mode_nonzero_exit_raises(self, tmp_path):
        """Script exiting non-zero should bubble up so `_guarded_fire`
        records the stderr tail in `last_error`."""
        from src.config import Config
        store = _store(tmp_path)
        ch = self._FakeChannel()
        sch = Scheduler(
            store=store,
            channels={"signal": ch},
            config=Config(db_path=tmp_path / "test.db"),
        )
        job = _sample_job(
            name="broken",
            mode="script",
            prompt_or_text="echo oops on stderr 1>&2; exit 7",
        )
        job.id = store.add(job)
        with pytest.raises(RuntimeError, match="exited 7"):
            await sch._fire(job)
        # No delivery on failure
        assert ch.texts == []
        assert ch.attachments == []

    @pytest.mark.asyncio
    async def test_script_mode_guarded_fire_records_error(self, tmp_path):
        """_guarded_fire (not _fire) should catch the RuntimeError and
        persist it to last_error, so the daemon doesn't crash."""
        from src.config import Config
        store = _store(tmp_path)
        ch = self._FakeChannel()
        sch = Scheduler(
            store=store,
            channels={"signal": ch},
            config=Config(db_path=tmp_path / "test.db"),
        )
        job = _sample_job(
            name="broken2",
            mode="script",
            prompt_or_text="exit 3",
        )
        job.id = store.add(job)
        # Should NOT raise — _guarded_fire swallows and records.
        await sch._guarded_fire(job)
        persisted = store.get_by_name("broken2")
        assert persisted is not None
        assert persisted.last_error is not None
        assert "exited 3" in persisted.last_error
