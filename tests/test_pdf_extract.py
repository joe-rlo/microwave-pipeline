"""Tests for PDF text extraction (src/channels/_pdf.py).

Covers the text-layer path and graceful degradation. The OCR fallback is
not exercised here — it requires a network call to OpenAI — but the
no-text-layer-without-key and malformed-input cases confirm we degrade to
`method="none"` instead of raising, which is what the channels rely on.
"""

import pymupdf
import pytest

from src.channels._pdf import extract_pdf_text


def _make_pdf(text: str, pages: int = 1) -> bytes:
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        # insert_textbox handles wrapping/newlines and keeps text on-page.
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def _blank_pdf(pages: int = 1) -> bytes:
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


class TestPdfExtract:
    @pytest.mark.asyncio
    async def test_text_layer_extraction(self):
        data = _make_pdf(
            "COMPREHENSIVE METABOLIC PANEL\n"
            "Glucose 92 mg/dL (ref 70-99)\n"
            "Sodium 140 mmol/L (ref 135-145)\n"
            "Potassium 4.3 mmol/L (ref 3.5-5.1)\n"
            "Creatinine 0.9 mg/dL (ref 0.7-1.3)\n"
            "Vitamin D 38 ng/mL (ref 30-100)\n"
        )
        r = await extract_pdf_text(data, openai_api_key="")
        assert r.method == "text"
        assert r.ok
        assert r.page_count == 1
        assert "Glucose" in r.text and "Vitamin D" in r.text
        assert not r.char_truncated
        assert not r.page_truncated

    @pytest.mark.asyncio
    async def test_char_truncation(self):
        # ~30 lines (~1.2k chars) — comfortably over the 500-char budget
        # below, and laid out as wrapped lines so it stays on the page.
        big = "\n".join(f"Analyte{i:02d} {i}.0 mg/dL (ref 1-9)" for i in range(30))
        data = _make_pdf(big)
        r = await extract_pdf_text(data, openai_api_key="", max_chars=500)
        assert r.method == "text"
        assert r.char_truncated
        assert "truncated" in r.text

    @pytest.mark.asyncio
    async def test_no_text_layer_without_ocr_key_degrades(self):
        # A blank (image-less, text-less) PDF stands in for a scan: no text
        # layer, and with no OpenAI key the OCR fallback is unavailable.
        r = await extract_pdf_text(_blank_pdf(), openai_api_key="")
        assert r.method == "none"
        assert not r.ok

    @pytest.mark.asyncio
    async def test_malformed_input_never_raises(self):
        r = await extract_pdf_text(b"this is not a pdf", openai_api_key="")
        assert r.method == "none"
        assert not r.ok
