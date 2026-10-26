from io import BytesIO

import pytest
from docx import Document
from pypdf import PdfWriter

from runbook_ai.ingest import IngestError, chunk_text, ingest_bytes

WORDS = " ".join(f"w{i:04d}" for i in range(600))  # about 3600 characters


def test_long_text_is_split_with_overlap():
    chunks = chunk_text(WORDS, source="a.txt", label="Doc", size=800, overlap=150)
    assert len(chunks) > 1
    assert all(len(c.text) <= 800 for c in chunks)
    assert chunks[1].text.split()[0] in chunks[0].text  # the overlap is real


def test_short_text_is_one_chunk():
    assert len(chunk_text("hello world", "a.txt", "Doc")) == 1


def test_blank_text_gives_no_chunks():
    assert chunk_text("   ", "a.txt", "Doc") == []


def test_overlap_must_be_small():
    with pytest.raises(ValueError):
        chunk_text("x", "a.txt", "Doc", size=100, overlap=60)


def test_txt_upload():
    chunks = ingest_bytes(b"Hello world. " * 5, "notes.txt")
    assert chunks[0].source == "notes.txt"


def test_markdown_with_headings_uses_headings():
    chunks = ingest_bytes(b"# T\n\n## One\nA\n\n## Two\nB\n", "g.md")
    sections = [c.section for c in chunks]
    assert "One" in sections and "Two" in sections


def test_docx_upload():
    doc = Document()
    doc.add_paragraph("Reset your VPN token in the portal.")
    buffer = BytesIO()
    doc.save(buffer)
    chunks = ingest_bytes(buffer.getvalue(), "guide.docx")
    assert "VPN token" in chunks[0].text


def test_pdf_without_text_is_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(IngestError):
        ingest_bytes(buffer.getvalue(), "scan.pdf")


def test_unsupported_type_is_rejected():
    with pytest.raises(IngestError):
        ingest_bytes(b"x", "tool.exe")


def test_oversized_file_is_rejected():
    with pytest.raises(IngestError):
        ingest_bytes(b"a" * (10 * 1024 * 1024 + 1), "big.txt")


def test_folder_parts_are_stripped_from_the_name():
    assert ingest_bytes(b"hello there", "../../evil.txt")[0].source == "evil.txt"


def test_empty_file_is_rejected():
    with pytest.raises(IngestError):
        ingest_bytes(b"   ", "empty.txt")