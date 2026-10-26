from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from .chunker import Chunk, split_into_chunks

ALLOWED = {".pdf", ".docx", ".txt", ".md"}
MAX_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_PAGES = 50


class IngestError(Exception):
    """A problem with an uploaded file that we can explain to the user."""


def chunk_text(text, source, label, page=None, size=800, overlap=150):
    """Cut text into overlapping chunks, preferring paragraph/word boundaries."""
    if overlap >= size // 2:
        raise ValueError("overlap must be smaller than half the chunk size")
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # try to cut at a paragraph break, else at a space
            cut = text.rfind("\n", start + size // 2, end)
            if cut == -1:
                cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(source=source, section=label, text=piece, page=page))
        if end >= len(text):
            break
        start = end - overlap
        space = text.find(" ", start, end)   # don't start in the middle of a word
        if space != -1:
            start = space + 1
    return chunks


def _read_pdf(data: bytes):
    reader = PdfReader(BytesIO(data))
    if reader.is_encrypted:
        raise IngestError("Password-protected PDFs are not supported.")
    if len(reader.pages) > MAX_PAGES:
        raise IngestError(f"PDF has more than {MAX_PAGES} pages.")
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def _read_docx(data: bytes):
    document = Document(BytesIO(data))
    return [(None, "\n".join(p.text for p in document.paragraphs))]


def ingest_bytes(data: bytes, filename: str) -> list[Chunk]:
    """Turn an uploaded file into chunks. Raises IngestError with a clear message."""
    name = Path(filename).name           # drop any folder parts from the name
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED:
        raise IngestError(f"Unsupported file type '{ext}'. Use PDF, DOCX, TXT or MD.")
    if len(data) > MAX_BYTES:
        raise IngestError("File is larger than 10 MB.")

    try:
        if ext == ".pdf":
            pages = _read_pdf(data)
        elif ext == ".docx":
            pages = _read_docx(data)
        else:
            pages = [(None, data.decode("utf-8", errors="replace"))]
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(f"Could not read this file ({exc.__class__.__name__}).") from exc

    chunks = []
    full_text = "\n".join(text for _, text in pages)
    if ext == ".md" and any(line.startswith("## ") for line in full_text.splitlines()):
        chunks = split_into_chunks(full_text, source=name)
    else:
        for page, text in pages:
            label = f"Page {page}" if page else "Document"
            chunks.extend(chunk_text(text, source=name, label=label, page=page))

    if not chunks:
        raise IngestError(
            "No readable text found. Scanned PDFs (images) are not supported, "
            "because there is no OCR."
        )
    return chunks