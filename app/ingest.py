"""Document loading, chunking and embedding.

Chunking strategy: fixed character window with overlap. Simple, deterministic
and surprisingly strong for syllabus/notice-style documents where the answer
is usually inside one page — far cheaper to reason about than fancy splitting.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import config, store

_model = None


def get_model():
    """Lazy-load the embedding model (downloads once to ~/.cache).

    The sentence-transformers import is deliberately inside this function:
    it pulls in torch (~30s first import), and unit tests should never pay that.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...]. Try pdfplumber, fall back to pypdf."""
    pages: list[tuple[int, str]] = []
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                pages.append((i, page.extract_text() or ""))
    except ImportError:
        pdfplumber = None  # fall through to pypdf
    except Exception:
        pdfplumber = None  # corrupt/encrypted pdf — try pypdf before giving up

    if not pages:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages, start=1):
                pages.append((i, page.extract_text() or ""))
        except Exception:
            raise RuntimeError(f"Could not extract text from {path.name}")

    # Keep only pages with actual content
    return [(p, t.strip()) for p, t in pages if t and t.strip()]


def load_txt(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(1, text)] if text.strip() else []


def load_file(path: Path) -> list[tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".txt", ".md"}:
        return load_txt(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def normalize(text: str) -> str:
    """Collapse the whitespace chaos that PDF extraction produces."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size character chunks with overlap, snapped to word boundaries."""
    if len(text) <= size:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        # Don't cut a word in half — slide back to the last space
        if end < len(text):
            sp = text.rfind(" ", start, end)
            if sp > start + size // 2:
                end = sp
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def ingest_file(path: Path) -> int:
    """Full pipeline for one file: extract -> normalize -> chunk -> embed -> store."""
    pages = load_file(path)
    chunks: list[dict] = []
    for page_num, page_text in pages:
        clean = normalize(page_text)
        for idx, piece in enumerate(chunk_text(clean, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            chunks.append({
                "id": f"{path.name}::p{page_num}::c{idx}",
                "text": piece,
                "metadata": {"source": path.name, "page": page_num, "chunk_index": idx},
            })

    if not chunks:
        return 0

    vectors = get_model().encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_loading_bar=False,
    ).tolist()

    return store.add_chunks(chunks, vectors)


def ingest_all() -> int:
    """Ingest every supported file in data/docs. Returns total chunks added."""
    if not config.DOCS_DIR.exists():
        return 0
    total = 0
    for path in sorted(config.DOCS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}:
            total += ingest_file(path)
    return total


if __name__ == "__main__":
    added = ingest_all()
    print(f"Ingested {added} new chunks. Store total: {store.count()} chunks "
          f"across {store.unique_docs()} documents.")
