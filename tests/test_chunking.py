"""Unit tests for the ingestion primitives (no network, no models)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import chunk_text, normalize  # noqa: E402
from app.store import _clean_meta  # noqa: E402


class TestNormalize:
    def test_collapses_runs_of_spaces(self):
        assert normalize("a    b\t\tc") == "a b c"

    def test_collapses_excess_newlines(self):
        assert normalize("a\n\n\n\n\nb") == "a\n\nb"

    def test_strips_null_bytes(self):
        assert normalize("a\x00b") == "ab"

    def test_strips_ends(self):
        assert normalize("   hello \n\n ") == "hello"


class TestChunkText:
    def test_short_text_single_chunk(self):
        assert chunk_text("hello world", 900, 150) == ["hello world"]

    def test_long_text_multiple_chunks(self):
        text = " ".join(["word"] * 1000)  # ~5000 chars
        chunks = chunk_text(text, 900, 150)
        assert len(chunks) > 1
        assert all(len(c) <= 900 for c in chunks)

    def test_overlap_present(self):
        text = " ".join(f"token{i}" for i in range(600))
        chunks = chunk_text(text, 400, 80)
        # with overlap, consecutive chunks must share some content
        assert len(chunks) >= 2
        overlap_zone = chunks[0][-200:]
        assert any(w in overlap_zone for w in chunks[1].split()[:5])

    def test_no_word_cut_at_end(self):
        """Chunks must END on whole words. (Chunk STARTS may begin mid-word
        when the overlap slice lands inside one — harmless for retrieval,
        the fragment only ever repeats content from the previous chunk.)"""
        text = "alpha " * 300
        for chunk in chunk_text(text, 300, 50):
            assert chunk.endswith("alpha")
            tokens = chunk.split()
            # at most the FIRST token may be an overlap fragment
            assert all(t == "alpha" for t in tokens[1:])

    def test_empty_text(self):
        assert chunk_text("", 900, 150) == []

    def test_reassembly_covers_all_content(self):
        text = " ".join(f"t{i}" for i in range(500))
        chunks = chunk_text(text, 350, 60)
        # every token must appear in at least one chunk
        all_tokens = set(text.split())
        seen = set()
        for c in chunks:
            seen.update(c.split())
        assert all_tokens == seen


class TestCleanMeta:
    def test_drops_none_values(self):
        out = _clean_meta({"source": "a.pdf", "page": None})
        assert "page" not in out or out.get("page") == ""

    def test_coerces_unsupported_types(self):
        out = _clean_meta({"source": "a.pdf", "weird": {"nested": 1}})
        assert isinstance(out["weird"], str)

    def test_fills_required_defaults(self):
        out = _clean_meta({})
        assert out["source"] == ""
        assert out["chunk_index"] == 0

    def test_keeps_scalars(self):
        out = _clean_meta({"page": 3, "chunk_index": 1, "source": "x"})
        assert out == {"page": 3, "chunk_index": 1, "source": "x"}
