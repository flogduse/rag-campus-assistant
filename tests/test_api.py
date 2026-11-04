"""API tests: full request cycle with mocked store + LLM (no model download, no network).

This is where the "RAG pipeline works end-to-end" claim is actually proven in CI:
question → retrieval (mocked) → prompt build → SSE frames → parsed client-side.
"""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, llm  # noqa: E402
from app import main as main_module  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)

FAKE_HITS = [
    {
        "index": 1,
        "text": "Mid-semester break begins Sep 24 and classes resume Oct 2.",
        "metadata": {"source": "academic-calendar.txt", "page": 1, "chunk_index": 0},
        "score": 0.83,
    },
    {
        "index": 2,
        "text": "The 50-mark internal splits as IA-1=10, IA-2=10, assignments=10.",
        "metadata": {"source": "academic-calendar.txt", "page": 2, "chunk_index": 1},
        "score": 0.71,
    },
]


@pytest.fixture(autouse=True)
def clear_rate_buckets():
    """Isolate the in-process rate limiter between tests."""
    main_module._rate_buckets.clear()
    yield
    main_module._rate_buckets.clear()


@pytest.fixture()
def fake_retrieval(monkeypatch):
    monkeypatch.setattr(llm, "retrieve", lambda q, history=None: FAKE_HITS)
    monkeypatch.setattr(
        "app.main.llm", llm
    )  # main.py imported the module object; patch its attribute
    yield


class TestChatEndpoint:
    def test_sse_stream_shape(self, fake_retrieval, monkeypatch):
        monkeypatch.setattr(
            llm, "stream_answer", lambda q, hits, history=None: iter(["The break is Sep 24 [1]."])
        )
        with client.stream("POST", "/api/chat", json={"question": "when is the break?"}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            frames = []
            for line in r.iter_lines():
                if line.startswith("data:"):
                    frames.append(json.loads(line[5:]))

        assert frames[0]["type"] == "sources"
        assert frames[0]["sources"][0]["source"] == "academic-calendar.txt"
        assert frames[-1] == {"type": "done"}
        tokens = "".join(f["t"] for f in frames if f["type"] == "token")
        assert "Sep 24" in tokens

    def test_sources_event_carries_scores_and_pages(self, fake_retrieval, monkeypatch):
        monkeypatch.setattr(llm, "stream_answer", lambda q, hits, history=None: iter(["ok"]))
        with client.stream("POST", "/api/chat", json={"question": "marks split"}) as r:
            lines = [l for l in r.iter_lines() if l.startswith("data:")]
        sources = json.loads(lines[0][5:])["sources"]
        assert sources[1]["page"] == 2
        assert sources[0]["score"] == 0.83

    def test_empty_page_is_sent_as_null(self, fake_retrieval, monkeypatch):
        """Chroma stores unknown pages as ""; the API contract says int | None."""
        hits = [{**FAKE_HITS[0], "metadata": {**FAKE_HITS[0]["metadata"], "page": ""}}]
        monkeypatch.setattr(llm, "retrieve", lambda q, history=None: hits)
        monkeypatch.setattr(llm, "stream_answer", lambda q, hits, history=None: iter(["ok"]))
        with client.stream("POST", "/api/chat", json={"question": "hi"}) as r:
            lines = [l for l in r.iter_lines() if l.startswith("data:")]
        assert json.loads(lines[0][5:])["sources"][0]["page"] is None

    def test_empty_retrieval_returns_guidance(self, fake_retrieval, monkeypatch):
        monkeypatch.setattr(llm, "retrieve", lambda q, history=None: [])
        with client.stream("POST", "/api/chat", json={"question": "zzz irrelevant"}) as r:
            lines = [l for l in r.iter_lines() if l.startswith("data:")]
        frames = [json.loads(l[5:]) for l in lines]
        tokens = "".join(f.get("t", "") for f in frames if f["type"] == "token")
        assert "could not find" in tokens.lower()

    def test_question_validation(self):
        r = client.post("/api/chat", json={"question": ""})
        assert r.status_code == 422  # min_length=1

    def test_blank_question_rejected(self):
        r = client.post("/api/chat", json={"question": "   "})
        assert r.status_code == 422

    def test_history_is_forwarded_to_retrieval_and_prompt(self, monkeypatch):
        seen = {}

        def fake_retrieve(question, history=None):
            seen["retrieve"] = (question, history)
            return FAKE_HITS

        def fake_stream(question, hits, history=None):
            seen["stream"] = history
            return iter(["ok"])

        monkeypatch.setattr(llm, "retrieve", fake_retrieve)
        monkeypatch.setattr(llm, "stream_answer", fake_stream)
        payload = {
            "question": "and when does it end?",
            "history": [
                {"role": "user", "content": "When is the mid-semester break?"},
                {"role": "assistant", "content": "It starts Sep 24 [1]."},
            ],
        }
        with client.stream("POST", "/api/chat", json=payload) as r:
            list(r.iter_lines())

        assert seen["retrieve"][1][0]["content"] == "When is the mid-semester break?"
        assert seen["stream"][1]["role"] == "assistant"

    def test_llm_error_becomes_stream_message(self, fake_retrieval, monkeypatch):
        def boom(q, hits, history=None):
            raise RuntimeError("provider down")
            yield  # pragma: no cover

        monkeypatch.setattr(llm, "stream_answer", boom)
        with client.stream("POST", "/api/chat", json={"question": "hi"}) as r:
            lines = [l for l in r.iter_lines() if l.startswith("data:")]
        frames = [json.loads(l[5:]) for l in lines]
        tokens = "".join(f.get("t", "") for f in frames if f["type"] == "token")
        assert "provider down" in tokens
        assert frames[-1] == {"type": "done"}  # stream still terminates cleanly


class TestRateLimit:
    def test_limit_returns_429_with_retry_after(self, fake_retrieval, monkeypatch):
        monkeypatch.setattr(config, "RATE_LIMIT", 2)
        monkeypatch.setattr(config, "RATE_WINDOW", 60)
        monkeypatch.setattr(llm, "stream_answer", lambda q, hits, history=None: iter(["ok"]))

        for _ in range(2):
            with client.stream("POST", "/api/chat", json={"question": "hi"}) as r:
                list(r.iter_lines())
            assert r.status_code == 200

        r = client.post("/api/chat", json={"question": "hi"})
        assert r.status_code == 429
        assert int(r.headers["retry-after"]) >= 1


class TestIngestProtection:
    def test_open_when_no_admin_token_configured(self, monkeypatch):
        monkeypatch.setattr(config, "ADMIN_TOKEN", "")
        monkeypatch.setattr(main_module.store, "reset", lambda: None)
        monkeypatch.setattr(main_module.ingest, "ingest_all", lambda: 3)
        monkeypatch.setattr(main_module.store, "count", lambda: 3)
        r = client.post("/api/ingest")
        assert r.status_code == 200
        assert r.json()["chunks"] == 3

    def test_rejects_missing_or_wrong_token(self, monkeypatch):
        monkeypatch.setattr(config, "ADMIN_TOKEN", "s3cret")
        assert client.post("/api/ingest").status_code == 401
        assert client.post("/api/ingest", headers={"X-Admin-Token": "nope"}).status_code == 401

    def test_accepts_matching_token(self, monkeypatch):
        monkeypatch.setattr(config, "ADMIN_TOKEN", "s3cret")
        monkeypatch.setattr(main_module.store, "reset", lambda: None)
        monkeypatch.setattr(main_module.ingest, "ingest_all", lambda: 2)
        monkeypatch.setattr(main_module.store, "count", lambda: 2)
        r = client.post("/api/ingest", headers={"X-Admin-Token": "s3cret"})
        assert r.status_code == 200


class _FakeModel:
    """Stand-in for SentenceTransformer so retrieval tests never load torch."""

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        return [_FakeVector()]


class _FakeVector(list):
    def __init__(self):
        super().__init__([0.1, 0.2, 0.3])

    def tolist(self):
        return list(self)


class TestRetrievalThreshold:
    def _stub(self, monkeypatch, distances):
        monkeypatch.setattr("app.ingest.get_model", lambda: _FakeModel())
        monkeypatch.setattr(
            llm.store,
            "query",
            lambda text, vector, k: {
                "documents": [[f"chunk {i}" for i in range(len(distances))]],
                "metadatas": [[{"source": "doc.txt", "page": 1, "chunk_index": i} for i in range(len(distances))]],
                "distances": [distances],
            },
        )

    def test_drops_hits_below_min_score(self, monkeypatch):
        self._stub(monkeypatch, distances=[0.2, 0.55, 0.7])  # scores .8, .45, .3
        monkeypatch.setattr(config, "MIN_SCORE", 0.4)
        hits = llm.retrieve("question")
        assert [h["text"] for h in hits] == ["chunk 0", "chunk 1"]

    def test_returns_nothing_when_all_irrelevant(self, monkeypatch):
        self._stub(monkeypatch, distances=[0.85, 0.9])
        monkeypatch.setattr(config, "MIN_SCORE", 0.4)
        assert llm.retrieve("totally off-topic") == []

    def test_citations_are_renumbered_after_filtering(self, monkeypatch):
        self._stub(monkeypatch, distances=[0.8, 0.2])  # first is dropped
        monkeypatch.setattr(config, "MIN_SCORE", 0.4)
        hits = llm.retrieve("question")
        assert [h["index"] for h in hits] == [1]
        assert hits[0]["text"] == "chunk 1"


class TestFollowUpQuery:
    def test_pronoun_follow_up_inherits_previous_question(self):
        history = [{"role": "user", "content": "When is the mid-semester break?"}]
        assert llm.retrieval_query("and when does it end?", history).startswith(
            "When is the mid-semester break?"
        )

    def test_standalone_question_is_unchanged(self):
        history = [{"role": "user", "content": "When is the mid-semester break?"}]
        q = "What are the hostel mess timings for dinner"
        assert llm.retrieval_query(q, history) == q

    def test_no_history_returns_question(self):
        assert llm.retrieval_query("hello", None) == "hello"

    def test_prompt_includes_earlier_turns(self):
        history = [{"role": "user", "content": "When is the break?"}]
        prompt = llm.build_prompt("and when does it end?", FAKE_HITS, history)
        assert "When is the break?" in prompt
        assert "Context:" in prompt


class TestGlyphTidy:
    def test_strips_typographic_characters_that_render_as_tofu(self):
        assert main_module._tidy("IA\u20111 \u202fAM\u00a0ok\u00ad") == "IA-1  AM ok"

    def test_stream_tokens_are_normalised(self, fake_retrieval, monkeypatch):
        monkeypatch.setattr(
            llm, "stream_answer", lambda q, hits, history=None: iter(["IA\u20111", " ends"])
        )
        with client.stream("POST", "/api/chat", json={"question": "ia?"}) as r:
            lines = [l for l in r.iter_lines() if l.startswith("data:")]
        frames = [json.loads(l[5:]) for l in lines]
        tokens = "".join(f["t"] for f in frames if f["type"] == "token")
        assert tokens == "IA-1 ends"


class TestStreamSanitising:
    def test_footnote_citation_split_across_tokens_is_normalized(self, fake_retrieval, monkeypatch):
        # A real model streams "[1", "†L9-L12]" as separate tokens.
        chunks = ["The break is Sep 24 [1", "†L9-L12]", " and classes resume Oct 2."]
        monkeypatch.setattr(llm, "stream_answer", lambda q, hits, history=None: iter(chunks))
        with client.stream("POST", "/api/chat", json={"question": "when is the break?"}) as r:
            lines = [l for l in r.iter_lines() if l.startswith("data:")]
        frames = [json.loads(l[5:]) for l in lines]
        tokens = "".join(f["t"] for f in frames if f["type"] == "token")
        assert "[1†" not in tokens
        assert tokens == "The break is Sep 24 [1] and classes resume Oct 2."

    def test_markdown_emphasis_is_not_shown_literally(self, fake_retrieval, monkeypatch):
        monkeypatch.setattr(llm, "stream_answer", lambda q, hits, history=None: iter(["**Sep 24**"]))
        with client.stream("POST", "/api/chat", json={"question": "when is the break?"}) as r:
            lines = [l for l in r.iter_lines() if l.startswith("data:")]
        frames = [json.loads(l[5:]) for l in lines]
        tokens = "".join(f["t"] for f in frames if f["type"] == "token")
        assert tokens == "Sep 24"

    def test_unclosed_bracket_is_flushed_at_end(self, fake_retrieval, monkeypatch):
        monkeypatch.setattr(llm, "stream_answer", lambda q, hits, history=None: iter(["see note [", "unfinished"]))
        with client.stream("POST", "/api/chat", json={"question": "hi"}) as r:
            lines = [l for l in r.iter_lines() if l.startswith("data:")]
        frames = [json.loads(l[5:]) for l in lines]
        tokens = "".join(f["t"] for f in frames if f["type"] == "token")
        assert tokens == "see note [unfinished"


class TestHealth:
    def test_health_shape(self):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert set(body.keys()) == {"status", "provider", "model", "chunks", "docs"}
