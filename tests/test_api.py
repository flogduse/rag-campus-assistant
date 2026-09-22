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

from app import llm  # noqa: E402
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


@pytest.fixture()
def fake_retrieval(monkeypatch):
    monkeypatch.setattr(llm, "retrieve", lambda q: FAKE_HITS)
    monkeypatch.setattr(
        "app.main.llm", llm
    )  # main.py imported the module object; patch its attribute
    yield


class TestChatEndpoint:
    def test_sse_stream_shape(self, fake_retrieval, monkeypatch):
        monkeypatch.setattr(
            llm, "stream_answer", lambda q, hits: iter(["The break is Sep 24 [1]."])
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
        monkeypatch.setattr(llm, "stream_answer", lambda q, hits: iter(["ok"]))
        with client.stream("POST", "/api/chat", json={"question": "marks split"}) as r:
            lines = [l for l in r.iter_lines() if l.startswith("data:")]
        sources = json.loads(lines[0][5:])["sources"]
        assert sources[1]["page"] == 2
        assert sources[0]["score"] == 0.83

    def test_empty_retrieval_returns_guidance(self, fake_retrieval, monkeypatch):
        monkeypatch.setattr(llm, "retrieve", lambda q: [])
        with client.stream("POST", "/api/chat", json={"question": "zzz irrelevant"}) as r:
            lines = [l for l in r.iter_lines() if l.startswith("data:")]
        frames = [json.loads(l[5:]) for l in lines]
        tokens = "".join(f.get("t", "") for f in frames if f["type"] == "token")
        assert "could not find" in tokens.lower()

    def test_question_validation(self):
        r = client.post("/api/chat", json={"question": ""})
        assert r.status_code == 422  # min_length=1

    def test_llm_error_becomes_stream_message(self, fake_retrieval, monkeypatch):
        def boom(q, hits):
            raise RuntimeError("provider down")
            yield  # pragma: no cover

        monkeypatch.setattr(llm, "stream_answer", boom)
        with client.stream("POST", "/api/chat", json={"question": "hi"}) as r:
            lines = [l for l in r.iter_lines() if l.startswith("data:")]
        frames = [json.loads(l[5:]) for l in lines]
        tokens = "".join(f.get("t", "") for f in frames if f["type"] == "token")
        assert "provider down" in tokens
        assert frames[-1] == {"type": "done"}  # stream still terminates cleanly


class TestHealth:
    def test_health_shape(self):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert set(body.keys()) == {"status", "provider", "model", "chunks", "docs"}
