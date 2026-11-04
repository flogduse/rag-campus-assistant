"""Retrieval and prompt tests that do not require ChromaDB or a model download."""
from __future__ import annotations

from app import llm


def _fake_model():
    class _Vec(list):
        def tolist(self):
            return list(self)

    class FakeModel:
        def encode(self, questions, **kwargs):
            return [_Vec([0.1, 0.2, 0.3]) for _ in questions]

    return FakeModel()


def test_lexical_overlap_counts_question_content_words():
    assert llm.lexical_overlap("internal marks split", "INTERNAL MARKS STRUCTURE") == 2 / 3
    assert llm.lexical_overlap("hostel quiet hours", "mess timings") == 0.0
    # Stopwords are ignored, so a question made only of them carries no signal.
    assert llm.lexical_overlap("what is the", "internal marks") == 0.0


def test_lexical_boost_lifts_exact_heading_match(monkeypatch):
    """A chunk that literally contains the query terms can outrank a chunk with
    a marginally better embedding similarity."""

    docs = [
        "unrelated notes about the library",
        "INTERNAL MARKS STRUCTURE: IA-1 carries 10 marks",
    ]
    # The unrelated chunk has the better cosine score before the lexical boost.
    raw = {
        "documents": [docs],
        "metadatas": [[{"source": "a.txt"}, {"source": "b.txt"}]],
        "distances": [[0.20, 0.25]],
    }
    monkeypatch.setattr(llm.store, "query", lambda *args, **kwargs: raw)
    monkeypatch.setattr("app.ingest.get_model", lambda: _fake_model())
    monkeypatch.setattr("app.llm.config.MIN_SCORE", 0.0)
    monkeypatch.setattr("app.llm.config.FETCH_K", 2)
    monkeypatch.setattr("app.llm.config.TOP_K", 2)
    monkeypatch.setattr("app.llm.config.LEXICAL_WEIGHT", 0.35)

    hits = llm.retrieve("internal marks")
    assert hits[0]["metadata"]["source"] == "b.txt"
    assert hits[0]["cosine"] == 0.75  # raw similarity is still reported


def test_retrieve_applies_hybrid_threshold_and_top_k(monkeypatch):
    docs = [
        "semester break begins in September",
        "hostel visitor entry closes at nine",
        "semester classes resume in October",
    ]
    raw = {
        "documents": [docs],
        "metadatas": [[{"source": f"doc-{i}.txt", "page": i + 1} for i in range(3)]],
        "distances": [[0.10, 0.20, 0.80]],
    }
    monkeypatch.setattr(llm.store, "query", lambda *args, **kwargs: raw)
    monkeypatch.setattr("app.ingest.get_model", lambda: _fake_model())
    monkeypatch.setattr("app.llm.config.MIN_SCORE", 0.35)
    monkeypatch.setattr("app.llm.config.FETCH_K", 3)
    monkeypatch.setattr("app.llm.config.TOP_K", 2)

    hits = llm.retrieve("semester dates")
    assert len(hits) == 2
    assert hits[0]["metadata"]["source"] == "doc-0.txt"
    assert hits[1]["metadata"]["source"] == "doc-1.txt"
    assert all(hit["score"] >= 0.35 for hit in hits)


def test_empty_store_returns_no_hits(monkeypatch):
    raw = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    monkeypatch.setattr(llm.store, "query", lambda *args, **kwargs: raw)
    monkeypatch.setattr("app.ingest.get_model", lambda: _fake_model())
    assert llm.retrieve("anything") == []


def test_retrieval_query_expands_pronoun_followups():
    history = [{"role": "user", "content": "When is the mid-semester break?"}]
    assert llm.retrieval_query("when is it?", history).startswith("When is the mid-semester break?")
    long_question = "What does the hostel handbook say about visitors on Sundays?"
    assert llm.retrieval_query(long_question, history) == long_question


def test_prompt_includes_recent_conversation_context():
    hits = [{
        "index": 1,
        "text": "The break begins on September 24.",
        "metadata": {"source": "calendar.txt", "page": 1},
        "score": 0.9,
    }]
    prompt = llm.build_prompt(
        "What about the following day?",
        hits,
        history=[{"role": "user", "content": "When does the break start?"}],
    )
    assert "When does the break start?" in prompt
    assert "September 24" in prompt
    assert "What about the following day?" in prompt
