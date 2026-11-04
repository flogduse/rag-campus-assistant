# Case study: RAG Campus Assistant

**Ask your college's documents a question, get an answer with a citation — or an
honest "I don't know".**

- Code: <https://github.com/flogduse/rag-campus-assistant>
- Stack: FastAPI · ChromaDB · sentence-transformers · plain JS (SSE)
- Role: solo — design, implementation, evaluation, deployment

---

## The problem

Campus information is scattered across a calendar, a hostel handbook and a
clubs document. The answer to "when is the mid-semester break?" exists, but it
is buried in a text file nobody opens. A general chatbot will confidently answer
from memory and be wrong about dates, marks and rules — the exact facts where
being wrong is expensive.

So the goal was narrow on purpose:

> Answer only from the provided documents, show the passages behind every
> answer, and say "I don't know" when the documents do not cover it.

That last requirement is the hard one. Retrieval systems fail quietly: they
return *something*, the LLM writes a fluent paragraph, and the user cannot tell
whether it came from page 4 or from the model's imagination.

## Constraints

- **No API key should be required to evaluate the core idea.** The retrieval
  path had to be demonstrable on its own.
- **Runs on a laptop and on free hosting.** No GPU, no managed vector database.
- **Every claim must be traceable** to a document and, where available, a page.
- **A small, boring dependency set.** No LangChain: the pipeline is about 400
  lines and is easier to reason about directly.

## Architecture

```
data/docs/*.pdf|txt|md
        │  load → chunk (900 chars, 150 overlap) + page metadata
        ▼
   ChromaDB (persistent, cosine)
        │  question → MiniLM embedding → FETCH_K=12 candidates
        ▼
   hybrid score = cosine + 0.35 × lexical overlap
        │  greedy MMR (λ=0.3 Jaccard penalty) → TOP_K=4
        ▼
   MIN_SCORE=0.35 cutoff  ──►  "I don't know" path
        │
        ▼
   Groq → OpenAI → Ollama → retrieval-only fallback
        │
        ▼
   SSE: sources → tokens → done
```

The browser never parses a JSON blob at the end. Sources arrive first (so the UI
can render citations immediately), tokens stream in, and a final `done` event
releases the input.

## Decisions worth defending

### Character-overlap chunking, not semantic splitting

The documents concentrate one answer per page, so deterministic 900-character
chunks with page metadata keep citations precise. Semantic splitting would add a
dependency and a failure mode (chunk boundaries that split a rule in half)
without a measurable gain on this corpus.

### MMR instead of plain top-k

Cosine top-k happily returns four near-identical chunks when one rule is
repeated across pages. Greedy MMR subtracts a redundancy penalty against the
already-selected set, so the model sees four *different* pieces of evidence.

### Hybrid lexical scoring — added because the eval caught a bug

Dense embeddings are good at paraphrase and bad at short keyword queries. "What
is the internal marks split?" scored **0.297** against the chunk headed
`INTERNAL MARKS STRUCTURE` — below the 0.35 cutoff, so the correct chunk was
silently dropped and the assistant answered "I don't know" to a question it
could answer.

The fix was not to lower the threshold (that would let off-topic noise through).
It was to add the signal dense retrieval lacks: the fraction of the question's
content words that literally appear in the chunk.

```
hybrid = cosine + 0.35 × lexical_overlap(question, chunk)
```

The same chunk now scores **0.531** and the answer is grounded. The raw cosine
is still returned alongside the hybrid score, so the UI can show both and the
ranking stays explainable.

### "I don't know" is a feature, not a failure

Chunks below `MIN_SCORE` are dropped before the prompt is built. An off-topic
question never reaches the model with loosely related context, so the model
cannot launder irrelevant passages into a confident answer.

### SSE instead of WebSockets

Token flow is one-way. SSE needs no extra dependency, survives every proxy, and
parses in a few dozen lines with `fetch()`.

### Provider detection with a real fallback

`Groq → OpenAI → Ollama → retrieval-only`. The same endpoint and the same UI work
with a free cloud key, a local model, or no model at all. This is what makes the
project demoable by someone who will not create an API key just to look at it.

## Evaluation

A golden set of 11 questions with two checks per question:

1. **Source hit** — did a chunk from the expected document appear in the top-k?
2. **Content hit** — does the retrieved text contain the expected keywords?

The second check exists because a large document can produce a *wrong chunk from
the right file*, which a source-only metric would score as a pass.

```bash
python -m app.ingest
python eval/run_eval.py            # exits non-zero under 80%
python eval/run_eval.py --verbose  # per-chunk scores
```

Current baseline on the bundled corpus:

```
Source hit-rate : 11/11 (100%)
Content hit-rate: 11/11 (100%)
```

The harness runs in CI on every push, alongside 45 unit/API tests. Its first
useful act was catching the `MIN_SCORE` regression described above — which is
the entire argument for having it.

## Testing

- 45 pytest tests, ~1.5 seconds, no network and no model download required.
- Gesture math and retrieval are faked at the boundary, so the API surface,
  prompt building, chunking edge cases, the `MIN_SCORE` cutoff, MMR selection and
  the lexical boost are all tested without a live LLM.
- CI job 1 runs the unit tests; job 2 builds the sample index and runs the
  retrieval eval.

## Limitations

- **Small corpus.** Three sample documents. The numbers above prove the plumbing
  works; they are not a claim about a real university's 500-page handbook.
- **Character chunking can split a table.** PDF tables are a known weak spot;
  `pdfplumber` extracts text, not structure.
- **No reranker.** A cross-encoder would likely improve precision, at the cost
  of latency and a much larger image.
- **The embedding model downloads on first run** (~90 MB). Acceptable for a
  demo, worth caching for a real deployment.
- **Conversation memory is shallow** — the last six turns, plus pronoun
  expansion before retrieval. It is not a stateful agent.

## What I would build next

1. **A hosted demo** so the project can be tried without cloning it.
2. **A reranker** on the top 12 candidates and a comparison against the current
   hybrid baseline.
3. **Per-answer faithfulness scoring** — does the generated sentence actually
   follow from the cited chunk? Retrieval hit-rate is only half the story.
4. **A larger golden set** drawn from real questions, not the ones I wrote after
   reading the documents (which is a conflict of interest the current set has).

## What this project demonstrates

- Designing a retrieval pipeline and knowing *why* each stage exists.
- Measuring instead of guessing — and letting the measurement change the design.
- Building for graceful degradation: no key, no model, no problem.
- Shipping the unglamorous parts too: tests, CI, Docker, rate limiting, an
  admin-token gate, and a documented eval harness.
