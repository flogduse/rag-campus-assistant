# RAG Campus Assistant

Ask questions about your college — syllabus, deadlines, hostel rules, clubs — and get
answers **strictly from your own documents**, with inline citations. This is Flagship
Project #1 from the portfolio roadmap (portfolio section: "Currently Building").

**Stack:** FastAPI · ChromaDB · sentence-transformers · plain-JS SSE frontend.
**LLM:** auto-detected — Groq (free key) → OpenAI → local Ollama → retrieval-only mode.

```
┌─────────────┐   chunks+page    ┌──────────────┐   cosine    ┌───────────────┐
│ data/docs/* │ ───────────────► │  ingest.py   │ ──────────► │   ChromaDB    │
│ pdf/txt/md  │  normalize       │  chunk+embed │  vectors    │ (persistent)  │
└─────────────┘                  └──────────────┘             └──────┬────────┘
                                                                     │ top-12
┌─────────────┐   POST /api/chat ┌──────────────┐    MMR top-4      │ (cosine)
│ static/ UI  │ ◄──────────────  │   main.py    │ ◄─────────────────┘
│ SSE reader  │  sources+tokens  │  (FastAPI)   │
└─────────────┘                  └──────┬───────┘
                                        │ prompt = system + [1..4] + question
                                        ▼
                              ┌───────────────────┐
                              │ llm.py (provider) │  Groq / OpenAI / Ollama
                              └───────────────────┘
```

## Quickstart

```bash
# 1) virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux

# 2) dependencies
pip install -r requirements.txt

# 3) (optional) LLM key — skip for retrieval-only mode
copy .env.example .env            # then edit .env

# 4) ingest documents (drop your PDFs into data/docs first — samples included)
python -m app.ingest

# 5) run
uvicorn app.main:app --port 8123
```

Open **http://localhost:8123** — the chat UI is served at the root.

## Without an API key

The system runs end-to-end in **retrieval-only mode**: `/api/chat` still embeds your
question, runs the MMR search and returns the top passages with scores. The UI shows
a red "RETRIEVAL-ONLY MODE" pill. Add a Groq key (free at console.groq.com) and the
same endpoint starts streaming real answers — no code changes.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | provider, model, chunk/doc counts |
| `/api/chat` | POST | `{question}` → SSE stream (sources → tokens → done) |
| `/api/ingest` | POST | wipe + re-ingest `data/docs` |
| `/docs` | GET | interactive OpenAPI docs |

## Testing

```bash
python -m pytest tests/ -v
```

Covers chunking edge cases, PDF fallback, metadata coercion, retrieval fallback,
and the full API via injected fakes.

## Design notes (interview ammunition)

- **Why character-overlap chunking?** Syllabus/notice documents concentrate answers
  in one page; deterministic chunks with page metadata make citations precise.
  Semantic splitting adds complexity without measurable gain here.
- **Why MMR?** Top-k cosine alone returns 4 near-duplicate chunks when a rule appears
  on multiple pages. Greedy MMR (λ=0.3 Jaccard redundancy penalty) diversifies.
- **Why string page metadata?** ChromaDB only allows scalar metadata; `None` page
  values from label-less PDFs would crash ingestion. Coerced at write, restored to
  `int|None` at read.
- **Why SSE over WebSockets?** One-way token flow, zero extra dependencies, works
  through every proxy, and `fetch` + ReadableStream parses it in 40 lines.

## Roadmap

- [ ] Evaluation harness: golden Q/A set + retrieval hit-rate scoring
- [ ] Conversational memory (follow-up questions)
- [ ] Streaming to the portfolio's "Currently Building" card — demo GIF on ship
