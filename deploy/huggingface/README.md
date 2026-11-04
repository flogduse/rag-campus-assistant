---
title: RAG Campus Assistant
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Answers questions from your documents with page-level citations
---

# RAG Campus Assistant

Ask questions about your college documents and get answers **lifted from those
documents**, with a citation on every claim.

- Retrieval runs fully offline (MiniLM embeddings + ChromaDB + MMR re-ranking).
- Add a `GROQ_API_KEY` (free tier) to turn on streamed, grounded answers.
- Without a key the Space still works in **retrieval-only mode** and returns the
  best-matching passages with similarity scores.

Source and case study: <https://github.com/flogduse/rag-campus-assistant>
