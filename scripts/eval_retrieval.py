#!/usr/bin/env python
"""Retrieval evaluation: hit-rate@k over a golden question set.

Run after ingesting the documents:

    python scripts/eval_retrieval.py            # uses data/eval/golden.json
    python scripts/eval_retrieval.py --threshold 0.9 --verbose

For each question it runs the real retrieval path (embedding -> ChromaDB ->
MMR -> MIN_SCORE filter) and checks two things:

  source   — the expected document appears among the retrieved sources
  keyword  — at least one expected keyword appears in the retrieved text

Exit status is 0 when the source hit-rate meets --threshold, else 1, so this
can gate a CI job or a release. No network and no LLM key required.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, llm, store  # noqa: E402


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["cases"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "data" / "eval" / "golden.json",
        help="path to the golden question set",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="minimum source hit-rate to exit 0 (default: 0.8)",
    )
    parser.add_argument("--verbose", action="store_true", help="print every hit")
    args = parser.parse_args()

    if store.count() == 0:
        print("Vector store is empty — run `python -m app.ingest` first.")
        return 2

    cases = load_cases(args.golden)
    print(
        f"Retrieval evaluation · {len(cases)} cases · "
        f"TOP_K={config.TOP_K} MIN_SCORE={config.MIN_SCORE} · "
        f"{store.count()} chunks\n"
    )

    source_hits = 0
    keyword_hits = 0

    for i, case in enumerate(cases, start=1):
        question = case["question"]
        expected = case["source"]
        keywords = [k.lower() for k in case.get("keywords", [])]

        hits = llm.retrieve(question)
        sources = [h["metadata"].get("source", "") for h in hits]
        texts = " ".join(h["text"] for h in hits).lower()

        source_ok = expected in sources
        keyword_ok = any(k in texts for k in keywords) if keywords else True
        source_hits += source_ok
        keyword_hits += keyword_ok

        mark = "✔" if source_ok and keyword_ok else "✘"
        top = f"{hits[0]['score']:.2f}" if hits else "—"
        print(
            f" {mark} {i:2}. {question[:58]:<58} "
            f"source={'✓' if source_ok else '✗'} "
            f"keyword={'✓' if keyword_ok else '✗'} "
            f"top={top} hits={len(hits)}"
        )
        if args.verbose:
            for h in hits:
                meta = h["metadata"]
                print(f"        [{h['index']}] {meta.get('source')} p.{meta.get('page')} "
                      f"({h['score']:.3f}) {h['text'][:70]!r}")
            if not hits:
                print("        (no chunks passed MIN_SCORE)")

    total = len(cases)
    source_rate = source_hits / total if total else 0.0
    keyword_rate = keyword_hits / total if total else 0.0

    print(
        f"\nSource hit-rate : {source_hits}/{total} ({source_rate:.0%})\n"
        f"Keyword recall  : {keyword_hits}/{total} ({keyword_rate:.0%})\n"
        f"Threshold       : {args.threshold:.0%}"
    )

    if source_rate < args.threshold:
        print("\nFAIL — below threshold. Lower MIN_SCORE, raise FETCH_K, or add/chunk documents better.")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
