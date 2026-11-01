"""Simple retrieval eval harness using golden Q/A set."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm


def main() -> None:
    golden_path = Path(__file__).parent / "golden.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    total = len(golden)
    hits_at_1 = 0
    hits_at_4 = 0
    for item in golden:
        q = item["question"]
        res = llm.retrieve(q)
        sources = [h["metadata"].get("source") for h in res]
        if sources and sources[0] in item["expected_sources"]:
            hits_at_1 += 1
        if any(s in item["expected_sources"] for s in sources):
            hits_at_4 += 1
    print(f"Eval on {total} questions")
    print(f"Hits@1: {hits_at_1}/{total} ({hits_at_1/total:.0%})")
    print(f"Hits@4: {hits_at_4}/{total} ({hits_at_4/total:.0%})")


if __name__ == "__main__":
    main()
