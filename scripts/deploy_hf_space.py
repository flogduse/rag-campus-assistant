#!/usr/bin/env python3
"""Publish this repository as a Docker Space on Hugging Face.

The Space runs the exact same image as local Docker, so the demo is not a
rewrite: retrieval, MMR, SSE streaming and the static UI are all identical.

Usage
-----
    pip install huggingface_hub
    huggingface-cli login                # or pass --token

    python scripts/deploy_hf_space.py \
        --repo-id flogduse/rag-campus-assistant \
        --groq-key gsk_...              # optional: enables streamed answers

Without ``--groq-key`` the Space still works in retrieval-only mode, which is
enough to prove the retrieval pipeline to a visitor.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE_README = ROOT / "deploy" / "huggingface" / "README.md"

# Never upload secrets, local state, or development-only folders.
IGNORE = [
    ".git/*",
    ".github/*",
    ".venv/*",
    ".env",
    "data/chroma/*",
    "__pycache__/*",
    "**/__pycache__/*",
    "*.pyc",
    ".pytest_cache/*",
    "tests/*",
    "eval/*",
    "scripts/*",
    "deploy/*",
    "docs/*",
    "*.log",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_SPACE_ID", "flogduse/rag-campus-assistant"),
        help="Space id, e.g. username/space-name",
    )
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF access token")
    parser.add_argument("--groq-key", default=os.getenv("GROQ_API_KEY"), help="Optional Groq key for streamed answers")
    parser.add_argument("--private", action="store_true", help="Create a private Space")
    args = parser.parse_args()

    if not args.token:
        print("error: pass --token or set HF_TOKEN (create one at https://huggingface.co/settings/tokens)", file=sys.stderr)
        return 2
    if not SPACE_README.exists():
        print(f"error: missing {SPACE_README}", file=sys.stderr)
        return 2

    api = HfApi(token=args.token)

    print(f"→ creating/updating Space {args.repo_id}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )

    print("→ uploading application (excluding local state and dev files)")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=str(ROOT),
        ignore_patterns=IGNORE,
        commit_message="Deploy RAG Campus Assistant",
    )

    # The Space reads its metadata from the root README.md, so overwrite the
    # GitHub README with the Space-specific one (front matter + app_port: 7860).
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="space",
        path_or_fileobj=str(SPACE_README),
        path_in_repo="README.md",
        commit_message="Add Space metadata",
    )

    if args.groq_key:
        print("→ setting GROQ_API_KEY secret")
        api.add_space_secret(repo_id=args.repo_id, key="GROQ_API_KEY", value=args.groq_key)
    else:
        print("i  no Groq key supplied — the Space will run in retrieval-only mode")

    url = f"https://huggingface.co/spaces/{args.repo_id}"
    print(f"✓ done: {url}")
    print("  first build takes a few minutes (torch + MiniLM download).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
