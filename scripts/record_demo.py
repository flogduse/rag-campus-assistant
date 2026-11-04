#!/usr/bin/env python3
"""Record a real browser demo of the running app and turn it into a GIF.

This is not a mocked animation: it drives the actual UI, sends a real question
to the running server, and captures the streamed answer as it renders.

Setup (dev only):
    pip install -r requirements-dev.txt
    python -m playwright install chromium

Usage:
    # terminal 1
    python -m app.ingest
    uvicorn app.main:app --port 8199

    # terminal 2
    python scripts/record_demo.py --url http://localhost:8199 \
        --out docs/demo.gif --poster docs/demo-poster.png

The GIF is written at 5 fps, capped at 960 px wide, with a poster frame that
can be used as a static preview where GIFs are not appropriate.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

FRAME_INTERVAL_MS = 200  # 5 fps
MAX_WIDTH = 960


def capture(url: str, question: str | None, seconds: float, frames_dir: Path) -> list[Path]:
    frames: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(url, wait_until="networkidle")

        # Wait until the health check has populated the header.
        page.wait_for_function(
            "() => { const el = document.getElementById('doc-count'); return el && el.textContent.trim() !== '—'; }",
            timeout=30_000,
        )
        page.wait_for_timeout(600)

        # 1. Welcome screen.
        for _ in range(6):
            frames.append(_shot(page, frames_dir, len(frames)))
            page.wait_for_timeout(FRAME_INTERVAL_MS)

        # 2. Ask a real question. Clicking a suggestion submits the form.
        if question:
            page.fill("#q", question)
            page.press("#q", "Enter")
        else:
            page.locator(".prompt-card").first.click()

        # 3. Wait for the first streamed token, so the GIF has no dead air.
        page.wait_for_function(
            "() => { const b = document.querySelector('article.msg.bot .bubble'); return b && b.textContent.trim().length > 0; }",
            timeout=60_000,
        )

        # 4. Capture the answer as it renders.
        deadline = page.evaluate("Date.now()") + int(seconds * 1000)
        while page.evaluate("Date.now()") < deadline:
            frames.append(_shot(page, frames_dir, len(frames)))
            page.wait_for_timeout(FRAME_INTERVAL_MS)

        # 5. Hold on the finished answer.
        for _ in range(8):
            frames.append(_shot(page, frames_dir, len(frames)))
            page.wait_for_timeout(FRAME_INTERVAL_MS)

        browser.close()

    return frames


def _shot(page, frames_dir: Path, index: int) -> Path:
    path = frames_dir / f"frame-{index:04d}.png"
    page.screenshot(path=str(path))
    return path


def build_gif(frames: list[Path], out: Path, poster: Path | None) -> None:
    images = [Image.open(f).convert("RGB") for f in frames]
    if not images:
        raise SystemExit("no frames captured")

    if images[0].width > MAX_WIDTH:
        height = round(images[0].height * MAX_WIDTH / images[0].width)
        images = [im.resize((MAX_WIDTH, height), Image.LANCZOS) for im in images]

    if poster:
        images[-1].save(poster)

    palette_images = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im in images]
    out.parent.mkdir(parents=True, exist_ok=True)
    palette_images[0].save(
        out,
        save_all=True,
        append_images=palette_images[1:],
        duration=FRAME_INTERVAL_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"wrote {out} ({len(images)} frames, {size_mb:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8199")
    parser.add_argument("--out", default="docs/demo.gif")
    parser.add_argument("--poster", default="docs/demo-poster.png")
    parser.add_argument("--question", default=None, help="question to type; default clicks the first suggestion")
    parser.add_argument("--seconds", type=float, default=14.0)
    args = parser.parse_args()

    frames_dir = Path(tempfile.mkdtemp(prefix="rag-demo-"))
    try:
        frames = capture(args.url, args.question, args.seconds, frames_dir)
        build_gif(frames, Path(args.out), Path(args.poster) if args.poster else None)
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
