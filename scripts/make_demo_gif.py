"""Render a captured terminal session to an animated GIF for the README.

    bash scripts/demo.sh > /tmp/demo.txt 2>&1
    venv/bin/python scripts/make_demo_gif.py /tmp/demo.txt docs/demo.gif

Why render rather than screen-record: the cockpit spends ~10 s inside the
simulator per pick and ~3 s inside the narration model, so a real-time capture
is nearly a minute of a mostly-static screen. This replays the SAME captured
bytes at a readable pace.

**The text is never synthesised.** The input is stdout from an actual
`scripts/demo.sh` run against the real bundle; this script only decides when
each line appears. Anything it cannot render (a control sequence, a tab) is
passed through as literal text rather than quietly dropped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# One dark palette, chosen for contrast on both GitHub themes.
BG = (13, 17, 23)
FG = (201, 209, 217)
DIM = (110, 118, 129)
ACCENT = (86, 211, 100)      # the engine's own lines
WARN = (210, 153, 34)        # WHY / narration
PROMPT = (88, 166, 255)      # the `>` prompt and typed input

FONT_CANDIDATES = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)

PAD = 18
LINE_H = 20
FONT_SIZE = 14
MAX_COLS = 92
VISIBLE_ROWS = 30

#: Milliseconds a frame holds. Lines that carry the payload linger.
FAST_MS = 55
SLOW_MS = 900
HOLD_MS = 2600


def _font() -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, FONT_SIZE)
            except OSError:
                continue
    raise SystemExit(
        "no monospace font found; add one to FONT_CANDIDATES. Tried:\n  "
        + "\n  ".join(FONT_CANDIDATES))


def colour_for(line: str) -> tuple[int, int, int]:
    stripped = line.strip()
    if stripped.startswith(">"):
        return PROMPT
    if stripped.startswith(("tier ", "E[$]")) or " E[$] " in line:
        return ACCENT
    if stripped.startswith("WHY") or stripped.startswith("separating axis"):
        return WARN
    if stripped.startswith(("note:", "stale:", "[round", "bundle", "session")):
        return DIM
    return FG


def dwell_for(line: str) -> int:
    """How long to hold the frame after this line lands."""
    stripped = line.strip()
    if not stripped:
        return FAST_MS
    if " E[$] " in line or stripped.startswith("WHY"):
        return SLOW_MS
    if stripped.startswith("[round") or "entries verified" in stripped:
        return SLOW_MS
    return FAST_MS


def wrap(line: str, cols: int) -> list[str]:
    line = line.replace("\t", "    ").rstrip("\n")
    if len(line) <= cols:
        return [line]
    out, rest = [], line
    while len(rest) > cols:
        out.append(rest[:cols])
        rest = rest[cols:]
    out.append(rest)
    return out


def render(lines: list[str], out_path: Path) -> Path:
    font = _font()
    width = PAD * 2 + int(font.getlength("M") * MAX_COLS)
    height = PAD * 2 + LINE_H * VISIBLE_ROWS

    frames: list[Image.Image] = []
    durations: list[int] = []

    def draw_window(window: list[str]) -> Image.Image:
        img = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(img)
        for i, text in enumerate(window):
            d.text((PAD, PAD + i * LINE_H), text,
                   font=font, fill=colour_for(text))
        return img

    shown: list[str] = []
    for raw in lines:
        for piece in wrap(raw, MAX_COLS):
            shown.append(piece)
            window = shown[-VISIBLE_ROWS:]
            frames.append(draw_window(window))
            durations.append(dwell_for(piece))

    if not frames:
        raise SystemExit("input had no lines to render")

    frames.append(frames[-1].copy())
    durations.append(HOLD_MS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=True,
    )
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path,
                    help="stdout from a real `bash scripts/demo.sh` run")
    ap.add_argument("out", type=Path, nargs="?", default=Path("docs/demo.gif"))
    args = ap.parse_args()

    if not args.capture.exists():
        print(f"error: {args.capture} not found. Capture one with:\n"
              f"    bash scripts/demo.sh > {args.capture} 2>&1", file=sys.stderr)
        return 1

    lines = args.capture.read_text(errors="replace").splitlines()
    path = render(lines, args.out)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path} ({len(lines)} lines, {size_kb:.0f} KB)")
    if size_kb > 8000:
        print("  WARNING: over 8 MB; GitHub will be slow to load it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
