#!/usr/bin/env python3
"""
Reel stage: a processed game folder -> output/reel.mp4 with broadcast overlays.

  python scripts/build_reel.py --game-dir Games/<game>             # engine clips
  python scripts/build_reel.py --game-dir Games/<game> --reviewed  # clips after vision review
  python scripts/build_reel.py --game-dir Games/<game> --no-overlays --cards

Overlays are drawn by the theme in overlays/themes/<name>/ using the game's league pack; they
need node and Playwright (README, "Install"). --no-overlays only joins the clips.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--game-dir", required=True, type=Path)
    ap.add_argument("--reviewed", action="store_true", help="use data/review/reel_main.json written by review_game.py --apply")
    ap.add_argument("--manifest", type=Path, help="any clip manifest (a JSON with a `clips` list of events that have a `path`)")
    ap.add_argument("--theme", default="baseline", help="overlay theme under overlays/themes/ (default: baseline)")
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("--cards", action="store_true", help="add an intro card and a final-score card")
    ap.add_argument("--output", type=Path, help="default: <game-dir>/output/reel.mp4")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    from hockey_extractor.reel import build_reel

    manifest = args.manifest
    if args.reviewed:
        manifest = args.game_dir / "data" / "review" / "reel_main.json"
        if not manifest.exists():
            print(f"no reviewed manifest at {manifest}; run scripts/review_game.py --apply first", file=sys.stderr)
            return 2
    try:
        res = build_reel(args.game_dir, manifest=manifest, theme=args.theme, overlays=not args.no_overlays,
                         cards=args.cards, output=args.output)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{res['clips']} clips, {res['overlays']} overlays -> {res['output']}", file=sys.stderr)
    print(res["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
