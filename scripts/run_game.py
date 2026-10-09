#!/usr/bin/env python3
"""
The whole path in one command: recording + box score -> engine clips -> reel with overlays.

  python scripts/run_game.py --video game.mp4 --box-score game.json

This needs no AI and no keys. Vision review is an optional middle step that turns on by itself
when it is configured, and is skipped quietly otherwise:

  --review auto    (default) agent if CLIP_REVIEW_AGENT_CMD is set, else api if
                   CLIP_REVIEW_API_KEY + CLIP_REVIEW_BASE_URL + CLIP_REVIEW_MODEL are set, else off
  --review off | api | agent | escalate

Other options are passed to the stage scripts: see process_game.py, review_game.py and
build_reel.py for the full list.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def review_mode(requested: str) -> str:
    """off | api | agent | escalate. `auto` looks at the environment and never errors."""
    if requested != "auto":
        return requested
    from clip_review.backends import api_configured

    agent = bool(os.environ.get("CLIP_REVIEW_AGENT_CMD", "").strip())
    if agent and api_configured():
        return "escalate"
    if agent:
        return "agent"
    return "api" if api_configured() else "off"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--box-score", type=Path)
    ap.add_argument("--hockeytech-game-id")
    ap.add_argument("--league")
    ap.add_argument("--games-dir", type=Path)
    ap.add_argument("--profile", default="auto")
    ap.add_argument("--sample-interval", type=int, default=0)
    ap.add_argument("--review", choices=("auto", "off", "api", "agent", "escalate"), default="auto")
    ap.add_argument("--adversary", action="store_true", help="with review: have a second reviewer challenge goals and majors")
    ap.add_argument("--theme", default="baseline")
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("--cards", action="store_true")
    args = ap.parse_args(argv)

    import process_game, review_game, build_reel

    pg = ["--video", str(args.video), "--profile", args.profile]
    for flag, val in (("--box-score", args.box_score), ("--hockeytech-game-id", args.hockeytech_game_id), ("--league", args.league),
                      ("--games-dir", args.games_dir)):
        if val:
            pg += [flag, str(val)]
    if args.sample_interval:
        pg += ["--sample-interval", str(args.sample_interval)]
    from io import StringIO
    import contextlib

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = process_game.main(pg)
    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    if code != 0 or not lines:
        print("engine stage failed; see the messages above", file=sys.stderr)
        return code or 1
    game_dir = Path(lines[-1])

    mode = review_mode(args.review)
    manifest_flag = []
    if mode != "off":
        rg = ["--game-dir", str(game_dir), "--video", str(args.video), "--backend", mode, "--apply"]
        if args.adversary:
            rg.append("--adversary")
        try:
            ok = review_game.main(rg) == 0 and (game_dir / "data" / "review" / "reel_main.json").exists()
        except Exception as exc:  # noqa: BLE001 - review is optional; the engine clips are always a valid reel
            print(f"vision review failed ({exc}); using the engine clips", file=sys.stderr)
            ok = False
        if ok:
            manifest_flag = ["--reviewed"]
        else:
            print("vision review did not finish; using the engine clips", file=sys.stderr)
    bg = ["--game-dir", str(game_dir), "--theme", args.theme, *manifest_flag]
    if args.no_overlays:
        bg.append("--no-overlays")
    if args.cards:
        bg.append("--cards")
    code = build_reel.main(bg)
    print(f"clips: {'reviewed (' + mode + ')' if manifest_flag else 'engine'}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
