#!/usr/bin/env python3
"""
Engine stage: one recording plus a box score -> a game folder with matched events and clips.

  python scripts/process_game.py --video game.mp4 --box-score game.json
  python scripts/process_game.py --video game.mp4 --hockeytech-game-id 4943 --league mhl

The box score is either a manual JSON file (format in docs/adding-a-league.md) or a HockeyTech
game id (needs HOCKEYTECH_API_KEY). The scorebug layout is detected from the video unless
--profile names one (see scorebug_profiles.py). Output goes under --games-dir (default
$HOCKEY_GAMES_DIR or ./Games); the last line printed is the game folder.

Needs no network, no account and no AI key when the box score is a file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def run(args: argparse.Namespace) -> int:
    import config
    from hockey_extractor import HighlightPipeline, FileManager
    from hockey_extractor.providers import HockeyTechProvider, ManualBoxScoreProvider

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    video = args.video.resolve()
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2
    if args.games_dir:
        config.GAMES_DIR = args.games_dir.resolve()
    config.GAMES_DIR.mkdir(parents=True, exist_ok=True)

    if args.box_score:
        provider = ManualBoxScoreProvider(args.box_score)
        fetcher = provider.create_fetcher()
        info = provider.game_info(video.name)
        config.BOX_SCORE_TIME_IS_ELAPSED = provider.time_is_elapsed
        if not config.FOLLOWED_TEAM:
            config.FOLLOWED_TEAM = provider.followed_team
        league = info["league"]
        box_source = str(args.box_score)
    else:
        if not args.league or not args.hockeytech_game_id:
            print("give --box-score FILE, or --league and --hockeytech-game-id", file=sys.stderr)
            return 2
        ht = HockeyTechProvider(args.league)
        fetcher = ht.fetch(args.hockeytech_game_id)
        summary = fetcher._box_score["SiteKit"]["Gamesummary"]
        meta = summary.get("meta") or {}
        from hockey_extractor import leagues
        pack = leagues.get(args.league) or {}
        info = {"date": str(meta.get("date") or ""), "date_formatted": str(meta.get("date") or ""),
                "home_team": str(meta.get("home_team") or "Home"), "away_team": str(meta.get("away_team") or "Away"),
                "league": str(pack.get("short") or args.league).upper(), "filename": video.name, "home_away": "home"}
        league = info["league"]
        box_source = f"hockeytech:{args.hockeytech_game_id}"

    folders = FileManager(config).create_game_folder_from_teams(
        date=info["date"], home_team=info["home_team"], away_team=info["away_team"], league=league,
        filename=video.name, home_away=info["home_away"], time_str="unknown")
    game_dir = Path(folders["game_dir"])

    profile_name, detection = args.profile, None
    if profile_name == "auto":
        try:
            from scorebug_detect import detect_scorebug_profile
            prof, detection = detect_scorebug_profile(video)
            profile_name = prof.execution_profile_name if prof else "auto"
        except Exception as exc:  # noqa: BLE001 - detection is a convenience; fall back to the catalog
            detection = {"method": "error", "error": str(exc)}
    selection = config.resolve_highlight_execution_selection(
        profile_name, game_info=info, source_game_info=dict(info),
        reel_mode=args.reel_mode or config.DEFAULT_REEL_MODE)
    profile = dict(selection["execution_profile"])
    if args.sample_interval:
        profile["sample_interval"] = args.sample_interval

    pipeline = HighlightPipeline(config=config, video_path=video, box_score_fetcher=fetcher, game_info_override=info,
                                 game_folders_override=folders, source_game_info_override=dict(info))
    result = pipeline.execute(**profile)

    data = game_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "run_config.json").write_text(json.dumps({
        "league": league, "time_is_elapsed": bool(config.BOX_SCORE_TIME_IS_ELAPSED), "followed_team": config.FOLLOWED_TEAM,
        "box_score_source": box_source, "execution_profile": selection["execution_profile_name"],
        "scorebug_detection": detection, "reel_mode": profile.get("reel_mode"),
        "events_found": result.events_found, "events_matched": result.events_matched, "clips_created": result.clips_created,
        "success": bool(result.success), "warnings": result.warnings, "errors": result.errors,
    }, indent=2, default=str), encoding="utf-8")
    print(f"matched {result.events_matched}/{result.events_found} events, {result.clips_created} clips "
          f"(profile {selection['execution_profile_name']})", file=sys.stderr)
    for err in result.errors or []:
        print(f"error: {err}", file=sys.stderr)
    print(game_dir)
    return 0 if result.success else 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", required=True, type=Path)
    src = ap.add_argument_group("box score (one of)")
    src.add_argument("--box-score", type=Path, help="manual box-score JSON")
    src.add_argument("--hockeytech-game-id", help="HockeyTech game id (needs --league)")
    ap.add_argument("--league", help="league pack id or short name")
    ap.add_argument("--games-dir", type=Path, help="default: $HOCKEY_GAMES_DIR or ./Games")
    ap.add_argument("--profile", default="auto", help="scorebug execution profile name, or auto (detect from the video)")
    ap.add_argument("--reel-mode", default="", help="goals_only (default), goals_with_pp_penalties, goals_with_all_penalties, ...")
    ap.add_argument("--sample-interval", type=int, default=0, help="seconds between OCR samples (default: the profile's)")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv=None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
