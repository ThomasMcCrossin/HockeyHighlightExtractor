"""
HockeyTech provider: fetch a game's box score from a HockeyTech league feed.

The league comes from a league pack whose `provider` block is
`{"type": "hockeytech", "client_code": "...", "league_id": "..."}`; the MHL pack is the worked
example. HockeyTech requires a public feed key: set HOCKEYTECH_API_KEY (the key your league's
own website uses). Set HOCKEYTECH_SEASON_ID to pin a season.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..box_score import BoxScoreFetcher
from ..goal import Goal
from .base import PreloadedBoxScoreFetcher


class HockeyTechProvider:
    def __init__(self, league: str, cache_dir: Optional[Path] = None, api_key: Optional[str] = None):
        self.league = league
        self.fetcher = BoxScoreFetcher(cache_dir=cache_dir, api_key=api_key)
        if self.fetcher._league_config(league) is None:
            raise ValueError(f"league '{league}' has no hockeytech provider block in its league pack")

    def fetch(self, game_id: str) -> PreloadedBoxScoreFetcher:
        """Fetch one game by HockeyTech game id and wrap it for the pipeline."""
        raw: Optional[Dict[str, Any]] = self.fetcher.fetch_box_score(self.league, str(game_id))
        if not raw:
            raise RuntimeError(f"HockeyTech returned no box score for game {game_id}")
        goals = self.fetcher.get_goals(raw)
        return PreloadedBoxScoreFetcher(str(game_id), raw, goals)

    def find_game_id(self, home_team: str, away_team: str, game_date: str) -> Optional[str]:
        return self.fetcher.find_game(self.league, home_team, away_team, game_date)
