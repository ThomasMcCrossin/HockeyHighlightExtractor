"""Shared piece of every box-score provider: a fetcher that returns data already in memory."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..box_score import BoxScoreFetcher
from ..goal import Goal

logger = logging.getLogger(__name__)


class PreloadedBoxScoreFetcher(BoxScoreFetcher):
    """BoxScoreFetcher that serves one pre-loaded game instead of calling an API.

    The pipeline treats it like the HockeyTech fetcher: `find_game` returns the game id,
    `fetch_box_score` returns a HockeyTech-shaped dict (`SiteKit.Gamesummary` with `meta`,
    `goals` and `penalties`) and `get_goals` returns typed goals.
    """

    def __init__(self, game_id: str, box_score: Dict, goals: List[Goal], cache_dir: Optional[Path] = None):
        super().__init__(cache_dir=cache_dir)
        self._game_id = game_id
        # Some pipeline components key off `_last_game_id` (legacy BoxScoreFetcher behaviour).
        self._last_game_id = game_id
        self._box_score = box_score
        self._goals = goals

    def find_game(self, league: str, home_team: str, away_team: str, game_date: str) -> Optional[str]:
        logger.info("Using pre-loaded game id: %s", self._game_id)
        return self._game_id

    def fetch_box_score(self, league: str, game_id: str) -> Optional[Dict]:
        logger.info("Using pre-loaded box score data")
        return self._box_score

    def get_goals(self, box_score: Dict) -> List[Goal]:
        return self._goals
