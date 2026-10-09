"""
Manual box-score provider: a plain JSON file you write (or export from any source).

This is the adapter to use for a league with no HockeyTech feed. The file format
(`manual-box-score/1`) is documented in docs/adding-a-league.md:

    {
      "format": "manual-box-score/1",
      "game_id": "2026-01-09-hawks-rams",
      "date": "2026-01-09",
      "league": "mhl",                       // league pack id or short name (optional)
      "home_team": "Harbour Hawks",
      "away_team": "Ridge Rams",
      "followed_team": "Harbour Hawks",      // optional; default: the home team
      "time_basis": "elapsed",               // "elapsed" (default) or "remaining" in the period
      "goals": [
        {"period": 1, "time": "05:12", "team": "Harbour Hawks",
         "scorer": {"name": "A. Player", "number": "9"},
         "assists": [{"name": "B. Player", "number": "12"}], "special": "PP"}
      ],
      "penalties": [
        {"period": 1, "time": "03:40", "team": "Ridge Rams",
         "player": {"name": "C. Player", "number": "4"}, "infraction": "Tripping", "minutes": 2}
      ]
    }

Optional context keys the engine reads for clock rules: "playoff" (bool), "game_number",
"schedule_notes", and "result" ({"overtime": bool, "shootout": bool}).

Names may be plain strings instead of {"name", "number"} objects. `special` is "PP", "SH", "EN"
or empty. Periods are 1-3, 4 for overtime, 5 for a shootout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..goal import Goal, GoalType
from .base import PreloadedBoxScoreFetcher

FORMAT = "manual-box-score/1"


def _person(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {"name": str(value.get("name") or "").strip(), "number": value.get("number")}
    return {"name": str(value or "").strip(), "number": None}


def _special(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in {"PP", "SH", "EN"} else ""


class ManualBoxScoreProvider:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError(f'{self.path}: expected "format": "{FORMAT}"')
        for key in ("date", "home_team", "away_team"):
            if not str(data.get(key) or "").strip():
                raise ValueError(f"{self.path}: missing {key}")
        if data.get("time_basis", "elapsed") not in {"elapsed", "remaining"}:
            raise ValueError(f'{self.path}: time_basis must be "elapsed" or "remaining"')
        self.data = data

    @property
    def game_id(self) -> str:
        return str(self.data.get("game_id") or f"{self.data['date']}-{self.data['home_team']}-{self.data['away_team']}")

    @property
    def time_is_elapsed(self) -> bool:
        return self.data.get("time_basis", "elapsed") == "elapsed"

    @property
    def followed_team(self) -> str:
        return str(self.data.get("followed_team") or self.data["home_team"])

    def game_info(self, filename: str) -> Dict[str, Any]:
        """Keyword arguments for hockey_extractor.models.GameInfo."""
        home = self.data["home_team"]
        away = self.data["away_team"]
        followed = self.followed_team.lower()
        home_away = "away" if followed and followed in away.lower() and followed not in home.lower() else "home"
        league = str(self.data.get("league") or "").strip()
        if league:
            from .. import leagues
            pack = leagues.get(league)
            league = str((pack or {}).get("short") or league).upper()
        else:
            from .. import leagues
            league = leagues.detect_league(home, away)
        result = self.data.get("result") or {}
        return {"date": str(self.data["date"]), "date_formatted": str(self.data["date"]), "home_team": home,
                "away_team": away, "league": league, "filename": filename, "home_away": home_away,
                "overtime": bool(result.get("overtime")), "shootout": bool(result.get("shootout"))}

    def goals(self) -> List[Dict[str, Any]]:
        out = []
        for g in self.data.get("goals") or []:
            assists = [_person(a) for a in (g.get("assists") or [])]
            out.append({
                "period": int(g.get("period", 1)),
                "time": str(g.get("time", "0:00")),
                "team": str(g.get("team", "")),
                "goal": _person(g.get("scorer")),
                "assist1": assists[0] if len(assists) > 0 else {},
                "assist2": assists[1] if len(assists) > 1 else {},
                "plus_minus": _special(g.get("special")),
            })
        return out

    def penalties(self) -> List[Dict[str, Any]]:
        return [{
            "period": int(p.get("period", 1)),
            "time": str(p.get("time", "0:00")),
            "team": str(p.get("team", "")),
            "player": _person(p.get("player")),
            "description": str(p.get("infraction", "")),
            "infraction": str(p.get("infraction", "")),
            "minutes": p.get("minutes", 2),
        } for p in self.data.get("penalties") or []]

    def box_score(self) -> Dict[str, Any]:
        """The game in HockeyTech's `SiteKit.Gamesummary` shape."""
        return {"SiteKit": {"Gamesummary": {
            "meta": {"game_id": self.game_id, "date": self.data["date"],
                     "home_team": self.data["home_team"], "away_team": self.data["away_team"]},
            "goals": self.goals(), "penalties": self.penalties(),
            "result": self.data.get("result") or {},
        }}, "_source_context": {k: self.data[k] for k in ("playoff", "schedule_notes", "result", "date", "game_number")
                               if self.data.get(k) not in (None, "")}}

    def typed_goals(self) -> List[Goal]:
        kinds = {"PP": GoalType.POWER_PLAY, "SH": GoalType.SHORT_HANDED, "EN": GoalType.EMPTY_NET}
        goals = []
        for g in self.goals():
            goals.append(Goal(period=g["period"], time=g["time"], team=g["team"], scorer=g["goal"]["name"] or "Unknown",
                              assist1=g["assist1"].get("name") or None, assist2=g["assist2"].get("name") or None,
                              goal_type=kinds.get(g["plus_minus"])))
        return goals

    def create_fetcher(self) -> PreloadedBoxScoreFetcher:
        return PreloadedBoxScoreFetcher(self.game_id, self.box_score(), self.typed_goals())
