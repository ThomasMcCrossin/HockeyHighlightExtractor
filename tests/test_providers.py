"""Box-score providers: the manual JSON format and the league packs behind HockeyTech."""

import json
from pathlib import Path

import pytest

from hockey_extractor import leagues
from hockey_extractor.box_score import BoxScoreFetcher
from hockey_extractor.goal import GoalType
from hockey_extractor.providers import ManualBoxScoreProvider

GAME = {
    "format": "manual-box-score/1",
    "game_id": "g1",
    "date": "2026-01-09",
    "league": "mhl",
    "home_team": "Harbour Hawks",
    "away_team": "Ridge Rams",
    "goals": [
        {"period": 1, "time": "05:12", "team": "Harbour Hawks", "scorer": {"name": "A Player", "number": "9"},
         "assists": ["B Player", {"name": "C Player"}], "special": "pp"},
        {"period": 3, "time": "19:01", "team": "Ridge Rams", "scorer": "D Player", "special": "EN"},
    ],
    "penalties": [{"period": 1, "time": "03:40", "team": "Ridge Rams", "player": "E Player",
                   "infraction": "Tripping", "minutes": 2}],
}


def write(tmp_path: Path, **over) -> Path:
    path = tmp_path / "game.json"
    path.write_text(json.dumps({**GAME, **over}), encoding="utf-8")
    return path


def test_manual_box_score_has_hockeytech_shape(tmp_path):
    p = ManualBoxScoreProvider(write(tmp_path))
    summary = p.box_score()["SiteKit"]["Gamesummary"]
    assert [g["plus_minus"] for g in summary["goals"]] == ["PP", "EN"]
    assert summary["goals"][0]["assist2"] == {"name": "C Player", "number": None}
    assert summary["penalties"][0]["player"]["name"] == "E Player"
    # the engine's own parser reads it
    events = BoxScoreFetcher().extract_events(p.box_score())
    assert [e["type"] for e in events].count("goal") == 2
    assert [e["type"] for e in events].count("penalty") == 1


def test_manual_typed_goals_and_fetcher(tmp_path):
    p = ManualBoxScoreProvider(write(tmp_path))
    goals = p.typed_goals()
    assert goals[0].goal_type == GoalType.POWER_PLAY and goals[0].assist2 == "C Player"
    assert goals[1].goal_type == GoalType.EMPTY_NET
    fetcher = p.create_fetcher()
    assert fetcher.find_game("mhl", "x", "y", "2026-01-09") == "g1"
    assert fetcher.fetch_box_score("mhl", "g1") == p.box_score()


def test_manual_game_info_and_followed_team(tmp_path):
    info = ManualBoxScoreProvider(write(tmp_path)).game_info("rec.mp4")
    assert info["league"] == "MHL" and info["home_away"] == "home"
    away = ManualBoxScoreProvider(write(tmp_path, followed_team="Ridge Rams")).game_info("rec.mp4")
    assert away["home_away"] == "away"


def test_manual_rejects_wrong_format(tmp_path):
    with pytest.raises(ValueError):
        ManualBoxScoreProvider(write(tmp_path, format="other"))
    with pytest.raises(ValueError):
        ManualBoxScoreProvider(write(tmp_path, time_basis="sideways"))


def test_league_packs_drive_hockeytech_config_and_detection():
    assert BoxScoreFetcher._league_config("MHL")["client_code"] == "mhl"
    assert BoxScoreFetcher._league_config("demo-soccer") is None
    assert leagues.detect_league("Truro Bearcats", "Nobody FC") == "MHL"
    assert leagues.detect_league("Nobody FC") == "Unknown"
    assert leagues.find_team("AMH")["name"] == "Amherst Ramblers"


def test_extra_league_dir(tmp_path, monkeypatch):
    pack = tmp_path / "mini"
    pack.mkdir()
    (pack / "league.json").write_text(json.dumps({
        "id": "mini", "short": "MIN", "sport": "hockey",
        "teams": [{"id": "ice-owls", "short": "OWL", "name": "Ice Owls", "city": "Frostville", "nickname": "Owls"}]}))
    monkeypatch.setenv("HOCKEY_LEAGUES_DIR", str(tmp_path))
    leagues._load_all.cache_clear()
    try:
        assert leagues.detect_league("Frostville Ice Owls") == "MIN"
    finally:
        monkeypatch.delenv("HOCKEY_LEAGUES_DIR")
        leagues._load_all.cache_clear()
