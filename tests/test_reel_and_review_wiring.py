"""The recording -> clips -> (optional review) -> reel wiring, without video or a browser."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from hockey_extractor import reel  # noqa: E402

CTX = reel.GameContext(
    home="Harbour Hawks", away="Ridge Rams", league="DHL", date="2026-01-09", time_is_elapsed=True,
    goals=[{"period": 1, "time": "05:12", "team": "Harbour Hawks"},
           {"period": 2, "time": "01:00", "team": "Ridge Rams"},
           {"period": 2, "time": "09:35", "team": "Harbour Hawks"}])


def league():
    return reel.load_league("DHL")


def test_goal_spec_score_clock_and_wording():
    ev = {"type": "goal", "period": 2, "time": "09:35", "team": "Harbour Hawks", "scorer": "Sam Rivera",
          "assist1": "Alex Tremblay", "assist2": "", "power_play": True}
    spec = reel.clip_spec(league(), CTX, ev)
    assert spec["kind"] == "score" and spec["side"] == "home"
    assert spec["score"] == {"home": 2, "away": 1}          # score after this goal
    assert spec["clock"]["time"] == "10:25"                  # elapsed 9:35 -> 10:25 left, as the pack's clock is remaining
    assert spec["headline"] == "POWER-PLAY GOAL" and spec["tags"] == ["PP"]
    assert spec["lines"] == ["Assists: Alex Tremblay"]
    assert spec["subject"]["name"] == "Sam Rivera"


def test_penalty_and_fight_specs():
    pen = {"type": "penalty", "period": 1, "time": "03:40", "team": "Ridge Rams",
           "player": {"name": "Pat Gallant"}, "infraction": "Tripping", "minutes": 2}
    spec = reel.clip_spec(league(), CTX, pen)
    assert spec["kind"] == "penalty" and spec["side"] == "away" and spec["lines"] == ["Tripping · 2 min"]
    fight = dict(pen, infraction="Fighting - Major", minutes=5)
    assert reel.clip_spec(league(), CTX, fight)["kind"] == "fight"
    assert reel.clip_spec(league(), CTX, {"type": "save"}) is None


def test_unknown_league_uses_neutral_pack():
    lg = reel.load_league("Unknown", "A", "B")
    spec = reel.clip_spec(lg, CTX, {"type": "goal", "period": 1, "time": "05:12", "team": "Harbour Hawks", "scorer": "X"})
    assert spec["home"]["logo"].endswith("fallback.png")


def test_display_time_follows_pack_clock():
    pack = {"clock": "elapsed"}
    assert reel.display_time(pack, 1, "05:12") == "5:12"
    assert reel.display_time({"clock": "remaining"}, 1, "05:12") == "14:48"
    assert reel.display_time({"clock": "remaining"}, 4, "01:00") == "4:00"


def test_event_offset_for_engine_and_reviewed_clips():
    assert reel._event_offset({"before_seconds": 32.0}) == 32.0
    assert reel._event_offset({"clip_video_start": 100.0, "video_time": 121.5, "before_seconds": 32}) == 21.5


def test_review_is_off_without_configuration(monkeypatch):
    import run_game
    for var in ("CLIP_REVIEW_AGENT_CMD", "CLIP_REVIEW_API_KEY", "CLIP_REVIEW_BASE_URL", "CLIP_REVIEW_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert run_game.review_mode("auto") == "off"
    monkeypatch.setenv("CLIP_REVIEW_AGENT_CMD", "agent {prompt}")
    assert run_game.review_mode("auto") == "agent"
    monkeypatch.setenv("CLIP_REVIEW_API_KEY", "k")
    monkeypatch.setenv("CLIP_REVIEW_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("CLIP_REVIEW_MODEL", "m")
    assert run_game.review_mode("auto") == "escalate"
    monkeypatch.delenv("CLIP_REVIEW_AGENT_CMD")
    assert run_game.review_mode("auto") == "api"
    assert run_game.review_mode("off") == "off"


def test_api_backend_has_no_default_provider(monkeypatch):
    from clip_review.backends import ApiBackend
    for var in ("CLIP_REVIEW_BASE_URL", "CLIP_REVIEW_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError):
        ApiBackend()


def test_apply_writes_reel_manifest_from_reviewed_windows(tmp_path):
    from clip_review.apply import write_overrides, apply_overrides
    game = tmp_path / "game"
    (game / "clips").mkdir(parents=True)
    clip = game / "clips" / "01.mp4"
    clip.write_bytes(b"0")
    ev = {"type": "goal", "period": 1, "time": "05:12", "team": "Harbour Hawks", "scorer": "S", "video_time": 80.0}
    inc = {"id": "01_goal_p1_05-12", "kind": "goal", "class": "goal", "period": 1, "time_elapsed": "05:12", "video_time": 80.0,
           "engine_in": 48.0, "engine_out": 83.0, "clip_path": "clips/01.mp4", "rows": [], "events": [ev]}
    results = {inc["id"]: {"final": {"status": "keep"}, "verdict": {"decision": "keep"}}}
    out = game / "data" / "review"
    ov = write_overrides(game, out, tmp_path / "v.mp4", [inc], results, {})
    applied = apply_overrides(game, out, tmp_path / "v.mp4", [inc], ov)
    manifest = json.loads(Path(applied["manifests"]["main"]).read_text())
    assert [c["clip_filename"] for c in manifest["clips"]] == ["01.mp4"]
    assert manifest["clips"][0]["review_status"] == "keep"
    assert reel.load_clips(game, Path(applied["manifests"]["main"]))[0]["_file"].endswith("01.mp4")


def test_stub_agent_follows_the_agent_command_protocol(tmp_path):
    """The example agent writes a verdict that the skill's checker accepts, from a prompt with spaces in paths."""
    import subprocess
    packet = tmp_path / "Games" / "Harbour Hawks vs Ridge Rams" / "packet"
    packet.mkdir(parents=True)
    bounds = {"lead_min": 15.0, "lead_max": 45, "tail_min": 8, "tail_max": 25, "len_min": 8, "len_max": 60, "near": 30,
              "relocate_max": 240, "drop_min_confidence": 0.8, "tail_min_replay": 5, "tail_trim": 16.0,
              "fight_pre": 10, "fight_post": 10, "engine_in": -32.0, "engine_out": 3.0, "video_from": -72.0, "video_to": 198.0}
    (packet / "incident.json").write_text(json.dumps({"incident_id": "01_goal_p1_05-12", "kind": "goal", "class": "goal",
                                                      "bounds": bounds}))
    out = packet / "out" / "verdict.json"
    prompt = f"Packet directory: {packet} . Start by reading it.\nWrite the verdict JSON to: {out}\nThen run something."
    subprocess.run([sys.executable, str(REPO / "examples" / "stub_review_agent.py"), prompt], check=True, cwd=packet)
    checked = subprocess.run([sys.executable, str(REPO / "skills" / "hockey-clip-review" / "scripts" / "check_verdict.py"),
                              str(out), "--packet", str(packet), "--json"], capture_output=True, text=True)
    assert json.loads(checked.stdout)["ok"], checked.stdout
