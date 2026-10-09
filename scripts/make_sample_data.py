#!/usr/bin/env python3
"""
Generate synthetic sample data: a small fake "broadcast" video and its box score.

  python scripts/make_sample_data.py [--out sample_data]

Writes <out>/sample_game.mp4 (4.5 minutes, 1280x720, 10 fps, a few MB) and
<out>/sample_game.json (a manual box score, format manual-box-score/1). The video is drawn
frame by frame: a flat rink with moving dots, a scorebug in the "flo_corner_period_first"
layout (period and game clock top-left, running score beside them), a frozen clock after each
whistle and a GOAL flash. It is not a real broadcast and contains no real footage, teams or
people; the teams are the fictional ones in overlays/leagues/demo-hockey.

The recording starts 4:00 into the 1st period, so it is also a partial-recording example.
Needs Pillow and ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
W, H, FPS = 1280, 720, 10
PERIOD_LEN = 20 * 60
START_ELAPSED = 4 * 60            # the recording starts 4:00 into the 1st period
DURATION = 270                    # seconds of video
FREEZE = 22                       # clock stays frozen this long after a goal or a penalty call

HOME, AWAY = "Harbour Hawks", "Ridge Rams"
# (elapsed in period 1, kind, team, details)
EVENTS = [
    ("05:12", "goal", HOME, {"scorer": "Sam Rivera", "number": "17", "assists": ["Alex Tremblay", "Jordan Leblanc"], "special": ""}),
    ("06:30", "penalty", AWAY, {"player": "Pat Gallant", "number": "4", "infraction": "Tripping", "minutes": 2}),
    ("07:15", "goal", HOME, {"scorer": "Alex Tremblay", "number": "9", "assists": ["Sam Rivera"], "special": "PP"}),
]


def secs(text: str) -> int:
    m, s = text.split(":")
    return int(m) * 60 + int(s)


def game_clock(t: float) -> int:
    """Elapsed seconds in the period at video time t: runs at real speed, frozen after each event."""
    elapsed = START_ELAPSED + t
    shift = 0.0
    for ev in EVENTS:
        at = secs(ev[0])
        if elapsed >= at + FREEZE + shift:
            shift += FREEZE
        elif elapsed >= at + shift:
            return at
    return int(START_ELAPSED + t - shift)


def video_time_of(elapsed_text: str) -> float:
    """Video second at which the game clock first reads `elapsed_text`."""
    target = secs(elapsed_text)
    t = 0.0
    while game_clock(t) < target:
        t += 0.1
    return t


def score_at(t: float):
    h = a = 0
    for ev in EVENTS:
        if ev[1] == "goal" and t >= video_time_of(ev[0]):
            if ev[2] == HOME:
                h += 1
            else:
                a += 1
    return h, a


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(REPO / "assets" / "fonts" / "BarlowSemiCondensed-SemiBold.ttf"), size)


def frame(t: float) -> Image.Image:
    im = Image.new("RGB", (W, H), (214, 232, 244))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 90), fill=(180, 200, 214))                   # boards
    d.line((W // 2, 90, W // 2, H), fill=(190, 40, 50), width=6)       # centre line
    d.ellipse((W // 2 - 70, H // 2 - 20, W // 2 + 70, H // 2 + 120), outline=(30, 80, 160), width=4)
    # players and puck: deterministic loops, nothing random
    for i in range(10):
        x = W / 2 + math.sin(t * 0.35 + i) * (300 + i * 12)
        y = 300 + math.cos(t * 0.5 + i * 1.7) * 120
        d.ellipse((x - 11, y - 11, x + 11, y + 11), fill=(20, 67, 122) if i % 2 else (179, 38, 45))
    px, py = W / 2 + math.sin(t * 0.9) * 260, 300 + math.cos(t * 1.1) * 100
    d.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(15, 15, 15))
    # goal flash for 8 s after the red light
    for ev in EVENTS:
        if ev[1] == "goal":
            g = video_time_of(ev[0])
            if g <= t < g + 8:
                d.rectangle((0, 0, W, H), outline=(255, 70, 70), width=14)
                d.text((W // 2, 200), "GOAL!", font=font(120), fill=(255, 70, 70), anchor="mm")
        if ev[1] == "penalty":
            p = video_time_of(ev[0])
            if p <= t < p + 6:
                d.text((W // 2, 200), "PENALTY", font=font(90), fill=(240, 160, 30), anchor="mm")
    # scorebug, top-left: period + clock box, then team abbreviations and score
    el = game_clock(t)
    remaining = PERIOD_LEN - el
    d.rectangle((30, 18, 600, 76), fill=(18, 28, 44))
    # the light box is larger than the OCR crop (46..198 x 28..66 at 1280x720) so no box edge is read as text
    d.rectangle((36, 20, 208, 74), fill=(244, 246, 250))
    d.text((52, 47), "1ST", font=font(34), fill=(10, 18, 30), anchor="lm")
    d.text((192, 47), f"{remaining // 60}:{remaining % 60:02d}", font=font(32), fill=(10, 18, 30), anchor="rm")
    h, a = score_at(t)
    d.text((220, 47), "HAW", font=font(30), fill=(255, 255, 255), anchor="lm")
    d.text((300, 47), str(h), font=font(34), fill=(240, 200, 60), anchor="lm")
    d.text((350, 47), "RRM", font=font(30), fill=(255, 255, 255), anchor="lm")
    d.text((430, 47), str(a), font=font(34), fill=(240, 200, 60), anchor="lm")
    d.text((470, 47), "SAMPLE", font=font(20), fill=(130, 150, 175), anchor="lm")
    return im


def box_score() -> dict:
    goals, penalties = [], []
    for el, kind, team, det in EVENTS:
        if kind == "goal":
            goals.append({"period": 1, "time": el, "team": team,
                          "scorer": {"name": det["scorer"], "number": det["number"]},
                          "assists": [{"name": n} for n in det["assists"]], "special": det["special"]})
        else:
            penalties.append({"period": 1, "time": el, "team": team,
                              "player": {"name": det["player"], "number": det["number"]},
                              "infraction": det["infraction"], "minutes": det["minutes"]})
    return {"format": "manual-box-score/1", "game_id": "sample-001", "date": "2026-01-09", "league": "demo-hockey",
            "home_team": HOME, "away_team": AWAY, "followed_team": HOME, "time_basis": "elapsed",
            "goals": goals, "penalties": penalties, "result": {"final_score": "2-0"}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=Path("sample_data"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    video = args.out / "sample_game.mp4"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", "2", "-i", "-",
           "-f", "lavfi", "-i", f"sine=frequency=180:sample_rate=48000:duration={DURATION}",
           "-vf", f"fps={FPS}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "48k", "-t", str(DURATION), "-movflags", "+faststart", str(video)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(DURATION * 2):                       # 2 drawn frames per second, ffmpeg repeats them to 10 fps
        proc.stdin.write(frame(i / 2).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        print("ffmpeg failed", file=sys.stderr)
        return 1
    (args.out / "sample_game.json").write_text(json.dumps(box_score(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {video} ({video.stat().st_size / 1e6:.1f} MB) and {args.out / 'sample_game.json'}", file=sys.stderr)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
