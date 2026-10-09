"""
Reel builder: engine (or reviewed) clips -> overlay specs -> overlay PNGs -> one reel.

Reads a processed game folder (`data/game_metadata.json`, `data/matched_events.json`,
`data/clips_manifest.json`) or a reviewed manifest (`data/review/reel_main.json`), wording each
clip from the league pack (see overlays/README.md), renders the PNGs through the overlay
renderer once, composites each over its clip with ffmpeg and joins the clips.

Needs ffmpeg and ffprobe; overlays also need node and Playwright (see README). With
`overlays=False` the clips are only joined, which needs nothing but ffmpeg.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import leagues

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
OVERLAY_SECONDS = 6.0
OVERLAY_LEAD_SECONDS = 0.5
FPS = 30
HEADLINES = {"": "GOAL", "PP": "POWER-PLAY GOAL", "SH": "SHORT-HANDED GOAL", "EN": "EMPTY-NET GOAL"}


# ---------------------------------------------------------------------------- wording

def _seconds(text: Any) -> Optional[int]:
    parts = str(text or "").replace(".", ":").split(":")
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return None


def _clock(sec: int) -> str:
    return f"{sec // 60}:{sec % 60:02d}"


def period_seconds(pack: Dict[str, Any], period: int) -> int:
    lengths = pack.get("period_minutes") or {}
    regular = float(lengths.get("regulation", 20))
    overtime = float(lengths.get("overtime", 5))
    return int((regular if period <= 3 else overtime) * 60)


def display_time(pack: Dict[str, Any], period: int, elapsed_text: Any, time_is_elapsed: bool = True) -> str:
    """The time as the league's broadcast shows it (`clock` in the pack: remaining or elapsed)."""
    sec = _seconds(elapsed_text)
    if sec is None:
        return str(elapsed_text or "")
    if (pack.get("clock", "remaining") == "remaining") == time_is_elapsed:
        sec = max(0, period_seconds(pack, period) - sec)
    return _clock(sec)


def _neutral_pack(home: str, away: str) -> Dict[str, Any]:
    return {"id": "neutral", "name": "", "short": "", "sport": "hockey", "logo": "", "primary": "#1f2d44",
            "secondary": "#c8102e", "fallback_logo": "assets/logos/fallback.png",
            "periods": {"1": "1st", "2": "2nd", "3": "3rd", "4": "OT", "5": "SO"}, "clock": "remaining",
            "teams": []}


def load_league(league: str, home: str = "", away: str = ""):
    from overlays.spec import League

    pack = leagues.get(league) if league and league != "Unknown" else None
    if pack is None:
        return League.from_pack(_neutral_pack(home, away))
    return League.from_pack(pack)


def _name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "").strip()
    return str(value or "").strip()


def _scoreline(goals: List[Dict[str, Any]], home: str, time_is_elapsed: bool, upto: Tuple[int, int]) -> Tuple[int, int]:
    """Score after every goal at or before (period, elapsed seconds) `upto`."""
    h = a = 0
    for g in goals:
        key = _goal_key(g, time_is_elapsed)
        if key is None or key > upto:
            continue
        if _same_team(g.get("team"), home):
            h += 1
        else:
            a += 1
    return h, a


def _goal_key(g: Dict[str, Any], time_is_elapsed: bool) -> Optional[Tuple[int, int]]:
    sec = _seconds(g.get("time"))
    if sec is None:
        return None
    period = int(g.get("period") or 1)
    # Sort by elapsed time; a remaining clock counts down, so flip it.
    return (period, sec if time_is_elapsed else -sec)


def _same_team(a: Any, b: Any) -> bool:
    x, y = str(a or "").strip().lower(), str(b or "").strip().lower()
    return bool(x) and bool(y) and (x == y or x in y or y in x)


@dataclass
class GameContext:
    home: str
    away: str
    league: str
    goals: List[Dict[str, Any]]
    time_is_elapsed: bool
    date: str


def game_context(game_dir: Path) -> GameContext:
    data = Path(game_dir) / "data"
    meta = json.loads((data / "game_metadata.json").read_text(encoding="utf-8")) if (data / "game_metadata.json").exists() else {}
    info = meta.get("game_info") or {}
    run = json.loads((data / "run_config.json").read_text(encoding="utf-8")) if (data / "run_config.json").exists() else {}
    summary = ((meta.get("box_score") or {}).get("SiteKit") or {}).get("Gamesummary") or {}
    goals = []
    for g in summary.get("goals") or []:
        goals.append({"period": g.get("period"), "time": g.get("time"), "team": g.get("team")})
    if not goals and (data / "matched_events.json").exists():
        events = json.loads((data / "matched_events.json").read_text(encoding="utf-8"))
        goals = [{"period": e.get("period"), "time": e.get("time"), "team": e.get("team")}
                 for e in events if e.get("type") == "goal"]
    return GameContext(home=str(info.get("home_team") or "Home"), away=str(info.get("away_team") or "Away"),
                       league=str(run.get("league") or info.get("league") or ""), goals=goals,
                       time_is_elapsed=bool(run.get("time_is_elapsed", True)), date=str(info.get("date") or ""))


def clip_spec(league, ctx: GameContext, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The overlay spec for one clip (a goal or penalty event dict), or None for other types."""
    pack = league.pack
    period = int(event.get("period") or 1)
    elapsed_sec = _seconds(event.get("time"))
    key = (period, (elapsed_sec or 0) if ctx.time_is_elapsed else -(elapsed_sec or 0))
    score = _scoreline(ctx.goals, ctx.home, ctx.time_is_elapsed, key)
    side = "home" if _same_team(event.get("team"), ctx.home) else "away"
    shown = display_time(pack, period, event.get("time"), ctx.time_is_elapsed)
    common = dict(home=ctx.home, away=ctx.away, side=side, period=period, time=shown, score=score,
                  context=str(pack.get("name") or ""))
    if event.get("type") == "goal":
        special = ("PP" if event.get("power_play") else "SH" if event.get("short_handed")
                   else "EN" if event.get("empty_net") else str(event.get("special") or "").upper())
        assists = [a for a in (_name(event.get("assist1")), _name(event.get("assist2"))) if a]
        return league.spec("score", headline=HEADLINES.get(special, "GOAL"), subject=(_name(event.get("scorer")), ""),
                           lines=[("Assists: " + ", ".join(assists)) if assists else "Unassisted"],
                           tags=[special] if special in {"PP", "SH", "EN"} else [],
                           badge="UNVERIFIED" if event.get("match_unreliable") else None, **common)
    if event.get("type") == "penalty":
        text = str(event.get("infraction") or "").strip()
        minutes = event.get("minutes")
        player = _name(event.get("player"))
        if "fight" in text.lower():
            return league.spec("fight", headline="FIGHTING MAJORS", subject=(player, ""),
                               lines=[f"{minutes} min"] if minutes else [], **{**common, "side": None})
        detail = " · ".join(x for x in (text, f"{minutes} min" if minutes else "") if x)
        return league.spec("penalty", headline="PENALTY", subject=(player, ""), lines=[detail] if detail else [], **common)
    return None


# ---------------------------------------------------------------------------- rendering

def render_overlays(specs: Dict[str, Dict[str, Any]], work_dir: Path, theme: str) -> Dict[str, Path]:
    """Render every spec with one renderer run. Returns {name: png path}."""
    spec_dir = work_dir / "specs"
    out_dir = work_dir / "png"
    shutil.rmtree(spec_dir, ignore_errors=True)
    spec_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in specs.items():
        (spec_dir / f"{name}.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    proc = subprocess.run(["node", str(REPO / "overlays" / "render.mjs"), "--theme", theme,
                           "--samples", str(spec_dir), "--out", str(out_dir)],
                          cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("overlay renderer failed (is `npm install` done and `npx playwright install chromium` run?): "
                           + (proc.stderr or proc.stdout)[-600:])
    return {name: out_dir / theme / f"{name}.png" for name in specs}


# ---------------------------------------------------------------------------- ffmpeg

def probe(path: Path) -> Dict[str, Any]:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), {})
    return {"width": int(video.get("width") or 0), "height": int(video.get("height") or 0),
            "duration": float(data["format"].get("duration") or 0), "audio": any(s["codec_type"] == "audio" for s in data["streams"])}


def _encode_args(config) -> List[str]:
    return ["-c:v", str(getattr(config, "OUTPUT_CODEC", "libx264")), "-preset", str(getattr(config, "OUTPUT_PRESET", "veryfast")),
            "-crf", str(getattr(config, "OUTPUT_CRF", 20)), "-pix_fmt", str(getattr(config, "OUTPUT_PIXEL_FORMAT", "yuv420p")),
            "-r", str(FPS), "-c:a", str(getattr(config, "OUTPUT_AUDIO_CODEC", "aac")), "-b:a", str(getattr(config, "OUTPUT_AUDIO_BITRATE", "192k")),
            "-ar", "48000", "-ac", "2", "-movflags", "+faststart"]


def composite_clip(clip: Path, png: Optional[Path], dest: Path, size: Tuple[int, int], overlay_at: float, config) -> None:
    """Normalise a clip to `size` / 30 fps / stereo 48 kHz and lay the overlay PNG over it."""
    w, h = size
    info = probe(clip)
    dur = max(info["duration"], 0.1)
    base = (f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,fps={FPS},format=yuv420p")
    cmd = ["nice", "-n", "10", "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(clip)]
    idx = 1
    if png is not None:
        shown = max(0.5, min(OVERLAY_SECONDS, dur - overlay_at))
        cmd += ["-loop", "1", "-framerate", str(FPS), "-t", f"{shown:.3f}", "-i", str(png)]
        fade = 0.25
        graph = (f"{base}[b];[{idx}:v]scale={w}:{h},format=rgba,fade=t=in:st=0:d={fade}:alpha=1,"
                 f"fade=t=out:st={max(0.0, shown - fade):.3f}:d={fade}:alpha=1,setpts=PTS+{overlay_at:.3f}/TB[o];"
                 f"[b][o]overlay=eof_action=pass:format=auto[v]")
        idx += 1
    else:
        graph = f"{base}[v]"
    if info["audio"]:
        graph += ";[0:a]aresample=48000,aformat=channel_layouts=stereo[a]"
    else:
        cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        graph += f";[{idx}:a]anull[a]"
    cmd += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}", *_encode_args(config), str(dest)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def card_clip(png: Path, dest: Path, size: Tuple[int, int], seconds: float, color: str, config) -> None:
    """A full-screen card (intro or final): the overlay PNG over a solid colour."""
    w, h = size
    cmd = ["nice", "-n", "10", "ffmpeg", "-v", "error", "-nostdin", "-y",
           "-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}:d={seconds}:r={FPS}",
           "-loop", "1", "-framerate", str(FPS), "-t", str(seconds), "-i", str(png),
           "-f", "lavfi", "-t", str(seconds), "-i", "anullsrc=r=48000:cl=stereo",
           "-filter_complex", f"[1:v]scale={w}:{h},format=rgba[o];[0:v][o]overlay=format=auto,format=yuv420p[v]",
           "-map", "[v]", "-map", "2:a", "-t", str(seconds), *_encode_args(config), str(dest)]
    subprocess.run(cmd, check=True)


def concat(parts: List[Path], dest: Path) -> None:
    listing = dest.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", "-movflags", "+faststart", str(dest)], check=True)
    finally:
        listing.unlink(missing_ok=True)


# ---------------------------------------------------------------------------- the reel

def load_clips(game_dir: Path, manifest: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Clip entries (event dicts with a `path`) from a manifest, in reel order."""
    game_dir = Path(game_dir)
    path = Path(manifest) if manifest else game_dir / "data" / "clips_manifest.json"
    clips = json.loads(path.read_text(encoding="utf-8")).get("clips", [])
    out = []
    for c in clips:
        p = Path(c["path"])
        full = p if p.is_absolute() else game_dir / p
        if full.exists():
            out.append(dict(c, _file=str(full)))
        else:
            logger.warning("clip missing, skipped: %s", full)
    return out


def _event_offset(c: Dict[str, Any]) -> float:
    """Seconds into the clip where the event happens."""
    if c.get("clip_video_start") is not None and c.get("video_time") is not None:
        return float(c["video_time"]) - float(c["clip_video_start"])
    return float(c.get("before_seconds") or 0.0)


def build_reel(game_dir: Path, *, manifest: Optional[Path] = None, theme: str = "baseline", overlays: bool = True,
               cards: bool = False, output: Optional[Path] = None, config=None) -> Dict[str, Any]:
    """Build the reel. Returns {"output", "clips", "overlays", "theme"}."""
    if config is None:
        import config as _config
        config = _config
    game_dir = Path(game_dir)
    clips = load_clips(game_dir, manifest)
    if not clips:
        raise RuntimeError(f"no clips to build a reel from in {manifest or game_dir / 'data' / 'clips_manifest.json'}")
    ctx = game_context(game_dir)
    league = load_league(ctx.league, ctx.home, ctx.away)
    work = game_dir / "output" / "reel_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    size = probe(Path(clips[0]["_file"]))
    size = (size["width"] or 1920, size["height"] or 1080)
    size = (size[0] - size[0] % 2, size[1] - size[1] % 2)

    specs: Dict[str, Dict[str, Any]] = {}
    if overlays:
        for i, c in enumerate(clips, 1):
            spec = clip_spec(league, ctx, c)
            if spec:
                specs[f"{i:02d}"] = spec
        if cards:
            hs, as_ = _scoreline(ctx.goals, ctx.home, ctx.time_is_elapsed, (99, 10 ** 6))
            specs["intro"] = league.spec("intro", home=ctx.home, away=ctx.away, headline="HIGHLIGHTS",
                                         lines=[ctx.date] if ctx.date else [], context=str(league.pack.get("name") or ""))
            specs["final"] = league.spec("final", home=ctx.home, away=ctx.away, score=(hs, as_), headline="FINAL",
                                         clock_text="Final", period=None, context=str(league.pack.get("name") or ""))
    pngs = render_overlays(specs, work, theme) if specs else {}

    parts: List[Path] = []
    primary = (league.pack.get("primary") or "#101820").lstrip("#")
    if "intro" in pngs:
        p = work / "00_intro.mp4"
        card_clip(pngs["intro"], p, size, 3, f"0x{primary}", config)
        parts.append(p)
    for i, c in enumerate(clips, 1):
        dest = work / f"{i:02d}.mp4"
        composite_clip(Path(c["_file"]), pngs.get(f"{i:02d}"), dest, size, max(0.0, _event_offset(c) - OVERLAY_LEAD_SECONDS), config)
        parts.append(dest)
    if "final" in pngs:
        p = work / "99_final.mp4"
        card_clip(pngs["final"], p, size, 4, f"0x{primary}", config)
        parts.append(p)
    out = Path(output) if output else game_dir / "output" / "reel.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    concat(parts, out)
    keep = game_dir / "output" / "overlays"
    shutil.rmtree(keep, ignore_errors=True)
    if pngs:
        keep.mkdir(parents=True, exist_ok=True)
        for name, png in pngs.items():
            shutil.copy2(png, keep / f"{name}.png")
    shutil.rmtree(work, ignore_errors=True)
    return {"output": str(out), "clips": len(clips), "overlays": len(specs), "theme": theme if specs else None}
