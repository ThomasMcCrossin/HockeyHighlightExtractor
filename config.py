"""
Engine configuration: working directories, encode settings, OCR health, clip windows and
scorebug execution profiles.

Everything here is a plain module attribute so the pipeline can read it with getattr().
Paths and the few secrets come from environment variables; nothing is read from a network
drive and nothing is printed on import. Override a value in code before building the
pipeline (the scripts do this for --games-dir) or edit it here.

Environment:
  HOCKEY_GAMES_DIR      where game folders are written (default: ./Games)
  HOCKEY_LOGS_DIR       log directory (default: ./logs)
  HOCKEY_FOLLOWED_TEAM  the team you follow (substring of its name); see docs/adding-a-league.md
  HOCKEYTECH_API_KEY    only for fetching box scores from a HockeyTech league
  RESEND_API_KEY, NOTIFICATION_EMAIL, NOTIFICATION_EMAIL_FROM
                        only for the optional major-penalty review email
"""

import os
from pathlib import Path

from scorebug_profiles import resolve_scorebug_profile

# ---------- Paths ----------
LOCAL_REPO_DIR = Path(__file__).resolve().parent
GAMES_DIR = Path(os.environ.get("HOCKEY_GAMES_DIR") or Path.cwd() / "Games")
LOGS_DIR = Path(os.environ.get("HOCKEY_LOGS_DIR") or Path.cwd() / "logs")
TEMP_DIR = Path(os.environ.get("HOCKEY_TEMP_DIR") or Path.cwd() / "temp")
# Optional extra folder searched for recordings (a synced or mounted directory).
GOOGLE_INPUT_DIR = Path(os.environ["HOCKEY_INPUT_DIR"]) if os.environ.get("HOCKEY_INPUT_DIR") else None
# The team the highlights follow ("our" team for power-play linking and descriptions). Empty:
# the home team of each game.
FOLLOWED_TEAM = os.environ.get("HOCKEY_FOLLOWED_TEAM", "").strip()
# Hashtags appended to the generated video description.
DESCRIPTION_HASHTAGS = os.environ.get("HOCKEY_DESCRIPTION_HASHTAGS", "#hockey #highlights").strip()


def ensure_output_directory():
    """Create the games output directory."""
    try:
        GAMES_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def ensure_logs_directory():
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def ensure_temp_directory():
    try:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def find_video_locations():
    """Directories searched for recordings when none is named."""
    locations = [Path.cwd(), Path.home() / "Downloads", Path.home() / "Desktop"]
    if GOOGLE_INPUT_DIR and GOOGLE_INPUT_DIR.exists():
        locations.insert(1, GOOGLE_INPUT_DIR)
    return locations


# ---------- Optional major-penalty review (Drive upload + email) ----------
# Used only by the "full_production" and "goals_with_approved_majors" reel modes, and only
# when configured. Without these the review clips stay local.
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
NOTIFICATION_EMAIL_TO = os.environ.get("NOTIFICATION_EMAIL", "")
NOTIFICATION_EMAIL_FROM = os.environ.get("NOTIFICATION_EMAIL_FROM", "onboarding@resend.dev")
MAJOR_REVIEW_FLAG_FILE = Path(os.environ.get("HOCKEY_MAJOR_REVIEW_FLAG") or Path.cwd() / "temp" / "major_review_active")
MAJOR_REVIEW_DRIVE_FOLDER_ID = os.environ.get("MAJOR_REVIEW_DRIVE_FOLDER_ID", "")
MAJOR_REVIEW_CHECK_INTERVAL_MINUTES = 5

# ---------- Video / analysis settings (unchanged) ----------
SUPPORTED_FORMATS = ['.ts', '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']

# ffmpeg / encoding defaults
# MoviePy uses ffmpeg under the hood; prefer modern H.264 with CRF (quality-based) encoding.
OUTPUT_CODEC = 'libx264'
OUTPUT_PRESET = 'veryfast'      # clip extraction needs practical encode speed
OUTPUT_CRF = 18                 # 18-20 is typical "visually lossless-ish" for 720p/1080p
OUTPUT_AUDIO_CODEC = 'aac'
OUTPUT_AUDIO_BITRATE = '192k'
OUTPUT_AUDIO_SAMPLE_RATE = 48000
OUTPUT_PIXEL_FORMAT = 'yuv420p'

AUDIO_SAMPLE_RATE = 22050
GOAL_ENERGY_THRESHOLD = 0.75
SAVE_ENERGY_THRESHOLD = 0.65
ANNOUNCER_EXCITEMENT_THRESHOLD = 0.7

MAX_HIGHLIGHT_CLIPS = 12
DEFAULT_CLIP_BEFORE_TIME = 15
DEFAULT_CLIP_AFTER_TIME = 4
BOX_SCORE_TIME_IS_ELAPSED = True  # Box scores list time elapsed in period

# ---------- OCR backend + health settings ----------
# Backends are tried in order for probing and fallback. "easyocr" is optional and
# only used when installed; otherwise it is skipped.
OCR_BACKENDS = ["tesseract", "easyocr"]
OCR_ENABLE_EASYOCR_FALLBACK = True
OCR_EASYOCR_LANGS = ["en"]
OCR_EASYOCR_GPU = False

# Health thresholds for hybrid behavior (probe + rerun sampling before failing).
OCR_MIN_SUCCESS_RATE = 0.05
OCR_MIN_PERIOD_RATE = 0.20
OCR_MIN_AVG_CONFIDENCE = 55.0
OCR_HEALTH_BAD_CONSECUTIVE_SAMPLES_RESET = 10

# Save scorebug-only crops for failed / low-confidence samples so FloHockey OCR
# issues can be diagnosed without storing full-frame images for every attempt.
OCR_DEBUG_SAVE_SCOREBUG_CROPS = True
OCR_DEBUG_SCOREBUG_CROP_DIRNAME = "ocr_scorebug_crops"
OCR_DEBUG_FAILURE_CROP_LIMIT = 40
OCR_DEBUG_LOW_CONFIDENCE_THRESHOLD = 65.0
OCR_DEBUG_LOW_CONFIDENCE_CROP_LIMIT = 25

# Local OCR refinement for low-confidence event matches.
EVENT_LOCAL_OCR_WINDOW_SECONDS = 60.0
EVENT_LOCAL_OCR_STEP_SECONDS = 0.5
EVENT_LOCAL_OCR_PERSISTENCE_WINDOW_SECONDS = 6.0
EVENT_LOCAL_OCR_MIN_HITS = 3
EVENT_LOCAL_OCR_MAX_DIFF_SECONDS = 6.0

# For recorded full-game workflows, an event cannot occur before its own elapsed
# game time relative to the detected puck-drop. This guard prevents P1 warmup
# clocks from matching real goals later in the game if OCR samples are taken too early.
EVENT_ENFORCE_MIN_VIDEO_TIME_FROM_GAME_START = True
EVENT_MIN_VIDEO_TIME_BUFFER_SECONDS = 240.0

# ---------- Penalty clip settings ----------
# PP contributing penalties (shown before powerplay goals)
# The penalty anchor is the first clock reading of the called time, i.e. the whistle, not the
# foul. Judged windows (the 2026-10 clip-review bake-off) start 5-10 s ahead of it, so
# the foul lead-in is in the clip, and run 10-16 s past it, so the referee's signal and the
# walk to the box are too. The old 2 s / 3 s window sat on the stoppage and scored 2.2/10.
PENALTY_PP_BEFORE_SECONDS = 9.0
PENALTY_PP_AFTER_SECONDS = 14.0

# Generic all-penalty mode (includes every penalty call in chronological order)
PENALTY_ALL_BEFORE_SECONDS = 9.0
PENALTY_ALL_AFTER_SECONDS = 14.0

# Goal clips refined from the scoreboard clock-stop are typically anchored at the
# whistle/stoppage, not the puck crossing the line. Give them more lead-in so the
# scoring play is actually visible.
GOAL_CLOCK_STOP_BEFORE_SECONDS = 32.0
# Judged celebration ends sit 10-20 s after the clock stop (median 14 s; the replay wipe
# follows at about +10..+19 s). The old 3 s tail cut the celebration on 81% of clips.
GOAL_CLOCK_STOP_AFTER_SECONDS = 16.0
# Goals the clock can't time (frozen Flo bug) are placed from the broadcast's celebration by a
# vision model (goal_locator.py; needs the optional vision endpoint, see hockey_extractor/vision.py). False disables.
GOAL_VISION_LOCATOR = True
GOAL_VISION_AFTER_SECONDS = 10.0
GOAL_FALLBACK_BEFORE_SECONDS = 20.0
GOAL_FALLBACK_AFTER_SECONDS = 4.0
GOAL_OT_BEFORE_SECONDS = 60.0
GOAL_OT_POWER_PLAY_BEFORE_SECONDS = 120.0
GOAL_OT_AFTER_SECONDS = 4.0
# Default goal timing rule: the goal moment is the first stable scoreboard clock
# freeze at the official goal time. Keep legacy near-match / projected fallbacks
# disabled unless a specific broken-scorebug run needs them.
GOAL_ENABLE_LEGACY_TIMING_FALLBACK = False
GOAL_CLOCK_STOP_ALLOW_CLOSE_SECONDS = 0
GOAL_ENABLE_PROJECTED_CLOCK_FALLBACK = False
GOAL_PROJECTED_CLOCK_FALLBACK_REQUIRES_UNRELIABLE = True
GOAL_LOCAL_OCR_ALLOW_CLOSE_SECONDS = 0
GOAL_ENABLE_LOCAL_OCR_CLOSEST_FALLBACK = False
GOAL_LOCAL_OCR_CLOSEST_FALLBACK_REQUIRES_UNRELIABLE = True

# Scrums and consequential stoppages (penalty_incidents.py): two or more penalties at one
# stoppage, or a lone major / misconduct, become ONE clip. The infraction happens before the
# whistle (judged major/scrum windows start 20-30 s ahead of the clock stop) and the officials
# sorting it out run 12-36 s after it.
SCRUM_BEFORE_SECONDS = 30.0
SCRUM_AFTER_SECONDS = 30.0

# 5-minute major settings (require manual review)
# Note: Clock freezes at penalty time for 10-30s while refs sort things out,
# so we need to go back further than the OCR timestamp to capture the incident
MAJOR_PENALTY_BEFORE_SECONDS = 30.0
MAJOR_PENALTY_AFTER_SECONDS = 90.0  # 1:30 after = ~2 min total clip
MAJOR_REVIEW_TIMEOUT_DAYS = 7

# ---------- Text burned into engine clips ----------
# Off by default: engine clips stay clean (and the vision reviewer sees the plain broadcast).
# Graphics come from the overlay system when the reel is built (scripts/build_reel.py,
# overlays/README.md). Turning this on burns a plain scorer/penalty caption into each clip.
OVERLAY_ENABLED = False
OVERLAY_FONT_SIZE = 42
OVERLAY_FONT = str(Path(__file__).resolve().parent / "assets" / "fonts" / "BarlowSemiCondensed-SemiBold.ttf")
OVERLAY_DURATION_SECONDS = 5.0

# ---------- Highlight execution profiles / reel modes ----------
# HockeyTech / MHL box score times are ELAPSED in period.
# Broadcast OCR clocks are REMAINING in period.
DEFAULT_REEL_MODE = "goals_only"
SUPPORTED_REEL_MODES = (
    "goals_only",
    "goals_with_pp_penalties",
    "goals_with_all_penalties",
    "goals_with_approved_majors",
    "full_production",
)

DEFAULT_HIGHLIGHT_EXECUTION_PROFILE = "flo_strip_recording"
HIGHLIGHT_EXECUTION_PROFILES = {
    # Dense, scorebug-first OCR profile for local Flo recordings.
    "flohockey_recording": {
        "sample_interval": 5,
        "tolerance_seconds": 30,
        "before_seconds": 8.0,
        "after_seconds": 6.0,
        "parallel_ocr": True,
        "ocr_workers": 4,
        "broadcast_type": "flohockey",
        "auto_detect_start": True,
        "reel_mode": DEFAULT_REEL_MODE,
    },
    # Faster FloHockey profile for multi-game backfills where 5-second OCR
    # sampling is too expensive but we still want the Flo-specific scorebug path.
    "flohockey_fast_recording": {
        "sample_interval": 15,
        "tolerance_seconds": 30,
        "before_seconds": 8.0,
        "after_seconds": 6.0,
        "parallel_ocr": True,
        "ocr_workers": 4,
        "broadcast_type": "flohockey",
        "auto_detect_start": True,
        "reel_mode": DEFAULT_REEL_MODE,
    },
    # Flo standard MHL strip (default): right-side period/clock block.
    "flo_strip_recording": {
        "sample_interval": 5,
        "tolerance_seconds": 30,
        "before_seconds": 8.0,
        "after_seconds": 6.0,
        "parallel_ocr": True,
        "ocr_workers": 4,
        "broadcast_type": "flo_strip",
        "auto_detect_start": True,
        "reel_mode": DEFAULT_REEL_MODE,
    },
    # 2026-27 Flo two-row top-left box: clock stacked over period.
    "flo_stacked_recording": {
        "sample_interval": 5,
        "tolerance_seconds": 30,
        "before_seconds": 8.0,
        "after_seconds": 6.0,
        "parallel_ocr": True,
        "ocr_workers": 4,
        "broadcast_type": "flo_stacked_topleft",
        "auto_detect_start": True,
        "reel_mode": DEFAULT_REEL_MODE,
    },
    # Flo top-left corner bar with period and clock first.
    "flo_corner_recording": {
        "sample_interval": 5,
        "tolerance_seconds": 30,
        "before_seconds": 8.0,
        "after_seconds": 6.0,
        "parallel_ocr": True,
        "ocr_workers": 4,
        "broadcast_type": "flo_corner_period_first",
        "auto_detect_start": True,
        "reel_mode": DEFAULT_REEL_MODE,
    },
    # MHL layout seen on one home venue's broadcasts: wide white Flo strip with right-side period/clock.
    "mhl_amherst_recording": {
        "sample_interval": 15,
        "tolerance_seconds": 30,
        "before_seconds": 8.0,
        "after_seconds": 6.0,
        "parallel_ocr": True,
        "ocr_workers": 4,
        "broadcast_type": "mhl_amherst",
        "auto_detect_start": True,
        "reel_mode": DEFAULT_REEL_MODE,
    },
    # MHL layout seen on another home venue's broadcasts: centered black banner with a tighter clock block.
    "mhl_summerside_recording": {
        "sample_interval": 15,
        "tolerance_seconds": 30,
        "before_seconds": 8.0,
        "after_seconds": 6.0,
        "parallel_ocr": True,
        "ocr_workers": 4,
        "broadcast_type": "mhl_summerside",
        "auto_detect_start": True,
        "reel_mode": DEFAULT_REEL_MODE,
    },
    # Backwards-compatible generic profile for non-Flo or manually tuned runs.
    "generic_recording": {
        "sample_interval": 30,
        "tolerance_seconds": 30,
        "before_seconds": 8.0,
        "after_seconds": 6.0,
        "parallel_ocr": True,
        "ocr_workers": 4,
        "broadcast_type": "auto",
        "auto_detect_start": True,
        "reel_mode": DEFAULT_REEL_MODE,
    },
    # Seeded non-standard MHL scorebug profile for one more home venue's broadcasts.
    "yarmouth_recording": {
        "sample_interval": 5,
        "tolerance_seconds": 30,
        "before_seconds": 8.0,
        "after_seconds": 6.0,
        "parallel_ocr": True,
        "ocr_workers": 4,
        "broadcast_type": "yarmouth",
        "auto_detect_start": True,
        "reel_mode": DEFAULT_REEL_MODE,
    },
}


def resolve_highlight_execution_selection(
    name: str | None = None,
    *,
    game_info: dict | None = None,
    source_game_info: dict | None = None,
    **overrides,
):
    """
    Resolve both the execution-profile settings and the matched scorebug profile.
    """
    selected = str(name or "").strip()
    scorebug_profile, scorebug_context = resolve_scorebug_profile(
        game_info=game_info,
        source_game_info=source_game_info,
    )
    if not selected or selected.lower() == "auto":
        selected = str(scorebug_profile.execution_profile_name or DEFAULT_HIGHLIGHT_EXECUTION_PROFILE).strip()
    if selected not in HIGHLIGHT_EXECUTION_PROFILES:
        raise ValueError(
            f"Unknown highlight execution profile '{selected}'. "
            f"Expected one of: {', '.join(sorted(HIGHLIGHT_EXECUTION_PROFILES))}"
        )

    profile = dict(HIGHLIGHT_EXECUTION_PROFILES[selected])
    for key, value in overrides.items():
        if value is not None:
            profile[key] = value

    return {
        "execution_profile_name": selected,
        "execution_profile": profile,
        "scorebug_profile": scorebug_profile.to_dict(),
        "scorebug_context": scorebug_context,
    }


def get_highlight_execution_profile(name: str | None = None, **overrides):
    """
    Return a copy of a named highlight execution profile.

    Callers can pass explicit overrides for one-off tuning without mutating the
    shared config dictionary.
    """
    selection = resolve_highlight_execution_selection(name, **overrides)
    return dict(selection["execution_profile"])
