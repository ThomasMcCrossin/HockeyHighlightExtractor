"""
🏒 Local-First Config — fast, reliable writes; optional post-run mirror to Google Drive.

This keeps all working files LOCAL (Games, logs, temp).
If Google Drive is detected, you can call mirror_game_to_gdrive(...) AFTER everything is done.
"""

import os
import shutil
from pathlib import Path

# ---------- Google Drive detection (no write test, just presence) ----------
# Set HIGHLIGHT_GOOGLE_ACCOUNT to your Drive account name to also check account-named mounts.
GOOGLE_ACCOUNT = os.environ.get("HIGHLIGHT_GOOGLE_ACCOUNT", "").strip()


def find_google_drive():
    """Find a Google Drive root for optional mirroring (read-only is fine)."""
    possible_drives = ['G:', 'J:', 'C:', 'D:', 'E:', 'F:', 'H:', 'I:', 'K:']
    print("🔍 Looking for a Google Drive (for optional mirroring).")
    for drive in possible_drives:
        for path in [
            Path(f"{drive}/My Drive"),
            Path(f"{drive}/Google Drive"),
            Path(f"{drive}/GoogleDrive"),
            Path(f"{drive}/Drive"),
            *([
                Path(f"{drive}/{GOOGLE_ACCOUNT}/My Drive"),
                Path(f"{drive}/GoogleDrive - {GOOGLE_ACCOUNT}"),
                Path(f"{drive}/My Drive - {GOOGLE_ACCOUNT}"),
            ] if GOOGLE_ACCOUNT else []),
        ]:
            projects_folder = path / "Projects"
            if projects_folder.exists():
                print(f"✅ Found Google Drive: {path}")
                return path
    print("⚠️ Google Drive not found — running local-only (that’s fine).")
    return None

# ---------- Local repo (script lives here) ----------
LOCAL_REPO_DIR = Path(__file__).parent
print(f"📁 Local Repository: {LOCAL_REPO_DIR}")

# ---------- Optional Google Drive locations (for MIRRORING only) ----------
GOOGLE_DRIVE = find_google_drive()
if GOOGLE_DRIVE:
    GOOGLE_HOCKEY_DIR = GOOGLE_DRIVE / "Projects" / "HockeyHighlights"
    GOOGLE_GAMES_DIR = GOOGLE_HOCKEY_DIR / "Games"     # mirror target
    GOOGLE_INPUT_DIR = GOOGLE_HOCKEY_DIR / "Videos"    # optional input source
else:
    GOOGLE_HOCKEY_DIR = None
    GOOGLE_GAMES_DIR = None
    GOOGLE_INPUT_DIR = None

# ---------- Working directories (ALWAYS LOCAL) ----------
# Your script reads/prints these; keeping the same names avoids breakage.
GAMES_DIR = LOCAL_REPO_DIR / "Games"   # script writes here
print(f"🎮 Games Output: Local ({GAMES_DIR})")

TEAMS_FILE = LOCAL_REPO_DIR / "teams.json"

# Logs and temp: always local
LOGS_DIR = LOCAL_REPO_DIR / "logs"
TEMP_DIR = LOCAL_REPO_DIR / "temp"

def ensure_logs_directory():
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"⚠️ Could not create logs directory {LOGS_DIR}: {e}")
        return False

def ensure_temp_directory():
    try:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"⚠️ Could not create temp directory {TEMP_DIR}: {e}")
        return False

# ---------- Video / analysis settings (unchanged) ----------
SUPPORTED_FORMATS = ['.ts', '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
OUTPUT_CODEC = 'mpeg4'

AUDIO_SAMPLE_RATE = 22050
GOAL_ENERGY_THRESHOLD = 0.75
SAVE_ENERGY_THRESHOLD = 0.65
ANNOUNCER_EXCITEMENT_THRESHOLD = 0.7

MAX_HIGHLIGHT_CLIPS = 12
DEFAULT_CLIP_BEFORE_TIME = 8
DEFAULT_CLIP_AFTER_TIME = 6

# ---------- Helpers used by your script (names preserved) ----------
def find_video_locations():
    """Return list of directories to search for videos (local-first, optionally Google Drive)."""
    video_locations = [
        LOCAL_REPO_DIR,                 # repo folder
        Path.home() / "Downloads",      # downloads
        Path.home() / "Desktop",        # desktop
    ]
    # Add Google Drive input location if available (read-only is fine)
    if GOOGLE_INPUT_DIR and GOOGLE_INPUT_DIR.exists():
        video_locations.insert(1, GOOGLE_INPUT_DIR)     # check GDrive early
        print(f"🔍 Will check Google Drive videos: {GOOGLE_INPUT_DIR}")
    return video_locations

def ensure_output_directory():
    """Ensure the local games output directory exists."""
    try:
        GAMES_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"⚠️ Could not create output directory {GAMES_DIR}: {e}")
        return False

# ---------- Optional post-run mirror (call this AFTER rendering finishes) ----------
def mirror_game_to_gdrive(local_game_dir: Path) -> Path | None:
    """
    Mirror a finished local game folder to Google Drive 'Games'.
    Returns the destination path (or None if GDrive is unavailable).
    Safe on read-only / sync-delayed setups: copies only; does not write during processing.
    """
    if not GOOGLE_GAMES_DIR:
        print("ℹ️ Skipping mirror: Google Drive not available.")
        return None

    dest = GOOGLE_GAMES_DIR / local_game_dir.name
    try:
        # Make sure parent exists
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Copy tree (overwrite newer files only)
        if dest.exists():
            # Incremental copy: mirror files
            for root, dirs, files in os.walk(local_game_dir):
                rel = Path(root).relative_to(local_game_dir)
                (dest / rel).mkdir(parents=True, exist_ok=True)
                for f in files:
                    src_f = Path(root) / f
                    dst_f = dest / rel / f
                    if not dst_f.exists() or src_f.stat().st_mtime > dst_f.stat().st_mtime:
                        shutil.copy2(src_f, dst_f)
        else:
            shutil.copytree(local_game_dir, dest)
        print(f"☁️ Mirrored to Google Drive: {dest}")
        return dest
    except Exception as e:
        print(f"⚠️ Mirror failed ({e}). Local output remains at: {local_game_dir}")
        return None

# ---------- Summary ----------
print("=" * 60)
print("🏒 LOCAL-FIRST SETUP SUMMARY")
print("=" * 60)
print(f"💻 Development: Local (Visual Studio)")
print(f"   Code: {LOCAL_REPO_DIR}")
print(f"   Teams: {TEAMS_FILE}")
if GOOGLE_DRIVE:
    print(f"☁️ Optional Mirror Target: {GOOGLE_GAMES_DIR}")
    if GOOGLE_INPUT_DIR:
        print(f"   Videos (read): {GOOGLE_INPUT_DIR}")
else:
    print("☁️ Optional Mirror Target: (none)")
print(f"📁 Output (write): {GAMES_DIR}")
print("=" * 60)


# ---------- Engine tuning synced from amherst-display (2026-09-26) ----------
# Timing windows, OCR health/debug knobs, reel modes and scorebug execution profiles
# used by hockey_extractor; see scorebug_profiles.py for the layout catalog.
OUTPUT_PRESET = 'veryfast'
OUTPUT_CRF = 18
OUTPUT_AUDIO_CODEC = 'aac'
OUTPUT_AUDIO_BITRATE = '192k'
OUTPUT_AUDIO_SAMPLE_RATE = 48000
OUTPUT_PIXEL_FORMAT = 'yuv420p'
BOX_SCORE_TIME_IS_ELAPSED = True
OCR_BACKENDS = ["tesseract", "easyocr"]
OCR_ENABLE_EASYOCR_FALLBACK = True
OCR_EASYOCR_LANGS = ["en"]
OCR_EASYOCR_GPU = False
OCR_MIN_SUCCESS_RATE = 0.05
OCR_MIN_PERIOD_RATE = 0.20
OCR_MIN_AVG_CONFIDENCE = 55.0
OCR_HEALTH_BAD_CONSECUTIVE_SAMPLES_RESET = 10
OCR_DEBUG_SAVE_SCOREBUG_CROPS = True
OCR_DEBUG_SCOREBUG_CROP_DIRNAME = "ocr_scorebug_crops"
OCR_DEBUG_FAILURE_CROP_LIMIT = 40
OCR_DEBUG_LOW_CONFIDENCE_THRESHOLD = 65.0
OCR_DEBUG_LOW_CONFIDENCE_CROP_LIMIT = 25
EVENT_LOCAL_OCR_WINDOW_SECONDS = 60.0
EVENT_LOCAL_OCR_STEP_SECONDS = 0.5
EVENT_LOCAL_OCR_PERSISTENCE_WINDOW_SECONDS = 6.0
EVENT_LOCAL_OCR_MIN_HITS = 3
EVENT_LOCAL_OCR_MAX_DIFF_SECONDS = 6.0
EVENT_ENFORCE_MIN_VIDEO_TIME_FROM_GAME_START = True
EVENT_MIN_VIDEO_TIME_BUFFER_SECONDS = 240.0
PENALTY_PP_BEFORE_SECONDS = 2.0
PENALTY_PP_AFTER_SECONDS = 3.0
GOAL_CLOCK_STOP_BEFORE_SECONDS = 32.0
GOAL_CLOCK_STOP_AFTER_SECONDS = 3.0
GOAL_FALLBACK_BEFORE_SECONDS = 20.0
GOAL_FALLBACK_AFTER_SECONDS = 4.0
GOAL_OT_BEFORE_SECONDS = 60.0
GOAL_OT_POWER_PLAY_BEFORE_SECONDS = 120.0
GOAL_OT_AFTER_SECONDS = 4.0
GOAL_ENABLE_LEGACY_TIMING_FALLBACK = False
GOAL_CLOCK_STOP_ALLOW_CLOSE_SECONDS = 0
GOAL_ENABLE_PROJECTED_CLOCK_FALLBACK = False
GOAL_PROJECTED_CLOCK_FALLBACK_REQUIRES_UNRELIABLE = True
GOAL_LOCAL_OCR_ALLOW_CLOSE_SECONDS = 0
GOAL_ENABLE_LOCAL_OCR_CLOSEST_FALLBACK = False
GOAL_LOCAL_OCR_CLOSEST_FALLBACK_REQUIRES_UNRELIABLE = True
MAJOR_PENALTY_BEFORE_SECONDS = 30.0
MAJOR_PENALTY_AFTER_SECONDS = 90.0
OVERLAY_ENABLED = True
OVERLAY_FONT_SIZE = 42
OVERLAY_FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
OVERLAY_DURATION_SECONDS = 5.0
DEFAULT_REEL_MODE = "goals_only"
SUPPORTED_REEL_MODES = (
    "goals_only",
    "goals_with_pp_penalties",
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
    # Amherst home MHL layout: wide white Flo strip with right-side period/clock.
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
    # Summerside home MHL layout: centered black banner with a tighter clock block.
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
    # Seeded non-standard MHL scorebug profile for Yarmouth home broadcasts.
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
