# 🏒 Hockey Highlight Extractor v2.0

**Box-score-based highlight detection with OCR time matching**

Automatically extract hockey game highlights by matching official box score events (goals, penalties) with video timestamps extracted via OCR.

---

## ✨ What's New in v2.0

### Complete Rewrite
- **Box-score-based detection** - Uses official game data instead of unreliable audio analysis
- **OCR time extraction** - Reads game clock from video scoreboard using Tesseract
- **Event matching** - Syncs box score events to exact video timestamps
- **Modular architecture** - Clean, maintainable code structure
- **Automated processing** - Watch folder mode for automatic video processing

### Why This Approach Works Better
| v1.0 (Audio-Based) | v2.0 (Box-Score-Based) |
|-------------------|------------------------|
| ❌ Required excellent audio quality | ✅ No audio dependency |
| ❌ Couldn't distinguish event types | ✅ Knows goals vs penalties |
| ❌ No team attribution | ✅ Knows which team scored |
| ❌ Never worked reliably | ✅ Ground truth from official data |

---

## 🎯 Features

- **Automatic box score fetching** via HockeyTech API (MHL & BSHL)
- **OCR scoreboard detection** to extract game time from video
- **Smart event matching** - Syncs box score events to video timestamps
- **Individual highlight clips** - One clip per goal/event
- **Compiled highlights reel** - All clips combined into single video
- **Watch folder automation** - Auto-process videos when added to folder
- **Detailed logging** - Debug info for troubleshooting

---

## 📋 Requirements

### System Requirements
- **Python 3.8+**
- **Tesseract OCR** (for scoreboard time extraction)
  - **macOS**: `brew install tesseract`
  - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
  - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Python Dependencies
Install via: `pip install -r requirements.txt`

**Core dependencies:**
- `moviepy` - Video processing
- `opencv-python` - Image processing
- `pytesseract` - OCR interface
- `requests` - API calls
- `watchdog` - File watching (for automation)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Tesseract OCR (system package)
# macOS:
brew install tesseract

# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# Windows: Download and install from GitHub

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure HockeyTech Access

Set `HOCKEYTECH_API_KEY` in your shell before running the extractor:

```bash
export HOCKEYTECH_API_KEY=your_key_here
```

### 3. Configure Paths

Edit `config.py` to set your directories:

```python
GAMES_DIR = LOCAL_REPO_DIR / "Games"   # Where outputs are saved
TEAMS_FILE = LOCAL_REPO_DIR / "teams.json"  # Team data
```

### 4. Run the Extractor

**Interactive Mode** (select video from list):
```bash
python main.py
```

**Watch Folder Mode** (automatic processing):
```bash
# Watch Downloads folder
python watch_folder.py

# Watch specific directory
python watch_folder.py /path/to/videos
```

---

## 📂 Project Structure

```
HockeyHighlightExtractor/
├── main.py                     # Main entry point (interactive)
├── watch_folder.py             # Watch folder automation
├── config.py                   # Configuration
├── requirements.txt            # Python dependencies
├── teams.json                  # MHL & BSHL team data
│
├── hockey_extractor/           # Core modules
│   ├── __init__.py
│   ├── video_processor.py      # Video loading and clip creation
│   ├── box_score.py            # HockeyTech API integration
│   ├── ocr_engine.py           # Scoreboard time extraction
│   ├── event_matcher.py        # Event-to-video matching
│   └── file_manager.py         # File organization
│
└── Games/                      # Output directory
    └── YYYY-MM-DD_Team1_vs_Team2/
        ├── output/             # Final highlights reel
        ├── clips/              # Individual highlight clips
        ├── data/               # Box score and metadata
        ├── logs/               # Processing logs
        └── source/             # Original video (moved after processing)
```

---

## 🎬 How It Works

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  1. PARSE FILENAME                                              │
│     Extract: date, teams, league from filename                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  2. FETCH BOX SCORE                                             │
│     HockeyTech API → goals, penalties, times                    │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  3. LOAD VIDEO                                                  │
│     MoviePy → video clip object                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  4. EXTRACT TIME (OCR)                                          │
│     Sample frames → Tesseract → game time from scoreboard       │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  5. MATCH EVENTS                                                │
│     Box score events ↔ Video timestamps                         │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  6. CREATE CLIPS                                                │
│     Extract 8s before + 6s after each event                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  7. COMPILE HIGHLIGHTS                                          │
│     Concatenate clips → final highlights reel                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎮 Usage Examples

### Example 1: Process a Single Video

```bash
$ python main.py

🏒 Hockey Highlight Extractor v2.0

Found 2 video(s):

1. 2025-01-15 Amherst Ramblers vs Truro Bearcats Home 7.00pm.ts (1234.5 MB)
2. 2025-01-20 Valley Wildcats vs Yarmouth Mariners Away 7.30pm.mp4 (987.3 MB)

Enter video number to process (1-2): 1

======================================================================
HOCKEY HIGHLIGHT EXTRACTOR v2.0
======================================================================
Processing: 2025-01-15 Amherst Ramblers vs Truro Bearcats Home 7.00pm.ts

======================================================================
STEP 1: PARSING GAME INFORMATION
======================================================================
📅 Date: 2025-01-15
🏒 League: MHL
🏠 Home: Amherst Ramblers
✈️  Away: Truro Bearcats
🎯 Perspective: Home

[... processing continues ...]
```

### Example 2: Watch Folder Automation

```bash
$ python watch_folder.py ~/Downloads

======================================================================
🏒 Hockey Highlight Extractor - Watch Folder Mode
======================================================================

👀 Watching directory: /Users/tom/Downloads
📋 Log file: logs/watch_folder_20250115_193045.log
📹 Supported formats: .ts, .mp4, .avi, .mov, .mkv
Waiting for new video files...
Press Ctrl+C to stop

📹 New video detected: game.mp4
🎬 PROCESSING: game.mp4
[... automatic processing ...]
✅ Successfully processed: game.mp4
```

---

## 📝 Video Filename Format

For best results, use this filename format (MHL standard):

```
YYYY-MM-DD Team1 vs Team2 Home/Away HH.MMam/pm.ext
```

**Examples:**
- `2025-01-15 Amherst Ramblers vs Truro Bearcats Home 7.00pm.ts`
- `2025-02-20 Valley Wildcats vs Yarmouth Mariners Away 3.30pm.mp4`

**Why this matters:**
- Automatic date extraction
- Team name matching for box score lookup
- League detection (MHL vs BSHL)

---

## ⚙️ Configuration

### Adjusting OCR Settings

If OCR is missing the scoreboard, adjust the ROI (region of interest):

```python
# In ocr_engine.py
def detect_scoreboard_roi(self, frame, method='auto'):
    # Change 'auto' to 'top' or 'bottom' depending on scoreboard location
    # Or manually set ROI:
    return (x, y, width, height)  # Pixel coordinates
```

### HockeyTech API Configuration

Update league IDs in `hockey_extractor/box_score.py`:

```python
LEAGUE_CONFIGS = {
    'MHL': {
        'client_code': 'mhl',
        'league_id': '2',  # Update with actual MHL league ID
    },
    'BSHL': {
        'client_code': 'bshl',
        'league_id': '1',  # Update with actual BSHL league ID
    }
}
```

### Clip Timing

Adjust how much video to include before/after events:

```python
# In main.py, STEP 6
created_clips = video_processor.create_highlight_clips(
    goal_events,
    game_folders['clips_dir'],
    before_seconds=8,   # ← Change this
    after_seconds=6     # ← Change this
)
```

---

## 🐛 Troubleshooting

### "Could not find game in league database"

**Causes:**
1. Game hasn't been played yet
2. Team names don't match league records
3. API is unavailable

**Solutions:**
- Check filename matches teams.json entries
- Verify game date is correct
- Try alternative team names/aliases

### "No timestamps extracted from video"

**Causes:**
1. Tesseract not installed
2. Scoreboard not visible/readable
3. Wrong ROI configuration

**Solutions:**
```bash
# Verify Tesseract is installed
tesseract --version

# Check scoreboard visibility
# Save debug frame to inspect ROI
ocr_engine.save_debug_frame(frame, Path('debug.jpg'), roi)
```

### "No events could be matched to video"

**Causes:**
1. OCR extraction failed
2. Video doesn't cover full game
3. Time format incompatible

**Solutions:**
- Review `data/video_timestamps.json` to check extracted times
- Adjust tolerance in event matching: `tolerance_seconds=60`
- Manually configure scoreboard ROI

---

## 🔧 Advanced Features

### Custom Event Filters

Process only specific event types:

```python
# Filter to goals only
goal_events = event_matcher.filter_events_by_type(events, ['goal'])

# Include penalties
highlight_events = event_matcher.filter_events_by_type(events, ['goal', 'penalty'])
```

### Timestamp Interpolation

Fill in missing OCR timestamps:

```python
enhanced_timestamps = event_matcher.estimate_missing_timestamps(
    video_timestamps,
    video_processor.duration
)
```

### Cache Management

Box scores are cached in `Games/*/data/` to avoid repeated API calls.

Clear cache:
```bash
rm Games/*/data/*_boxscore.json
```

---

## 📊 Output Files

After processing, each game folder contains:

```
Games/2025-01-15_Amherst_Ramblers_vs_Truro_Bearcats/
│
├── output/
│   └── highlights.mp4                  # Final highlights reel
│
├── clips/
│   ├── 01_GOAL_P1_Amherst_Ramblers.mp4
│   ├── 02_GOAL_P2_Truro_Bearcats.mp4
│   └── ...
│
├── data/
│   ├── game_metadata.json              # Game info + box score
│   ├── matched_events.json             # Events with video times
│   ├── video_timestamps.json           # OCR extracted times
│   └── MHL_1234_boxscore.json          # Cached box score
│
├── logs/
│   └── processing.log                  # Detailed debug log
│
└── source/
    └── original_video.ts               # Moved from input location
```

---

## 🤝 Contributing

This project is designed for local use with MHL and BSHL games. To adapt for other leagues:

1. Add league configuration to `box_score.py`
2. Update `teams.json` with team data
3. Adjust filename parser in `file_manager.py`
4. Configure OCR ROI for different scoreboard layouts

---

## 📜 License

MIT License - Free to use and modify

---

## 🆘 Support

**Issues?**
1. Check logs in `Games/*/logs/processing.log`
2. Review `data/video_timestamps.json` for OCR accuracy
3. Verify Tesseract installation: `tesseract --version`
4. Ensure video filename matches expected format

**Questions?**
- Review this README
- Check code comments in `hockey_extractor/` modules
- Examine example output files

---

## 📈 Version History

### v2.0.0 (Current)
- Complete rewrite with box-score-based detection
- OCR time extraction from scoreboard
- Modular architecture
- Watch folder automation
- HockeyTech API integration

### v1.0.0 (Deprecated)
- Audio-based detection (unreliable, removed)

---

**Made with 🏒 for hockey highlight extraction**
