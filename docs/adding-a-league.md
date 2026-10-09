# Adding a league

A league is three pieces of data: a **league pack** (teams, codes, colours, logos, wording), a
**box-score source** (a data provider adapter) and, when its broadcasts use an unfamiliar scoreboard,
a **scorebug profile**. None of it needs changes to the engine or the overlay renderer.

The MHL (HockeyTech, Flo broadcasts) is the complete worked example:
`overlays/leagues/mhl/league.json`, the HockeyTech provider, and the MHL scorebug profiles in
`scorebug_profiles.py`. `overlays/leagues/demo-hockey/` is a fictional league that uses the manual
box-score format and is what the quickstart runs.

## 1. League pack

Create `overlays/leagues/<id>/league.json`, or a folder of your own and point
`HOCKEY_LEAGUES_DIR` at its parent (several folders can be joined with `:`; a pack with the same
`id` replaces the bundled one).

```json
{
  "id": "demo-hockey",
  "name": "Demo Hockey League",
  "short": "DHL",
  "sport": "hockey",
  "logo": "overlays/leagues/demo-hockey/logos/league.svg",
  "primary": "#1b2a41",
  "secondary": "#e0a526",
  "fallback_logo": "assets/logos/fallback.png",
  "periods": {"1": "1st", "2": "2nd", "3": "3rd", "4": "OT", "5": "SO"},
  "clock": "remaining",
  "period_minutes": {"regulation": 20, "overtime": 5},
  "provider": {"type": "manual"},
  "teams": [
    {"id": "harbour-hawks", "provider_id": "1", "short": "HAW", "city": "Harbour", "nickname": "Hawks",
     "name": "Harbour Hawks", "logo": "overlays/leagues/demo-hockey/logos/harbour-hawks.svg",
     "primary": "#14437a", "secondary": "#0b2447", "text": "#ffffff"}
  ]
}
```

| field | meaning |
|---|---|
| `id`, `short` | the pack's name and its short form; a box score's `league` may use either |
| `sport` | `hockey` packs are searched for league detection; the overlay contract also takes other sports |
| `periods` | the period labels the overlays show |
| `clock` | `remaining` or `elapsed`: which way the league's broadcasts count. Overlay times are converted to it |
| `period_minutes` | period lengths for that conversion (default 20 and 5) |
| `provider` | where box scores come from; see section 2 |
| team `short` | the code on the overlay scoreline |
| team `provider_id` | the id the box-score source uses for the team |
| team `text` | a colour that is readable on `primary`; light team colours need a dark `text` |

A team resolves by `short`, `provider_id`, `id`, `name`, `nickname` or `city`, so a box score can use
whichever it has. A team the pack does not know still gets an overlay, with neutral colours and
the placeholder logo.

Logo paths are relative to the repository root (or absolute) and may be PNG or SVG. This
repository does not ship team or league logos: they belong to their owners. Put yours at the
paths the pack names. Any missing file falls back to `assets/logos/fallback.png`.

## 2. Box-score source

### Manual JSON

Works for any league and any source (a scoresheet you typed, a scrape, an export). Set
`"format": "manual-box-score/1"`.

```json
{
  "format": "manual-box-score/1",
  "game_id": "2026-01-09-hawks-rams",
  "date": "2026-01-09",
  "league": "demo-hockey",
  "home_team": "Harbour Hawks",
  "away_team": "Ridge Rams",
  "followed_team": "Harbour Hawks",
  "time_basis": "elapsed",
  "playoff": false,
  "result": {"final_score": "2-0", "overtime": false, "shootout": false},
  "goals": [
    {"period": 1, "time": "05:12", "team": "Harbour Hawks",
     "scorer": {"name": "Sam Rivera", "number": "17"},
     "assists": [{"name": "Alex Tremblay"}, "Jordan Leblanc"], "special": "PP"}
  ],
  "penalties": [
    {"period": 1, "time": "06:30", "team": "Ridge Rams",
     "player": {"name": "Pat Gallant", "number": "4"}, "infraction": "Tripping", "minutes": 2}
  ]
}
```

- `time` is the game clock at the event, in the box score's own convention: elapsed in the
  period by default, or time remaining with `"time_basis": "remaining"`.
- `period`: 1 to 3, 4 for overtime, 5 for a shootout.
- `special` on a goal: `PP`, `SH`, `EN` or empty. `infraction` containing "fight" makes the
  penalty a fight in the reel.
- `followed_team` is the team the highlights are about (default: the home team). The engine uses
  it to link power-play goals to the penalty that caused them; `--team` overrides it.
- Names are plain strings or `{"name", "number"}`.

`scripts/make_sample_data.py` writes a complete example.

### HockeyTech

HockeyTech runs the feeds behind many junior and minor leagues. Add a `provider` block with the
league's client code and league id:

```json
"provider": {"type": "hockeytech", "client_code": "mhl", "league_id": "1"}
```

Then pass `--league <id>` and `--hockeytech-game-id <id>`. HockeyTech requires the public feed key
that the league's own web site uses; set it as `HOCKEYTECH_API_KEY` (not stored in the repo).
`HOCKEYTECH_SEASON_ID` pins a season, otherwise the feed default applies. The adapter converts
HockeyTech's game summary into the same shape the manual provider produces.

### Another source

A provider is any object that returns a `PreloadedBoxScoreFetcher(game_id, box_score, goals)`
(`hockey_extractor/providers/base.py`): `box_score` is a dict of the shape
`{"SiteKit": {"Gamesummary": {"meta": {...}, "goals": [...], "penalties": [...]}}}` and `goals` is
a list of `hockey_extractor.goal.Goal`. `providers/manual.py` is about 100 lines and is the model
to copy.

## 3. Scorebug profile

The engine reads the period and clock from the broadcast's scoreboard graphic. Layouts it knows
are in `scorebug_profiles.py` (the catalog) and `hockey_extractor/ocr_engine.py`
(`SCOREBUG_BOX_LAYOUTS`). If your broadcaster's graphic matches one, nothing to do: detection picks
it from the video, or pass `--profile`. If not, add it as configuration:

1. **Describe the boxes.** In `SCOREBUG_BOX_LAYOUTS`, add an entry: the period and the clock as
   boxes in fractions of the frame, `(x, y, width, height)`, read left to right. They are cropped
   and stitched before OCR, so a graphic that stacks the period under the clock reads like a
   one-line banner.
2. **Add a profile** to `SCOREBUG_PROFILES` in `scorebug_profiles.py`: an id, a description, the
   execution profile it uses, and optionally the league or team it applies to.
3. **Add an execution profile** in `config.py` (`HIGHLIGHT_EXECUTION_PROFILES`): copy a neighbour
   and set `broadcast_type` to your layout. The sample interval, tolerance and clip lead are
   there.
4. **Add a reference crop** to `assets/scorebugs/<profile id>.png` (used when an optional vision
   model chooses between layouts) and a real crop with the expected reading to
   `tests/fixtures/scorebugs/` and its `manifest.json`, so `tests/test_scorebug_layouts.py`
   guards the layout.

Use real crops only from footage you may share, or crop tightly to the graphic. The scorebug
crops in the repository are small crops of the graphic alone.

## Checklist

- [ ] `league.json` with teams, `periods`, `clock` and a `provider`
- [ ] logos at the paths the pack names, or accept the placeholder
- [ ] a box score in the manual format or a HockeyTech id
- [ ] a short test recording: `scripts/process_game.py --video clip.mp4 --box-score game.json`;
      check `data/ocr_sampling_log.txt` reads the period and clock
- [ ] a profile and layout if the log shows the scoreboard is not read
- [ ] `scripts/build_reel.py --game-dir Games/<game>` and look at `output/overlays/`
