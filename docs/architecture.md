# Architecture

The pipeline has four stages. Each reads files the previous one wrote, so any stage can be re-run
alone, and a stage you do not need can be skipped.

```
 recording.mp4 ---+
                  |     1. engine                 2. review (optional)         3. reel
 box score -------+--> process_game.py  ------>  review_game.py --apply  ---> build_reel.py
 (JSON / HockeyTech)    Games/<game>/data/...     data/review/reel_main.json   output/reel.mp4
                        Games/<game>/clips/...
```

`scripts/run_game.py` chains them. Stage 2 runs only when configured (see
[vision-review.md](vision-review.md)); without it, stage 3 reads the engine's own clip list.

## Stage 1: engine (`hockey_extractor/`)

`HighlightPipeline` in `pipeline.py` runs these steps:

1. **Parse and set up.** Game info (teams, date, league) comes from the box-score provider, not
   from the file name. A game folder is created under the games directory.
2. **Box score.** A provider (`providers/`) returns goals and penalties in HockeyTech's
   `SiteKit.Gamesummary` shape. Providers: `ManualBoxScoreProvider` (a JSON file) and
   `HockeyTechProvider` (a league feed). Both return a `PreloadedBoxScoreFetcher`.
3. **Video and game start.** The scorebug layout is chosen by `scorebug_detect.py` (OCR vote over
   a few frames; an optional vision model breaks ties). The game start is found from the first
   readable clock.
4. **OCR sampling.** Frames are sampled every few seconds, the scorebug period and clock are
   cropped with a layout from `ocr_engine.SCOREBUG_BOX_LAYOUTS` (or a profile's region) and read
   with Tesseract (EasyOCR as an optional fallback). Readings are normalised against clock rules,
   so a misread digit is outvoted by its neighbours.
5. **Match.** Each box-score event is matched to a video time by its period and clock
   (`event_matcher.py`). Goals are then refined by finding where the clock stopped; when the
   scorebug is frozen, `goal_locator.py` places the goal from the celebration if a vision
   endpoint is configured. Penalties are located the same way. Events outside a partial
   recording are left unmatched.
6. **Clips.** `video_processor.py` cuts a window around each matched event with ffmpeg and writes
   `data/clips_manifest.json`. Windows are set in `config.py` (goals start 32 s before a
   clock-stop time).
7. **Optional extras.** Major penalties can pause for a human approval (Drive upload and an
   e-mail); only the `goals_with_approved_majors` and `full_production` reel modes use it. A
   plain highlights reel (`output/highlights.mp4`) and a text description are written too.

What a game folder holds:

| path | content |
|---|---|
| `data/game_metadata.json` | game info and the box score as the engine read it |
| `data/matched_events.json` | every goal and penalty with `video_time`, match confidence, how it was refined |
| `data/clips_manifest.json` | the clips that were cut: event fields plus `path`, `before_seconds`, `after_seconds` |
| `data/run_config.json` | what `process_game.py` decided: league, followed team, profile, counts |
| `data/ocr_sampling_log.*`, `video_timestamps.json` | the OCR readings, for debugging |
| `data/SCOREBOARD_ALERT.txt` | written when the scorebug looked stuck or glitching |
| `clips/` | engine clips |
| `data/review/` | review packets, `overrides.json`, `reel_main.json`, reviewed `clips/` |
| `output/` | `reel.mp4`, `highlights.mp4`, `overlays/` (the PNGs used) |

## Stage 2: vision review (`clip_review/`, `skills/hockey-clip-review/`)

Builds one incident per goal or penalty group the engine placed, shows a reviewer contact sheets of
frames around it, and lets the reviewer overrule the engine's window within code-enforced bounds.
The outputs are `overrides.json` (every decision) and `reel_main.json` (the clips the reel should
use, with reviewed windows). Everything else about review is in [vision-review.md](vision-review.md).

The contract between stage 2 and stage 3 is a manifest: a JSON file with a `clips` list, each entry
an event (`type`, `period`, `time`, `team`, scorer or player, ...) plus `path` (relative to the
game folder or absolute). Optional `clip_video_start` and `video_time` say where in the clip the
event happens; otherwise `before_seconds` does. Any tool can write one and pass it to
`build_reel.py --manifest`.

Incident classes today: `goal`, `minor`, `major`, `fight`. Each has bounds in
`clip_review/incidents.py` (`BOUNDS`). A new class, such as a scrum (several penalties at one
stoppage), needs a `BOUNDS` entry, a rule in `penalty_class` or `build_incidents`, and a
paragraph in the skill's `SKILL.md`; the packet, review and apply code do not care how many
classes there are.

## Stage 3: reel (`hockey_extractor/reel.py`, `overlays/`)

For each clip the reel builder:

1. words an overlay spec from the event, the running score and the league pack
   (`clip_spec`): headline, scorer, assists, power-play tag, period and clock as the league shows
   them;
2. renders all specs in one run of `overlays/render.mjs` with the chosen theme, which also checks
   layout (clipped text, off-frame, over the broadcast scorebug zone);
3. composites each PNG over its clip with ffmpeg, from half a second before the event for six
   seconds, with fades, and normalises every clip to the same size, frame rate and audio so they
   can be joined without re-encoding;
4. concatenates the clips, with optional intro and final-score cards.

The renderer and themes know nothing about hockey data; the builder knows nothing about fonts or
layout. The spec in [overlays/README.md](../overlays/README.md) is the only thing between them.

## League data

One league pack (`overlays/leagues/<id>/league.json`) describes a league for every stage: team
names and codes (league detection, score-keeping), colours and logos (overlays), period labels
and clock direction (overlay wording), and the box-score provider (`provider` block). Extra pack
folders load from `HOCKEY_LEAGUES_DIR`. See [adding-a-league.md](adding-a-league.md).

## Configuration

- `config.py`: paths (`HOCKEY_GAMES_DIR`, `HOCKEY_LOGS_DIR`), the followed team
  (`HOCKEY_FOLLOWED_TEAM`), encode settings, OCR health thresholds, clip windows, reel modes and
  the scorebug execution profiles. Importing it prints nothing and reaches no network.
- Environment: `HOCKEYTECH_API_KEY`, `HOCKEYTECH_SEASON_ID` (HockeyTech only);
  `CLIP_REVIEW_*` (vision, optional); `RESEND_API_KEY`, `NOTIFICATION_EMAIL`,
  `HIGHLIGHTS_MAJOR_REVIEW_FOLDER_ID`, `GOOGLE_APPLICATION_CREDENTIALS` (major-penalty review,
  optional).

## Time

HockeyTech and the manual format give box-score times as elapsed in the period; broadcast
clocks count down. The engine converts using the period length (20 minutes, 5 for overtime).
`time_basis` in the manual format and `BOX_SCORE_TIME_IS_ELAPSED` in `config.py` pick the box
score's convention; the league pack's `clock` picks what the overlays show.
