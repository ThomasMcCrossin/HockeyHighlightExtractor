# Hockey Highlight Extractor

Turns a recording of a hockey broadcast and the game's box score into a highlight reel with
broadcast-style overlays. It works for any league. The Maritime Junior Hockey League (MHL, a
HockeyTech league) is the fully built worked example; adding another league is data, not code
([docs/adding-a-league.md](docs/adding-a-league.md)).

```
recording + box score
        |
        v
  engine (OCR of the broadcast scorebug, box-score matching)  -->  clips around each goal/penalty
        |
        v   (optional, only when configured)
  vision review (an agent or a vision API re-picks each clip's in/out points)
        |
        v
  reel builder (league pack + theme -> overlays, ffmpeg composite)  -->  output/reel.mp4
```

## Two tiers

**Engine only (default).** Needs no AI and no account. The engine reads the game clock off the
broadcast scorebug with OCR, matches each goal and penalty in the box score to a moment in the
video and cuts a clip around it. This is what runs when you follow the quickstart. Expect
clips that contain the event, cut by rules learned from judged highlights: about 32 s of build-up
and 16 s of celebration around each goal, the foul and the referee's call on minor penalties, and
one longer clip for a scrum (several penalties at one stoppage, or any major). In the 2026-10-08
bake-off, blinded judges scored the earlier fixed-window engine 4.56 out of 10; the current rules
were tuned on that judged data afterwards (see "Results").

**With vision review (optional).** A reviewer looks at frames from the recording and chooses
better in/out points: the build-up to a goal, the end of the celebration, the call on a penalty.
It can also drop a clip whose event is not in the recording. The reviewer is any
OpenAI-compatible vision API or any agent command-line tool that can read images and run
commands. It turns on by itself when you set an agent command or an API key, endpoint and model,
and is skipped without a message otherwise. The best agent reviewer scored 7.09 out of 10 in the
same test. Details, costs and the failure modes: [docs/vision-review.md](docs/vision-review.md).

The judged figures here are the 2026-10-08 bake-off; the engine-only number predates the
learned clip windows and has not been re-judged since.

## Quickstart

You need Python 3.10+, `ffmpeg`, `tesseract` (the OCR program) and Node 20+.

```bash
# 1. Python dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Overlay renderer (Node + a headless browser, needed only for overlays)
npm install
npx playwright install chromium

# 3. Make a small synthetic game (about 3.5 MB, no real footage) and its box score
.venv/bin/python scripts/make_sample_data.py

# 4. Recording + box score -> clips -> reel with overlays
.venv/bin/python scripts/run_game.py --video sample_data/sample_game.mp4 \
    --box-score sample_data/sample_game.json --cards
```

The last line printed is the reel (`Games/<date>_<home>_vs_<away>/output/reel.mp4`). The sample
run takes a couple of minutes: the engine matches 3 of 3 box-score events, cuts 2 goal
clips, and the reel adds a lower third to each goal plus an intro and a final-score card. The
sample game is fictional (the "Demo Hockey League" pack in `overlays/leagues/demo-hockey`).

Without Node, run `run_game.py` with `--no-overlays` to get the clips joined with no graphics.
`python -m pytest -q` runs the tests (no video, browser or network needed).

### Your own game

```bash
# a box score you wrote or exported (format: docs/adding-a-league.md)
.venv/bin/python scripts/run_game.py --video game.mp4 --box-score game.json

# or fetch it from a HockeyTech league (the MHL pack is the example)
export HOCKEYTECH_API_KEY=...        # the public feed key your league's own site uses
.venv/bin/python scripts/run_game.py --video game.mp4 --league mhl --hockeytech-game-id 4943 --team Truro
```

The scorebug layout is detected from the video. If detection picks wrongly, name one with
`--profile` (profiles are listed in `config.py`, layouts in `scorebug_profiles.py`). A recording
may start late or end early: goals outside it are reported as unmatched, not clipped at its edge.

### Turn on vision review

Set one of these and run the same command; there is no other switch.

```bash
# an agent CLI that can read images and run shell commands (the command is yours to choose)
export CLIP_REVIEW_AGENT_CMD='<your-agent> <flags> {prompt}'

# or any OpenAI-compatible vision endpoint
export CLIP_REVIEW_API_KEY=... CLIP_REVIEW_BASE_URL=https://.../v1 CLIP_REVIEW_MODEL=...
```

`--review off` forces it off; `--review api|agent|escalate` forces a backend. See
[docs/vision-review.md](docs/vision-review.md). To see the plumbing without a model, use
`CLIP_REVIEW_AGENT_CMD='python {repo}/examples/stub_review_agent.py {prompt}'`.

## Stages and scripts

| stage | script | what it does |
|---|---|---|
| all of it | `scripts/run_game.py` | recording + box score -> engine clips -> optional review -> reel |
| engine | `scripts/process_game.py` | OCR the scorebug, match events, cut clips; writes the game folder |
| review | `scripts/review_game.py` | vision review of a game folder; `--apply` writes reviewed clips and a reel manifest |
| reel | `scripts/build_reel.py` | clips + overlays -> `output/reel.mp4`; `--reviewed` uses the review result |
| demo data | `scripts/make_sample_data.py` | synthetic recording and box score |
| leak check | `scripts/leak_scan.sh` | scans the tree and diff for paths, keys, e-mail addresses and media |

## Layout

```
hockey_extractor/    engine: OCR, matching, clips, league packs, providers, reel builder
  providers/           box-score adapters: HockeyTech and manual JSON
clip_review/         vision review: packets, backends, review, apply
skills/              portable agent skill for clip review (SKILL.md, frame and verdict tools)
overlays/            overlay contract, renderer, league packs, themes
scorebug_profiles.py, scorebug_detect.py, goal_locator.py   scorebug layouts and detection
config.py            engine settings and scorebug execution profiles
assets/              scorebug reference crops, fonts (OFL), placeholder logo
docs/                architecture, adding a league, vision review, overlays, history
tests/               engine, provider and wiring tests
```

## Results

The 2026-10-08 bake-off compared vision reviewers and the engine on 39 incidents from recorded
games (blinded judges, 0 to 10 highlight score; full table, method and caveats in
[docs/vision-review.md](docs/vision-review.md)).

| clips from | highlight score | tokens per incident |
|---|---|---|
| best agent reviewer | 7.09 | 672k |
| lean agent reviewer (thinking off, 20 tool calls) | 6.39 | 360k |
| other agent reviewers | 5.60 to 6.04 | 199k to 1.32M |
| API first, agent only where unsure (`escalate`) | 5.39 | 229k |
| engine only | 4.56 | none |
| one-shot vision API (no agent) | 3.88 | 11k |

Every agent reviewer beat the engine. The one-shot API reviewer did not: it needs the agent's
ability to go back for more frames. The ranking is highlight quality, not whether the goal is in
the clip; the engine usually does contain the event.

After the bake-off the engine's clip windows were re-tuned from the judges' data (goal tail 3 s ->
16 s, minor penalties -2/+3 s -> -9/+14 s, scrums as one clip). On an offline check against the
judged windows, clips covering build-up, moment and ending went from 26% to 81% (87% on held-out
games). That is a coverage check, not a new blinded judging run.

## Notes

- Box-score times in HockeyTech are elapsed in the period; broadcast clocks count down. The
  engine converts, and each league pack states which one its overlays show.
- The project works from recordings you already have; it does not download streams.
- Logos are not shipped. League packs point at logo paths you supply; a missing file falls back
  to a generated placeholder ([assets/logos/README.md](assets/logos/README.md)).
- History: [docs/history/](docs/history/) keeps the notes from the original rewrite.

## License

MIT, as stated in earlier versions of this README. A `LICENSE` file has not been added yet.
