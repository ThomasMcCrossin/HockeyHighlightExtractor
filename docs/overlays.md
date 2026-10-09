# Overlays

Overlays are the graphics laid over clips in the reel: a lower third for each goal or penalty,
and optional full-screen intro and final-score cards. They are drawn from three independent parts,
so a league, a look and the pipeline can each change without touching the others.

| part | where | knows about |
|---|---|---|
| **spec** | built by `hockey_extractor/reel.py` | the event, the score, the league's wording |
| **league pack** | `overlays/leagues/<id>/league.json` | teams, colours, logos, period labels |
| **theme** | `overlays/themes/<name>/theme.mjs` | layout and type only; no hockey, no league |

The contract (spec fields, theme API, the rules every theme must meet) is
[overlays/README.md](../overlays/README.md). This page covers how the pieces are used.

## What the pipeline produces

For each clip in the reel, `clip_spec()` builds a spec:

- **Goal**: kind `score`, headline `GOAL` (or `POWER-PLAY GOAL`, `SHORT-HANDED GOAL`,
  `EMPTY-NET GOAL`), the scorer, `Assists: ...` or `Unassisted`, the score after the goal and the
  period and clock as the league shows them. A goal whose video time the engine could not verify
  carries an `UNVERIFIED` badge.
- **Penalty**: kind `penalty`, the player, the infraction and minutes. A fight becomes kind
  `fight` (`FIGHTING MAJORS`).
- **Cards** (`--cards`): kind `intro` and kind `final` with the final score, drawn over the league's
  primary colour.

The producer does all the wording; a theme lays out the text it is given. The running score is
counted from the box score's goals in game order, so a clip's overlay shows the score after its
own goal even when the reel skips other goals.

The overlay appears half a second before the event and stays for six seconds, with a quarter-second
fade at each end (`OVERLAY_LEAD_SECONDS` and `OVERLAY_SECONDS` in `reel.py`). Where the event falls
in the clip comes from the clip's `before_seconds` (engine clips) or its `clip_video_start` and
`video_time` (reviewed clips).

## Running the renderer on its own

```bash
npm install && npx playwright install chromium
python3 overlays/samples.py                      # rebuild the sample specs from the league packs
node overlays/render.mjs --theme baseline        # PNGs and checks.json in overlays/out/baseline/
node overlays/render.mjs --theme baseline --spec my_spec.json --output my.png
```

`overlays/samples/` holds twelve specs (goal, penalty, fight, save, intermission, final, intro and
others, including a fictional soccer league and a team with a light colour), all with fictional
players. The renderer writes `checks.json`: for every sample, any text that is clipped, off the
frame, or over the broadcast scorebug zone (top-left, 720 by 190 pixels at 1080p). The
baseline theme has no findings.

From Python:

```python
from overlays.spec import League, render
league = League("demo-hockey")                    # any pack under overlays/leagues or HOCKEY_LEAGUES_DIR
spec = league.spec("score", home="HAW", away="RRM", side="home", score=(1, 0), period="1", time="14:48",
                   headline="GOAL", subject=("Sam Rivera", "17"), lines=["Assists: Alex Tremblay"])
render(spec, "baseline", "goal.png")              # transparent 1920x1080
```

The PNG is always 1920 by 1080; the reel builder scales it to the recording's size.

## Writing a theme

Copy `overlays/themes/baseline/` to `overlays/themes/<name>/` and edit `theme.mjs`. A theme exports
`meta` and `render(spec, ctx)`, which returns `{css, html}` for a transparent page. Escape every
string from the spec with `ctx.esc`; load a bundled image with `ctx.asset`; load a font with
`ctx.fontFace` (open-licence fonts only, with the licence file beside them). Bebas Neue and Barlow
Semi Condensed are built in. No network at render time and no JavaScript animation: the output is a
still image. Text must not be clipped, nothing may leave the 96 by 54 pixel title-safe margin, and
lower thirds stay in the bottom third and clear of the scorebug zone; the renderer checks the first
three. Pick a theme with `--theme <name>`.

Only the `baseline` theme is bundled: it is a plain reference that exercises every kind.

## Logos and fonts

No team or league logos are shipped. A pack's `logo` paths resolve against the repository root (or
are absolute); a missing file is replaced by `assets/logos/fallback.png`, a generated placeholder.
The MHL pack names `assets/logos/mhl/<team-slug>.png` and `assets/logos/league-mhl.png`; add them if
you have the right to use them. The `demo-hockey` and `demo-soccer` packs use small invented SVG
shields in their own folders. Fonts in `assets/fonts/` are under the SIL Open Font License; the
licence texts are beside them.
