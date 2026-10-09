# Vision review

Optional. The engine alone runs with no AI and no key. Vision review adds a reviewer that looks at
frames from the recording and re-picks each clip's in and out points. It is off unless you
configure it, and nothing complains when it is off.

## Why it exists

The engine finds each event by reading the scorebug clock and then cuts a fixed window around that
moment. The moment is usually right and the window is usually wrong for the play: goal clips often
end as the celebration starts, minor-penalty clips often start after the call, and a frozen or
lagging scorebug can misplace an event by tens of seconds. A reviewer that can see the picture can
pick the build-up, the end of the celebration, the call, and can notice a replay or a scoreboard
outage.

## Two tiers

| | engine only | engine + vision review |
|---|---|---|
| needs | OCR (Tesseract) | an agent command or a vision API, plus a few minutes per game |
| clip windows | fixed, per event type | chosen from the play, within bounds |
| bad scorebug | event may be misplaced | reviewer can relocate it or drop the clip |
| judged highlight quality (0 to 10, 2026-10-08) | 4.56 | 5.4 to 7.1 for agent reviewers |
| cost | none | 200k to 1.3M tokens per incident for agents; 11k for the one-shot API |

An engine-improvement effort is raising the engine-only tier separately; the next engine sync
changes the first row's numbers, not the structure.

## Turning it on

There is one switch: configuration being present. `scripts/run_game.py --review auto` (the default)
looks at the environment, never fails because of it, and uses:

| set | backend |
|---|---|
| `CLIP_REVIEW_AGENT_CMD` | `agent` |
| `CLIP_REVIEW_API_KEY`, `CLIP_REVIEW_BASE_URL`, `CLIP_REVIEW_MODEL` | `api` |
| both | `escalate` |
| neither | none; the engine's clips go into the reel |

`--review off` disables it even when configured. To run it on a game you already processed:

```bash
python scripts/review_game.py --game-dir "Games/<game>" --video recording.mp4 --apply
python scripts/build_reel.py   --game-dir "Games/<game>" --reviewed
```

## Backends

**api**: any OpenAI-compatible chat-completions endpoint that accepts images. No provider is
assumed. Set `CLIP_REVIEW_API_KEY`, `CLIP_REVIEW_BASE_URL` (the root that serves `/chat/completions`)
and `CLIP_REVIEW_MODEL`. `CLIP_REVIEW_API_EXTRA` is a JSON object merged into every request, for
provider-specific switches (for example turning a model's thinking mode off). It makes two passes:
coarse contact sheets, then dense sheets around the chosen boundaries. It is cheap and fast and, in the
bake-off, it was the weakest reviewer: it cannot go back for a frame it did not get.

**agent**: any command-line agent that can read image files and run shell commands. The command is
yours; nothing in the code names a model or a vendor.

```bash
export CLIP_REVIEW_AGENT_CMD='<agent> <flags> {prompt}'
export CLIP_REVIEW_ADVERSARY_CMD='<another agent> <flags> {prompt}'   # optional
```

The command runs inside the incident's packet directory. `{prompt}` is replaced by the prompt
(without it the prompt goes to standard input); `{skill}` by the skill directory and `{repo}`
by the repository root. The prompt tells the agent to read `skills/hockey-clip-review/SKILL.md`,
look at the contact sheets, pull more frames with the skill's `frames.py` if it needs them, write a
verdict file and run `check_verdict.py` until it prints OK. A command that works for any agent
that can do that is a valid reviewer. Token and cost figures are read from the agent's
output where it prints them (a single JSON result, or a stream of JSON events).

`examples/stub_review_agent.py` is a reviewer that uses no model; it exists to show the protocol
and to test the plumbing:

```bash
export CLIP_REVIEW_AGENT_CMD='python {repo}/examples/stub_review_agent.py {prompt}'
```

**escalate**: the api backend for every incident, handing off to the agent command only when the
call is weak (a fight or major, a game with a scorebug alert, an unsure or failed verdict, or
confidence under `CLIP_REVIEW_ESCALATE_MIN_CONFIDENCE`, default 0.6). `summary.json` reports the
hand-off rate. In the bake-off it handed off 59% of incidents and scored 5.39, below the agent
alone; use it to bound spending, not to improve quality.

### Agent budget

The skill tells agents to stop after about 20 turns and 12 frame pulls. An agent that loops can
still spend, so put a hard guard in the command if your agent has one (a maximum number of turns).
`skills/hockey-clip-review/harness/pi-tool-budget.ts` is such a guard for the pi agent: it blocks
frame pulls after `HCR_MAX_TOOL_CALLS` tool calls (default 24) and stops the run `HCR_TOOL_GRACE`
(8) calls later. Before the guards, one agent used 4.06M tokens and 73 turns on a single
fight; with them, 479k and 18 turns, and every budgeted run still returned a valid verdict.

## How an agent overrules the engine

Per incident the pipeline:

1. **Packet** (code): `incident.json` (the box-score rows, the engine's anchor time and window,
   neighbouring events, a scorebug-alert flag, the authority bounds below) and contact sheets of
   frames, numbered by seconds from the anchor. Goals: 75 s before to 40 s after at 1 s steps (120
   before and 60 after at 1.5 s when the game had a scorebug alert). Minors: 60 before, 25 after.
   Majors and fights: 120 either side.
2. **Review**: the reviewer writes a `hockey-clip-review/verdict@1`: `keep`, `adjust` (new in and
   out points from the play: for a goal, from the start of the scoring play to the end of the
   celebration), `relocate` (the event is more than 30 s from the anchor; needs frame evidence),
   `drop` (the event is not in the recording; needs evidence) or `unsure`.
3. **Adversary** (`--adversary`; goals, majors and fights): a second reviewer gets the proposed
   final clip and tries to refute it (event not in the clip, cut before the puck crosses, starts
   mid-play, replay included, fight cut off, wrong incident). A dispute triggers a fresh review that
   must answer the objection; if still disputed the clip is `held_for_human`: the engine window
   stays and the incident is listed in the summary.
4. **Apply** (`--apply`): writes `data/review/overrides.json`, cuts reviewed clips into
   `data/review/clips/` and writes `reel_main.json`. Statuses: `override` (the reviewer's window
   replaces the engine's), `drop` (the clip leaves the reel), and `keep`, `unsure`, `rejected`,
   `no_verdict`, `held_for_human` (the engine's window stays).

The reviewer has authority, not the last word. The code enforces bounds on every verdict
(`clip_review/incidents.py`, `skills/hockey-clip-review/scripts/check_verdict.py`), the same
checks the agents run on themselves:

| class | in-point | after the event | length |
|---|---|---|---|
| goal | 15 to 45 s before the goal (floor `CLIP_REVIEW_MIN_LEAD_S`) | 8 to 25 s (5 s if a replay cuts in); ends at most 16 s after | 8 to 60 s |
| minor penalty | 1 to 20 s before the foul | 2 to 25 s; ends at most 12 s after | 9 to 40 s |
| major | 2 to 45 s before | 5 to 60 s; ends at most 35 s after | 10 to 90 s |
| fight | at most 10 s before the gloves drop | at most 10 s after the players are separated | 10 to 75 s |

Only box-score incidents can be reviewed (the reviewer cannot invent a highlight). A relocation
needs two evidence frames and stays within 240 s of the anchor (360 s for majors and fights). A
window outside the bounds is clamped; a verdict that still fails the checks is discarded and the
engine window kept. `CLIP_REVIEW_TAIL_TRIM=0` turns the rule-based trim of dead air off.

Results are cached per (backend, incident, prompt), so a re-run pays only for what changed.
`data/review/summary.json` lists each incident's status, engine and final window, disputes, wall
time and tokens. The command exits 0 once the summary is written, even when some incidents failed;
those keep the engine window.

Reel modes (`--reel-mode`): `goals` (default), `with-rough` (fights and majors join the main
reel in game order) and `separate-rough` (a second manifest, `reel_rough_stuff.json`).

Incident classes are `goal`, `minor`, `major`, `fight`, `scrum` (several penalties at one stoppage; `penalty_incidents.py`). Adding another means
a `BOUNDS` entry and a classification rule in `clip_review/incidents.py` and a paragraph in the
skill; see [architecture.md](architecture.md).

## Costs

Measured on the same 39 incidents:

| reviewer | tokens per incident | median wall time |
|---|---|---|
| agent, strongest | 672k | 278 s |
| agent, lean (thinking off, 20 tool calls) | 360k | 109 s |
| agent, two others | 814k, 1.32M | 505 s, 383 s |
| agent, smallest | 199k | 123 s |
| api (one-shot) | 11k | 23 s |

A game has a handful of goals and several penalties, each an incident. Dollar cost depends on the provider and the plan: some agents
run on a flat-rate plan (cost 0 per incident), and a harness that reports a list price for a model
it mapped to is not a bill. The api backend records tokens only. Estimate with
tokens per incident times your provider's price, and use `--kinds goal` to review goals only.

## Results (2026-10-08 bake-off)

Question: which reviewer should be allowed to overrule the engine, and is an adversary worth it?
Corpus: 39 incidents from recorded games of the 2026-27 season and the 2026 playoffs (goals in normal
games, goals in games with scorebug alerts, minors, majors and fights). Every reviewer got the same packet; blinded
model judges scored each resulting clip 0 to 10 (overall plus build-up, moment, ending and
cleanliness), two judges per clip and a third on the 10 most disputed (88 verdicts).

| contestant | highlight score | event-seen set (32) | tokens / incident |
|---|---|---|---|
| agent A (strongest) | 7.09 | 7.48 | 672k |
| agent B (lean) | 6.39 | 6.95 | 360k |
| agent C | 6.04 | 6.56 | 814k |
| agent D | 5.81 | 6.65 | 1.32M |
| agent E | 5.60 | 6.24 | 199k |
| escalate (59% hand-off) | 5.39 | 5.82 | 229k |
| engine, no review | 4.56 | 5.37 | none |
| api, one-shot | 3.88 | 4.62 | 11k |

What to take from it:

- The ranking is highlight quality, not "the goal is in the clip" (that is the floor). Every agent
  reviewer beat the engine; the one-shot api reviewer did not.
- Agent A leads mostly on penalties. On goals A and B are level (difference 0.53, 95% interval
  -0.03 to +1.21).
- The engine's minor-penalty clips score 2.2 (the windows land after the call) and its goal clips cut
  the celebration on 42% of clips.
- Judges agree on scores (mean difference 0.84, Spearman 0.76) but pick the same best clip about
  half the time. Compare mean scores, not win rates.
- Goal clips should show the build-up. Before the 15 s floor the reviewers' median build-up
  ranged from 8 s to 20 s, against 32 s for the engine.
- Judges flagged 34 to 38% of agent clips as too long (dead air after the celebration or call). The
  rule-based tail trim cuts 100 of 257 judged agent clips, median 32.0 to 31.0 s; it also shortened
  clips the judges had not flagged, so it is a trade, and `CLIP_REVIEW_TAIL_TRIM=0` turns it off.
- Five incidents had no goal in the footage (a stream outage slate); dropping them was the right
  answer, and only some agents did.
- Not fixed by review: when scorebug detection picks the wrong layout on a broadcast, zero events
  match. Check `data/ocr_sampling_log.txt` and pass `--profile`.

The model names behind the lettered agents are deliberately left out: they are configuration, and
they will have changed by the time you read this. Measure your own choices on a few games with the
same packets; `summary.json` has the tokens and wall time.
