#!/usr/bin/env python3
"""
A reviewer that uses no model: it only shows how an agent command plugs in.

  export CLIP_REVIEW_AGENT_CMD='python {repo}/examples/stub_review_agent.py {prompt}'
  python scripts/review_game.py --game-dir Games/<game> --video rec.mp4 --backend agent --apply

The command receives the prompt (see clip_review/backends.py REVIEW_PROMPT) as its argument. A real
agent reads SKILL.md, looks at the contact sheets in the packet and decides. This stub skips the
looking: it keeps the event where the engine put it and sets a fixed window of 20 s before and
8 s after for a goal, so it exercises packets, verdict checking, overrides and the reel manifest.
It is a test double, not a reviewer.
"""
import json
import re
import sys
from pathlib import Path

prompt = " ".join(sys.argv[1:]) or sys.stdin.read()
# game folders can contain spaces, so match to the delimiters rather than to whitespace
packet = Path(re.search(r"Packet directory: (.+?) \. ", prompt).group(1))
out = Path(re.search(r"Write the verdict JSON to: (.+?)\n", prompt).group(1))
incident = json.loads((packet / "incident.json").read_text())
b = incident["bounds"]
lead = max(b["lead_min"], 20.0) if incident["class"] == "goal" else max(b["lead_min"], 5.0)
tail = max(b["tail_min"], 8.0)
verdict = {
    "schema": "hockey-clip-review/verdict@1", "role": "reviewer", "incident_id": incident["incident_id"],
    "decision": "adjust", "event_visible": True, "event_t": 0.0, "in_t": -lead, "out_t": tail,
    "in_kind": "other", "out_kind": "other", "scorebug": "unknown",
    "evidence": [{"t": 0.0, "what": "stub: event assumed at the engine's anchor"}],
    "confidence": 0.5, "reason": "stub reviewer: fixed window around the engine's anchor",
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(verdict, indent=2))
print(f"VERDICT {out}")
