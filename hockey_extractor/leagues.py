"""
League packs: the one description of a league (teams, codes, colours, logos, period labels,
box-score provider) shared by the engine and the overlay renderer.

A pack is `overlays/leagues/<id>/league.json` (see overlays/README.md). Extra pack folders can
be added with the HOCKEY_LEAGUES_DIR environment variable (os.pathsep separated); a pack there
with the same id replaces the bundled one.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

BUNDLED_DIR = Path(__file__).resolve().parents[1] / "overlays" / "leagues"


def pack_dirs() -> List[Path]:
    extra = [Path(p).expanduser() for p in os.environ.get("HOCKEY_LEAGUES_DIR", "").split(os.pathsep) if p]
    return [BUNDLED_DIR, *extra]


@lru_cache(maxsize=None)
def _load_all(dirs: tuple) -> Dict[str, Dict[str, Any]]:
    packs: Dict[str, Dict[str, Any]] = {}
    for root in dirs:
        for f in sorted(Path(root).glob("*/league.json")):
            try:
                pack = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            packs[str(pack.get("id") or f.parent.name)] = pack
    return packs


def load_all() -> Dict[str, Dict[str, Any]]:
    return _load_all(tuple(str(d) for d in pack_dirs()))


def get(league_id: str) -> Optional[Dict[str, Any]]:
    wanted = str(league_id or "").strip().lower()
    for key, pack in load_all().items():
        if key.lower() == wanted or str(pack.get("short", "")).lower() == wanted:
            return pack
    return None


def hockey_packs() -> List[Dict[str, Any]]:
    return [p for p in load_all().values() if p.get("sport", "hockey") == "hockey"]


def provider_config(league_id: str) -> Dict[str, Any]:
    """The pack's `provider` block ({} when the league has none)."""
    pack = get(league_id)
    return dict((pack or {}).get("provider") or {})


def find_team(name: str, league_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Resolve a team by short code, id, name, nickname or city. Returns the team plus its league id."""
    key = str(name or "").strip().lower()
    if not key:
        return None
    packs = [get(league_id)] if league_id else hockey_packs()
    for pack in packs:
        for t in (pack or {}).get("teams", []):
            fields = {str(t.get(f, "")).lower() for f in ("short", "provider_id", "id", "name", "nickname", "city")}
            if key in fields:
                return dict(t, league=pack["id"])
    return None


def detect_league(*team_names: str) -> str:
    """League id whose pack knows any of the team names (full name, or city + nickname contained
    in the text); 'Unknown' when none does."""
    for pack in hockey_packs():
        for t in pack.get("teams", []):
            for name in team_names:
                n = str(name or "").strip().lower()
                if n and (n == str(t.get("name", "")).lower() or n == str(t.get("short", "")).lower()
                          or str(t.get("name", "")).lower() in n
                          or (str(t.get("city", "")).lower() and n == str(t.get("city", "")).lower())
                          or (str(t.get("nickname", "")).lower() and n == str(t.get("nickname", "")).lower())):
                    return str(pack.get("short") or pack["id"]).upper()
    return "Unknown"
