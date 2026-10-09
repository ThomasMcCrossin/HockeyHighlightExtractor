"""
The one optional vision endpoint shared by three jobs: scorebug layout choice when the OCR vote
is weak (scorebug_detect.py), goal placement when the scorebug clock is frozen (goal_locator.py)
and the api review backend (clip_review/).

Nothing here is required. With no endpoint configured every caller skips quietly and the engine
runs on OCR alone. Any OpenAI-compatible chat-completions endpoint that accepts images works:

  CLIP_REVIEW_API_KEY    bearer key
  CLIP_REVIEW_BASE_URL   endpoint root that serves /chat/completions
  CLIP_REVIEW_MODEL      vision-capable model name
  CLIP_REVIEW_API_EXTRA  optional JSON object merged into every request body
                         (provider-specific switches, e.g. turning thinking off)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def configured() -> bool:
    return bool(_env("CLIP_REVIEW_API_KEY") and _env("CLIP_REVIEW_BASE_URL") and _env("CLIP_REVIEW_MODEL"))


def settings() -> Optional[Tuple[str, str, str]]:
    """(api_key, base_url, model), or None when the endpoint is not fully configured."""
    if not configured():
        return None
    return _env("CLIP_REVIEW_API_KEY"), _env("CLIP_REVIEW_BASE_URL").rstrip("/"), _env("CLIP_REVIEW_MODEL")


def extra_body() -> Dict[str, Any]:
    raw = _env("CLIP_REVIEW_API_EXTRA")
    return json.loads(raw) if raw else {}
