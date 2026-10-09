"""
Optional Google Drive settings for the major-penalty review upload.

The upload only runs for the reel modes that need a human to approve major penalties and
only when these are set. The default reel mode never touches Drive.

  HIGHLIGHTS_MAJOR_REVIEW_FOLDER_ID   folder id or folder URL for review clips
  GOOGLE_APPLICATION_CREDENTIALS      service-account key file
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Optional


def normalize_drive_folder_id(value: str) -> str:
    raw = str(value or "").strip()
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", raw)
    return match.group(1) if match else raw


@dataclass(frozen=True)
class ResolvedDriveConfig:
    major_review_folder_id: str
    credentials_path: str


def resolve_drive_config(env: Optional[Dict[str, str]] = None) -> ResolvedDriveConfig:
    source = env if env is not None else os.environ
    folder = source.get("HIGHLIGHTS_MAJOR_REVIEW_FOLDER_ID") or source.get("MAJOR_REVIEW_DRIVE_FOLDER_ID") or ""
    return ResolvedDriveConfig(
        major_review_folder_id=normalize_drive_folder_id(folder),
        credentials_path=str(source.get("GOOGLE_APPLICATION_CREDENTIALS", "") or "").strip(),
    )
