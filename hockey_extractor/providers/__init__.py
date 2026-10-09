"""Box-score data providers. Each yields a fetcher the pipeline can use in place of an API call."""

from .base import PreloadedBoxScoreFetcher
from .hockeytech import HockeyTechProvider
from .manual import FORMAT as MANUAL_FORMAT, ManualBoxScoreProvider

__all__ = ["PreloadedBoxScoreFetcher", "HockeyTechProvider", "ManualBoxScoreProvider", "MANUAL_FORMAT"]
