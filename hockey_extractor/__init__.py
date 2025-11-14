"""
Hockey Highlight Extractor - Box Score Based Detection

A modular system for extracting hockey highlights using box score data
and OCR-based time matching.
"""

__version__ = "2.0.0"
__author__ = "Thomas McCrossin"

from .video_processor import VideoProcessor
from .box_score import BoxScoreFetcher
from .ocr_engine import OCREngine
from .event_matcher import EventMatcher
from .file_manager import FileManager

__all__ = [
    'VideoProcessor',
    'BoxScoreFetcher',
    'OCREngine',
    'EventMatcher',
    'FileManager',
]
