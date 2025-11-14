"""
OCR Engine - Extracts game time from video scoreboards using Tesseract OCR
"""

import logging
import re
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import numpy as np
import cv2

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("pytesseract not installed - OCR functionality disabled")

logger = logging.getLogger(__name__)


class OCREngine:
    """Extracts time information from video scoreboards"""

    def __init__(self, config=None):
        """
        Initialize OCR Engine

        Args:
            config: Optional configuration object
        """
        self.config = config
        self.scoreboard_roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)

        if not TESSERACT_AVAILABLE:
            logger.error("pytesseract not available - install with: pip install pytesseract")

    def detect_scoreboard_roi(
        self,
        frame: np.ndarray,
        method: str = 'auto'
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect scoreboard region in frame

        Args:
            frame: Video frame (RGB or BGR)
            method: Detection method ('auto', 'top', 'bottom')

        Returns:
            ROI as (x, y, width, height) or None
        """
        try:
            height, width = frame.shape[:2]

            if method == 'top':
                # Assume scoreboard is in top portion of frame
                return (0, 0, width, int(height * 0.15))

            elif method == 'bottom':
                # Assume scoreboard is in bottom portion
                y_start = int(height * 0.85)
                return (0, y_start, width, height - y_start)

            else:  # 'auto'
                # Default to top 15% of frame (most common for hockey)
                roi = (0, 0, width, int(height * 0.15))
                logger.info(f"Auto-detected scoreboard ROI: {roi}")
                return roi

        except Exception as e:
            logger.error(f"Failed to detect scoreboard ROI: {e}")
            return None

    def set_scoreboard_roi(self, x: int, y: int, width: int, height: int):
        """
        Manually set scoreboard ROI

        Args:
            x: X coordinate
            y: Y coordinate
            width: Width
            height: Height
        """
        self.scoreboard_roi = (x, y, width, height)
        logger.info(f"Scoreboard ROI set to: {self.scoreboard_roi}")

    def extract_time_from_frame(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[Tuple[int, str]]:
        """
        Extract game time from video frame

        Args:
            frame: Video frame (RGB or BGR)
            roi: Optional region of interest (x, y, w, h). Uses stored ROI if None.

        Returns:
            Tuple of (period, time_string) or None if extraction failed
            Example: (1, "15:23") for Period 1, 15:23 remaining
        """
        if not TESSERACT_AVAILABLE:
            return None

        try:
            # Use provided ROI or stored ROI
            if roi is None:
                roi = self.scoreboard_roi

            # Auto-detect ROI if not set
            if roi is None:
                roi = self.detect_scoreboard_roi(frame)

            if roi is None:
                logger.warning("No ROI available for time extraction")
                return None

            # Extract ROI from frame
            x, y, w, h = roi
            scoreboard = frame[y:y+h, x:x+w]

            # Preprocess for better OCR
            processed = self._preprocess_for_ocr(scoreboard)

            # Run OCR
            text = pytesseract.image_to_string(
                processed,
                config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789:. PeriodOT'
            )

            logger.debug(f"OCR raw text: {text}")

            # Parse time and period from text
            result = self._parse_time_text(text)

            if result:
                period, time_str = result
                logger.debug(f"Extracted: Period {period}, Time {time_str}")
                return result

            return None

        except Exception as e:
            logger.error(f"Failed to extract time from frame: {e}")
            return None

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy

        Args:
            image: Input image (RGB or BGR)

        Returns:
            Preprocessed grayscale image
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Resize for better OCR (if too small)
            height = gray.shape[0]
            if height < 50:
                scale = 50 / height
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # Apply bilateral filter to reduce noise while keeping edges sharp
            denoised = cv2.bilateralFilter(gray, 5, 50, 50)

            # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            # Apply thresholding
            # Try adaptive thresholding first
            binary = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

            return binary

        except Exception as e:
            logger.warning(f"Preprocessing failed, using original: {e}")
            return image

    def _parse_time_text(self, text: str) -> Optional[Tuple[int, str]]:
        """
        Parse period and time from OCR text

        Args:
            text: Raw OCR text

        Returns:
            Tuple of (period, time_string) or None
        """
        # Clean up text
        text = text.strip().upper()

        # Patterns to match
        # Examples: "1st 15:23", "P2 12:00", "Period 3 5:45", "OT 4:23"
        patterns = [
            r'(?:PERIOD\s*)?(\d)[SNRT][TD]?\s*(\d{1,2}:\d{2})',  # "1st 15:23" or "Period 1st 15:23"
            r'P(?:ERIOD)?\s*(\d)\s*(\d{1,2}:\d{2})',             # "P1 15:23" or "Period 1 15:23"
            r'(\d)\s*(\d{1,2}:\d{2})',                           # "1 15:23"
            r'(OT)\s*(\d{1,2}:\d{2})',                           # "OT 4:23"
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                period_str, time_str = match.groups()

                # Convert period to int (OT = 4)
                if period_str.upper() == 'OT':
                    period = 4
                else:
                    try:
                        period = int(period_str)
                    except ValueError:
                        continue

                # Validate time format
                if self._validate_time_format(time_str):
                    return (period, time_str)

        # Try to find just a time string if period not found
        time_match = re.search(r'(\d{1,2}:\d{2})', text)
        if time_match:
            time_str = time_match.group(1)
            if self._validate_time_format(time_str):
                # Default to period 1 if we can't determine period
                logger.warning(f"Found time {time_str} but no period - defaulting to P1")
                return (1, time_str)

        logger.debug(f"Could not parse time from: {text}")
        return None

    def _validate_time_format(self, time_str: str) -> bool:
        """
        Validate time string format (MM:SS)

        Args:
            time_str: Time string to validate

        Returns:
            True if valid format
        """
        try:
            parts = time_str.split(':')
            if len(parts) != 2:
                return False

            minutes = int(parts[0])
            seconds = int(parts[1])

            # Hockey periods are 20 minutes
            return 0 <= minutes <= 20 and 0 <= seconds <= 59

        except (ValueError, AttributeError):
            return False

    def sample_video_times(
        self,
        video_processor,
        sample_interval: int = 30,
        max_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Sample time from video at regular intervals

        Args:
            video_processor: VideoProcessor instance with loaded video
            sample_interval: Seconds between samples
            max_samples: Maximum number of samples (None for all)

        Returns:
            List of dictionaries with {video_time, period, game_time}
        """
        timestamps = []

        try:
            duration = video_processor.duration
            current_time = 0.0

            sample_count = 0

            while current_time < duration:
                # Check max samples limit
                if max_samples and sample_count >= max_samples:
                    break

                # Get frame at current time
                frame = video_processor.get_frame_at_time(current_time)

                if frame is not None:
                    # Extract time from frame
                    result = self.extract_time_from_frame(frame)

                    if result:
                        period, game_time = result
                        timestamps.append({
                            'video_time': current_time,
                            'period': period,
                            'game_time': game_time,
                            'game_time_seconds': self._time_to_seconds(game_time)
                        })
                        logger.debug(f"Sample at {current_time:.1f}s: P{period} {game_time}")

                # Move to next sample
                current_time += sample_interval
                sample_count += 1

            logger.info(f"Sampled {len(timestamps)} timestamps from video")
            return timestamps

        except Exception as e:
            logger.error(f"Failed to sample video times: {e}")
            return []

    def _time_to_seconds(self, time_str: str) -> int:
        """
        Convert MM:SS time string to seconds

        Args:
            time_str: Time string in MM:SS format

        Returns:
            Time in seconds
        """
        try:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
        except (ValueError, AttributeError):
            pass

        return 0

    def save_debug_frame(self, frame: np.ndarray, output_path: Path, roi: Optional[Tuple] = None):
        """
        Save frame with ROI highlighted for debugging

        Args:
            frame: Video frame
            output_path: Where to save image
            roi: Optional ROI to highlight
        """
        try:
            # Make a copy
            debug_frame = frame.copy()

            # Draw ROI if provided
            if roi:
                x, y, w, h = roi
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Save
            cv2.imwrite(str(output_path), debug_frame)
            logger.info(f"Saved debug frame to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save debug frame: {e}")
