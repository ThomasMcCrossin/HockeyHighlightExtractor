"""
Event Matcher - Syncs box score events with video timestamps using OCR data
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class EventMatcher:
    """Matches box score events to video timestamps"""

    # Hockey period lengths (minutes)
    PERIOD_LENGTH = 20  # Regular periods
    OT_LENGTH = 20      # Overtime (can vary)

    def __init__(self, config=None):
        """
        Initialize EventMatcher

        Args:
            config: Optional configuration object
        """
        self.config = config

    def match_events_to_video(
        self,
        events: List[Dict],
        video_timestamps: List[Dict],
        tolerance_seconds: int = 30
    ) -> List[Dict]:
        """
        Match box score events to video timestamps

        Args:
            events: List of event dicts from box score (with period, time)
            video_timestamps: List of sampled video timestamps (with video_time, period, game_time)
            tolerance_seconds: Maximum time difference for matching (seconds)

        Returns:
            List of events with video_time added
        """
        matched_events = []

        if not video_timestamps:
            logger.warning("No video timestamps available for matching")
            return events

        logger.info(f"Matching {len(events)} events to {len(video_timestamps)} video timestamps")

        for event in events:
            try:
                # Find closest video timestamp for this event
                video_time = self._find_closest_timestamp(
                    event,
                    video_timestamps,
                    tolerance_seconds
                )

                if video_time is not None:
                    # Create new event dict with video_time
                    matched_event = event.copy()
                    matched_event['video_time'] = video_time
                    matched_events.append(matched_event)

                    logger.debug(
                        f"Matched {event['type']} at P{event['period']} {event['time']} "
                        f"to video time {video_time:.1f}s"
                    )
                else:
                    logger.warning(
                        f"Could not match {event['type']} at P{event['period']} {event['time']}"
                    )
                    # Still include event but without video_time
                    matched_events.append(event)

            except Exception as e:
                logger.error(f"Error matching event: {e}")
                matched_events.append(event)

        # Count successful matches
        successful = sum(1 for e in matched_events if e.get('video_time') is not None)
        logger.info(f"Successfully matched {successful}/{len(events)} events")

        return matched_events

    def _find_closest_timestamp(
        self,
        event: Dict,
        video_timestamps: List[Dict],
        tolerance_seconds: int
    ) -> Optional[float]:
        """
        Find the closest video timestamp for a box score event

        Args:
            event: Event dictionary with period and time
            video_timestamps: List of video timestamp dictionaries
            tolerance_seconds: Maximum allowed time difference

        Returns:
            Video time in seconds or None if no match found
        """
        event_period = event.get('period')
        event_time = event.get('time', '00:00')

        # Convert event time to seconds
        event_seconds = self._time_to_seconds(event_time)

        # Filter timestamps for matching period
        period_timestamps = [
            ts for ts in video_timestamps
            if ts.get('period') == event_period
        ]

        if not period_timestamps:
            # Try interpolation if we have timestamps before and after this period
            return self._interpolate_timestamp(event, video_timestamps)

        # Find timestamp with closest game time
        best_match = None
        best_diff = float('inf')

        for ts in period_timestamps:
            ts_seconds = ts.get('game_time_seconds', 0)

            # Calculate time difference
            # Note: Hockey clocks count DOWN, so we need to handle this
            time_diff = abs(event_seconds - ts_seconds)

            if time_diff < best_diff:
                best_diff = time_diff
                best_match = ts

        # Check if match is within tolerance
        if best_match and best_diff <= tolerance_seconds:
            return best_match['video_time']

        # If exact period match failed, try interpolation
        return self._interpolate_timestamp(event, video_timestamps)

    def _interpolate_timestamp(
        self,
        event: Dict,
        video_timestamps: List[Dict]
    ) -> Optional[float]:
        """
        Interpolate video timestamp when exact period match not found

        Args:
            event: Event dictionary
            video_timestamps: List of video timestamps

        Returns:
            Interpolated video time or None
        """
        try:
            event_period = event.get('period')
            event_time = event.get('time', '00:00')
            event_seconds = self._time_to_seconds(event_time)

            # Convert event to absolute game time (seconds from game start)
            event_game_seconds = self._event_to_absolute_time(event_period, event_seconds)

            # Find timestamps before and after the event
            before = None
            after = None

            for ts in video_timestamps:
                ts_game_seconds = self._event_to_absolute_time(
                    ts['period'],
                    ts['game_time_seconds']
                )

                if ts_game_seconds <= event_game_seconds:
                    if before is None or ts_game_seconds > before['abs_time']:
                        before = {
                            'video_time': ts['video_time'],
                            'abs_time': ts_game_seconds
                        }

                if ts_game_seconds >= event_game_seconds:
                    if after is None or ts_game_seconds < after['abs_time']:
                        after = {
                            'video_time': ts['video_time'],
                            'abs_time': ts_game_seconds
                        }

            # Interpolate between before and after
            if before and after:
                # Linear interpolation
                total_time_diff = after['abs_time'] - before['abs_time']
                event_offset = event_game_seconds - before['abs_time']

                if total_time_diff > 0:
                    ratio = event_offset / total_time_diff
                    video_time_diff = after['video_time'] - before['video_time']
                    interpolated_time = before['video_time'] + (ratio * video_time_diff)

                    logger.debug(
                        f"Interpolated P{event_period} {event_time} to {interpolated_time:.1f}s"
                    )
                    return interpolated_time

            # If only before or after exists, use that
            if before:
                logger.debug(f"Using nearest timestamp before event: {before['video_time']:.1f}s")
                return before['video_time']

            if after:
                logger.debug(f"Using nearest timestamp after event: {after['video_time']:.1f}s")
                return after['video_time']

            return None

        except Exception as e:
            logger.error(f"Interpolation failed: {e}")
            return None

    def _event_to_absolute_time(self, period: int, time_seconds: int) -> int:
        """
        Convert period + time to absolute game time (seconds from start)

        Args:
            period: Period number (1, 2, 3, 4=OT)
            time_seconds: Time remaining in period (seconds)

        Returns:
            Absolute game time in seconds
        """
        # Calculate time elapsed in previous periods
        if period == 1:
            previous_periods_time = 0
        elif period == 2:
            previous_periods_time = self.PERIOD_LENGTH * 60
        elif period == 3:
            previous_periods_time = self.PERIOD_LENGTH * 60 * 2
        else:  # OT (period 4+)
            previous_periods_time = self.PERIOD_LENGTH * 60 * 3
            # Add any additional OT periods
            if period > 4:
                previous_periods_time += (period - 4) * self.OT_LENGTH * 60

        # Hockey clocks count DOWN, so we need to invert
        # Time remaining = period_length - time_elapsed
        time_elapsed = (self.PERIOD_LENGTH * 60) - time_seconds

        return previous_periods_time + time_elapsed

    def _time_to_seconds(self, time_str: str) -> int:
        """
        Convert MM:SS time string to seconds

        Args:
            time_str: Time in MM:SS format

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

    def filter_events_by_type(
        self,
        events: List[Dict],
        event_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Filter events by type

        Args:
            events: List of events
            event_types: Types to include (None for all). E.g., ['goal']

        Returns:
            Filtered event list
        """
        if event_types is None:
            return events

        return [e for e in events if e.get('type') in event_types]

    def sort_events_by_video_time(self, events: List[Dict]) -> List[Dict]:
        """
        Sort events by video timestamp

        Args:
            events: List of events

        Returns:
            Sorted event list
        """
        # Only sort events that have video_time
        with_time = [e for e in events if e.get('video_time') is not None]
        without_time = [e for e in events if e.get('video_time') is None]

        # Sort those with time
        with_time.sort(key=lambda e: e['video_time'])

        # Return sorted + unsorted
        return with_time + without_time

    def estimate_missing_timestamps(
        self,
        video_timestamps: List[Dict],
        video_duration: float
    ) -> List[Dict]:
        """
        Fill in missing timestamps using linear interpolation

        This can help when OCR misses some frames

        Args:
            video_timestamps: Existing timestamps
            video_duration: Total video duration in seconds

        Returns:
            Enhanced timestamp list with interpolated values
        """
        if len(video_timestamps) < 2:
            return video_timestamps

        enhanced = video_timestamps.copy()

        # Sort by video time
        enhanced.sort(key=lambda t: t['video_time'])

        # Find gaps and interpolate
        i = 0
        while i < len(enhanced) - 1:
            current = enhanced[i]
            next_ts = enhanced[i + 1]

            video_gap = next_ts['video_time'] - current['video_time']

            # If gap is large (>60 seconds), interpolate
            if video_gap > 60:
                # Check if this might be a period break
                if current['period'] != next_ts['period']:
                    logger.debug(
                        f"Detected period break: P{current['period']} -> P{next_ts['period']}"
                    )
                    # Don't interpolate across period breaks
                    i += 1
                    continue

                # Interpolate timestamps in the gap
                num_interpolated = int(video_gap / 30)  # Every 30 seconds

                for j in range(1, num_interpolated + 1):
                    ratio = j / (num_interpolated + 1)

                    interp_video_time = current['video_time'] + (ratio * video_gap)

                    # Estimate game time (counting down)
                    game_time_diff = current['game_time_seconds'] - next_ts['game_time_seconds']
                    interp_game_time_sec = current['game_time_seconds'] - int(ratio * game_time_diff)

                    enhanced.append({
                        'video_time': interp_video_time,
                        'period': current['period'],
                        'game_time': f"{interp_game_time_sec//60}:{interp_game_time_sec%60:02d}",
                        'game_time_seconds': interp_game_time_sec,
                        'interpolated': True
                    })

            i += 1

        # Re-sort after adding interpolated timestamps
        enhanced.sort(key=lambda t: t['video_time'])

        logger.info(f"Enhanced timestamps: {len(video_timestamps)} -> {len(enhanced)}")

        return enhanced
