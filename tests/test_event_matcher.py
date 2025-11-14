"""
Unit tests for EventMatcher
"""

import pytest
from hockey_extractor.event_matcher import EventMatcher


class TestEventMatcher:
    """Tests for EventMatcher"""

    def setup_method(self):
        """Setup test fixtures"""
        self.matcher = EventMatcher()

    def test_time_to_seconds(self):
        """Test time string to seconds conversion"""
        assert self.matcher._time_to_seconds('15:23') == 923
        assert self.matcher._time_to_seconds('00:00') == 0
        assert self.matcher._time_to_seconds('20:00') == 1200
        assert self.matcher._time_to_seconds('5:45') == 345

    def test_time_to_seconds_invalid(self):
        """Test invalid time string handling"""
        assert self.matcher._time_to_seconds('invalid') == 0
        assert self.matcher._time_to_seconds('') == 0
        assert self.matcher._time_to_seconds('25') == 0

    def test_event_to_absolute_time(self):
        """Test converting period + time to absolute game time"""
        # Period 1, 15:00 remaining = 5 minutes elapsed
        assert self.matcher._event_to_absolute_time(1, 15 * 60) == 5 * 60

        # Period 2, 10:00 remaining = 20min (P1) + 10min elapsed = 30min
        assert self.matcher._event_to_absolute_time(2, 10 * 60) == 30 * 60

        # Period 3, 5:00 remaining = 40min (P1+P2) + 15min elapsed = 55min
        assert self.matcher._event_to_absolute_time(3, 5 * 60) == 55 * 60

    def test_filter_events_by_type(self):
        """Test filtering events by type"""
        events = [
            {'type': 'goal', 'period': 1, 'time': '10:00', 'team': 'Team1'},
            {'type': 'penalty', 'period': 1, 'time': '8:00', 'team': 'Team2'},
            {'type': 'goal', 'period': 2, 'time': '15:00', 'team': 'Team1'},
        ]

        # Filter goals only
        goals = self.matcher.filter_events_by_type(events, ['goal'])
        assert len(goals) == 2
        assert all(e['type'] == 'goal' for e in goals)

        # Filter penalties only
        penalties = self.matcher.filter_events_by_type(events, ['penalty'])
        assert len(penalties) == 1
        assert penalties[0]['type'] == 'penalty'

        # Filter all types
        all_events = self.matcher.filter_events_by_type(events, None)
        assert len(all_events) == 3

    def test_sort_events_by_video_time(self):
        """Test sorting events by video time"""
        events = [
            {'type': 'goal', 'period': 2, 'time': '15:00', 'video_time': 2000.0},
            {'type': 'goal', 'period': 1, 'time': '10:00', 'video_time': 500.0},
            {'type': 'goal', 'period': 3, 'time': '5:00', 'video_time': 3500.0},
            {'type': 'penalty', 'period': 1, 'time': '8:00'},  # No video_time
        ]

        sorted_events = self.matcher.sort_events_by_video_time(events)

        # Events with video_time should be sorted first
        assert sorted_events[0]['video_time'] == 500.0
        assert sorted_events[1]['video_time'] == 2000.0
        assert sorted_events[2]['video_time'] == 3500.0

        # Event without video_time should be last
        assert sorted_events[3].get('video_time') is None

    def test_find_closest_timestamp_with_confidence_exact_match(self):
        """Test finding closest timestamp with exact match"""
        event = {
            'type': 'goal',
            'period': 2,
            'time': '15:00',  # 15 * 60 = 900 seconds
            'team': 'Team1'
        }

        video_timestamps = [
            {
                'video_time': 500.0,
                'period': 1,
                'game_time': '10:00',
                'game_time_seconds': 600
            },
            {
                'video_time': 2000.0,
                'period': 2,
                'game_time': '15:00',
                'game_time_seconds': 900
            },
        ]

        result = self.matcher._find_closest_timestamp_with_confidence(
            event,
            video_timestamps,
            tolerance_seconds=30
        )

        assert result is not None
        video_time, confidence, time_diff = result

        assert video_time == 2000.0
        assert confidence == 1.0  # Exact match
        assert time_diff == 0

    def test_find_closest_timestamp_with_confidence_near_match(self):
        """Test finding closest timestamp with near match"""
        event = {
            'type': 'goal',
            'period': 2,
            'time': '15:00',  # 900 seconds
            'team': 'Team1'
        }

        video_timestamps = [
            {
                'video_time': 2000.0,
                'period': 2,
                'game_time': '14:50',  # 890 seconds, 10 second diff
                'game_time_seconds': 890
            },
        ]

        result = self.matcher._find_closest_timestamp_with_confidence(
            event,
            video_timestamps,
            tolerance_seconds=30
        )

        assert result is not None
        video_time, confidence, time_diff = result

        assert video_time == 2000.0
        assert 0.6 < confidence < 0.7  # Reduced confidence for 10s diff with 30s tolerance
        assert time_diff == 10

    def test_find_closest_timestamp_no_match(self):
        """Test finding timestamp when no match within tolerance"""
        event = {
            'type': 'goal',
            'period': 2,
            'time': '15:00',
            'team': 'Team1'
        }

        video_timestamps = [
            {
                'video_time': 500.0,
                'period': 1,  # Different period
                'game_time': '10:00',
                'game_time_seconds': 600
            },
        ]

        result = self.matcher._find_closest_timestamp_with_confidence(
            event,
            video_timestamps,
            tolerance_seconds=10
        )

        # Should return interpolated result or None
        # Exact behavior depends on implementation
        assert result is None or (result is not None and result[1] < 0.6)

    def test_estimate_missing_timestamps(self):
        """Test filling in missing timestamps via interpolation"""
        timestamps = [
            {'video_time': 0.0, 'period': 1, 'game_time': '20:00', 'game_time_seconds': 1200},
            {'video_time': 120.0, 'period': 1, 'game_time': '18:00', 'game_time_seconds': 1080},
            # Large gap here
            {'video_time': 500.0, 'period': 1, 'game_time': '10:00', 'game_time_seconds': 600},
        ]

        enhanced = self.matcher.estimate_missing_timestamps(timestamps, video_duration=600.0)

        # Should have more timestamps after interpolation
        assert len(enhanced) >= len(timestamps)

        # Should be sorted by video_time
        for i in range(len(enhanced) - 1):
            assert enhanced[i]['video_time'] <= enhanced[i + 1]['video_time']


class TestEventMatchingIntegration:
    """Integration tests for event matching"""

    def setup_method(self):
        """Setup test fixtures"""
        self.matcher = EventMatcher()

    def test_match_events_to_video_full_pipeline(self):
        """Test complete event matching pipeline"""
        events = [
            {'type': 'goal', 'period': 1, 'time': '15:00', 'team': 'Team1', 'scorer': 'Player1'},
            {'type': 'goal', 'period': 2, 'time': '10:00', 'team': 'Team2', 'scorer': 'Player2'},
        ]

        video_timestamps = [
            {'video_time': 300.0, 'period': 1, 'game_time': '15:00', 'game_time_seconds': 900},
            {'video_time': 1500.0, 'period': 2, 'game_time': '10:00', 'game_time_seconds': 600},
        ]

        matched_events = self.matcher.match_events_to_video(
            events,
            video_timestamps,
            tolerance_seconds=30
        )

        # All events should be matched
        assert len(matched_events) == 2

        # Check first event
        assert matched_events[0]['video_time'] is not None
        assert matched_events[0]['match_confidence'] is not None
        assert 0.0 <= matched_events[0]['match_confidence'] <= 1.0

        # Check second event
        assert matched_events[1]['video_time'] is not None
        assert matched_events[1]['match_confidence'] is not None
