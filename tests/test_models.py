"""
Unit tests for domain models
"""

import pytest
from hockey_extractor.models import GameInfo, Event, VideoTimestamp, PipelineResult


class TestGameInfo:
    """Tests for GameInfo model"""

    def test_valid_game_info(self):
        """Test creating valid GameInfo"""
        game_info = GameInfo(
            date='2025-01-15',
            home_team='Amherst Ramblers',
            away_team='Truro Bearcats',
            league='MHL',
            filename='test.mp4',
            home_away='home',
            time='7.00pm'
        )

        assert game_info.date == '2025-01-15'
        assert game_info.home_team == 'Amherst Ramblers'
        assert game_info.league == 'MHL'
        assert game_info.date_formatted is not None

    def test_invalid_date_format(self):
        """Test that invalid date format raises ValueError"""
        with pytest.raises(ValueError, match="Invalid date format"):
            GameInfo(
                date='01-15-2025',  # Wrong format
                home_team='Team1',
                away_team='Team2',
                league='MHL',
                filename='test.mp4'
            )

    def test_invalid_league(self):
        """Test that invalid league raises ValueError"""
        with pytest.raises(ValueError, match="Invalid league"):
            GameInfo(
                date='2025-01-15',
                home_team='Team1',
                away_team='Team2',
                league='NHL',  # Invalid league
                filename='test.mp4'
            )

    def test_empty_team_name(self):
        """Test that empty team names raise ValueError"""
        with pytest.raises(ValueError, match="cannot be empty"):
            GameInfo(
                date='2025-01-15',
                home_team='',  # Empty team name
                away_team='Team2',
                league='MHL',
                filename='test.mp4'
            )


class TestEvent:
    """Tests for Event model"""

    def test_valid_goal_event(self):
        """Test creating valid goal event"""
        event = Event(
            type='goal',
            period=2,
            time='15:23',
            team='Amherst Ramblers',
            scorer='John Smith',
            video_time=1823.5,
            match_confidence=0.95
        )

        assert event.type == 'goal'
        assert event.period == 2
        assert event.scorer == 'John Smith'
        assert event.match_confidence == 0.95

    def test_valid_penalty_event(self):
        """Test creating valid penalty event"""
        event = Event(
            type='penalty',
            period=1,
            time='12:00',
            team='Truro Bearcats',
            player='Jane Doe',
            infraction='Tripping',
            minutes=2
        )

        assert event.type == 'penalty'
        assert event.player == 'Jane Doe'
        assert event.minutes == 2

    def test_invalid_event_type(self):
        """Test that invalid event type raises ValueError"""
        with pytest.raises(ValueError, match="Invalid event type"):
            Event(
                type='save',  # Invalid type
                period=1,
                time='10:00',
                team='Team1'
            )

    def test_invalid_period(self):
        """Test that invalid period raises ValueError"""
        with pytest.raises(ValueError, match="Invalid period"):
            Event(
                type='goal',
                period=10,  # Invalid period
                time='10:00',
                team='Team1',
                scorer='Player'
            )

    def test_invalid_time_format(self):
        """Test that invalid time format raises ValueError"""
        with pytest.raises(ValueError, match="Invalid time format"):
            Event(
                type='goal',
                period=1,
                time='25:70',  # Invalid time
                team='Team1',
                scorer='Player'
            )

    def test_goal_without_scorer(self):
        """Test that goal without scorer raises ValueError"""
        with pytest.raises(ValueError, match="must have a scorer"):
            Event(
                type='goal',
                period=1,
                time='10:00',
                team='Team1',
                scorer=None  # Missing scorer
            )

    def test_to_dict_conversion(self):
        """Test converting event to dictionary"""
        event = Event(
            type='goal',
            period=2,
            time='15:23',
            team='Team1',
            scorer='John Smith',
            video_time=1823.5,
            match_confidence=0.95
        )

        event_dict = event.to_dict()

        assert event_dict['type'] == 'goal'
        assert event_dict['period'] == 2
        assert event_dict['scorer'] == 'John Smith'
        assert event_dict['video_time'] == 1823.5
        assert event_dict['match_confidence'] == 0.95

    def test_from_dict_conversion(self):
        """Test creating event from dictionary"""
        event_dict = {
            'type': 'goal',
            'period': 2,
            'time': '15:23',
            'team': 'Team1',
            'scorer': 'John Smith',
            'video_time': 1823.5,
            'match_confidence': 0.95
        }

        event = Event.from_dict(event_dict)

        assert event.type == 'goal'
        assert event.period == 2
        assert event.scorer == 'John Smith'
        assert event.video_time == 1823.5


class TestVideoTimestamp:
    """Tests for VideoTimestamp model"""

    def test_valid_timestamp(self):
        """Test creating valid video timestamp"""
        timestamp = VideoTimestamp(
            video_time=1823.5,
            period=2,
            game_time='15:23',
            game_time_seconds=923
        )

        assert timestamp.video_time == 1823.5
        assert timestamp.period == 2
        assert timestamp.game_time == '15:23'
        assert timestamp.game_time_seconds == 923

    def test_inconsistent_time_seconds(self):
        """Test that inconsistent time_seconds raises ValueError"""
        with pytest.raises(ValueError, match="Inconsistent game_time_seconds"):
            VideoTimestamp(
                video_time=1823.5,
                period=2,
                game_time='15:23',
                game_time_seconds=999  # Doesn't match 15:23
            )

    def test_negative_video_time(self):
        """Test that negative video_time raises ValueError"""
        with pytest.raises(ValueError, match="must be >= 0"):
            VideoTimestamp(
                video_time=-10.0,  # Negative time
                period=2,
                game_time='15:23',
                game_time_seconds=923
            )


class TestPipelineResult:
    """Tests for PipelineResult model"""

    def test_valid_result(self):
        """Test creating valid pipeline result"""
        game_info = GameInfo(
            date='2025-01-15',
            home_team='Team1',
            away_team='Team2',
            league='MHL',
            filename='test.mp4'
        )

        result = PipelineResult(
            success=True,
            game_info=game_info,
            events_found=10,
            events_matched=8,
            clips_created=8,
            highlights_path='/path/to/highlights.mp4'
        )

        assert result.success is True
        assert result.events_found == 10
        assert result.events_matched == 8
        assert result.match_rate() == 80.0

    def test_events_matched_exceeds_found(self):
        """Test that events_matched > events_found raises ValueError"""
        game_info = GameInfo(
            date='2025-01-15',
            home_team='Team1',
            away_team='Team2',
            league='MHL',
            filename='test.mp4'
        )

        with pytest.raises(ValueError, match="cannot exceed"):
            PipelineResult(
                success=True,
                game_info=game_info,
                events_found=5,
                events_matched=10,  # More than found
                clips_created=5
            )

    def test_match_rate_calculation(self):
        """Test match rate calculation"""
        game_info = GameInfo(
            date='2025-01-15',
            home_team='Team1',
            away_team='Team2',
            league='MHL',
            filename='test.mp4'
        )

        result = PipelineResult(
            success=True,
            game_info=game_info,
            events_found=10,
            events_matched=7,
            clips_created=7
        )

        assert result.match_rate() == 70.0

    def test_match_rate_zero_events(self):
        """Test match rate with zero events"""
        game_info = GameInfo(
            date='2025-01-15',
            home_team='Team1',
            away_team='Team2',
            league='MHL',
            filename='test.mp4'
        )

        result = PipelineResult(
            success=True,
            game_info=game_info,
            events_found=0,
            events_matched=0,
            clips_created=0
        )

        assert result.match_rate() == 0.0
