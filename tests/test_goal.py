"""
Unit tests for Goal model and related classes
"""

import pytest
import sys
import os

# Add parent directory to path for direct imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hockey_extractor.goal import Goal, GoalType, GoalSummary


class TestGoalType:
    """Tests for GoalType enum"""

    def test_from_string_power_play(self):
        """Test parsing power play goal type"""
        assert GoalType.from_string('PP') == GoalType.POWER_PLAY
        assert GoalType.from_string('PPG') == GoalType.POWER_PLAY
        assert GoalType.from_string('power play') == GoalType.POWER_PLAY

    def test_from_string_short_handed(self):
        """Test parsing short-handed goal type"""
        assert GoalType.from_string('SH') == GoalType.SHORT_HANDED
        assert GoalType.from_string('SHG') == GoalType.SHORT_HANDED

    def test_from_string_empty_net(self):
        """Test parsing empty net goal type"""
        assert GoalType.from_string('EN') == GoalType.EMPTY_NET
        assert GoalType.from_string('ENG') == GoalType.EMPTY_NET

    def test_from_string_none(self):
        """Test parsing empty/null values"""
        assert GoalType.from_string(None) is None
        assert GoalType.from_string('') is None

    def test_from_string_unknown(self):
        """Test parsing unknown values"""
        assert GoalType.from_string('XYZ') is None


class TestGoal:
    """Tests for Goal model"""

    def test_valid_goal(self):
        """Test creating valid goal"""
        goal = Goal(
            period=2,
            time='15:23',
            team='Amherst Ramblers',
            scorer='John Smith',
            assist1='Jane Doe',
            assist2='Bob Wilson',
            goal_type=GoalType.POWER_PLAY,
        )

        assert goal.period == 2
        assert goal.time == '15:23'
        assert goal.team == 'Amherst Ramblers'
        assert goal.scorer == 'John Smith'
        assert goal.assist1 == 'Jane Doe'
        assert goal.assist2 == 'Bob Wilson'
        assert goal.goal_type == GoalType.POWER_PLAY

    def test_valid_goal_no_assists(self):
        """Test creating goal without assists"""
        goal = Goal(
            period=1,
            time='10:00',
            team='Team A',
            scorer='Player 1',
        )

        assert goal.scorer == 'Player 1'
        assert goal.assist1 is None
        assert goal.assist2 is None
        assert goal.has_assists is False
        assert goal.assist_count == 0

    def test_goal_with_one_assist(self):
        """Test goal with single assist"""
        goal = Goal(
            period=1,
            time='10:00',
            team='Team A',
            scorer='Player 1',
            assist1='Player 2',
        )

        assert goal.has_assists is True
        assert goal.assist_count == 1

    def test_invalid_period(self):
        """Test that invalid period raises ValueError"""
        with pytest.raises(ValueError, match="Invalid period"):
            Goal(
                period=10,
                time='10:00',
                team='Team A',
                scorer='Player 1',
            )

    def test_invalid_time_format(self):
        """Test that invalid time format raises ValueError"""
        with pytest.raises(ValueError, match="Invalid time format"):
            Goal(
                period=1,
                time='invalid',
                team='Team A',
                scorer='Player 1',
            )

    def test_invalid_time_values(self):
        """Test that invalid time values raise ValueError"""
        with pytest.raises(ValueError, match="Invalid minutes"):
            Goal(
                period=1,
                time='25:00',
                team='Team A',
                scorer='Player 1',
            )

    def test_empty_team(self):
        """Test that empty team raises ValueError"""
        with pytest.raises(ValueError, match="Team cannot be empty"):
            Goal(
                period=1,
                time='10:00',
                team='',
                scorer='Player 1',
            )

    def test_empty_scorer(self):
        """Test that empty scorer raises ValueError"""
        with pytest.raises(ValueError, match="Scorer cannot be empty"):
            Goal(
                period=1,
                time='10:00',
                team='Team A',
                scorer='',
            )

    def test_time_seconds_property(self):
        """Test time_seconds property"""
        goal = Goal(
            period=1,
            time='15:30',
            team='Team A',
            scorer='Player 1',
        )
        assert goal.time_seconds == 930  # 15*60 + 30

    def test_absolute_game_seconds_property(self):
        """Test absolute_game_seconds property"""
        # Period 1, 15:00 remaining = 5 minutes elapsed = 300 seconds
        goal = Goal(
            period=1,
            time='15:00',
            team='Team A',
            scorer='Player 1',
        )
        assert goal.absolute_game_seconds == 300

        # Period 2, 10:00 remaining = 20 + 10 minutes = 30 minutes = 1800 seconds
        goal2 = Goal(
            period=2,
            time='10:00',
            team='Team A',
            scorer='Player 1',
        )
        assert goal2.absolute_game_seconds == 1800

    def test_is_special_teams(self):
        """Test is_special_teams property"""
        pp_goal = Goal(
            period=1,
            time='10:00',
            team='Team A',
            scorer='Player 1',
            goal_type=GoalType.POWER_PLAY,
        )
        assert pp_goal.is_special_teams is True

        sh_goal = Goal(
            period=1,
            time='10:00',
            team='Team A',
            scorer='Player 1',
            goal_type=GoalType.SHORT_HANDED,
        )
        assert sh_goal.is_special_teams is True

        es_goal = Goal(
            period=1,
            time='10:00',
            team='Team A',
            scorer='Player 1',
            goal_type=GoalType.EVEN_STRENGTH,
        )
        assert es_goal.is_special_teams is False

    def test_is_matched(self):
        """Test is_matched property"""
        goal = Goal(
            period=1,
            time='10:00',
            team='Team A',
            scorer='Player 1',
        )
        assert goal.is_matched is False

        matched_goal = goal.with_video_time(500.0, 0.95)
        assert matched_goal.is_matched is True
        assert matched_goal.video_time == 500.0
        assert matched_goal.match_confidence == 0.95

    def test_to_dict(self):
        """Test converting goal to dictionary"""
        goal = Goal(
            period=2,
            time='15:23',
            team='Team A',
            scorer='Player 1',
            assist1='Player 2',
            goal_type=GoalType.POWER_PLAY,
            video_time=1500.0,
            match_confidence=0.9,
        )

        goal_dict = goal.to_dict()

        assert goal_dict['type'] == 'goal'
        assert goal_dict['period'] == 2
        assert goal_dict['time'] == '15:23'
        assert goal_dict['scorer'] == 'Player 1'
        assert goal_dict['assist1'] == 'Player 2'
        assert goal_dict['special'] == 'PP'
        assert goal_dict['video_time'] == 1500.0
        assert goal_dict['match_confidence'] == 0.9

    def test_from_dict(self):
        """Test creating goal from dictionary"""
        goal_dict = {
            'period': 2,
            'time': '15:23',
            'team': 'Team A',
            'scorer': 'Player 1',
            'assist1': 'Player 2',
            'special': 'PP',
            'video_time': 1500.0,
            'match_confidence': 0.9,
        }

        goal = Goal.from_dict(goal_dict)

        assert goal.period == 2
        assert goal.time == '15:23'
        assert goal.scorer == 'Player 1'
        assert goal.goal_type == GoalType.POWER_PLAY
        assert goal.video_time == 1500.0

    def test_str_representation(self):
        """Test string representation"""
        goal = Goal(
            period=2,
            time='15:23',
            team='Amherst',
            scorer='Smith',
            assist1='Doe',
            goal_type=GoalType.POWER_PLAY,
        )

        goal_str = str(goal)
        assert 'P2' in goal_str
        assert '15:23' in goal_str
        assert 'Smith' in goal_str
        assert '[PP]' in goal_str
        assert 'Doe' in goal_str


class TestGoalSummary:
    """Tests for GoalSummary model"""

    def test_empty_summary(self):
        """Test empty goal summary"""
        summary = GoalSummary(
            home_team='Team A',
            away_team='Team B',
        )

        assert summary.home_score == 0
        assert summary.away_score == 0
        assert summary.total_goals == 0

    def test_summary_with_goals(self):
        """Test summary with goals"""
        goals = [
            Goal(period=1, time='15:00', team='Team A', scorer='P1'),
            Goal(period=1, time='10:00', team='Team B', scorer='P2'),
            Goal(period=2, time='15:00', team='Team A', scorer='P3'),
        ]

        summary = GoalSummary(
            home_team='Team A',
            away_team='Team B',
            goals=goals,
        )

        assert summary.home_score == 2
        assert summary.away_score == 1
        assert summary.total_goals == 3

    def test_goals_in_period(self):
        """Test filtering goals by period"""
        goals = [
            Goal(period=1, time='15:00', team='Team A', scorer='P1'),
            Goal(period=1, time='10:00', team='Team B', scorer='P2'),
            Goal(period=2, time='15:00', team='Team A', scorer='P3'),
        ]

        summary = GoalSummary(
            home_team='Team A',
            away_team='Team B',
            goals=goals,
        )

        period1_goals = summary.goals_in_period(1)
        assert len(period1_goals) == 2

        period2_goals = summary.goals_in_period(2)
        assert len(period2_goals) == 1

    def test_power_play_goals(self):
        """Test filtering power play goals"""
        goals = [
            Goal(period=1, time='15:00', team='Team A', scorer='P1', goal_type=GoalType.POWER_PLAY),
            Goal(period=1, time='10:00', team='Team B', scorer='P2'),
            Goal(period=2, time='15:00', team='Team A', scorer='P3', goal_type=GoalType.POWER_PLAY),
        ]

        summary = GoalSummary(
            home_team='Team A',
            away_team='Team B',
            goals=goals,
        )

        pp_goals = summary.power_play_goals
        assert len(pp_goals) == 2

    def test_to_dict(self):
        """Test converting summary to dictionary"""
        goals = [
            Goal(period=1, time='15:00', team='Team A', scorer='P1'),
        ]

        summary = GoalSummary(
            home_team='Team A',
            away_team='Team B',
            goals=goals,
        )

        summary_dict = summary.to_dict()

        assert summary_dict['home_team'] == 'Team A'
        assert summary_dict['away_team'] == 'Team B'
        assert summary_dict['home_score'] == 1
        assert summary_dict['away_score'] == 0
        assert len(summary_dict['goals']) == 1

    def test_from_dict(self):
        """Test creating summary from dictionary"""
        summary_dict = {
            'home_team': 'Team A',
            'away_team': 'Team B',
            'goals': [
                {'period': 1, 'time': '15:00', 'team': 'Team A', 'scorer': 'P1'},
            ]
        }

        summary = GoalSummary.from_dict(summary_dict)

        assert summary.home_team == 'Team A'
        assert summary.away_team == 'Team B'
        assert len(summary.goals) == 1
