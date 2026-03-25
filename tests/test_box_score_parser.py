"""
Unit tests for BoxScoreParser
"""

import pytest
import sys
import os

# Add parent directory to path for direct imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hockey_extractor.box_score_parser import BoxScoreParser
from hockey_extractor.goal import Goal, GoalType


class TestBoxScoreParser:
    """Tests for BoxScoreParser"""

    def setup_method(self):
        """Setup test fixtures"""
        self.parser = BoxScoreParser()

    def test_parse_empty_box_score(self):
        """Test parsing empty box score"""
        goals = self.parser.parse_goals({})
        assert goals == []

    def test_parse_goals_hockeytech_format(self):
        """Test parsing goals from HockeyTech API format"""
        box_score = {
            'SiteKit': {
                'Gamesummary': {
                    'goals': [
                        {
                            'period': 1,
                            'time': '15:23',
                            'team': 'Amherst Ramblers',
                            'goal': {'name': 'John Smith'},
                            'assist1': {'name': 'Jane Doe'},
                            'assist2': {'name': 'Bob Wilson'},
                            'plus_minus': 'PP'
                        },
                        {
                            'period': 2,
                            'time': '10:00',
                            'team': 'Truro Bearcats',
                            'goal': {'name': 'Player X'},
                            'assist1': {'name': ''},
                            'assist2': {'name': ''},
                            'plus_minus': ''
                        }
                    ]
                }
            }
        }

        goals = self.parser.parse_goals(box_score)

        assert len(goals) == 2

        # Check first goal
        assert goals[0].period == 1
        assert goals[0].time == '15:23'
        assert goals[0].team == 'Amherst Ramblers'
        assert goals[0].scorer == 'John Smith'
        assert goals[0].assist1 == 'Jane Doe'
        assert goals[0].assist2 == 'Bob Wilson'
        assert goals[0].goal_type == GoalType.POWER_PLAY

        # Check second goal
        assert goals[1].period == 2
        assert goals[1].scorer == 'Player X'
        assert goals[1].assist1 is None
        assert goals[1].goal_type is None

    def test_parse_goals_direct_format(self):
        """Test parsing goals with direct field names"""
        box_score = {
            'goals': [
                {
                    'period': 1,
                    'time': '12:00',
                    'team': 'Team A',
                    'scorer': 'Player 1',
                    'assist1': 'Player 2',
                    'special': 'SH'
                }
            ]
        }

        goals = self.parser.parse_goals(box_score)

        assert len(goals) == 1
        assert goals[0].scorer == 'Player 1'
        assert goals[0].goal_type == GoalType.SHORT_HANDED

    def test_parse_goals_alternative_format(self):
        """Test parsing goals with scoring_plays format"""
        box_score = {
            'SiteKit': {
                'Gamesummary': {
                    'scoring_plays': [
                        {
                            'period': 1,
                            'time': '8:30',
                            'team_name': 'Team X',
                            'scorer_name': 'Scorer Y',
                        }
                    ]
                }
            }
        }

        goals = self.parser.parse_goals(box_score)

        assert len(goals) == 1
        assert goals[0].team == 'Team X'

    def test_parse_goal_summary(self):
        """Test parsing goal summary with team context"""
        box_score = {
            'goals': [
                {'period': 1, 'time': '15:00', 'team': 'Team A', 'scorer': 'P1'},
                {'period': 2, 'time': '10:00', 'team': 'Team B', 'scorer': 'P2'},
            ]
        }

        summary = self.parser.parse_goal_summary(
            box_score,
            home_team='Team A',
            away_team='Team B'
        )

        assert summary.home_team == 'Team A'
        assert summary.away_team == 'Team B'
        assert summary.home_score == 1
        assert summary.away_score == 1
        assert summary.total_goals == 2

    def test_goals_to_event_dicts(self):
        """Test converting goals to event dictionaries"""
        goals = [
            Goal(period=1, time='15:00', team='Team A', scorer='P1'),
            Goal(period=2, time='10:00', team='Team B', scorer='P2'),
        ]

        events = self.parser.goals_to_event_dicts(goals)

        assert len(events) == 2
        assert all(e['type'] == 'goal' for e in events)
        assert events[0]['period'] == 1
        assert events[0]['time'] == '15:00'
        assert events[0]['scorer'] == 'P1'

    def test_goals_sorted_by_time(self):
        """Test that goals are sorted by period and time"""
        box_score = {
            'goals': [
                {'period': 2, 'time': '10:00', 'team': 'Team A', 'scorer': 'P2'},
                {'period': 1, 'time': '5:00', 'team': 'Team A', 'scorer': 'P1a'},
                {'period': 1, 'time': '15:00', 'team': 'Team A', 'scorer': 'P1b'},
            ]
        }

        goals = self.parser.parse_goals(box_score)

        # Current parser normalizes goals in period order, then by elapsed clock time.
        assert goals[0].period == 1
        assert goals[0].time == '5:00'
        assert goals[1].period == 1
        assert goals[1].time == '15:00'
        assert goals[2].period == 2

    def test_handles_missing_fields_gracefully(self):
        """Test that parser handles missing fields without crashing"""
        box_score = {
            'goals': [
                {'period': 1, 'time': '15:00'},  # Missing team and scorer
                {'period': 1, 'team': 'Team A', 'scorer': 'P1'},  # Missing time
                {'time': '10:00', 'team': 'Team A', 'scorer': 'P1'},  # Missing period
            ]
        }

        # Should not raise exception
        goals = self.parser.parse_goals(box_score)

        # None of these should parse successfully due to missing required fields
        assert len(goals) == 0

    def test_extracts_boolean_goal_types(self):
        """Test parsing goal types from boolean flags"""
        box_score = {
            'goals': [
                {
                    'period': 1,
                    'time': '15:00',
                    'team': 'Team A',
                    'scorer': 'P1',
                    'power_play': True
                },
                {
                    'period': 2,
                    'time': '10:00',
                    'team': 'Team A',
                    'scorer': 'P2',
                    'empty_net': True
                }
            ]
        }

        goals = self.parser.parse_goals(box_score)

        assert len(goals) == 2
        assert goals[0].goal_type == GoalType.POWER_PLAY
        assert goals[1].goal_type == GoalType.EMPTY_NET

    def test_extracts_assists_from_array(self):
        """Test parsing assists from assists array"""
        box_score = {
            'goals': [
                {
                    'period': 1,
                    'time': '15:00',
                    'team': 'Team A',
                    'scorer': 'Scorer',
                    'assists': ['Assist1', 'Assist2']
                }
            ]
        }

        goals = self.parser.parse_goals(box_score)

        assert len(goals) == 1
        assert goals[0].assist1 == 'Assist1'
        assert goals[0].assist2 == 'Assist2'
