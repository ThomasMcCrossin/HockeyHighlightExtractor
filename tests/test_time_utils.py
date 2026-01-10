"""
Unit tests for time utilities
"""

import pytest
import sys
import os

# Add parent directory to path for direct imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hockey_extractor.time_utils import (
    GameTime,
    parse_time_string,
    time_string_to_seconds,
    seconds_to_time_string,
    period_time_to_absolute_seconds,
    absolute_seconds_to_period_time,
    format_period,
    parse_period_string,
    PERIOD_LENGTH_SECONDS,
)


class TestParseTimeString:
    """Tests for parse_time_string function"""

    def test_valid_time(self):
        """Test parsing valid time strings"""
        assert parse_time_string('15:23') == (15, 23)
        assert parse_time_string('5:45') == (5, 45)
        assert parse_time_string('0:00') == (0, 0)
        assert parse_time_string('20:00') == (20, 0)

    def test_invalid_time(self):
        """Test parsing invalid time strings"""
        assert parse_time_string('invalid') == (None, None)
        assert parse_time_string('') == (None, None)
        assert parse_time_string('25') == (None, None)


class TestTimeStringToSeconds:
    """Tests for time_string_to_seconds function"""

    def test_conversion(self):
        """Test time to seconds conversion"""
        assert time_string_to_seconds('15:23') == 923
        assert time_string_to_seconds('0:00') == 0
        assert time_string_to_seconds('20:00') == 1200
        assert time_string_to_seconds('5:45') == 345

    def test_invalid_returns_zero(self):
        """Test that invalid values return 0"""
        assert time_string_to_seconds('invalid') == 0
        assert time_string_to_seconds('') == 0


class TestSecondsToTimeString:
    """Tests for seconds_to_time_string function"""

    def test_conversion(self):
        """Test seconds to time string conversion"""
        assert seconds_to_time_string(923) == '15:23'
        assert seconds_to_time_string(0) == '0:00'
        assert seconds_to_time_string(1200) == '20:00'
        assert seconds_to_time_string(345) == '5:45'

    def test_negative_becomes_zero(self):
        """Test that negative values become 0:00"""
        assert seconds_to_time_string(-100) == '0:00'


class TestPeriodTimeToAbsoluteSeconds:
    """Tests for period_time_to_absolute_seconds function"""

    def test_period_1(self):
        """Test period 1 conversions"""
        # 15:00 remaining = 5 min elapsed = 300 seconds
        assert period_time_to_absolute_seconds(1, 15 * 60) == 300

        # 0:00 remaining = 20 min elapsed = 1200 seconds
        assert period_time_to_absolute_seconds(1, 0) == 1200

    def test_period_2(self):
        """Test period 2 conversions"""
        # P2, 10:00 remaining = 20 (P1) + 10 min = 30 min = 1800 seconds
        assert period_time_to_absolute_seconds(2, 10 * 60) == 1800

    def test_period_3(self):
        """Test period 3 conversions"""
        # P3, 5:00 remaining = 40 (P1+P2) + 15 min = 55 min = 3300 seconds
        assert period_time_to_absolute_seconds(3, 5 * 60) == 3300

    def test_overtime(self):
        """Test overtime conversions"""
        # OT (P4), 15:00 remaining = 60 (3 periods) + 5 min = 65 min = 3900 seconds
        assert period_time_to_absolute_seconds(4, 15 * 60) == 3900


class TestAbsoluteSecondsToPeriodTime:
    """Tests for absolute_seconds_to_period_time function"""

    def test_period_1(self):
        """Test period 1 conversions"""
        # 300 seconds = 5 min elapsed in P1 = 15:00 remaining
        period, time_remaining = absolute_seconds_to_period_time(300)
        assert period == 1
        assert time_remaining == 15 * 60

    def test_period_2(self):
        """Test period 2 conversions"""
        # 1800 seconds = 30 min = P2, 10:00 remaining
        period, time_remaining = absolute_seconds_to_period_time(1800)
        assert period == 2
        assert time_remaining == 10 * 60

    def test_period_3(self):
        """Test period 3 conversions"""
        # 3300 seconds = 55 min = P3, 5:00 remaining
        period, time_remaining = absolute_seconds_to_period_time(3300)
        assert period == 3
        assert time_remaining == 5 * 60

    def test_negative_returns_start(self):
        """Test that negative values return start of game"""
        period, time_remaining = absolute_seconds_to_period_time(-100)
        assert period == 1
        assert time_remaining == PERIOD_LENGTH_SECONDS


class TestFormatPeriod:
    """Tests for format_period function"""

    def test_regular_periods(self):
        """Test formatting regular periods"""
        assert format_period(1) == '1st'
        assert format_period(2) == '2nd'
        assert format_period(3) == '3rd'

    def test_overtime(self):
        """Test formatting overtime periods"""
        assert format_period(4) == 'OT'
        assert format_period(5) == '2OT'


class TestParsePeriodString:
    """Tests for parse_period_string function"""

    def test_numeric(self):
        """Test parsing numeric period strings"""
        assert parse_period_string('1') == 1
        assert parse_period_string('2') == 2
        assert parse_period_string('3') == 3

    def test_ordinal(self):
        """Test parsing ordinal period strings"""
        assert parse_period_string('1st') == 1
        assert parse_period_string('2nd') == 2
        assert parse_period_string('3rd') == 3

    def test_prefix_format(self):
        """Test parsing P1, P2 format"""
        assert parse_period_string('P1') == 1
        assert parse_period_string('P2') == 2
        assert parse_period_string('P3') == 3

    def test_overtime(self):
        """Test parsing overtime strings"""
        assert parse_period_string('OT') == 4
        assert parse_period_string('1OT') == 4
        assert parse_period_string('2OT') == 5

    def test_case_insensitive(self):
        """Test case insensitivity"""
        assert parse_period_string('ot') == 4
        assert parse_period_string('OT') == 4
        assert parse_period_string('p1') == 1


class TestGameTime:
    """Tests for GameTime dataclass"""

    def test_valid_game_time(self):
        """Test creating valid GameTime"""
        gt = GameTime(period=2, time_remaining='15:00')

        assert gt.period == 2
        assert gt.time_remaining == '15:00'
        assert gt.time_remaining_seconds == 900

    def test_time_elapsed_in_period(self):
        """Test time_elapsed_in_period property"""
        gt = GameTime(period=1, time_remaining='15:00')
        assert gt.time_elapsed_in_period == 300  # 5 minutes elapsed

    def test_absolute_seconds(self):
        """Test absolute_seconds property"""
        gt1 = GameTime(period=1, time_remaining='15:00')
        assert gt1.absolute_seconds == 300  # 5 min = 300 sec

        gt2 = GameTime(period=2, time_remaining='10:00')
        assert gt2.absolute_seconds == 1800  # 20 + 10 min = 30 min = 1800 sec

    def test_invalid_period(self):
        """Test that invalid period raises ValueError"""
        with pytest.raises(ValueError, match="Invalid period"):
            GameTime(period=10, time_remaining='15:00')

    def test_invalid_time_format(self):
        """Test that invalid time format raises ValueError"""
        with pytest.raises(ValueError, match="Invalid time format"):
            GameTime(period=1, time_remaining='invalid')

    def test_str_representation(self):
        """Test string representation"""
        gt = GameTime(period=2, time_remaining='15:00')
        assert str(gt) == 'P2 15:00'

    def test_from_period_and_seconds(self):
        """Test creating GameTime from period and seconds"""
        gt = GameTime.from_period_and_seconds(2, 900)
        assert gt.period == 2
        assert gt.time_remaining == '15:00'
