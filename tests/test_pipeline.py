"""
Unit tests for HighlightPipeline
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from hockey_extractor.pipeline import HighlightPipeline
from hockey_extractor.models import GameInfo, PipelineResult


class TestHighlightPipeline:
    """Tests for HighlightPipeline orchestrator"""

    def test_pipeline_initialization(self):
        """Test pipeline can be initialized"""
        mock_config = Mock()
        video_path = Path('/fake/video.mp4')

        # Create pipeline with mocked dependencies
        pipeline = HighlightPipeline(
            mock_config,
            video_path,
            file_manager=Mock(),
            box_score_fetcher=Mock(),
            video_processor=Mock(),
            ocr_engine=Mock(),
            event_matcher=Mock()
        )

        assert pipeline.video_path == video_path
        assert pipeline.config == mock_config
        assert pipeline.game_info is None
        assert len(pipeline.events) == 0

    def test_pipeline_context_manager(self):
        """Test pipeline can be used as context manager"""
        mock_config = Mock()
        video_path = Path('/fake/video.mp4')

        mock_video_processor = Mock()
        mock_video_processor.cleanup = Mock()

        with HighlightPipeline(
            mock_config,
            video_path,
            file_manager=Mock(),
            box_score_fetcher=Mock(),
            video_processor=mock_video_processor,
            ocr_engine=Mock(),
            event_matcher=Mock()
        ) as pipeline:
            assert pipeline is not None

        # Cleanup should be called on exit
        mock_video_processor.cleanup.assert_called_once()

    def test_step1_parse_and_setup_success(self):
        """Test step 1: Parse filename and setup folders"""
        mock_config = Mock()
        video_path = Path('/fake/2025-01-15 Team1 vs Team2 Home 7.00pm.mp4')

        # Mock FileManager
        mock_file_manager = Mock()
        mock_file_manager.parse_mhl_filename.return_value = {
            'date': '2025-01-15',
            'home_team': 'Team1',
            'away_team': 'Team2',
            'league': 'MHL',
            'filename': video_path.name,
            'home_away': 'home',
            'time': '7.00pm'
        }
        mock_file_manager.create_game_folder.return_value = {
            'game_dir': Path('/output/game'),
            'clips_dir': Path('/output/game/clips'),
            'output_dir': Path('/output/game/output'),
            'data_dir': Path('/output/game/data'),
            'logs_dir': Path('/output/game/logs'),
            'source_dir': Path('/output/game/source')
        }

        pipeline = HighlightPipeline(
            mock_config,
            video_path,
            file_manager=mock_file_manager,
            box_score_fetcher=Mock(),
            video_processor=Mock(),
            ocr_engine=Mock(),
            event_matcher=Mock()
        )

        # Execute step 1
        pipeline._step1_parse_and_setup()

        # Verify
        assert pipeline.game_info is not None
        assert pipeline.game_info.date == '2025-01-15'
        assert pipeline.game_info.home_team == 'Team1'
        assert pipeline.game_info.league == 'MHL'
        assert pipeline.game_folders is not None
        assert 'game_dir' in pipeline.game_folders

    def test_step2_fetch_box_score_success(self):
        """Test step 2: Fetch box score from API"""
        mock_config = Mock()
        video_path = Path('/fake/video.mp4')

        # Mock BoxScoreFetcher
        mock_fetcher = Mock()
        mock_fetcher.find_game.return_value = 'game123'
        mock_fetcher.fetch_box_score.return_value = {'game_data': 'test'}
        mock_fetcher.extract_events.return_value = [
            {'type': 'goal', 'period': 1, 'time': '10:00', 'team': 'Team1'}
        ]

        # Mock FileManager
        mock_file_manager = Mock()

        pipeline = HighlightPipeline(
            mock_config,
            video_path,
            file_manager=mock_file_manager,
            box_score_fetcher=mock_fetcher,
            video_processor=Mock(),
            ocr_engine=Mock(),
            event_matcher=Mock()
        )

        # Setup game_info (required for step 2)
        pipeline.game_info = GameInfo(
            date='2025-01-15',
            home_team='Team1',
            away_team='Team2',
            league='MHL',
            filename='video.mp4'
        )
        pipeline.game_folders = {'data_dir': Path('/tmp')}

        # Execute step 2
        pipeline._step2_fetch_box_score()

        # Verify
        assert pipeline.box_score is not None
        assert len(pipeline.events) == 1
        assert pipeline.events[0]['type'] == 'goal'

    def test_step2_fetch_box_score_no_game_found(self):
        """Test step 2: Handle case where game not found"""
        mock_config = Mock()
        video_path = Path('/fake/video.mp4')

        # Mock BoxScoreFetcher
        mock_fetcher = Mock()
        mock_fetcher.find_game.return_value = None  # Game not found

        pipeline = HighlightPipeline(
            mock_config,
            video_path,
            file_manager=Mock(),
            box_score_fetcher=mock_fetcher,
            video_processor=Mock(),
            ocr_engine=Mock(),
            event_matcher=Mock()
        )

        # Setup game_info
        pipeline.game_info = GameInfo(
            date='2025-01-15',
            home_team='Team1',
            away_team='Team2',
            league='MHL',
            filename='video.mp4'
        )
        pipeline.game_folders = {'data_dir': Path('/tmp')}

        # Should raise ValueError
        with pytest.raises(ValueError, match="Could not find game"):
            pipeline._step2_fetch_box_score()

    def test_create_result(self):
        """Test creating PipelineResult"""
        mock_config = Mock()
        video_path = Path('/fake/video.mp4')

        pipeline = HighlightPipeline(
            mock_config,
            video_path,
            file_manager=Mock(),
            box_score_fetcher=Mock(),
            video_processor=Mock(),
            ocr_engine=Mock(),
            event_matcher=Mock()
        )

        # Setup some state
        pipeline.game_info = GameInfo(
            date='2025-01-15',
            home_team='Team1',
            away_team='Team2',
            league='MHL',
            filename='video.mp4'
        )
        pipeline.events = [
            {'type': 'goal', 'period': 1, 'time': '10:00'},
            {'type': 'goal', 'period': 2, 'time': '15:00'}
        ]
        pipeline.matched_events = [
            {'type': 'goal', 'period': 1, 'time': '10:00', 'video_time': 500.0},
            {'type': 'goal', 'period': 2, 'time': '15:00'}  # No video_time
        ]
        pipeline.created_clips = [
            ({'type': 'goal'}, Path('/clip1.mp4'))
        ]

        pipeline._pipeline_start_time = 0.0
        import time
        time.sleep(0.01)  # Small delay to get duration > 0

        # Create result
        result = pipeline._create_result(
            success=True,
            errors=[],
            warnings=['test warning'],
            highlights_path=Path('/output/highlights.mp4')
        )

        # Verify
        assert result.success is True
        assert result.events_found == 2
        assert result.events_matched == 1  # Only one has video_time
        assert result.clips_created == 1
        assert result.highlights_path == '/output/highlights.mp4'
        assert len(result.warnings) == 1
        assert result.total_duration_seconds > 0

    def test_execute_integration_mock(self):
        """Integration test: Execute full pipeline with mocks"""
        mock_config = Mock()
        mock_config.MAX_HIGHLIGHT_CLIPS = 10
        video_path = Path('/fake/2025-01-15 Team1 vs Team2 Home 7.00pm.mp4')

        # Mock FileManager
        mock_file_manager = Mock()
        mock_file_manager.parse_mhl_filename.return_value = {
            'date': '2025-01-15',
            'home_team': 'Team1',
            'away_team': 'Team2',
            'league': 'MHL',
            'filename': video_path.name,
            'home_away': 'home',
            'time': '7.00pm'
        }
        mock_file_manager.create_game_folder.return_value = {
            'game_dir': Path('/output/game'),
            'clips_dir': Path('/output/game/clips'),
            'output_dir': Path('/output/game/output'),
            'data_dir': Path('/output/game/data'),
            'logs_dir': Path('/output/game/logs'),
            'source_dir': Path('/output/game/source')
        }
        mock_file_manager.save_game_metadata = Mock()
        mock_file_manager.save_events = Mock()

        # Mock BoxScoreFetcher
        mock_fetcher = Mock()
        mock_fetcher.find_game.return_value = 'game123'
        mock_fetcher.fetch_box_score.return_value = {'game_data': 'test'}
        mock_fetcher.extract_events.return_value = [
            {'type': 'goal', 'period': 1, 'time': '10:00', 'team': 'Team1', 'scorer': 'Player1'}
        ]

        # Mock VideoProcessor
        mock_video = Mock()
        mock_video.load_video.return_value = True
        mock_video.duration = 3600.0
        mock_video.fps = 30.0
        mock_video.create_highlight_clips.return_value = [
            ({'type': 'goal'}, Path('/clip1.mp4'))
        ]
        mock_video.create_highlights_reel.return_value = True
        mock_video.cleanup = Mock()

        # Mock OCREngine
        mock_ocr = Mock()
        mock_ocr.sample_video_times.return_value = [
            {'video_time': 500.0, 'period': 1, 'game_time': '10:00', 'game_time_seconds': 600}
        ]

        # Mock EventMatcher
        mock_matcher = Mock()
        mock_matcher.estimate_missing_timestamps.return_value = [
            {'video_time': 500.0, 'period': 1, 'game_time': '10:00', 'game_time_seconds': 600}
        ]
        mock_matcher.match_events_to_video.return_value = [
            {
                'type': 'goal',
                'period': 1,
                'time': '10:00',
                'team': 'Team1',
                'scorer': 'Player1',
                'video_time': 500.0,
                'match_confidence': 0.95
            }
        ]
        mock_matcher.sort_events_by_video_time.return_value = [
            {
                'type': 'goal',
                'period': 1,
                'time': '10:00',
                'team': 'Team1',
                'scorer': 'Player1',
                'video_time': 500.0
            }
        ]
        mock_matcher.filter_events_by_type.return_value = [
            {
                'type': 'goal',
                'period': 1,
                'time': '10:00',
                'team': 'Team1',
                'scorer': 'Player1',
                'video_time': 500.0
            }
        ]

        # Create pipeline
        pipeline = HighlightPipeline(
            mock_config,
            video_path,
            file_manager=mock_file_manager,
            box_score_fetcher=mock_fetcher,
            video_processor=mock_video,
            ocr_engine=mock_ocr,
            event_matcher=mock_matcher
        )

        # Execute
        result = pipeline.execute()

        # Verify
        assert result.success is True
        assert result.events_found == 1
        assert result.events_matched == 1
        assert result.clips_created == 1
        assert result.match_rate() == 100.0

        # Verify cleanup was called
        mock_video.cleanup.assert_called()
