#!/usr/bin/env python3
"""
Hockey Highlight Extractor v2.0
Box-score-based highlight detection with OCR time matching

Main entry point for processing hockey game videos
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Import configuration
import config

# Import our modules
from hockey_extractor import (
    VideoProcessor,
    BoxScoreFetcher,
    OCREngine,
    EventMatcher,
    FileManager
)


def setup_logging(log_file: Path = None) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        log_file: Optional log file path

    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger('HockeyExtractor')
    logger.setLevel(logging.DEBUG)

    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (DEBUG level)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def process_video(video_path: Path, logger: logging.Logger) -> bool:
    """
    Process a single video file

    Args:
        video_path: Path to video file
        logger: Logger instance

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("=" * 70)
        logger.info("HOCKEY HIGHLIGHT EXTRACTOR v2.0")
        logger.info("Box-Score-Based Detection")
        logger.info("=" * 70)
        logger.info(f"Processing: {video_path.name}")

        # Initialize file manager
        file_manager = FileManager(config)

        # Parse filename
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: PARSING GAME INFORMATION")
        logger.info("=" * 70)

        game_info = file_manager.parse_mhl_filename(video_path.name)
        if not game_info:
            logger.warning("MHL format not detected, using generic parser")
            game_info = file_manager.parse_generic_hockey_filename(video_path.name)

        logger.info(f"📅 Date: {game_info['date']}")
        logger.info(f"🏒 League: {game_info['league']}")
        logger.info(f"🏠 Home: {game_info['home_team']}")
        logger.info(f"✈️  Away: {game_info['away_team']}")
        logger.info(f"🎯 Perspective: {game_info['home_away'].title()}")

        # Create folder structure
        game_folders = file_manager.create_game_folder(game_info)
        logger.info(f"\n📁 Output folder: {game_folders['game_dir']}")

        # Setup dedicated log file
        log_file = game_folders['logs_dir'] / 'processing.log'
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # STEP 2: Fetch box score
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: FETCHING BOX SCORE")
        logger.info("=" * 70)

        box_score_fetcher = BoxScoreFetcher(cache_dir=game_folders['data_dir'])

        # Find game ID
        game_id = box_score_fetcher.find_game(
            game_info['league'],
            game_info['home_team'],
            game_info['away_team'],
            game_info['date']
        )

        if not game_id:
            logger.error("❌ Could not find game in league database")
            logger.error("   This could mean:")
            logger.error("   1. Game hasn't been played yet")
            logger.error("   2. Team names in filename don't match league records")
            logger.error("   3. League API is unavailable")
            logger.info("\n💡 You can still process the video without box scores")
            logger.info("   by using OCR-only mode (feature coming soon)")
            return False

        # Fetch box score
        box_score = box_score_fetcher.fetch_box_score(game_info['league'], game_id)

        if not box_score:
            logger.error("❌ Failed to fetch box score")
            return False

        # Extract events
        events = box_score_fetcher.extract_events(box_score)

        if not events:
            logger.warning("⚠️  No events found in box score")
            logger.info("   This might be a scoreless game or data issue")

        logger.info(f"✅ Found {len(events)} events")
        for event in events[:5]:  # Show first 5
            logger.info(f"   - P{event['period']} {event['time']}: {event['type'].upper()}")
        if len(events) > 5:
            logger.info(f"   ... and {len(events) - 5} more")

        # Save game metadata
        file_manager.save_game_metadata(game_folders, game_info, box_score)

        # STEP 3: Load video
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: LOADING VIDEO")
        logger.info("=" * 70)

        video_processor = VideoProcessor(video_path, config)

        if not video_processor.load_video():
            logger.error("❌ Failed to load video")
            return False

        logger.info(f"✅ Video loaded: {video_processor.duration:.1f}s @ {video_processor.fps:.1f} FPS")

        # STEP 4: Extract time from video using OCR
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: EXTRACTING TIME FROM VIDEO (OCR)")
        logger.info("=" * 70)

        ocr_engine = OCREngine(config)

        # Sample video at intervals to build timestamp map
        logger.info("Sampling video frames for time extraction...")
        logger.info("This may take a few minutes depending on video length...")

        video_timestamps = ocr_engine.sample_video_times(
            video_processor,
            sample_interval=30,  # Sample every 30 seconds
            max_samples=None     # Sample entire video
        )

        if not video_timestamps:
            logger.warning("⚠️  No timestamps extracted from video")
            logger.warning("   OCR may have failed to detect scoreboard")
            logger.info("\n💡 Tips:")
            logger.info("   1. Ensure scoreboard is visible in video")
            logger.info("   2. Check that tesseract-ocr is installed on your system")
            logger.info("   3. Try adjusting ROI settings if scoreboard is in unusual position")
            return False

        logger.info(f"✅ Extracted {len(video_timestamps)} timestamps from video")

        # Save debug info
        debug_file = game_folders['data_dir'] / 'video_timestamps.json'
        import json
        with open(debug_file, 'w') as f:
            json.dump(video_timestamps, f, indent=2)

        # STEP 5: Match events to video
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: MATCHING EVENTS TO VIDEO")
        logger.info("=" * 70)

        event_matcher = EventMatcher(config)

        # Enhance timestamps with interpolation
        video_timestamps = event_matcher.estimate_missing_timestamps(
            video_timestamps,
            video_processor.duration
        )

        # Match events
        matched_events = event_matcher.match_events_to_video(
            events,
            video_timestamps,
            tolerance_seconds=30
        )

        # Filter to only events with successful matches
        valid_events = [e for e in matched_events if e.get('video_time') is not None]

        if not valid_events:
            logger.error("❌ No events could be matched to video timestamps")
            logger.error("   This could mean:")
            logger.error("   1. OCR failed to extract accurate times")
            logger.error("   2. Video doesn't cover the entire game")
            logger.error("   3. Scoreboard format is incompatible")
            return False

        logger.info(f"✅ Matched {len(valid_events)}/{len(events)} events to video")

        # Sort by video time
        valid_events = event_matcher.sort_events_by_video_time(valid_events)

        # Save matched events
        file_manager.save_events(game_folders, valid_events)

        # STEP 6: Create highlight clips
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: CREATING HIGHLIGHT CLIPS")
        logger.info("=" * 70)

        # Filter to goals only (can include penalties later)
        goal_events = event_matcher.filter_events_by_type(valid_events, ['goal'])

        if not goal_events:
            logger.warning("⚠️  No goals found in matched events")
            logger.info("   Creating clips for all events instead...")
            goal_events = valid_events

        logger.info(f"Creating {len(goal_events)} highlight clips...")

        created_clips = video_processor.create_highlight_clips(
            goal_events,
            game_folders['clips_dir'],
            before_seconds=8,
            after_seconds=6
        )

        if not created_clips:
            logger.error("❌ No clips could be created")
            return False

        logger.info(f"✅ Created {len(created_clips)} clips")

        # STEP 7: Create highlights reel
        logger.info("\n" + "=" * 70)
        logger.info("STEP 7: CREATING HIGHLIGHTS REEL")
        logger.info("=" * 70)

        clip_paths = [clip_path for _, clip_path in created_clips]
        highlights_path = game_folders['output_dir'] / 'highlights.mp4'

        success = video_processor.create_highlights_reel(
            clip_paths,
            highlights_path,
            max_clips=config.MAX_HIGHLIGHT_CLIPS if hasattr(config, 'MAX_HIGHLIGHT_CLIPS') else None
        )

        # Cleanup
        video_processor.cleanup()

        # Generate summary report
        logger.info("\n" + "=" * 70)
        logger.info("PROCESSING COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"\n📊 Summary:")
        logger.info(f"   Events in box score: {len(events)}")
        logger.info(f"   Events matched to video: {len(valid_events)}")
        logger.info(f"   Highlight clips created: {len(created_clips)}")
        logger.info(f"\n📁 Output:")
        logger.info(f"   Highlights reel: {highlights_path}")
        logger.info(f"   Individual clips: {game_folders['clips_dir']}")
        logger.info(f"   Game data: {game_folders['data_dir']}")
        logger.info(f"   Logs: {log_file}")

        if success:
            logger.info("\n✅ SUCCESS!")
        else:
            logger.warning("\n⚠️  Completed with warnings")

        return success

    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Processing interrupted by user")
        return False

    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    """Main entry point"""
    print("🏒 Hockey Highlight Extractor v2.0")
    print()

    # Setup basic logging
    logger = setup_logging()

    # Find video files
    file_manager = FileManager(config)
    video_files = file_manager.find_video_files()

    if not video_files:
        logger.error("No video files found in configured locations")
        logger.info("\nSearched:")
        logger.info(f"  - {config.LOCAL_REPO_DIR}")
        logger.info(f"  - {Path.home() / 'Downloads'}")
        logger.info(f"  - {Path.home() / 'Desktop'}")

        if hasattr(config, 'GOOGLE_INPUT_DIR') and config.GOOGLE_INPUT_DIR:
            logger.info(f"  - {config.GOOGLE_INPUT_DIR}")

        logger.info(f"\nSupported formats: {', '.join(config.SUPPORTED_FORMATS)}")
        input("\nPress Enter to exit...")
        return

    # Display found videos
    logger.info(f"Found {len(video_files)} video(s):\n")
    for i, video_file in enumerate(video_files, 1):
        size_mb = video_file.stat().st_size / (1024 * 1024)
        logger.info(f"{i}. {video_file.name} ({size_mb:.1f} MB)")

    # Select video
    if len(video_files) == 1:
        selected_video = video_files[0]
        logger.info(f"\nProcessing: {selected_video.name}")
    else:
        try:
            choice = input(f"\nEnter video number to process (1-{len(video_files)}): ")
            index = int(choice) - 1

            if 0 <= index < len(video_files):
                selected_video = video_files[index]
            else:
                logger.error("Invalid selection")
                input("\nPress Enter to exit...")
                return

        except ValueError:
            logger.error("Invalid input")
            input("\nPress Enter to exit...")
            return

    # Process the video
    success = process_video(selected_video, logger)

    # Wait for user before exiting
    if success:
        input("\n\n✅ Press Enter to exit...")
    else:
        input("\n\n❌ Press Enter to exit...")


if __name__ == "__main__":
    main()
