#!/usr/bin/env python3
"""
Hockey Highlight Extractor v2.0
Box-score-based highlight detection with OCR time matching

Main entry point for processing hockey game videos
Uses the HighlightPipeline orchestrator for clean, testable execution
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Import configuration
import config

# Import pipeline orchestrator
from hockey_extractor import HighlightPipeline, FileManager


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

    # Clear existing handlers
    logger.handlers.clear()

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
    Process a single video file using the pipeline orchestrator

    Args:
        video_path: Path to video file
        logger: Logger instance

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create pipeline
        pipeline = HighlightPipeline(config, video_path)

        # Execute pipeline with default settings
        result = pipeline.execute(
            sample_interval=30,      # Sample every 30 seconds
            tolerance_seconds=30,    # 30 second tolerance for matching
            before_seconds=8.0,      # 8 seconds before event
            after_seconds=6.0,       # 6 seconds after event
            max_clips=config.MAX_HIGHLIGHT_CLIPS if hasattr(config, 'MAX_HIGHLIGHT_CLIPS') else None,
            parallel_ocr=True,       # Use parallel processing
            ocr_workers=4            # 4 worker threads
        )

        # Log result summary
        if result.success:
            logger.info("\n✅ SUCCESS!")
            logger.info(f"\n📊 Final Statistics:")
            logger.info(f"   Match rate: {result.match_rate():.1f}%")
            logger.info(f"   Total duration: {result.total_duration_seconds:.1f}s")

            if result.warnings:
                logger.info(f"\n⚠️  Warnings ({len(result.warnings)}):")
                for warning in result.warnings:
                    logger.info(f"   - {warning}")

            return True
        else:
            logger.error("\n❌ PROCESSING FAILED")
            if result.errors:
                logger.error(f"\n❌ Errors ({len(result.errors)}):")
                for error in result.errors:
                    logger.error(f"   - {error}")
            return False

    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Processing interrupted by user")
        return False

    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
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
