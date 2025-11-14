#!/usr/bin/env python3
"""
Watch Folder - Automatic video processing when files are added to watched directory

This script monitors a specified directory for new video files and automatically
processes them when detected.
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Set

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("❌ watchdog not installed")
    print("   Install with: pip install watchdog")
    sys.exit(1)

# Import configuration and main processor
import config
from main import process_video, setup_logging


class VideoFileHandler(FileSystemEventHandler):
    """Handler for new video file events"""

    def __init__(self, logger: logging.Logger, supported_formats: list):
        """
        Initialize handler

        Args:
            logger: Logger instance
            supported_formats: List of supported video formats
        """
        self.logger = logger
        self.supported_formats = [fmt.lower() for fmt in supported_formats]
        self.processing_queue: Set[Path] = set()
        self.processed_files: Set[Path] = set()

    def on_created(self, event):
        """
        Called when a file is created

        Args:
            event: File system event
        """
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Check if this is a supported video format
        if file_path.suffix.lower() not in self.supported_formats:
            return

        self.logger.info(f"📹 New video detected: {file_path.name}")

        # Wait for file to finish writing
        if self._wait_for_file_stable(file_path):
            self._process_video_file(file_path)
        else:
            self.logger.warning(f"⚠️  File not stable, skipping: {file_path.name}")

    def on_moved(self, event):
        """
        Called when a file is moved

        Args:
            event: File system event
        """
        if event.is_directory:
            return

        dest_path = Path(event.dest_path)

        # Check if destination is a supported video format
        if dest_path.suffix.lower() not in self.supported_formats:
            return

        self.logger.info(f"📹 Video moved to watch folder: {dest_path.name}")

        # Wait for file to finish moving
        if self._wait_for_file_stable(dest_path):
            self._process_video_file(dest_path)
        else:
            self.logger.warning(f"⚠️  File not stable, skipping: {dest_path.name}")

    def _wait_for_file_stable(self, file_path: Path, timeout: int = 60, check_interval: int = 2) -> bool:
        """
        Wait for file to finish writing by checking if size is stable

        Args:
            file_path: Path to file
            timeout: Maximum seconds to wait
            check_interval: Seconds between checks

        Returns:
            True if file is stable, False if timeout
        """
        self.logger.debug(f"Waiting for file to stabilize: {file_path.name}")

        start_time = time.time()
        last_size = -1

        while time.time() - start_time < timeout:
            try:
                if not file_path.exists():
                    time.sleep(check_interval)
                    continue

                current_size = file_path.stat().st_size

                # Check if size hasn't changed
                if current_size == last_size and current_size > 0:
                    self.logger.debug(f"File stable at {current_size} bytes")
                    return True

                last_size = current_size
                time.sleep(check_interval)

            except Exception as e:
                self.logger.warning(f"Error checking file size: {e}")
                time.sleep(check_interval)

        self.logger.warning(f"Timeout waiting for file to stabilize")
        return False

    def _process_video_file(self, file_path: Path):
        """
        Process a video file

        Args:
            file_path: Path to video file
        """
        # Skip if already processed or in queue
        if file_path in self.processed_files:
            self.logger.info(f"Already processed: {file_path.name}")
            return

        if file_path in self.processing_queue:
            self.logger.info(f"Already in queue: {file_path.name}")
            return

        # Add to queue
        self.processing_queue.add(file_path)

        try:
            self.logger.info(f"\n{'=' * 70}")
            self.logger.info(f"🎬 PROCESSING: {file_path.name}")
            self.logger.info(f"{'=' * 70}\n")

            # Process the video
            success = process_video(file_path, self.logger)

            # Mark as processed
            self.processed_files.add(file_path)

            if success:
                self.logger.info(f"\n✅ Successfully processed: {file_path.name}\n")
            else:
                self.logger.error(f"\n❌ Failed to process: {file_path.name}\n")

        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {e}")

        finally:
            # Remove from queue
            self.processing_queue.discard(file_path)


def main():
    """Main entry point for watch folder"""
    print("=" * 70)
    print("🏒 Hockey Highlight Extractor - Watch Folder Mode")
    print("=" * 70)
    print()

    # Setup logging
    log_dir = config.LOCAL_REPO_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"watch_folder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file)

    # Determine watch directory
    watch_dir = None

    # Check for command line argument
    if len(sys.argv) > 1:
        watch_dir = Path(sys.argv[1])
        if not watch_dir.exists():
            logger.error(f"Directory does not exist: {watch_dir}")
            return

    # Otherwise use default locations
    if watch_dir is None:
        # Try Google Drive input directory first
        if hasattr(config, 'GOOGLE_INPUT_DIR') and config.GOOGLE_INPUT_DIR:
            if config.GOOGLE_INPUT_DIR.exists():
                watch_dir = config.GOOGLE_INPUT_DIR

        # Fallback to Downloads
        if watch_dir is None:
            watch_dir = Path.home() / "Downloads"

    logger.info(f"👀 Watching directory: {watch_dir}")
    logger.info(f"📋 Log file: {log_file}")
    logger.info(f"📹 Supported formats: {', '.join(config.SUPPORTED_FORMATS)}")
    logger.info("\nWaiting for new video files...")
    logger.info("Press Ctrl+C to stop\n")

    # Create event handler
    event_handler = VideoFileHandler(logger, config.SUPPORTED_FORMATS)

    # Create observer
    observer = Observer()
    observer.schedule(event_handler, str(watch_dir), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Stopping watch folder...")
        observer.stop()

    observer.join()
    logger.info("✅ Watch folder stopped")


if __name__ == "__main__":
    if not WATCHDOG_AVAILABLE:
        print("❌ watchdog package not installed")
        print("   Install with: pip install watchdog")
        sys.exit(1)

    main()
