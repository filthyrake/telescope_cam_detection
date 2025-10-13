#!/usr/bin/env python3
"""
Stream Health Watchdog - Monitors camera streams and auto-restarts on failure

Detects frozen streams by monitoring frame activity and automatically restarts
the telescope detection service (and Neolink container if needed).

Usage:
    python scripts/stream_watchdog.py [--check-interval 30] [--freeze-threshold 120]
"""

import argparse
import logging
import subprocess
import sys
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stream_watchdog')


class StreamWatchdog:
    """Monitors camera stream health and triggers recovery actions"""

    def __init__(
        self,
        check_interval: int = 30,
        freeze_threshold: int = 120,
        max_restarts_per_hour: int = 5,
        config_path: str = "config/config.yaml"
    ):
        """
        Args:
            check_interval: Seconds between health checks
            freeze_threshold: Seconds of inactivity before declaring stream frozen
            max_restarts_per_hour: Maximum service restarts per hour (safety limit)
            config_path: Path to config.yaml file
        """
        self.check_interval = check_interval
        self.freeze_threshold = freeze_threshold
        self.max_restarts_per_hour = max_restarts_per_hour
        self.config_path = config_path

        self.last_activity: Dict[str, datetime] = {}
        self.restart_history: list = []
        self.recovery_attempts: Dict[str, int] = {}

        logger.info(f"Stream watchdog initialized:")
        logger.info(f"  Check interval: {check_interval}s")
        logger.info(f"  Freeze threshold: {freeze_threshold}s")
        logger.info(f"  Max restarts/hour: {max_restarts_per_hour}")

    def get_last_activity_from_logs(self, camera_id: str) -> Optional[datetime]:
        """Check recent logs for camera activity"""
        try:
            # Look for recent frame captures or detections for this camera
            cmd = [
                'journalctl',
                '-u', 'telescope_detection.service',
                '--since', '5 minutes ago',
                '--no-pager',
                '-n', '500'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                logger.warning(f"Failed to read journal: {result.stderr}")
                return None

            # Parse logs for camera activity (frame captures, detections, etc.)
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):  # Start from most recent
                if camera_id in line:
                    # Look for indicators of active streaming
                    if any(marker in line for marker in [
                        'Capture loop',
                        'Successfully connected',
                        'Frame size:',
                        'Saved snapshot',
                        f'[{camera_id}]'
                    ]):
                        # Extract timestamp from journalctl line
                        # Format: "Oct 06 18:55:02 aihost telescope-detection[107341]: ..."
                        try:
                            parts = line.split()
                            if len(parts) >= 3:
                                # Parse timestamp: combine current year with log's month, day, and time
                                log_time_str = f"{datetime.now().year} {parts[0]} {parts[1]} {parts[2]}"
                                # Example: "2024 Oct 06 18:55:02"
                                log_time = datetime.strptime(log_time_str, "%Y %b %d %H:%M:%S")
                                return log_time
                        except Exception as e:
                            logger.debug(f"Failed to parse timestamp: {e}")

            return None

        except subprocess.TimeoutExpired:
            logger.warning("Journal query timed out")
            return None
        except Exception as e:
            logger.error(f"Error checking logs: {e}")
            return None

    def check_camera_health(self, camera_id: str) -> bool:
        """Check if camera stream is healthy"""
        last_activity = self.get_last_activity_from_logs(camera_id)

        if last_activity:
            self.last_activity[camera_id] = last_activity
            age = (datetime.now() - last_activity).total_seconds()
            logger.debug(f"[{camera_id}] Last activity: {age:.0f}s ago")
            return age < self.freeze_threshold

        # No activity found - check if we have historical data
        if camera_id in self.last_activity:
            age = (datetime.now() - self.last_activity[camera_id]).total_seconds()
            logger.warning(f"[{camera_id}] No recent activity in logs, last known: {age:.0f}s ago")
            return age < self.freeze_threshold

        # First check - assume healthy to avoid false positive on startup
        logger.info(f"[{camera_id}] First health check - assuming healthy")
        self.last_activity[camera_id] = datetime.now()
        return True

    def can_restart(self) -> bool:
        """Check if we're within restart rate limit"""
        # Clean old restart records (older than 1 hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.restart_history = [ts for ts in self.restart_history if ts > cutoff]

        if len(self.restart_history) >= self.max_restarts_per_hour:
            logger.error(f"⚠️ Restart rate limit exceeded ({self.max_restarts_per_hour}/hour)")
            logger.error("Too many restarts - possible recurring issue. Manual intervention required.")
            return False

        return True

    def restart_telescope_service(self) -> bool:
        """Restart the telescope detection service"""
        logger.warning("🔄 Restarting telescope detection service...")
        try:
            result = subprocess.run(
                ['sudo', 'systemctl', 'restart', 'telescope_detection.service'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.info("✅ Service restarted successfully")
                self.restart_history.append(datetime.now())
                time.sleep(10)  # Give service time to start
                return True
            else:
                logger.error(f"❌ Service restart failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Service restart failed: {e}")
            return False

    def restart_neolink_container(self) -> bool:
        """Restart the Neolink Docker container"""
        logger.warning("🔄 Restarting Neolink container...")
        try:
            result = subprocess.run(
                ['sudo', 'docker', 'restart', 'neolink'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.info("✅ Neolink container restarted")
                time.sleep(15)  # Give container time to reconnect to camera
                return True
            else:
                logger.error(f"❌ Container restart failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Container restart failed: {e}")
            return False

    def recover_frozen_stream(self, camera_id: str) -> bool:
        """Execute recovery procedure for frozen stream"""
        if not self.can_restart():
            return False

        attempts = self.recovery_attempts.get(camera_id, 0)

        if attempts == 0:
            # First attempt - restart telescope service
            logger.warning(f"🚨 [{camera_id}] Stream frozen - attempting service restart")
            if self.restart_telescope_service():
                self.recovery_attempts[camera_id] = 1
                return True

        elif attempts == 1:
            # Second attempt - restart Neolink container (if cam2)
            if camera_id == 'cam2':
                logger.warning(f"🚨 [{camera_id}] Still frozen - attempting Neolink restart")
                if self.restart_neolink_container():
                    # Restart telescope service again after Neolink restart
                    time.sleep(5)
                    self.restart_telescope_service()
                    self.recovery_attempts[camera_id] = 2
                    return True
            else:
                logger.error(f"❌ [{camera_id}] Service restart didn't help - manual check needed")
                self.recovery_attempts[camera_id] = 99  # Stop trying

        else:
            # Too many attempts - give up
            logger.error(f"❌ [{camera_id}] Recovery failed after multiple attempts")
            logger.error("Manual intervention required - check camera and network connectivity")
            self.recovery_attempts[camera_id] = 99

        return False

    def get_camera_ids_from_config(self) -> List[str]:
        """
        Parse camera IDs from config.yaml file.

        Returns:
            List of enabled camera IDs
        """
        try:
            config_file = Path(__file__).parent.parent / self.config_path
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_file}")
                return ['cam1', 'cam2']  # Fallback to defaults

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Extract enabled camera IDs
            cameras = config.get('cameras', [])
            camera_ids = [
                cam['id'] for cam in cameras
                if cam.get('enabled', True)  # Default to enabled if not specified
            ]

            if camera_ids:
                logger.info(f"Loaded {len(camera_ids)} camera(s) from config: {', '.join(camera_ids)}")
                return camera_ids
            else:
                logger.warning("No cameras found in config, using defaults")
                return ['cam1', 'cam2']

        except Exception as e:
            logger.error(f"Error reading config file: {e}")
            logger.warning("Falling back to default camera IDs: cam1, cam2")
            return ['cam1', 'cam2']

    def run(self):
        """Main watchdog loop"""
        logger.info("🐕 Stream watchdog started")

        # Give service time to start on initial launch
        time.sleep(30)

        # Get camera IDs from config
        cameras = self.get_camera_ids_from_config()

        while True:
            try:
                all_healthy = True

                for camera_id in cameras:
                    is_healthy = self.check_camera_health(camera_id)

                    if is_healthy:
                        # Reset recovery attempts on successful health check
                        if camera_id in self.recovery_attempts:
                            if self.recovery_attempts[camera_id] < 99:
                                logger.info(f"✅ [{camera_id}] Stream recovered")
                            del self.recovery_attempts[camera_id]
                    else:
                        all_healthy = False
                        age = (datetime.now() - self.last_activity.get(camera_id, datetime.now())).total_seconds()
                        logger.warning(f"⚠️ [{camera_id}] Stream appears frozen (no activity for {age:.0f}s)")

                        # Attempt recovery
                        self.recover_frozen_stream(camera_id)

                if all_healthy:
                    logger.debug("✅ All camera streams healthy")

                # Wait before next check
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("Watchdog stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in watchdog loop: {e}", exc_info=True)
                time.sleep(self.check_interval)


def main():
    parser = argparse.ArgumentParser(
        description='Monitor camera stream health and auto-restart on failure'
    )
    parser.add_argument(
        '--check-interval',
        type=int,
        default=30,
        help='Seconds between health checks (default: 30)'
    )
    parser.add_argument(
        '--freeze-threshold',
        type=int,
        default=120,
        help='Seconds of inactivity before declaring stream frozen (default: 120)'
    )
    parser.add_argument(
        '--max-restarts-per-hour',
        type=int,
        default=5,
        help='Maximum service restarts per hour (default: 5)'
    )

    args = parser.parse_args()

    watchdog = StreamWatchdog(
        check_interval=args.check_interval,
        freeze_threshold=args.freeze_threshold,
        max_restarts_per_hour=args.max_restarts_per_hour
    )

    try:
        watchdog.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
