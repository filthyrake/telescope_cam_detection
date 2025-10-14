"""
Camera Health Monitor
Monitors camera health and triggers automatic restarts on failure.
"""

import time
import logging
from typing import Dict, List, Optional, Callable
from threading import Thread, Event, Lock
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CameraHealthMetrics:
    """Health metrics for a single camera."""
    camera_id: str
    camera_name: str
    is_connected: bool = False
    fps: float = 0.0
    last_frame_time: float = 0.0
    consecutive_errors: int = 0
    uptime_seconds: float = 0.0
    restart_count: int = 0
    last_restart_time: Optional[float] = None
    restart_attempts_in_cycle: int = 0
    last_health_check_time: float = field(default_factory=time.time)
    start_time: float = field(default_factory=time.time)

    @property
    def last_frame_age_seconds(self) -> float:
        """Time since last frame was received."""
        if self.last_frame_time == 0:
            return float('inf')
        return time.time() - self.last_frame_time

    @property
    def health_score(self) -> int:
        """
        Calculate composite health score (0-100).

        Returns:
            Health score from 0 (failed) to 100 (healthy)
        """
        if not self.is_connected:
            return 0

        score = 100.0

        # Penalize low FPS (expect at least 10 FPS)
        if self.fps < 10:
            score -= (10 - self.fps) * 5  # -5 points per FPS below 10

        # Penalize stale frames
        frame_age = self.last_frame_age_seconds
        if frame_age > 5:
            score -= min(50, frame_age * 2)  # Up to -50 points for very stale frames

        # Penalize consecutive errors
        if self.consecutive_errors > 0:
            score -= min(30, self.consecutive_errors * 5)  # Up to -30 points

        return max(0, int(score))

    @property
    def status(self) -> str:
        """
        Get camera status string.

        Returns:
            One of: "healthy", "degraded", "failed", "restarting"
        """
        if self.restart_attempts_in_cycle > 0:
            return "restarting"

        if not self.is_connected:
            return "failed"

        score = self.health_score
        if score >= 80:
            return "healthy"
        elif score >= 40:
            return "degraded"
        else:
            return "failed"


class CameraHealthMonitor:
    """
    Monitors camera health and triggers automatic restarts on failure.

    Features:
    - Monitors FPS, connection status, frame age, error rate
    - Automatic restart with exponential backoff
    - Configurable health check parameters
    - Thread-safe health status access
    """

    def __init__(
        self,
        frame_sources: List,
        restart_callback: Callable[[int], bool],
        config: Optional[Dict] = None
    ):
        """
        Initialize camera health monitor.

        Args:
            frame_sources: List of RTSPStreamCapture instances to monitor
            restart_callback: Function to call to restart a camera (takes camera index, returns success)
            config: Health monitoring configuration dict
        """
        self.frame_sources = frame_sources
        self.restart_callback = restart_callback

        # Configuration with defaults
        default_config = {
            'enabled': True,
            'check_interval_seconds': 10,
            'min_fps': 5,
            'max_frame_age_seconds': 30,
            'max_consecutive_errors': 5,
            'auto_restart': True,
            'max_restart_attempts': 10,
            'restart_cooldown_seconds': 300,
            'backoff_multiplier': 2,
            'initial_backoff_seconds': 5,
            'alert_on_failure': True,
            'alert_on_recovery': True
        }

        self.config = {**default_config, **(config or {})}

        # Health metrics for each camera
        self.health_metrics: Dict[str, CameraHealthMetrics] = {}
        self._metrics_lock = Lock()

        # Monitoring thread
        self.monitor_thread: Optional[Thread] = None
        self.stop_event = Event()

        # Initialize metrics for all cameras
        for i, frame_source in enumerate(frame_sources):
            camera_id = frame_source.camera_id
            camera_name = frame_source.camera_name

            metrics = CameraHealthMetrics(
                camera_id=camera_id,
                camera_name=camera_name
            )

            with self._metrics_lock:
                self.health_metrics[camera_id] = metrics

        logger.info(f"Camera health monitor initialized for {len(frame_sources)} camera(s)")
        if self.config['auto_restart']:
            logger.info(f"  Auto-restart enabled (max {self.config['max_restart_attempts']} attempts)")
        else:
            logger.info("  Auto-restart disabled (monitoring only)")

    def start(self):
        """Start health monitoring thread."""
        if not self.config['enabled']:
            logger.info("Camera health monitoring disabled")
            return

        self.stop_event.clear()
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True, name="HealthMonitor")
        self.monitor_thread.start()
        logger.info("Camera health monitor started")

    def stop(self):
        """Stop health monitoring thread."""
        logger.info("Stopping camera health monitor...")
        self.stop_event.set()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        logger.info("Camera health monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop running in separate thread."""
        logger.info("Health monitoring loop started")
        check_interval = self.config['check_interval_seconds']

        while not self.stop_event.is_set():
            try:
                # Check health of all cameras
                for i, frame_source in enumerate(self.frame_sources):
                    if frame_source is None:
                        continue

                    camera_id = frame_source.camera_id

                    # Update health metrics
                    self._update_health_metrics(frame_source)

                    # Check if camera needs restart
                    with self._metrics_lock:
                        metrics = self.health_metrics.get(camera_id)
                        if metrics is None:
                            continue

                    # Check health status
                    if self._should_restart_camera(metrics):
                        logger.warning(f"[{camera_id}] Health check failed, attempting restart")
                        self._attempt_restart(i, metrics)

                # Sleep until next check
                self.stop_event.wait(timeout=check_interval)

            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}", exc_info=True)
                self.stop_event.wait(timeout=1.0)

    def _update_health_metrics(self, frame_source):
        """
        Update health metrics for a camera.

        Args:
            frame_source: RTSPStreamCapture instance
        """
        camera_id = frame_source.camera_id

        try:
            # Get stats from frame source
            stats = frame_source.get_stats()

            with self._metrics_lock:
                metrics = self.health_metrics.get(camera_id)
                if metrics is None:
                    return

                # Update metrics
                metrics.is_connected = stats.get('is_connected', False)
                metrics.fps = stats.get('fps', 0.0)

                # Update last frame time (use current time if connected and FPS > 0)
                if metrics.is_connected and metrics.fps > 0:
                    metrics.last_frame_time = time.time()

                # Calculate uptime
                metrics.uptime_seconds = time.time() - metrics.start_time
                metrics.last_health_check_time = time.time()

                # Log health status periodically (every 5 checks = ~50s with 10s interval)
                if int(metrics.uptime_seconds) % 50 == 0:
                    logger.debug(
                        f"[{camera_id}] Health: {metrics.status} "
                        f"(score={metrics.health_score}, fps={metrics.fps:.1f}, "
                        f"frame_age={metrics.last_frame_age_seconds:.1f}s)"
                    )

        except Exception as e:
            logger.error(f"[{camera_id}] Error updating health metrics: {e}")

    def _should_restart_camera(self, metrics: CameraHealthMetrics) -> bool:
        """
        Determine if camera should be restarted based on health metrics.

        Args:
            metrics: Camera health metrics

        Returns:
            True if camera should be restarted, False otherwise
        """
        if not self.config['auto_restart']:
            return False

        # Don't restart if already attempting restart
        if metrics.restart_attempts_in_cycle > 0:
            return False

        # Don't restart if we've exceeded max restart attempts
        if metrics.restart_count >= self.config['max_restart_attempts']:
            # Check if cooldown period has passed
            if metrics.last_restart_time is None:
                return False

            cooldown = self.config['restart_cooldown_seconds']
            time_since_last_restart = time.time() - metrics.last_restart_time

            if time_since_last_restart < cooldown:
                return False

            # Reset restart count after cooldown
            metrics.restart_count = 0
            logger.info(f"[{metrics.camera_id}] Restart cooldown expired, resetting restart counter")

        # Check health indicators
        failures = []

        # Check 1: Connection status
        if not metrics.is_connected:
            failures.append("disconnected")

        # Check 2: FPS too low
        if metrics.fps < self.config['min_fps']:
            failures.append(f"low FPS ({metrics.fps:.1f})")

        # Check 3: Stale frames
        frame_age = metrics.last_frame_age_seconds
        if frame_age > self.config['max_frame_age_seconds']:
            failures.append(f"stale frames ({frame_age:.0f}s)")

        # Check 4: Too many consecutive errors
        if metrics.consecutive_errors > self.config['max_consecutive_errors']:
            failures.append(f"errors ({metrics.consecutive_errors})")

        # Restart if any failures detected
        if failures:
            logger.warning(
                f"[{metrics.camera_id}] Health check failures: {', '.join(failures)}"
            )
            return True

        return False

    def _attempt_restart(self, camera_index: int, metrics: CameraHealthMetrics):
        """
        Attempt to restart a camera with exponential backoff.

        Args:
            camera_index: Index of camera in frame_sources list
            metrics: Camera health metrics
        """
        camera_id = metrics.camera_id

        # Calculate backoff delay
        backoff_delay = self._calculate_backoff_delay(metrics.restart_count)

        logger.info(
            f"[{camera_id}] Restarting camera "
            f"(attempt {metrics.restart_count + 1}/{self.config['max_restart_attempts']}, "
            f"backoff={backoff_delay:.0f}s)"
        )

        # Mark as restarting
        with self._metrics_lock:
            metrics.restart_attempts_in_cycle += 1

        # Wait for backoff delay
        if backoff_delay > 0:
            logger.info(f"[{camera_id}] Waiting {backoff_delay:.0f}s before restart...")
            self.stop_event.wait(timeout=backoff_delay)

        # Check if monitor was stopped during wait
        if self.stop_event.is_set():
            return

        # Attempt restart
        try:
            success = self.restart_callback(camera_index)

            with self._metrics_lock:
                if success:
                    logger.info(f"[{camera_id}] ✓ Camera restart successful")

                    # Reset metrics after successful restart
                    metrics.restart_count += 1
                    metrics.last_restart_time = time.time()
                    metrics.restart_attempts_in_cycle = 0
                    metrics.consecutive_errors = 0
                    metrics.start_time = time.time()  # Reset uptime

                    # Trigger recovery alert if configured
                    if self.config['alert_on_recovery']:
                        # TODO: Implement alert system (issue #73)
                        pass
                else:
                    logger.error(f"[{camera_id}] ✗ Camera restart failed")

                    metrics.restart_count += 1
                    metrics.last_restart_time = time.time()
                    metrics.restart_attempts_in_cycle = 0
                    metrics.consecutive_errors += 1

                    # Trigger failure alert if configured
                    if self.config['alert_on_failure']:
                        # TODO: Implement alert system (issue #73)
                        pass

                    # Check if we've exhausted restart attempts
                    if metrics.restart_count >= self.config['max_restart_attempts']:
                        logger.error(
                            f"[{camera_id}] ✗ Max restart attempts reached "
                            f"({self.config['max_restart_attempts']}), giving up"
                        )

        except Exception as e:
            logger.error(f"[{camera_id}] Error during camera restart: {e}", exc_info=True)
            with self._metrics_lock:
                metrics.restart_attempts_in_cycle = 0
                metrics.consecutive_errors += 1

    def _calculate_backoff_delay(self, restart_count: int) -> float:
        """
        Calculate exponential backoff delay for restart attempts.

        Args:
            restart_count: Number of restart attempts so far

        Returns:
            Backoff delay in seconds
        """
        initial = self.config['initial_backoff_seconds']
        multiplier = self.config['backoff_multiplier']

        # Exponential backoff: 5s, 10s, 20s, 40s, 80s, 160s, 300s (capped)
        delay = initial * (multiplier ** restart_count)
        max_delay = self.config['restart_cooldown_seconds']

        return min(delay, max_delay)

    def get_camera_health(self, camera_id: str) -> Optional[Dict]:
        """
        Get health status for a specific camera.

        Args:
            camera_id: Camera ID

        Returns:
            Dict with health status, or None if camera not found
        """
        with self._metrics_lock:
            metrics = self.health_metrics.get(camera_id)
            if metrics is None:
                return None

            return {
                'camera_id': metrics.camera_id,
                'camera_name': metrics.camera_name,
                'status': metrics.status,
                'is_connected': metrics.is_connected,
                'fps': metrics.fps,
                'last_frame_age_seconds': metrics.last_frame_age_seconds,
                'consecutive_errors': metrics.consecutive_errors,
                'uptime_seconds': metrics.uptime_seconds,
                'restart_count': metrics.restart_count,
                'last_restart': (
                    datetime.fromtimestamp(metrics.last_restart_time).isoformat()
                    if metrics.last_restart_time else None
                ),
                'health_score': metrics.health_score
            }

    def get_all_cameras_health(self) -> List[Dict]:
        """
        Get health status for all cameras.

        Returns:
            List of health status dicts
        """
        with self._metrics_lock:
            return [
                self.get_camera_health(camera_id)
                for camera_id in self.health_metrics.keys()
            ]

    def get_health_summary(self) -> Dict:
        """
        Get summary of health status for all cameras.

        Returns:
            Dict with summary statistics
        """
        with self._metrics_lock:
            cameras = list(self.health_metrics.values())

            total = len(cameras)
            healthy = sum(1 for m in cameras if m.status == "healthy")
            degraded = sum(1 for m in cameras if m.status == "degraded")
            failed = sum(1 for m in cameras if m.status == "failed")
            restarting = sum(1 for m in cameras if m.status == "restarting")

            return {
                'total_cameras': total,
                'healthy': healthy,
                'degraded': degraded,
                'failed': failed,
                'restarting': restarting,
                'cameras': [
                    {
                        'id': m.camera_id,
                        'name': m.camera_name,
                        'status': m.status,
                        'health_score': m.health_score
                    }
                    for m in cameras
                ]
            }

    def increment_error_count(self, camera_id: str):
        """
        Increment consecutive error count for a camera.

        Args:
            camera_id: Camera ID
        """
        with self._metrics_lock:
            metrics = self.health_metrics.get(camera_id)
            if metrics:
                metrics.consecutive_errors += 1

    def reset_error_count(self, camera_id: str):
        """
        Reset consecutive error count for a camera.

        Args:
            camera_id: Camera ID
        """
        with self._metrics_lock:
            metrics = self.health_metrics.get(camera_id)
            if metrics:
                metrics.consecutive_errors = 0
