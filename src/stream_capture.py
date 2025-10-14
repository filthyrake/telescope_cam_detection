"""
RTSP Stream Capture Module
Handles video stream capture from Reolink RLC-410W camera with minimal latency.
"""

import cv2
import os
import time
import logging
from typing import Optional, Tuple
import numpy as np
from queue import Queue, Full
from threading import Thread, Event, Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RTSP connection timeout in microseconds (5 seconds)
RTSP_TIMEOUT_MICROSECONDS = 5_000_000


class RTSPStreamCapture:
    """
    Captures RTSP stream with focus on low latency.
    Implements frame dropping to always process the latest frame.
    """

    def __init__(
        self,
        rtsp_url: str,
        frame_queue: Queue,
        target_width: int = 1280,
        target_height: int = 720,
        buffer_size: int = 1,
        camera_id: str = "default",
        camera_name: str = "Default Camera",
        use_tcp: bool = False
    ):
        """
        Initialize RTSP stream capture.

        Args:
            rtsp_url: RTSP URL of the camera
            frame_queue: Queue to put captured frames
            target_width: Resize width for inference
            target_height: Resize height for inference
            buffer_size: Number of frames to buffer (keep at 1 for lowest latency)
            camera_id: Unique identifier for this camera
            camera_name: Human-readable name for this camera
            use_tcp: Use TCP transport instead of UDP (reduces tearing)
        """
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.target_width = target_width
        self.target_height = target_height
        self.buffer_size = buffer_size
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.use_tcp = use_tcp

        self.capture: Optional[cv2.VideoCapture] = None
        self.stop_event = Event()
        self.capture_thread: Optional[Thread] = None
        self.is_connected = False

        # Latest frame for video streaming (thread-safe access)
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = Lock()  # Protects latest_frame

        # Performance metrics
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_fps_check = time.time()
        self.fps = 0.0

    def connect(self) -> bool:
        """
        Connect to the RTSP stream with short timeout.

        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f"[{self.camera_id}] Connecting to RTSP stream: {self.rtsp_url}")
        if self.use_tcp:
            logger.info(f"[{self.camera_id}] Using TCP transport (rtsp_transport=tcp)")

        # Save existing environment variable to restore later (avoid global pollution)
        old_opencv_options = os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS')

        try:
            # OpenCV VideoCapture with optimized settings for low latency
            # Use FFMPEG backend with TCP transport if requested
            if self.use_tcp:
                # Set RTSP transport to TCP via environment variable (OpenCV+FFMPEG)
                # Set shorter timeout (5 seconds) for non-blocking startup
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = f'rtsp_transport;tcp|timeout;{RTSP_TIMEOUT_MICROSECONDS}'
            else:
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = f'timeout;{RTSP_TIMEOUT_MICROSECONDS}'

            self.capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        finally:
            # Restore original environment variable to avoid affecting other cameras
            if old_opencv_options is not None:
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = old_opencv_options
            else:
                # Remove if it didn't exist before
                os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS', None)

        # Set low latency options
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        if not self.capture.isOpened():
            logger.error(f"[{self.camera_id}] Failed to open RTSP stream")
            return False

        # Test read
        ret, frame = self.capture.read()
        if not ret or frame is None:
            logger.error(f"[{self.camera_id}] Failed to read test frame from stream")
            return False

        self.is_connected = True
        logger.info(f"[{self.camera_id}] Successfully connected. Frame size: {frame.shape[1]}x{frame.shape[0]}")
        return True

    def start(self) -> bool:
        """
        Start the capture thread.
        Non-blocking: starts immediately and handles connection in background.

        Returns:
            Always returns True (camera will connect in background if not immediately available)
        """
        # Try initial connection (non-blocking with short timeout)
        if not self.is_connected:
            logger.info(f"[{self.camera_id}] Starting capture thread (will connect in background if needed)")
            # Don't block on connection - let background thread handle it

        self.stop_event.clear()
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info(f"[{self.camera_id}] Capture thread started")
        return True  # Always succeed - connection happens in background

    def stop(self):
        """Stop the capture thread and release resources."""
        logger.info(f"[{self.camera_id}] Stopping capture thread...")
        self.stop_event.set()

        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)

        if self.capture:
            self.capture.release()

        self.is_connected = False
        logger.info(f"[{self.camera_id}] Capture thread stopped")

    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        logger.info(f"[{self.camera_id}] Capture loop started")
        consecutive_failures = 0
        max_failures = 30  # Reconnect after 30 consecutive failures
        retry_delay = 5.0  # Seconds to wait before retrying connection

        while not self.stop_event.is_set():
            try:
                # If not connected, attempt to connect
                if not self.is_connected or self.capture is None:
                    logger.info(f"[{self.camera_id}] Attempting to connect...")
                    if self.connect():
                        logger.info(f"[{self.camera_id}] Connected successfully!")
                        consecutive_failures = 0
                    else:
                        logger.warning(f"[{self.camera_id}] Connection failed, will retry in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue

                ret, frame = self.capture.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame (attempt {consecutive_failures}/{max_failures})")

                    if consecutive_failures >= max_failures:
                        logger.error("Too many consecutive failures, attempting reconnect...")
                        if self._reconnect():
                            consecutive_failures = 0
                        else:
                            time.sleep(1.0)
                    continue

                consecutive_failures = 0

                # Resize frame for inference (reduces latency)
                if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                    frame = cv2.resize(frame, (self.target_width, self.target_height))

                # Store latest frame for video streaming (thread-safe)
                with self.frame_lock:
                    self.latest_frame = frame.copy()

                # Add timestamp to frame metadata
                timestamp = time.time()

                # Try to put frame in queue, drop if full (non-blocking)
                try:
                    self.frame_queue.put_nowait({
                        'frame': frame,
                        'timestamp': timestamp,
                        'frame_id': self.frame_count,
                        'camera_id': self.camera_id,
                        'camera_name': self.camera_name
                    })
                    self.frame_count += 1
                except Full:
                    self.dropped_frames += 1

                # Calculate FPS every second
                if time.time() - self.last_fps_check >= 1.0:
                    self.fps = self.frame_count / (time.time() - self.last_fps_check)
                    self.last_fps_check = time.time()
                    self.frame_count = 0

                    if self.dropped_frames > 0:
                        logger.debug(f"FPS: {self.fps:.1f}, Dropped: {self.dropped_frames}")
                        self.dropped_frames = 0

            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)

    def _reconnect(self) -> bool:
        """
        Attempt to reconnect to the stream.

        Returns:
            True if reconnection successful, False otherwise
        """
        logger.info(f"[{self.camera_id}] Attempting to reconnect...")

        # Always release existing capture, even if it failed
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception as e:
                logger.warning(f"[{self.camera_id}] Error releasing capture: {e}")
            finally:
                self.capture = None

        time.sleep(2.0)  # Wait before reconnecting
        return self.connect()

    def get_stats(self) -> dict:
        """
        Get capture statistics.

        Returns:
            Dictionary containing capture stats
        """
        stats = {
            'is_connected': self.is_connected,
            'fps': self.fps,
            'total_frames': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'queue_size': self.frame_queue.qsize()
        }

        # Add memory usage info if psutil is available
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            stats['memory_mb'] = mem_info.rss / 1024 / 1024
            stats['memory_percent'] = process.memory_percent()
        except ImportError:
            pass  # psutil not available

        return stats


def create_rtsp_url(
    camera_ip: str,
    username: str = "admin",
    password: str = "",
    stream_type: str = "main",
    protocol: str = "rtsp",
    camera_id: Optional[str] = None
) -> str:
    """
    Create stream URL for Reolink camera with various protocol options.

    Args:
        camera_ip: IP address of the camera (or Neolink host for 'neolink' protocol)
        username: Camera username (ignored for neolink protocol)
        password: Camera password (ignored for neolink protocol)
        stream_type: 'main' for high quality or 'sub' for lower quality
        protocol: Protocol to use:
            - 'rtsp' or 'rtsp-udp': Standard RTSP over UDP (default)
            - 'rtsp-tcp': RTSP over TCP (reduces tearing, more reliable)
            - 'onvif': ONVIF RTSP path (alternative protocol)
            - 'h265': H.265/HEVC encoding (if camera supports)
            - 'neolink': Neolink RTSP bridge (best quality for E1 Pro)
                         Note: For neolink, camera_ip should be the Neolink host
                         (e.g., '127.0.0.1' if running locally), not the camera's IP.
                         For neolink, username and password are ignored in the URL as
                         Neolink authenticates upstream.
        camera_id: Camera ID (required for neolink protocol, ignored for others)

    Returns:
        Stream URL string

    Raises:
        ValueError: If neolink protocol is used without a valid camera_id
    """
    protocol = protocol.lower()

    # Neolink RTSP bridge (best quality for E1 Pro WiFi cameras)
    if protocol == "neolink":
        # Validate camera_id for neolink protocol
        if not camera_id:
            raise ValueError(
                "Neolink protocol requires a valid camera_id. "
                "Camera ID must match the name configured in neolink-config.toml"
            )

        # Neolink serves on port 8554 by default
        # Format: rtsp://<neolink_host>:8554/<camera_name>/<stream>
        stream_name = "mainStream" if stream_type == "main" else "subStream"
        url = f"rtsp://{camera_ip}:8554/{camera_id}/{stream_name}"
        return url

    # ONVIF uses different path format
    elif protocol == "onvif":
        # ONVIF Profile S stream path
        # Channel 1 = main stream, Channel 2 = sub stream
        channel = "101" if stream_type == "main" else "102"
        url = f"rtsp://{username}:{password}@{camera_ip}:554/Streaming/Channels/{channel}"
        return url

    # H.265 encoding option
    elif protocol == "h265":
        stream_path = "h265Preview_01_main" if stream_type == "main" else "h265Preview_01_sub"
        url = f"rtsp://{username}:{password}@{camera_ip}:554/{stream_path}"
        return url

    # Standard H.264 RTSP (default, or rtsp-tcp - handled in RTSPStreamCapture)
    else:
        stream_path = "h264Preview_01_main" if stream_type == "main" else "h264Preview_01_sub"
        url = f"rtsp://{username}:{password}@{camera_ip}:554/{stream_path}"
        # Note: TCP transport is handled in RTSPStreamCapture via use_tcp parameter
        return url


if __name__ == "__main__":
    # Test the stream capture
    from queue import Queue

    # Camera configuration - UPDATE THESE FOR YOUR CAMERA
    CAMERA_IP = "192.168.1.100"  # Replace with your camera IP
    USERNAME = "admin"
    PASSWORD = "your_password_here"  # Replace with your camera password

    rtsp_url = create_rtsp_url(CAMERA_IP, USERNAME, PASSWORD, "main")
    frame_queue = Queue(maxsize=2)

    capture = RTSPStreamCapture(rtsp_url, frame_queue)

    try:
        if capture.start():
            logger.info("Stream capture started. Press Ctrl+C to stop...")

            # Display frames
            while True:
                if not frame_queue.empty():
                    frame_data = frame_queue.get()
                    frame = frame_data['frame']

                    # Display FPS
                    cv2.putText(
                        frame,
                        f"FPS: {capture.fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                    cv2.imshow('RTSP Stream', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                time.sleep(0.01)
        else:
            logger.error("Failed to start stream capture")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        capture.stop()
        cv2.destroyAllWindows()
        logger.info("Test completed")
