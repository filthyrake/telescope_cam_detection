"""
RTSP Stream Capture Module
Handles video stream capture from Reolink RLC-410W camera with minimal latency.
"""

import cv2
import time
import logging
from typing import Optional, Tuple
import numpy as np
from queue import Queue, Full
from threading import Thread, Event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        # Latest frame for video streaming
        self.latest_frame: Optional[np.ndarray] = None

        # Performance metrics
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_fps_check = time.time()
        self.fps = 0.0

    def connect(self) -> bool:
        """
        Connect to the RTSP stream.

        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f"[{self.camera_id}] Connecting to RTSP stream: {self.rtsp_url}")
        if self.use_tcp:
            logger.info(f"[{self.camera_id}] Using TCP transport (rtsp_transport=tcp)")

        # OpenCV VideoCapture with optimized settings for low latency
        # Use FFMPEG backend with TCP transport if requested
        if self.use_tcp:
            # Set RTSP transport to TCP via environment variable (OpenCV+FFMPEG)
            import os
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

        self.capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

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

        Returns:
            True if started successfully, False otherwise
        """
        if not self.is_connected:
            if not self.connect():
                return False

        self.stop_event.clear()
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info(f"[{self.camera_id}] Capture thread started")
        return True

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

        while not self.stop_event.is_set():
            try:
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

                # Store latest frame for video streaming
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
        logger.info("Attempting to reconnect...")

        if self.capture:
            self.capture.release()

        time.sleep(2.0)  # Wait before reconnecting
        return self.connect()

    def get_stats(self) -> dict:
        """
        Get capture statistics.

        Returns:
            Dictionary containing capture stats
        """
        return {
            'is_connected': self.is_connected,
            'fps': self.fps,
            'total_frames': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'queue_size': self.frame_queue.qsize()
        }


def create_rtsp_url(
    camera_ip: str,
    username: str = "admin",
    password: str = "",
    stream_type: str = "main",
    protocol: str = "rtsp"
) -> str:
    """
    Create stream URL for Reolink camera with various protocol options.

    Args:
        camera_ip: IP address of the camera
        username: Camera username
        password: Camera password
        stream_type: 'main' for high quality or 'sub' for lower quality
        protocol: Protocol to use:
            - 'rtsp' or 'rtsp-udp': Standard RTSP over UDP (default)
            - 'rtsp-tcp': RTSP over TCP (reduces tearing, more reliable)
            - 'onvif': ONVIF RTSP path (alternative protocol)
            - 'h265': H.265/HEVC encoding (if camera supports)

    Returns:
        Stream URL string
    """
    protocol = protocol.lower()

    # ONVIF uses different path format
    if protocol == "onvif":
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
