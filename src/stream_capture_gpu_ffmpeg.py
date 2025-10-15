"""
GPU-Accelerated RTSP Stream Capture using FFmpeg subprocess
Uses FFmpeg with NVDEC (h264_cuvid) for hardware-accelerated video decoding.
Reduces CPU usage from ~1000% to near-zero by using GPU video decoders.
"""

import subprocess
import time
import logging
import numpy as np
import torch
from typing import Optional, Union
from queue import Queue, Full
from threading import Thread, Event, Lock

logger = logging.getLogger(__name__)


class RTSPStreamCaptureGPU:
    """
    GPU-accelerated RTSP stream capture using FFmpeg with NVDEC.
    Uses h264_cuvid decoder to offload H.264 decoding to GPU.
    """

    def __init__(
        self,
        rtsp_url: str,
        frame_queue: Queue,
        target_width: int = 1280,
        target_height: int = 720,
        camera_id: str = "default",
        camera_name: str = "Default Camera",
        use_tcp: bool = False,
        buffer_size: Optional[int] = None,
        max_failures: int = 30,
        retry_delay: float = 5.0,
        keep_frames_on_gpu: bool = True,
        device: Union[str, torch.device] = "cuda:0"
    ):
        """
        Initialize GPU-accelerated RTSP stream capture.

        Args:
            rtsp_url: RTSP URL of the camera
            frame_queue: Queue to put captured frames
            target_width: Resize width for inference
            target_height: Resize height for inference
            camera_id: Unique identifier for this camera
            camera_name: Human-readable name for this camera
            use_tcp: Use TCP transport instead of UDP
            buffer_size: Accepted for API compatibility, but not used (FFmpeg uses internal buffering)
            max_failures: Reconnect after N consecutive failures
            retry_delay: Seconds to wait before retrying connection
            keep_frames_on_gpu: If True, store frames as GPU tensors (reduces CPU usage)
            device: GPU device to use for frame storage
        """
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.target_width = target_width
        self.target_height = target_height
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.use_tcp = use_tcp
        self.buffer_size = buffer_size
        self.max_failures = max_failures
        self.retry_delay = retry_delay
        self.keep_frames_on_gpu = keep_frames_on_gpu
        self.device = torch.device(device) if isinstance(device, str) else device

        # Warn if buffer_size is set (not used by FFmpeg subprocess)
        if buffer_size is not None:
            logger.warning(
                f"[{self.camera_id}] buffer_size parameter is not used by FFmpeg GPU capture "
                f"(FFmpeg uses internal buffering). Ignoring buffer_size={buffer_size}"
            )

        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.stop_event = Event()
        self.capture_thread: Optional[Thread] = None
        self.is_connected = False

        # Latest frame for video streaming (thread-safe access)
        # Can be either np.ndarray or torch.Tensor depending on keep_frames_on_gpu
        self.latest_frame: Optional[Union[np.ndarray, torch.Tensor]] = None
        self.frame_lock = Lock()

        # Performance metrics
        self.frame_count = 0  # Reset every second for FPS calculation
        self.frame_id_counter = 0  # Never reset - monotonically increasing frame ID
        self.dropped_frames = 0
        self.last_fps_check = time.time()
        self.fps = 0.0

    def _build_ffmpeg_command(self) -> list:
        """
        Build FFmpeg command with GPU-accelerated decoding.

        Returns:
            List of command arguments for subprocess
        """
        cmd = [
            'ffmpeg',
            '-hwaccel', 'cuda',  # Enable CUDA hardware acceleration
            '-c:v', 'h264_cuvid',  # Use NVIDIA CUVID H.264 decoder (GPU decode!)
        ]

        # RTSP options
        if self.use_tcp:
            cmd.extend(['-rtsp_transport', 'tcp'])

        cmd.extend([
            '-i', self.rtsp_url,  # Input RTSP URL
            '-f', 'rawvideo',  # Output raw video
            '-pix_fmt', 'bgr24',  # OpenCV-compatible pixel format
            '-s', f'{self.target_width}x{self.target_height}',  # Resize
            'pipe:1'  # Output to stdout
        ])

        return cmd

    def connect(self) -> bool:
        """
        Connect to the RTSP stream with GPU-accelerated decoding.

        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f"[{self.camera_id}] Connecting to RTSP stream (GPU decode via FFmpeg): {self.rtsp_url}")

        try:
            # Build FFmpeg command
            cmd = self._build_ffmpeg_command()

            logger.info(f"[{self.camera_id}] Starting FFmpeg with GPU decoder (h264_cuvid)")

            # Start FFmpeg process
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8  # Large buffer for video data
            )

            # Wait a moment for FFmpeg to initialize
            time.sleep(0.5)

            # Check if process is still running
            if self.ffmpeg_process.poll() is not None:
                stderr = self.ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                logger.error(f"[{self.camera_id}] FFmpeg failed to start: {stderr[:500]}")
                return False

            self.is_connected = True
            logger.info(f"[{self.camera_id}] ✓ GPU decode enabled (FFmpeg + h264_cuvid)")
            return True

        except Exception as e:
            logger.error(f"[{self.camera_id}] Connection error: {e}")
            return False

    def start(self) -> bool:
        """
        Start the capture thread.

        Returns:
            Always returns True (connection happens in background)
        """
        self.stop_event.clear()
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info(f"[{self.camera_id}] GPU capture thread started")
        return True

    def stop(self):
        """Stop the capture thread and release resources."""
        logger.info(f"[{self.camera_id}] Stopping GPU capture thread...")
        self.stop_event.set()

        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)
            if self.capture_thread.is_alive():
                logger.error(f"[{self.camera_id}] CRITICAL: Capture thread did not stop after 5s timeout (thread may be blocked)")
                logger.error(f"[{self.camera_id}] Thread is orphaned and will continue running - potential resource leak (Issue #96)")

        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5.0)
                logger.info(f"[{self.camera_id}] FFmpeg process terminated successfully")
            except subprocess.TimeoutExpired:
                logger.warning(f"[{self.camera_id}] FFmpeg did not terminate gracefully, forcing kill...")
                try:
                    self.ffmpeg_process.kill()
                    self.ffmpeg_process.wait(timeout=2.0)  # Ensure it's really dead (Issue #97)
                    logger.info(f"[{self.camera_id}] FFmpeg process killed successfully")
                except subprocess.TimeoutExpired:
                    logger.error(f"[{self.camera_id}] CRITICAL: FFmpeg process did not die after kill() - zombie process (Issue #97)")
                    # Don't set to None - keep reference so we know there's a zombie
                    return
                except Exception as e:
                    logger.error(f"[{self.camera_id}] Error killing FFmpeg: {e}")
                    return
            except Exception as e:
                logger.error(f"[{self.camera_id}] Error terminating FFmpeg: {e}")
                return

            # Only set to None if we successfully terminated the process
            self.ffmpeg_process = None

        self.is_connected = False
        logger.info(f"[{self.camera_id}] GPU capture stopped")

    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        logger.info(f"[{self.camera_id}] GPU capture loop started")
        consecutive_failures = 0

        # Calculate frame size
        frame_size = self.target_height * self.target_width * 3  # BGR24 = 3 bytes per pixel

        while not self.stop_event.is_set():
            try:
                # If not connected, attempt to connect
                if not self.is_connected or self.ffmpeg_process is None:
                    logger.info(f"[{self.camera_id}] Attempting to connect...")
                    if self.connect():
                        logger.info(f"[{self.camera_id}] Connected successfully!")
                        consecutive_failures = 0
                    else:
                        logger.warning(f"[{self.camera_id}] Connection failed, retrying in {self.retry_delay}s...")
                        time.sleep(self.retry_delay)
                        continue

                # Read frame from FFmpeg stdout
                raw_frame = self.ffmpeg_process.stdout.read(frame_size)

                if len(raw_frame) != frame_size:
                    consecutive_failures += 1
                    logger.warning(f"[{self.camera_id}] Failed to read frame ({consecutive_failures}/{self.max_failures})")

                    if consecutive_failures >= self.max_failures:
                        logger.error(f"[{self.camera_id}] Too many failures, reconnecting...")
                        self._reconnect()
                        consecutive_failures = 0
                    continue

                # Convert raw bytes to numpy array OR GPU tensor
                if self.keep_frames_on_gpu:
                    # Convert directly to GPU tensor (HxWxC BGR format)
                    img_np = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.target_height, self.target_width, 3))
                    img = torch.from_numpy(img_np).to(self.device, non_blocking=True)
                else:
                    # Keep as NumPy array (backward compatibility)
                    img = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.target_height, self.target_width, 3))

                # Store latest frame for video streaming
                with self.frame_lock:
                    if self.keep_frames_on_gpu:
                        self.latest_frame = img.clone()
                    else:
                        self.latest_frame = img.copy()

                # Add timestamp
                timestamp = time.time()

                # Clone GPU tensor before putting in queue to avoid race conditions (Issue #95)
                # Multiple threads may access the same tensor, causing CUDA errors
                if self.keep_frames_on_gpu:
                    frame_for_queue = img.clone()
                else:
                    frame_for_queue = img.copy()

                # Put frame in queue (non-blocking)
                try:
                    self.frame_queue.put_nowait({
                        'frame': frame_for_queue,
                        'timestamp': timestamp,
                        'frame_id': self.frame_id_counter,  # Use separate counter for frame IDs
                        'camera_id': self.camera_id,
                        'camera_name': self.camera_name,
                        'is_gpu_tensor': self.keep_frames_on_gpu  # Indicate format for downstream
                    })
                    self.frame_count += 1  # For FPS calculation (reset every second)
                    self.frame_id_counter += 1  # Monotonically increasing (never reset)
                except Full:
                    self.dropped_frames += 1

                # Calculate FPS
                if time.time() - self.last_fps_check >= 1.0:
                    self.fps = self.frame_count / (time.time() - self.last_fps_check)
                    self.last_fps_check = time.time()
                    self.frame_count = 0

                    if self.dropped_frames > 0:
                        logger.debug(f"[{self.camera_id}] FPS: {self.fps:.1f}, Dropped: {self.dropped_frames}")
                        self.dropped_frames = 0

                consecutive_failures = 0

            except Exception as e:
                consecutive_failures += 1
                logger.warning(f"[{self.camera_id}] Stream error ({consecutive_failures}/{self.max_failures}): {e}")

                if consecutive_failures >= self.max_failures:
                    logger.error(f"[{self.camera_id}] Too many failures, reconnecting...")
                    self._reconnect()
                    consecutive_failures = 0

                time.sleep(1.0)

    def _reconnect(self) -> bool:
        """
        Attempt to reconnect to the stream.

        Returns:
            True if reconnection successful, False otherwise
        """
        logger.info(f"[{self.camera_id}] Attempting to reconnect...")

        # Close existing connection
        if self.ffmpeg_process is not None:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2.0)
                logger.debug(f"[{self.camera_id}] FFmpeg terminated for reconnect")
            except subprocess.TimeoutExpired:
                logger.warning(f"[{self.camera_id}] FFmpeg did not terminate, forcing kill for reconnect...")
                try:
                    self.ffmpeg_process.kill()
                    self.ffmpeg_process.wait(timeout=2.0)  # Ensure it's dead (Issue #97)
                    logger.debug(f"[{self.camera_id}] FFmpeg killed for reconnect")
                except subprocess.TimeoutExpired:
                    logger.error(f"[{self.camera_id}] CRITICAL: FFmpeg zombie process after kill() (Issue #97)")
                    # Don't clear reference - we have a zombie
                    self.is_connected = False
                    return False
                except Exception as e:
                    logger.error(f"[{self.camera_id}] Error killing FFmpeg for reconnect: {e}")
                    self.is_connected = False
                    return False
            except Exception as e:
                logger.error(f"[{self.camera_id}] Error terminating FFmpeg for reconnect: {e}")
                self.is_connected = False
                return False

            # Only clear reference if process successfully terminated
            self.ffmpeg_process = None

        self.is_connected = False
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

    def get_latest_frame_as_numpy(self) -> Optional[np.ndarray]:
        """
        Get latest frame as NumPy array (for web streaming).
        Converts from GPU tensor if needed.

        Returns:
            Latest frame as NumPy array (BGR format), or None if no frame available
        """
        with self.frame_lock:
            if self.latest_frame is None:
                return None

            if isinstance(self.latest_frame, torch.Tensor):
                # Convert GPU tensor to NumPy (on CPU)
                return self.latest_frame.cpu().numpy()
            else:
                # Already NumPy array
                return self.latest_frame
