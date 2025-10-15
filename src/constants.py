"""
System Constants Module
Centralizes hardcoded "magic numbers" for better maintainability.
Values that should be user-configurable are loaded from config.yaml instead.
"""

# ============================================================================
# Queue Timeouts (seconds)
# ============================================================================
# Timeout for getting items from queues (blocking operations)
QUEUE_GET_TIMEOUT_SECONDS = 0.1

# Timeout for thread join operations
THREAD_JOIN_TIMEOUT_SECONDS = 5.0


# ============================================================================
# Logging and Monitoring
# ============================================================================
# Log dropped frames/results every N occurrences (reduces spam)
LOG_DROPPED_EVERY_N = 10


# ============================================================================
# RTSP Stream Capture
# ============================================================================
# RTSP connection timeout in microseconds (5 seconds)
RTSP_TIMEOUT_MICROSECONDS = 5_000_000

# Default values for RTSP reconnection (can be overridden by config)
DEFAULT_MAX_RTSP_FAILURES = 30
DEFAULT_RTSP_RETRY_DELAY_SECONDS = 5.0
DEFAULT_RTSP_RECONNECT_DELAY_SECONDS = 2.0


# ============================================================================
# Error Handling Sleep Intervals (seconds)
# ============================================================================
# Sleep duration when errors occur in processing loops
ERROR_SLEEP_SECONDS = 0.1

# Sleep duration after failed RTSP reads
RTSP_FAILURE_SLEEP_SECONDS = 1.0


# ============================================================================
# Web Server / Video Streaming
# ============================================================================
# Default MJPEG streaming frame rate (FPS)
DEFAULT_MJPEG_FPS = 30

# Default JPEG compression quality (0-100)
DEFAULT_JPEG_QUALITY = 85

# WebSocket heartbeat interval (seconds)
WEBSOCKET_HEARTBEAT_INTERVAL_SECONDS = 1.0


# ============================================================================
# Performance Monitoring
# ============================================================================
# FPS calculation interval (seconds)
FPS_CALCULATION_INTERVAL_SECONDS = 1.0

# Minimum time delta for rate calculations (avoids division by zero)
MIN_TIME_DELTA = 0.001
