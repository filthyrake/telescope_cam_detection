#!/usr/bin/env python3
"""
Security verification test for RTSP URL credential redaction.

This test verifies that the security issue described in the GitHub issue is fixed:
- RTSP URLs with embedded credentials (rtsp://user:pass@host) are not logged in plaintext
- Credentials are redacted to rtsp://***:***@host in all log messages
- Both stream_capture.py and stream_capture_gpu_ffmpeg.py modules are protected
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import logging
from io import StringIO
from unittest.mock import MagicMock, patch
from queue import Queue

# Mock dependencies before importing
sys.modules['cv2'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['torch'] = MagicMock()

from src.stream_capture import RTSPStreamCapture, RTSPURLFilter
from src.stream_capture_gpu_ffmpeg import RTSPStreamCaptureGPU


class TestRTSPCredentialSecurity(unittest.TestCase):
    """Security tests for RTSP URL credential redaction."""
    
    def setUp(self):
        """Set up logging capture for each test."""
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        
        # Store original handlers so we can restore them
        self.original_handlers = {
            'src.stream_capture': logging.getLogger('src.stream_capture').handlers[:],
            'src.stream_capture_gpu_ffmpeg': logging.getLogger('src.stream_capture_gpu_ffmpeg').handlers[:]
        }
    
    def tearDown(self):
        """Restore original handlers."""
        for logger_name, handlers in self.original_handlers.items():
            logger = logging.getLogger(logger_name)
            logger.handlers = handlers
    
    def get_log_output(self):
        """Get the captured log output."""
        self.handler.flush()
        return self.log_stream.getvalue()
    
    def test_stream_capture_line_96_security_fix(self):
        """
        Test that line 96 in stream_capture.py does not leak credentials.
        This is the primary issue mentioned in the GitHub issue.
        """
        # Set up logger - add handler but keep filters
        logger = logging.getLogger('src.stream_capture')
        logger.addHandler(self.handler)
        logger.setLevel(logging.INFO)
        
        # Simulate what line 96 does: logs the RTSP URL
        camera_id = "test_cam"
        rtsp_url = "rtsp://admin:mysecretpassword@192.168.1.100:554/stream"
        logger.info(f"[{camera_id}] Connecting to RTSP stream: {rtsp_url}")
        
        output = self.get_log_output()
        
        # Verify password is NOT in logs
        self.assertNotIn("mysecretpassword", output,
                        "Password should not appear in logs")
        self.assertNotIn("admin:mysecretpassword", output,
                        "Credentials should not appear in logs")
        
        # Verify URL is properly redacted
        self.assertIn("rtsp://***:***@192.168.1.100:554/stream", output,
                     "URL should be redacted to rtsp://***:***@host")
    
    def test_gpu_ffmpeg_line_129_security_fix(self):
        """
        Test that line 129 in stream_capture_gpu_ffmpeg.py does not leak credentials.
        This module also logs RTSP URLs and needs the same protection.
        """
        # Set up logger - add handler but keep filters
        logger = logging.getLogger('src.stream_capture_gpu_ffmpeg')
        logger.addHandler(self.handler)
        logger.setLevel(logging.INFO)
        
        # Test the filter directly on this logger
        test_url = "rtsp://user:topsecret123@10.0.0.1:554/h265Preview_01_main"
        logger.info(f"Connecting to RTSP stream (GPU decode via FFmpeg): {test_url}")
        
        output = self.get_log_output()
        
        # Verify password is NOT in logs
        self.assertNotIn("topsecret123", output,
                        "Password should not appear in GPU module logs")
        
        # Verify URL is properly redacted
        self.assertIn("rtsp://***:***@10.0.0.1", output,
                     "URL should be redacted in GPU module")
    
    def test_filter_attached_to_both_modules(self):
        """Verify that RTSPURLFilter is attached to both logging modules."""
        # Check stream_capture module
        sc_logger = logging.getLogger('src.stream_capture')
        sc_has_filter = any(f.__class__.__name__ == 'RTSPURLFilter' for f in sc_logger.filters)
        self.assertTrue(sc_has_filter,
                       "src.stream_capture module should have RTSPURLFilter")
        
        # Check stream_capture_gpu_ffmpeg module
        gpu_logger = logging.getLogger('src.stream_capture_gpu_ffmpeg')
        gpu_has_filter = any(f.__class__.__name__ == 'RTSPURLFilter' for f in gpu_logger.filters)
        self.assertTrue(gpu_has_filter,
                       "src.stream_capture_gpu_ffmpeg module should have RTSPURLFilter")
    
    def test_security_issue_summary(self):
        """
        Integration test that verifies the complete security fix.
        
        Original Issue:
        - RTSP URLs like rtsp://user:pass@host were logged in plaintext
        - This could expose credentials in log files
        - Affected lines: stream_capture.py:96, stream_capture_gpu_ffmpeg.py:129
        
        Fix:
        - Added RTSPURLFilter logging filter
        - Filter redacts credentials to rtsp://***:***@host
        - Applied to both affected modules
        """
        test_cases = [
            ("rtsp://admin:password123@192.168.1.100:554/stream",
             "rtsp://***:***@192.168.1.100:554/stream"),
            ("rtsp://user:p@ss!word@10.0.0.1/main",
             "rtsp://***:***@10.0.0.1/main"),
            ("rtsp://test:@192.168.1.50:554/sub",
             "rtsp://***:***@192.168.1.50:554/sub"),
        ]
        
        filter_obj = RTSPURLFilter()
        
        for original, expected in test_cases:
            redacted = filter_obj._redact_credentials(original)
            self.assertEqual(redacted, expected,
                           f"URL {original} should be redacted to {expected}")
            self.assertNotIn("password", redacted,
                           "Password should never appear in redacted URL")
            self.assertNotIn("p@ss!word", redacted,
                           "Special character passwords should be redacted")


if __name__ == '__main__':
    print("="*70)
    print("Security Test: RTSP URL Credential Redaction")
    print("="*70)
    print()
    print("Testing fix for: RTSP URLs with credentials logged and stored in memory")
    print()
    
    # Run tests
    unittest.main(verbosity=2)
