#!/usr/bin/env python3
"""
Tests for RTSP URL credential redaction in logging.
Validates that credentials are properly redacted from log messages.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import logging
from io import StringIO
from unittest.mock import MagicMock

# Mock cv2 and numpy to avoid dependency issues during testing
sys.modules['cv2'] = MagicMock()
sys.modules['numpy'] = MagicMock()

from src.stream_capture import RTSPURLFilter


class TestRTSPURLFilter(unittest.TestCase):
    """Tests for RTSPURLFilter logging filter."""
    
    def setUp(self):
        """Set up test logger with filter and string handler."""
        self.logger = logging.getLogger('test_rtsp_filter')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Create string handler to capture log output
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self.handler)
        
        # Add RTSP URL filter
        self.filter = RTSPURLFilter()
        self.logger.addFilter(self.filter)
    
    def tearDown(self):
        """Clean up logger."""
        self.logger.handlers = []
        self.logger.filters = []
    
    def get_log_output(self):
        """Get the logged output as a string."""
        self.handler.flush()
        return self.log_stream.getvalue()
    
    def test_standard_rtsp_url_redaction(self):
        """Test redaction of standard RTSP URL with credentials."""
        url = "rtsp://admin:mypassword123@192.168.1.100:554/stream"
        self.logger.info(f"Connecting to {url}")
        
        output = self.get_log_output()
        self.assertNotIn("mypassword123", output)
        self.assertNotIn("admin:mypassword123", output)
        self.assertIn("rtsp://***:***@192.168.1.100:554/stream", output)
    
    def test_multiple_urls_in_message(self):
        """Test redaction of multiple RTSP URLs in single message."""
        msg = "Cameras: rtsp://user1:pass1@10.0.0.1/stream and rtsp://user2:pass2@10.0.0.2/stream"
        self.logger.info(msg)
        
        output = self.get_log_output()
        self.assertNotIn("pass1", output)
        self.assertNotIn("pass2", output)
        self.assertNotIn("user1", output)
        self.assertNotIn("user2", output)
        self.assertIn("rtsp://***:***@10.0.0.1/stream", output)
        self.assertIn("rtsp://***:***@10.0.0.2/stream", output)
    
    def test_special_characters_in_credentials(self):
        """Test redaction with special characters in password."""
        url = "rtsp://admin:p@ss!w0rd#123@192.168.1.100:554/stream"
        self.logger.info(f"URL: {url}")
        
        output = self.get_log_output()
        self.assertNotIn("p@ss!w0rd#123", output)
        self.assertIn("rtsp://***:***@192.168.1.100:554/stream", output)
    
    def test_empty_password(self):
        """Test redaction with empty password."""
        url = "rtsp://admin:@192.168.1.100:554/stream"
        self.logger.info(f"URL: {url}")
        
        output = self.get_log_output()
        # Empty password case should still be redacted
        self.assertIn("rtsp://***:***@192.168.1.100:554/stream", output)
    
    def test_onvif_url_format(self):
        """Test redaction of ONVIF-style RTSP URLs."""
        url = "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101"
        self.logger.info(f"ONVIF URL: {url}")
        
        output = self.get_log_output()
        self.assertNotIn("password", output)
        self.assertIn("rtsp://***:***@192.168.1.100:554/Streaming/Channels/101", output)
    
    def test_h265_url_format(self):
        """Test redaction of H.265 RTSP URLs."""
        url = "rtsp://admin:secret@192.168.1.100:554/h265Preview_01_main"
        self.logger.info(f"H265 URL: {url}")
        
        output = self.get_log_output()
        self.assertNotIn("secret", output)
        self.assertIn("rtsp://***:***@192.168.1.100:554/h265Preview_01_main", output)
    
    def test_url_without_credentials(self):
        """Test that URLs without credentials are not modified."""
        url = "rtsp://192.168.1.100:8554/camera1/mainStream"
        self.logger.info(f"Neolink URL: {url}")
        
        output = self.get_log_output()
        # Should remain unchanged since no credentials
        self.assertIn("rtsp://192.168.1.100:8554/camera1/mainStream", output)
    
    def test_non_rtsp_urls_unaffected(self):
        """Test that non-RTSP URLs are not affected."""
        msg = "HTTP URL: http://admin:password@example.com/api"
        self.logger.info(msg)
        
        output = self.get_log_output()
        # HTTP URLs should not be redacted
        self.assertIn("http://admin:password@example.com/api", output)
    
    def test_case_insensitive_rtsp(self):
        """Test that RTSP is matched case-insensitively."""
        url = "RTSP://admin:password@192.168.1.100:554/stream"
        self.logger.info(f"Uppercase: {url}")
        
        output = self.get_log_output()
        self.assertNotIn("password", output)
        # The protocol part is preserved but credentials are redacted
        self.assertIn("RTSP://***:***@192.168.1.100:554/stream", output)
    
    def test_error_message_with_url(self):
        """Test redaction in error messages."""
        url = "rtsp://admin:secretpass@192.168.1.100:554/stream"
        self.logger.error(f"Failed to connect to {url}")
        
        output = self.get_log_output()
        self.assertNotIn("secretpass", output)
        self.assertIn("rtsp://***:***@192.168.1.100:554/stream", output)
    
    def test_exception_with_url(self):
        """Test redaction in exception messages."""
        url = "rtsp://admin:mypass@192.168.1.100:554/stream"
        try:
            raise ConnectionError(f"Cannot connect to {url}")
        except ConnectionError as e:
            self.logger.error(f"Connection failed: {e}", exc_info=True)
        
        output = self.get_log_output()
        self.assertNotIn("mypass", output)
        self.assertIn("rtsp://***:***@192.168.1.100:554/stream", output)
    
    def test_formatted_logging(self):
        """Test redaction with formatted logging arguments."""
        username = "admin"
        password = "testpass123"
        ip = "192.168.1.100"
        url = f"rtsp://{username}:{password}@{ip}:554/stream"
        
        self.logger.info("Connecting to %s", url)
        
        output = self.get_log_output()
        self.assertNotIn("testpass123", output)
        self.assertIn("rtsp://***:***@192.168.1.100:554/stream", output)
    
    def test_debug_level_messages(self):
        """Test that filter works for all log levels."""
        url = "rtsp://user:debugpass@192.168.1.100:554/stream"
        
        self.logger.debug(f"Debug: {url}")
        self.logger.info(f"Info: {url}")
        self.logger.warning(f"Warning: {url}")
        self.logger.error(f"Error: {url}")
        self.logger.critical(f"Critical: {url}")
        
        output = self.get_log_output()
        # Should not appear in any log level
        self.assertNotIn("debugpass", output)
        # Should appear redacted in all
        self.assertEqual(output.count("rtsp://***:***@192.168.1.100:554/stream"), 5)
    
    def test_url_in_mixed_content(self):
        """Test redaction when URL is mixed with other content."""
        msg = "Camera config: {id: 'cam1', url: 'rtsp://admin:hideme@10.0.0.1/stream', fps: 30}"
        self.logger.info(msg)
        
        output = self.get_log_output()
        self.assertNotIn("hideme", output)
        self.assertIn("rtsp://***:***@10.0.0.1/stream", output)
        # Other content should remain
        self.assertIn("id: 'cam1'", output)
        self.assertIn("fps: 30", output)


class TestRTSPURLFilterDirect(unittest.TestCase):
    """Direct tests of the RTSPURLFilter._redact_credentials method."""
    
    def setUp(self):
        """Set up filter instance."""
        self.filter = RTSPURLFilter()
    
    def test_redact_credentials_method(self):
        """Test the _redact_credentials method directly."""
        text = "rtsp://admin:password@192.168.1.100:554/stream"
        result = self.filter._redact_credentials(text)
        
        self.assertEqual(result, "rtsp://***:***@192.168.1.100:554/stream")
        self.assertNotIn("password", result)
    
    def test_redact_with_no_urls(self):
        """Test text without URLs is unchanged."""
        text = "This is just regular text with no URLs"
        result = self.filter._redact_credentials(text)
        
        self.assertEqual(result, text)
    
    def test_redact_preserves_other_text(self):
        """Test that non-URL text is preserved."""
        text = "Connecting to camera rtsp://admin:pass@10.0.0.1/stream with timeout 5s"
        result = self.filter._redact_credentials(text)
        
        self.assertIn("Connecting to camera", result)
        self.assertIn("with timeout 5s", result)
        self.assertIn("rtsp://***:***@10.0.0.1/stream", result)
        self.assertNotIn("pass", result)


if __name__ == '__main__':
    unittest.main()
