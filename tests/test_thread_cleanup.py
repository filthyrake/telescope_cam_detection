#!/usr/bin/env python3
"""
Tests for thread cleanup verification in stream_capture and detection_processor.
Validates that threads are properly terminated and handled when stuck.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import time
import logging
from unittest.mock import patch, MagicMock
from threading import Thread, Event
from queue import Queue

# Mock torch module before importing detection_processor
sys.modules['torch'] = MagicMock()

from src.stream_capture import RTSPStreamCapture
from src.detection_processor import DetectionProcessor
from src.constants import THREAD_JOIN_TIMEOUT_SECONDS

# Suppress logging during tests
logging.basicConfig(level=logging.CRITICAL)


class TestStreamCaptureThreadCleanup(unittest.TestCase):
    """Tests for RTSPStreamCapture thread cleanup."""

    def test_thread_created_as_daemon(self):
        """Test that capture threads are created as daemon threads."""
        frame_queue = Queue()
        capture = RTSPStreamCapture(
            rtsp_url="rtsp://fake:fake@192.168.1.1:554/stream",
            frame_queue=frame_queue,
            camera_id="test_cam"
        )
        
        # Mock the cv2.VideoCapture to avoid actual connection
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = False
            mock_capture.return_value = mock_cap_instance
            
            # Start the capture
            capture.start()
            
            # Thread should be created as daemon
            self.assertTrue(capture.capture_thread.daemon)
            
            # Clean up
            capture.stop()

    def test_normal_thread_cleanup(self):
        """Test that threads stop normally within timeout."""
        frame_queue = Queue()
        capture = RTSPStreamCapture(
            rtsp_url="rtsp://fake:fake@192.168.1.1:554/stream",
            frame_queue=frame_queue,
            camera_id="test_cam"
        )
        
        # Mock the cv2.VideoCapture to avoid actual connection
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = False
            mock_capture.return_value = mock_cap_instance
            
            # Start the capture (thread will run but connection will fail quickly)
            capture.start()
            
            # Give thread a moment to start
            time.sleep(0.1)
            
            # Stop should complete within normal timeout
            start_time = time.time()
            capture.stop()
            stop_duration = time.time() - start_time
            
            # Should stop quickly since thread isn't actually blocked
            self.assertLess(stop_duration, THREAD_JOIN_TIMEOUT_SECONDS + 1)
            
            # Thread should be stopped
            self.assertFalse(capture.capture_thread.is_alive())

    def test_stuck_thread_retry_logic(self):
        """Test that stuck threads trigger retry logic with extended timeout."""
        frame_queue = Queue()
        capture = RTSPStreamCapture(
            rtsp_url="rtsp://fake:fake@192.168.1.1:554/stream",
            frame_queue=frame_queue,
            camera_id="test_cam"
        )
        
        # Create a thread that will never stop
        def infinite_loop():
            stop_event = Event()
            while not stop_event.wait(timeout=0.1):
                pass
        
        # Replace the capture thread with our infinite loop thread (daemon=True like real implementation)
        capture.capture_thread = Thread(target=infinite_loop, daemon=True)
        capture.capture_thread.start()
        
        # Verify thread is a daemon (as it is in real implementation)
        self.assertTrue(capture.capture_thread.daemon)
        
        # Track join calls
        join_calls = []
        
        def mock_join(timeout=None):
            join_calls.append(timeout)
            time.sleep(0.01)
            return None
        
        capture.capture_thread.join = mock_join
        
        # Stop should attempt join twice
        with patch('src.stream_capture.THREAD_JOIN_TIMEOUT_SECONDS', 0.1):
            capture.stop()
        
        # Should have called join twice: once with 0.1s, once with 0.2s
        self.assertEqual(len(join_calls), 2)
        self.assertAlmostEqual(join_calls[0], 0.1, places=1)
        self.assertAlmostEqual(join_calls[1], 0.2, places=1)
        
        # Thread is still alive but won't block shutdown (daemon=True)
        self.assertTrue(capture.capture_thread.is_alive())

    def test_thread_stops_on_second_attempt(self):
        """Test that thread stopping on second attempt is handled correctly."""
        frame_queue = Queue()
        capture = RTSPStreamCapture(
            rtsp_url="rtsp://fake:fake@192.168.1.1:554/stream",
            frame_queue=frame_queue,
            camera_id="test_cam"
        )
        
        # Create a thread that stops after a delay
        stop_flag = Event()
        
        def delayed_stop():
            # Wait for the stop flag, but with a timeout
            stop_flag.wait(timeout=0.5)
        
        capture.capture_thread = Thread(target=delayed_stop, daemon=True)
        capture.capture_thread.start()
        
        # Mock join to simulate first attempt failing, second succeeding
        join_count = [0]
        original_join = capture.capture_thread.join
        
        def mock_join(timeout=None):
            join_count[0] += 1
            if join_count[0] == 1:
                # First join - don't actually stop
                time.sleep(0.01)
            else:
                # Second join - signal stop and join for real
                stop_flag.set()
                original_join(timeout=timeout)
        
        capture.capture_thread.join = mock_join
        
        # Stop should succeed on second attempt
        with patch('src.stream_capture.THREAD_JOIN_TIMEOUT_SECONDS', 0.1):
            capture.stop()
        
        # Thread should be stopped
        self.assertFalse(capture.capture_thread.is_alive())


class TestDetectionProcessorThreadCleanup(unittest.TestCase):
    """Tests for DetectionProcessor thread cleanup."""

    def test_processor_thread_created_as_daemon(self):
        """Test that processor threads are created as daemon threads."""
        input_queue = Queue()
        output_queue = Queue()
        
        processor = DetectionProcessor(
            input_queue=input_queue,
            output_queue=output_queue
        )
        
        # Start the processor
        processor.start()
        
        # Thread should be created as daemon
        self.assertTrue(processor.processor_thread.daemon)
        
        # Clean up
        processor.stop()

    def test_normal_thread_cleanup(self):
        """Test that processor threads stop normally within timeout."""
        input_queue = Queue()
        output_queue = Queue()
        
        processor = DetectionProcessor(
            input_queue=input_queue,
            output_queue=output_queue
        )
        
        # Start the processor
        processor.start()
        
        # Give thread a moment to start
        time.sleep(0.1)
        
        # Stop should complete within normal timeout
        start_time = time.time()
        processor.stop()
        stop_duration = time.time() - start_time
        
        # Should stop quickly
        self.assertLess(stop_duration, THREAD_JOIN_TIMEOUT_SECONDS + 1)
        
        # Thread should be stopped
        self.assertFalse(processor.processor_thread.is_alive())

    def test_stuck_processor_thread_retry_logic(self):
        """Test that stuck processor threads trigger retry logic with extended timeout."""
        input_queue = Queue()
        output_queue = Queue()
        
        processor = DetectionProcessor(
            input_queue=input_queue,
            output_queue=output_queue
        )
        
        # Create a thread that will never stop
        def infinite_loop():
            stop_event = Event()
            while not stop_event.wait(timeout=0.1):
                pass
        
        # Replace the processor thread with our infinite loop thread (daemon=True like real implementation)
        processor.processor_thread = Thread(target=infinite_loop, daemon=True)
        processor.processor_thread.start()
        
        # Verify thread is a daemon (as it is in real implementation)
        self.assertTrue(processor.processor_thread.daemon)
        
        # Track join calls
        join_calls = []
        
        def mock_join(timeout=None):
            join_calls.append(timeout)
            time.sleep(0.01)
            return None
        
        processor.processor_thread.join = mock_join
        
        # Stop should attempt join twice
        with patch('src.detection_processor.THREAD_JOIN_TIMEOUT_SECONDS', 0.1):
            processor.stop()
        
        # Should have called join twice: once with 0.1s, once with 0.2s
        self.assertEqual(len(join_calls), 2)
        self.assertAlmostEqual(join_calls[0], 0.1, places=1)
        self.assertAlmostEqual(join_calls[1], 0.2, places=1)
        
        # Thread is still alive but won't block shutdown (daemon=True)
        self.assertTrue(processor.processor_thread.is_alive())

    def test_processor_thread_stops_on_second_attempt(self):
        """Test that processor thread stopping on second attempt is handled."""
        input_queue = Queue()
        output_queue = Queue()
        
        processor = DetectionProcessor(
            input_queue=input_queue,
            output_queue=output_queue
        )
        
        # Create a thread that stops after a delay
        stop_flag = Event()
        
        def delayed_stop():
            stop_flag.wait(timeout=0.5)
        
        processor.processor_thread = Thread(target=delayed_stop, daemon=True)
        processor.processor_thread.start()
        
        # Mock join to simulate first attempt failing, second succeeding
        join_count = [0]
        original_join = processor.processor_thread.join
        
        def mock_join(timeout=None):
            join_count[0] += 1
            if join_count[0] == 1:
                time.sleep(0.01)
            else:
                stop_flag.set()
                original_join(timeout=timeout)
        
        processor.processor_thread.join = mock_join
        
        # Stop should succeed on second attempt
        with patch('src.detection_processor.THREAD_JOIN_TIMEOUT_SECONDS', 0.1):
            processor.stop()
        
        # Thread should be stopped
        self.assertFalse(processor.processor_thread.is_alive())


if __name__ == '__main__':
    unittest.main()
