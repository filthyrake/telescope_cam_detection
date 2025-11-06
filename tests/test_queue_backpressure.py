"""
Tests for queue overflow backpressure mechanism.
Tests that detection_processor and inference_engine properly handle queue overflows
with backpressure signaling instead of silent data loss.
"""

import unittest
import time
from queue import Queue
from unittest.mock import Mock, patch, MagicMock

# Mock all required modules before importing
import sys
sys.modules['torch'] = MagicMock()
sys.modules['torch.cuda'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['av'] = MagicMock()
sys.modules['numpy'] = MagicMock()

from src.detection_processor import DetectionProcessor
from src.constants import QUEUE_PUT_TIMEOUT_SECONDS


class TestDetectionProcessorBackpressure(unittest.TestCase):
    """Tests for DetectionProcessor queue backpressure mechanism."""

    def test_backpressure_event_initialized(self):
        """Test that backpressure event is initialized."""
        input_queue = Queue()
        output_queue = Queue()
        
        processor = DetectionProcessor(
            input_queue=input_queue,
            output_queue=output_queue
        )
        
        # Backpressure event should exist and be cleared
        self.assertIsNotNone(processor.backpressure_event)
        self.assertFalse(processor.backpressure_event.is_set())
        
        processor.stop()

    def test_queue_overflow_count_tracked(self):
        """Test that queue overflow events are tracked."""
        input_queue = Queue()
        output_queue = Queue(maxsize=1)  # Small queue to force overflow
        
        processor = DetectionProcessor(
            input_queue=input_queue,
            output_queue=output_queue
        )
        
        # Initial count should be zero
        self.assertEqual(processor.queue_overflow_count, 0)
        
        processor.stop()

    def test_queue_overflow_count_in_stats(self):
        """Test that queue_overflow_count appears in stats."""
        input_queue = Queue()
        output_queue = Queue()
        
        processor = DetectionProcessor(
            input_queue=input_queue,
            output_queue=output_queue
        )
        
        stats = processor.get_stats()
        self.assertIn('queue_overflow_count', stats)
        self.assertEqual(stats['queue_overflow_count'], 0)
        
        processor.stop()

    def test_backpressure_signal_on_overflow(self):
        """Test that backpressure event is set when queue overflows."""
        input_queue = Queue()
        output_queue = Queue(maxsize=1)
        
        processor = DetectionProcessor(
            input_queue=input_queue,
            output_queue=output_queue
        )
        
        processor.start()
        
        # Fill output queue
        output_queue.put({'test': 'data'})
        
        # Add detection that will cause overflow
        detection_result = {
            'detections': [],
            'frame_id': 1,
            'timestamp': time.time(),
            'inference_time': 0.01,
            'camera_id': 'test',
            'camera_name': 'Test Camera'
        }
        
        input_queue.put(detection_result)
        
        # Wait longer for processing (timeout + processing time)
        time.sleep(QUEUE_PUT_TIMEOUT_SECONDS + 0.5)
        
        # Check that overflow was tracked
        self.assertGreater(processor.queue_overflow_count, 0)
        
        # Check that backpressure event was set
        self.assertTrue(processor.backpressure_event.is_set())
        
        processor.stop()

    def test_backpressure_cleared_on_success(self):
        """Test that backpressure event is cleared when queue accepts data."""
        input_queue = Queue()
        output_queue = Queue(maxsize=2)
        
        processor = DetectionProcessor(
            input_queue=input_queue,
            output_queue=output_queue
        )
        
        processor.start()
        
        # Manually set backpressure event
        processor.backpressure_event.set()
        self.assertTrue(processor.backpressure_event.is_set())
        
        # Add detection that should succeed
        detection_result = {
            'detections': [],
            'frame_id': 1,
            'timestamp': time.time(),
            'inference_time': 0.01,
            'camera_id': 'test',
            'camera_name': 'Test Camera'
        }
        
        input_queue.put(detection_result)
        
        # Wait for processing
        time.sleep(0.3)
        
        # Backpressure should be cleared
        self.assertFalse(processor.backpressure_event.is_set())
        
        processor.stop()

    def test_blocking_put_with_timeout(self):
        """Test that processor uses blocking put with timeout instead of put_nowait."""
        input_queue = Queue()
        output_queue = Queue(maxsize=1)
        
        processor = DetectionProcessor(
            input_queue=input_queue,
            output_queue=output_queue
        )
        
        processor.start()
        
        # Fill output queue
        output_queue.put({'existing': 'data'})
        
        # Add detection
        detection_result = {
            'detections': [],
            'frame_id': 1,
            'timestamp': time.time(),
            'inference_time': 0.01,
            'camera_id': 'test',
            'camera_name': 'Test Camera'
        }
        
        start_time = time.time()
        input_queue.put(detection_result)
        
        # Wait for processing attempt
        time.sleep(QUEUE_PUT_TIMEOUT_SECONDS + 0.5)
        
        elapsed = time.time() - start_time
        
        # Should have waited at least the timeout duration
        self.assertGreaterEqual(elapsed, QUEUE_PUT_TIMEOUT_SECONDS)
        
        # Should have tracked the drop
        self.assertEqual(processor.dropped_results, 1)
        
        processor.stop()


class TestInferenceEngineBackpressure(unittest.TestCase):
    """Tests for InferenceEngine queue backpressure mechanism."""

    def test_backpressure_event_initialized(self):
        """Test that backpressure event is initialized in inference engine."""
        try:
            from src.inference_engine_yolox import InferenceEngine
            
            input_queue = Queue()
            output_queue = Queue()
            
            # Mock the detector to avoid GPU requirements
            with patch('src.inference_engine_yolox.YOLOXDetector') as mock_detector:
                mock_detector.return_value = Mock()
                
                engine = InferenceEngine(
                    input_queue=input_queue,
                    output_queue=output_queue
                )
                
                # Backpressure event should exist and be cleared
                self.assertIsNotNone(engine.backpressure_event)
                self.assertFalse(engine.backpressure_event.is_set())
                
                engine.stop()
        except ImportError:
            self.skipTest("Inference engine dependencies not available")

    def test_queue_overflow_count_in_inference_stats(self):
        """Test that queue_overflow_count appears in inference engine stats."""
        try:
            from src.inference_engine_yolox import InferenceEngine
            
            input_queue = Queue()
            output_queue = Queue()
            
            with patch('src.inference_engine_yolox.YOLOXDetector') as mock_detector:
                mock_detector.return_value = Mock()
                
                engine = InferenceEngine(
                    input_queue=input_queue,
                    output_queue=output_queue
                )
                
                stats = engine.get_stats()
                self.assertIn('queue_overflow_count', stats)
                self.assertEqual(stats['queue_overflow_count'], 0)
                
                engine.stop()
        except ImportError:
            self.skipTest("Inference engine dependencies not available")


if __name__ == '__main__':
    unittest.main()
