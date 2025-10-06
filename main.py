#!/usr/bin/env python3
"""
Telescope Detection System - Main Application
Integrates all components for real-time detection monitoring.
"""

import sys
import time
import logging
import signal
from pathlib import Path
from queue import Queue
import yaml

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stream_capture import RTSPStreamCapture, create_rtsp_url
from inference_engine_yolox import InferenceEngine  # YOLOX version (47x faster!)
from detection_processor import DetectionProcessor
from web_server import WebServer
from snapshot_saver import SnapshotSaver
from two_stage_pipeline_yolox import TwoStageDetectionPipeline  # YOLOX-compatible
from species_classifier import SpeciesClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TelescopeDetectionSystem:
    """
    Main application class that orchestrates all components.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the telescope detection system.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = None

        # Components
        self.stream_capture = None
        self.inference_engine = None
        self.detection_processor = None
        self.web_server = None

        # Queues for inter-component communication
        self.frame_queue = None
        self.inference_queue = None
        self.detection_queue = None

        # Shutdown flag
        self.shutdown_requested = False

    def load_config(self) -> bool:
        """
        Load configuration from YAML file.

        Returns:
            True if config loaded successfully, False otherwise
        """
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                return False

            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)

            logger.info("Configuration loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    def initialize_components(self) -> bool:
        """
        Initialize all system components.

        Returns:
            True if all components initialized successfully, False otherwise
        """
        try:
            # Create queues
            frame_queue_size = self.config.get('performance', {}).get('frame_queue_size', 2)
            detection_queue_size = self.config.get('performance', {}).get('detection_queue_size', 10)

            self.frame_queue = Queue(maxsize=frame_queue_size)
            self.inference_queue = Queue(maxsize=detection_queue_size)
            self.detection_queue = Queue(maxsize=detection_queue_size)

            logger.info("Queues created")

            # Initialize RTSP stream capture
            camera_config = self.config['camera']
            rtsp_url = create_rtsp_url(
                camera_ip=camera_config['ip'],
                username=camera_config['username'],
                password=camera_config['password'],
                stream_type=camera_config.get('stream', 'main')
            )

            self.stream_capture = RTSPStreamCapture(
                rtsp_url=rtsp_url,
                frame_queue=self.frame_queue,
                target_width=camera_config.get('target_width', 1280),
                target_height=camera_config.get('target_height', 720),
                buffer_size=camera_config.get('buffer_size', 1)
            )

            logger.info("Stream capture initialized")

            # Initialize two-stage pipeline (if enabled)
            detection_config = self.config['detection']
            two_stage_pipeline = None

            if detection_config.get('use_two_stage', False):
                logger.info("Initializing two-stage detection pipeline (YOLOX + iNaturalist)...")

                # Get species classification config
                species_config = self.config.get('species_classification', {})
                inat_config = species_config.get('inat_classifier', {})
                enhancement_config = species_config.get('enhancement', {})

                # Pass device to enhancement config if not specified
                if enhancement_config and 'device' not in enhancement_config:
                    enhancement_config['device'] = detection_config['device']

                # Initialize pipeline
                two_stage_pipeline = TwoStageDetectionPipeline(
                    enable_species_classification=detection_config.get('enable_species_classification', True),
                    stage2_confidence_threshold=species_config.get('confidence_threshold', 0.3),
                    device=detection_config['device'],
                    enhancement_config=enhancement_config if enhancement_config else None
                )

                # Initialize iNaturalist species classifier
                if detection_config.get('enable_species_classification', False):
                    logger.info("Loading iNaturalist species classifier...")

                    model_name = inat_config.get('model_name', 'eva02_large_patch14_clip_336.merged2b_ft_inat21')
                    taxonomy_file = inat_config.get('taxonomy_file', 'models/inat2021_taxonomy.json')
                    input_size = inat_config.get('input_size', 336)
                    use_hierarchical = inat_config.get('use_hierarchical', True)

                    # Create universal classifier (handles all animals)
                    inat_classifier = SpeciesClassifier(
                        model_name=model_name,
                        checkpoint_path=None,  # Use pretrained from timm
                        device=detection_config['device'],
                        confidence_threshold=inat_config.get('confidence_threshold', 0.3),
                        taxonomy_file=taxonomy_file,
                        input_size=input_size,
                        use_hierarchical=use_hierarchical
                    )

                    # Load the model (10,000 classes)
                    if inat_classifier.load_model(num_classes=10000):
                        # Add classifier for all animal groups
                        # iNaturalist covers all species, so we use one universal classifier
                        two_stage_pipeline.add_species_classifier('bird', inat_classifier)
                        two_stage_pipeline.add_species_classifier('mammal', inat_classifier)
                        two_stage_pipeline.add_species_classifier('reptile', inat_classifier)
                        logger.info(f"✅ iNaturalist classifier loaded ({model_name})")
                        logger.info(f"   Taxonomy: {taxonomy_file}")
                        logger.info(f"   Species count: 10,000")
                    else:
                        logger.warning("Failed to load iNaturalist classifier, Stage 2 disabled")
                        two_stage_pipeline = None

                logger.info("Two-stage pipeline initialized")

            # Initialize YOLOX inference engine (Stage 1: Fast detection)
            model_config_dict = detection_config.get('model', {})

            logger.info(f"Using YOLOX (Apache 2.0) for Stage 1 detection")
            logger.info(f"  Expected inference time: 11-21ms (47x faster than GroundingDINO)")

            self.inference_engine = InferenceEngine(
                model_name=model_config_dict.get('name', 'yolox-s'),
                model_path=model_config_dict.get('weights', 'models/yolox/yolox_s.pth'),
                device=detection_config['device'],
                conf_threshold=detection_config.get('conf_threshold', 0.25),
                nms_threshold=detection_config.get('nms_threshold', 0.45),
                input_queue=self.frame_queue,
                output_queue=self.inference_queue,
                min_box_area=detection_config.get('min_box_area', 0),
                max_det=detection_config.get('max_detections', 300),
                use_two_stage=two_stage_pipeline is not None,
                two_stage_pipeline=two_stage_pipeline,
                class_confidence_overrides=detection_config.get('class_confidence_overrides', {}),
                wildlife_only=detection_config.get('wildlife_only', True)
            )

            logger.info("YOLOX inference engine initialized")

            # Initialize snapshot saver (if enabled)
            snapshot_config = self.config.get('snapshots', {})
            snapshot_saver = None

            if snapshot_config.get('enabled', False):
                snapshot_saver = SnapshotSaver(
                    output_dir=snapshot_config.get('output_dir', 'clips'),
                    save_mode=snapshot_config.get('save_mode', 'image'),
                    trigger_classes=snapshot_config.get('trigger_classes'),
                    min_confidence=snapshot_config.get('min_confidence', 0.6),
                    cooldown_seconds=snapshot_config.get('cooldown_seconds', 30),
                    clip_duration=snapshot_config.get('clip_duration', 10),
                    pre_buffer_seconds=snapshot_config.get('pre_buffer_seconds', 5),
                    fps=30,
                    save_annotated=snapshot_config.get('save_annotated', True)
                )
                logger.info("Snapshot saver initialized")

            # Initialize detection processor
            history_size = self.config.get('performance', {}).get('history_size', 30)

            self.detection_processor = DetectionProcessor(
                input_queue=self.inference_queue,
                output_queue=self.detection_queue,
                detection_history_size=history_size,
                snapshot_saver=snapshot_saver,
                frame_source=self.stream_capture
            )

            logger.info("Detection processor initialized")

            # Initialize web server
            web_config = self.config['web']

            # Create a wrapper to provide latest frame to web server
            class FrameSource:
                def __init__(self, capture):
                    self.capture = capture
                    self.latest_frame = None

                def update(self):
                    # This would be called periodically to update latest frame
                    # For now, web server will get frames from video feed endpoint
                    pass

            self.web_server = WebServer(
                detection_queue=self.detection_queue,
                frame_source=self.stream_capture,  # Pass stream capture for video
                host=web_config.get('host', '0.0.0.0'),
                port=web_config.get('port', 8000)
            )

            logger.info("Web server initialized")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False

    def start(self) -> bool:
        """
        Start all system components.

        Returns:
            True if all components started successfully, False otherwise
        """
        try:
            logger.info("Starting Telescope Detection System...")

            # Start stream capture
            if not self.stream_capture.start():
                logger.error("Failed to start stream capture")
                return False

            logger.info("Stream capture started")

            # Start inference engine
            if not self.inference_engine.start():
                logger.error("Failed to start inference engine")
                self.stream_capture.stop()
                return False

            logger.info("Inference engine started")

            # Start detection processor
            if not self.detection_processor.start():
                logger.error("Failed to start detection processor")
                self.stream_capture.stop()
                self.inference_engine.stop()
                return False

            logger.info("Detection processor started")

            # Start web server (blocking)
            logger.info(f"Starting web server on http://{self.config['web']['host']}:{self.config['web']['port']}")
            logger.info("=" * 80)
            logger.info("System is running!")
            logger.info(f"Open browser to: http://localhost:{self.config['web']['port']}")
            logger.info("Press Ctrl+C to stop")
            logger.info("=" * 80)

            # Run web server in current thread (blocking)
            self.web_server.run()

            return True

        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self.stop()
            return False

    def stop(self):
        """Stop all system components gracefully."""
        logger.info("Stopping Telescope Detection System...")

        if self.detection_processor:
            self.detection_processor.stop()

        if self.inference_engine:
            self.inference_engine.stop()

        if self.stream_capture:
            self.stream_capture.stop()

        logger.info("System stopped")

    def print_stats(self):
        """Print system statistics."""
        logger.info("=" * 80)
        logger.info("System Statistics:")

        if self.stream_capture:
            stats = self.stream_capture.get_stats()
            logger.info(f"  Stream Capture:")
            logger.info(f"    - Connected: {stats['is_connected']}")
            logger.info(f"    - FPS: {stats['fps']:.1f}")
            logger.info(f"    - Dropped frames: {stats['dropped_frames']}")

        if self.inference_engine:
            stats = self.inference_engine.get_stats()
            logger.info(f"  Inference Engine:")
            logger.info(f"    - Device: {stats['device']}")
            logger.info(f"    - FPS: {stats['fps']:.1f}")
            logger.info(f"    - Avg inference time: {stats['avg_inference_time_ms']:.1f}ms")

        if self.detection_processor:
            stats = self.detection_processor.get_stats()
            logger.info(f"  Detection Processor:")
            logger.info(f"    - Processed: {stats['processed_count']}")
            logger.info(f"    - History size: {stats['history_size']}")

        logger.info("=" * 80)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Shutdown signal received")
    sys.exit(0)


def main():
    """Main entry point."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and start system
    system = TelescopeDetectionSystem()

    try:
        # Load configuration
        if not system.load_config():
            logger.error("Failed to load configuration")
            sys.exit(1)

        # Initialize components
        if not system.initialize_components():
            logger.error("Failed to initialize components")
            sys.exit(1)

        # Start system (blocking)
        if not system.start():
            logger.error("Failed to start system")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        system.stop()


if __name__ == "__main__":
    main()
