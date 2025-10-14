#!/usr/bin/env python3
"""
Backyard Computer Vision System - Main Application
Integrates all components for real-time detection monitoring.
"""

import sys
import time
import logging
import signal
from pathlib import Path
from queue import Queue
from typing import Optional
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

        # Components (now lists for multi-camera support)
        self.stream_captures = []  # List of RTSPStreamCapture instances
        self.inference_engines = []  # List of InferenceEngine instances
        self.detection_processors = []  # List of DetectionProcessor instances
        self.web_server = None

        # Queues for inter-component communication (per-camera)
        self.frame_queues = []  # List of frame queues
        self.inference_queues = []  # List of inference queues
        self.detection_queue = None  # Single queue for all detections

        # Shutdown flag
        self.shutdown_requested = False

    def load_credentials(self) -> dict:
        """
        Load camera credentials from separate YAML file.

        Returns:
            Dictionary with camera credentials, or empty dict if file not found
        """
        credentials_file = Path("camera_credentials.yaml")
        if not credentials_file.exists():
            logger.error(f"Credentials file not found: {credentials_file}")
            logger.error("Please copy camera_credentials.example.yaml to camera_credentials.yaml and fill in your credentials")
            return {}

        try:
            with open(credentials_file, 'r') as f:
                credentials = yaml.safe_load(f)
            logger.info("Camera credentials loaded successfully")
            return credentials
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            return {}

    def load_config(self) -> bool:
        """
        Load configuration from YAML file and merge with credentials.

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

            # Load and merge credentials
            credentials = self.load_credentials()
            if not credentials or 'cameras' not in credentials:
                logger.error("Failed to load camera credentials")
                return False

            # Merge credentials into camera configs
            for camera in self.config.get('cameras', []):
                camera_id = camera.get('id')
                if camera_id in credentials['cameras']:
                    camera['username'] = credentials['cameras'][camera_id]['username']
                    camera['password'] = credentials['cameras'][camera_id]['password']
                else:
                    logger.error(f"No credentials found for camera: {camera_id}")
                    return False

            logger.info("Camera credentials merged successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    def validate_config(self) -> bool:
        """
        Validate configuration for required fields and value ranges.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            logger.info("Validating configuration...")

            # Validate cameras array exists
            if 'cameras' not in self.config or not self.config['cameras']:
                logger.error("Configuration validation failed: 'cameras' array is missing or empty")
                return False

            # Validate each camera configuration
            for camera in self.config['cameras']:
                camera_id = camera.get('id', 'unknown')

                # Required fields
                required_fields = ['id', 'name', 'ip']
                for field in required_fields:
                    if field not in camera:
                        logger.error(f"Camera '{camera_id}' is missing required field: '{field}'")
                        return False

                # Validate IP address format (basic check)
                ip = camera['ip']
                if not ip or not isinstance(ip, str):
                    logger.error(f"Camera '{camera_id}' has invalid IP address: {ip}")
                    return False

            # Validate web server port
            web_config = self.config.get('web', {})
            port = web_config.get('port', 8000)
            if not isinstance(port, int) or port < 1 or port > 65535:
                logger.error(f"Invalid web server port: {port} (must be 1-65535)")
                return False

            # Validate detection configuration
            detection_config = self.config.get('detection', {})
            conf_threshold = detection_config.get('conf_threshold', 0.25)
            if not isinstance(conf_threshold, (int, float)) or conf_threshold < 0.0 or conf_threshold > 1.0:
                logger.error(f"Invalid conf_threshold: {conf_threshold} (must be 0.0-1.0)")
                return False

            # Validate min_box_area
            min_box_area = detection_config.get('min_box_area', 0)
            if not isinstance(min_box_area, int) or min_box_area < 0:
                logger.error(f"Invalid min_box_area: {min_box_area} (must be >= 0)")
                return False

            logger.info("✓ Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    def validate_model_files(self) -> bool:
        """
        Validate that required model files exist before loading.

        Returns:
            True if all required model files exist, False otherwise
        """
        try:
            logger.info("Validating model files...")

            detection_config = self.config.get('detection', {})
            model_config = detection_config.get('model', {})

            # Check YOLOX model weights
            model_path = model_config.get('weights', 'models/yolox/yolox_s.pth')
            model_file = Path(model_path)

            if not model_file.exists():
                logger.error(f"❌ Model weights file not found: {model_path}")
                logger.error(f"   Absolute path: {model_file.absolute()}")
                logger.error(f"   Please download YOLOX weights:")
                logger.error(f"   wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth")
                logger.error(f"   mv yolox_s.pth {model_path}")
                return False

            logger.info(f"✓ YOLOX model weights found: {model_path}")

            # Check taxonomy file if Stage 2 is enabled
            if detection_config.get('use_two_stage', False):
                species_config = self.config.get('species_classification', {})
                inat_config = species_config.get('inat_classifier', {})
                taxonomy_file = inat_config.get('taxonomy_file', 'models/inat2021_taxonomy.json')
                taxonomy_path = Path(taxonomy_file)

                if not taxonomy_path.exists():
                    logger.error(f"❌ Taxonomy file not found: {taxonomy_file}")
                    logger.error(f"   Absolute path: {taxonomy_path.absolute()}")
                    logger.error(f"   Please download iNaturalist taxonomy:")
                    logger.error(f"   See docs/features/STAGE2_SETUP.md for instructions")
                    return False

                logger.info(f"✓ Taxonomy file found: {taxonomy_file}")

                # Check Real-ESRGAN model if enhancement is enabled
                enhancement_config = species_config.get('enhancement', {})
                if enhancement_config.get('enabled', False) and enhancement_config.get('method', '') == 'realesrgan':
                    realesrgan_config = enhancement_config.get('realesrgan', {})
                    model_path_esrgan = realesrgan_config.get('model_path', 'models/enhancement/RealESRGAN_x4plus.pth')
                    esrgan_path = Path(model_path_esrgan)

                    if not esrgan_path.exists():
                        logger.warning(f"⚠️  Real-ESRGAN model not found: {model_path_esrgan}")
                        logger.warning(f"   Stage 2 enhancement will be disabled")
                        logger.warning(f"   Download from: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
                    else:
                        logger.info(f"✓ Real-ESRGAN model found: {model_path_esrgan}")

            logger.info("✓ Model file validation passed")
            return True

        except Exception as e:
            logger.error(f"Model file validation error: {e}")
            return False

    def _create_shared_queue(self) -> Queue:
        """
        Create shared detection queue for all cameras.

        Returns:
            Queue instance for detections
        """
        detection_queue_size = self.config.get('performance', {}).get('detection_queue_size', 10)
        return Queue(maxsize=detection_queue_size)

    def _get_enabled_cameras(self) -> list:
        """
        Get list of enabled cameras from configuration.

        Returns:
            List of enabled camera configurations

        Raises:
            ValueError: If no cameras are configured or enabled
        """
        cameras_config = self.config.get('cameras', [])
        if not cameras_config:
            raise ValueError('No cameras configured in config.yaml. Expected "cameras" array with at least one camera entry.')

        enabled_cameras = [cam for cam in cameras_config if cam.get('enabled', True)]
        if not enabled_cameras:
            raise ValueError("No enabled cameras found in configuration")

        return enabled_cameras

    def _initialize_snapshot_saver(self) -> Optional[SnapshotSaver]:
        """
        Initialize shared snapshot saver if enabled in configuration.

        Returns:
            SnapshotSaver instance if enabled, None otherwise
        """
        snapshot_config = self.config.get('snapshots', {})

        if not snapshot_config.get('enabled', False):
            return None

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
        logger.info("Shared snapshot saver initialized")
        return snapshot_saver

    def _create_per_camera_queues(self) -> tuple:
        """
        Create per-camera queues for frame and inference data.

        Returns:
            Tuple of (frame_queue, inference_queue)
        """
        frame_queue_size = self.config.get('performance', {}).get('frame_queue_size', 2)
        detection_queue_size = self.config.get('performance', {}).get('detection_queue_size', 10)

        frame_queue = Queue(maxsize=frame_queue_size)
        inference_queue = Queue(maxsize=detection_queue_size)

        return frame_queue, inference_queue

    def _create_stream_capture(self, camera_config: dict, frame_queue: Queue) -> RTSPStreamCapture:
        """
        Create RTSP stream capture instance for a camera.

        Args:
            camera_config: Camera configuration dictionary
            frame_queue: Queue for captured frames

        Returns:
            RTSPStreamCapture instance
        """
        camera_id = camera_config.get('id', 'default')
        camera_name = camera_config.get('name', 'Default Camera')

        # Build RTSP URL
        rtsp_url = create_rtsp_url(
            camera_ip=camera_config['ip'],
            username=camera_config['username'],
            password=camera_config['password'],
            stream_type=camera_config.get('stream', 'main'),
            protocol=camera_config.get('protocol', 'rtsp'),
            camera_id=camera_id
        )

        # Determine if TCP transport should be used
        use_tcp = camera_config.get('protocol', 'rtsp').lower() == 'rtsp-tcp'

        stream_capture = RTSPStreamCapture(
            rtsp_url=rtsp_url,
            frame_queue=frame_queue,
            target_width=camera_config.get('target_width', 1280),
            target_height=camera_config.get('target_height', 720),
            buffer_size=camera_config.get('buffer_size', 1),
            camera_id=camera_id,
            camera_name=camera_name,
            use_tcp=use_tcp
        )

        logger.info(f"  [{camera_id}] Stream capture initialized")
        return stream_capture

    def _merge_camera_detection_config(self, camera_config: dict, detection_config: dict) -> dict:
        """
        Merge camera-specific detection overrides with global settings.

        Args:
            camera_config: Camera configuration dictionary
            detection_config: Global detection configuration

        Returns:
            Merged detection configuration for this camera
        """
        camera_id = camera_config.get('id', 'default')
        camera_detection_config = detection_config.copy()
        camera_overrides = camera_config.get('detection_overrides', {})

        if not camera_overrides:
            return camera_detection_config

        logger.info(f"  [{camera_id}] Applying per-camera detection overrides")

        # Override scalar values
        for key in ['conf_threshold', 'min_box_area', 'max_detections', 'nms_threshold']:
            if key in camera_overrides:
                camera_detection_config[key] = camera_overrides[key]
                logger.info(f"    {key}: {camera_overrides[key]}")

        # Merge per-class confidence overrides (camera overrides take precedence)
        if 'class_confidence_overrides' in camera_overrides:
            merged_class_overrides = camera_detection_config.get('class_confidence_overrides', {}).copy()
            camera_class_overrides = camera_overrides['class_confidence_overrides']
            merged_class_overrides.update(camera_class_overrides)
            camera_detection_config['class_confidence_overrides'] = merged_class_overrides
            logger.info(f"    class_confidence_overrides: {camera_class_overrides}")

        # Merge per-class size constraints (camera overrides take precedence)
        if 'class_size_constraints' in camera_overrides:
            merged_size_constraints = camera_detection_config.get('class_size_constraints', {}).copy()
            camera_size_constraints = camera_overrides['class_size_constraints']
            merged_size_constraints.update(camera_size_constraints)
            camera_detection_config['class_size_constraints'] = merged_size_constraints
            logger.info(f"    class_size_constraints: {camera_size_constraints}")

        return camera_detection_config

    def _initialize_two_stage_pipeline(self, camera_config: dict, camera_detection_config: dict) -> Optional[TwoStageDetectionPipeline]:
        """
        Initialize two-stage detection pipeline for a camera.

        Args:
            camera_config: Camera configuration dictionary
            camera_detection_config: Merged detection configuration for this camera

        Returns:
            TwoStageDetectionPipeline instance if successful, None otherwise
        """
        camera_id = camera_config.get('id', 'default')
        logger.info(f"  [{camera_id}] Initializing Stage 2 pipeline...")

        # Get species classification config
        species_config = self.config.get('species_classification', {})
        inat_config = species_config.get('inat_classifier', {})
        enhancement_config = species_config.get('enhancement', {})

        # Pass device to enhancement config if not specified
        if enhancement_config and 'device' not in enhancement_config:
            enhancement_config['device'] = camera_detection_config['device']

        # Get preprocessing config (camera-specific overrides global)
        global_preprocessing = species_config.get('preprocessing', {})
        camera_preprocessing = camera_config.get('stage2_preprocessing', {})

        # Merge preprocessing configs (camera overrides global)
        crop_padding_percent = camera_preprocessing.get('crop_padding_percent',
                                                        global_preprocessing.get('crop_padding_percent', 20))
        min_crop_size = camera_preprocessing.get('min_crop_size',
                                                global_preprocessing.get('min_crop_size', 64))

        if camera_preprocessing:
            logger.info(f"  [{camera_id}] Stage 2 preprocessing: padding={crop_padding_percent}%, min_size={min_crop_size}px")

        # Initialize pipeline for this camera
        camera_two_stage_pipeline = TwoStageDetectionPipeline(
            enable_species_classification=camera_detection_config.get('enable_species_classification', True),
            stage2_confidence_threshold=species_config.get('confidence_threshold', 0.3),
            device=camera_detection_config['device'],
            enhancement_config=enhancement_config if enhancement_config else None,
            crop_padding_percent=crop_padding_percent,
            min_crop_size=min_crop_size
        )

        # Initialize iNaturalist species classifier for this camera
        if camera_detection_config.get('enable_species_classification', False):
            model_name = inat_config.get('model_name', 'eva02_large_patch14_clip_336.merged2b_ft_inat21')
            taxonomy_file = inat_config.get('taxonomy_file', 'models/inat2021_taxonomy.json')
            input_size_inat = inat_config.get('input_size', 336)
            use_hierarchical = inat_config.get('use_hierarchical', True)

            # Get geographic filter settings
            geo_filter_config = species_config.get('geographic_filter', {})
            enable_geo_filter = geo_filter_config.get('enabled', False)
            allowed_species = geo_filter_config.get('allowed_species', [])

            # Create classifier instance for this camera
            inat_classifier = SpeciesClassifier(
                model_name=model_name,
                checkpoint_path=None,  # Use pretrained from timm
                device=camera_detection_config['device'],
                confidence_threshold=inat_config.get('confidence_threshold', 0.3),
                taxonomy_file=taxonomy_file,
                input_size=input_size_inat,
                use_hierarchical=use_hierarchical,
                allowed_species=allowed_species,
                enable_geographic_filter=enable_geo_filter
            )

            # Load the model (10,000 classes)
            if inat_classifier.load_model(num_classes=10000):
                # Add classifier for all animal groups
                # iNaturalist covers all species, so we use one universal classifier
                camera_two_stage_pipeline.add_species_classifier('bird', inat_classifier)
                camera_two_stage_pipeline.add_species_classifier('mammal', inat_classifier)
                camera_two_stage_pipeline.add_species_classifier('reptile', inat_classifier)
                logger.info(f"  [{camera_id}] ✅ iNaturalist classifier loaded")
            else:
                logger.warning(f"  [{camera_id}] Failed to load iNaturalist classifier, Stage 2 disabled")
                return None

        return camera_two_stage_pipeline

    def _create_inference_engine(self, camera_config: dict, camera_detection_config: dict,
                                  frame_queue: Queue, inference_queue: Queue,
                                  two_stage_pipeline: Optional[TwoStageDetectionPipeline]) -> InferenceEngine:
        """
        Create YOLOX inference engine for a camera.

        Args:
            camera_config: Camera configuration dictionary
            camera_detection_config: Merged detection configuration for this camera
            frame_queue: Input queue for frames
            inference_queue: Output queue for inference results
            two_stage_pipeline: Two-stage pipeline instance (or None)

        Returns:
            InferenceEngine instance
        """
        camera_id = camera_config.get('id', 'default')
        model_config_dict = self.config.get('detection', {}).get('model', {})
        input_size = camera_detection_config.get('input_size', [640, 640])
        input_size_tuple = tuple(input_size)

        inference_engine = InferenceEngine(
            model_name=model_config_dict.get('name', 'yolox-s'),
            model_path=model_config_dict.get('weights', 'models/yolox/yolox_s.pth'),
            device=camera_detection_config['device'],
            conf_threshold=camera_detection_config.get('conf_threshold', 0.25),
            nms_threshold=camera_detection_config.get('nms_threshold', 0.45),
            input_size=input_size_tuple,
            input_queue=frame_queue,
            output_queue=inference_queue,
            min_box_area=camera_detection_config.get('min_box_area', 0),
            max_det=camera_detection_config.get('max_detections', 300),
            use_two_stage=two_stage_pipeline is not None,
            two_stage_pipeline=two_stage_pipeline,
            class_confidence_overrides=camera_detection_config.get('class_confidence_overrides', {}),
            class_size_constraints=camera_detection_config.get('class_size_constraints', {}),
            wildlife_only=camera_detection_config.get('wildlife_only', True)
        )

        logger.info(f"  [{camera_id}] YOLOX inference engine initialized")
        return inference_engine

    def _create_detection_processor(self, camera_config: dict, inference_queue: Queue,
                                     stream_capture: RTSPStreamCapture,
                                     snapshot_saver: Optional[SnapshotSaver]) -> DetectionProcessor:
        """
        Create detection processor for a camera.

        Args:
            camera_config: Camera configuration dictionary
            inference_queue: Input queue for inference results
            stream_capture: Stream capture instance for frame access
            snapshot_saver: Shared snapshot saver instance (or None)

        Returns:
            DetectionProcessor instance
        """
        camera_id = camera_config.get('id', 'default')
        history_size = self.config.get('performance', {}).get('history_size', 30)

        # Get motion filter configuration
        motion_filter_config = self.config.get('motion_filter', {})
        enable_motion_filter = motion_filter_config.get('enabled', False)
        motion_filter_params = {k: v for k, v in motion_filter_config.items() if k != 'enabled'}

        # Get time-of-day filter configuration
        time_of_day_filter_config = self.config.get('time_of_day_filter', {})
        enable_time_of_day_filter = time_of_day_filter_config.get('enabled', False)
        time_of_day_filter_params = {k: v for k, v in time_of_day_filter_config.items() if k != 'enabled'}

        detection_processor = DetectionProcessor(
            input_queue=inference_queue,
            output_queue=self.detection_queue,  # All cameras write to shared detection queue
            detection_history_size=history_size,
            snapshot_saver=snapshot_saver,  # Shared snapshot saver
            frame_source=stream_capture,
            enable_motion_filter=enable_motion_filter,
            motion_filter_config=motion_filter_params,
            enable_time_of_day_filter=enable_time_of_day_filter,
            time_of_day_filter_config=time_of_day_filter_params
        )

        logger.info(f"  [{camera_id}] Detection processor initialized")
        return detection_processor

    def _initialize_camera_pipeline(self, camera_config: dict, snapshot_saver: Optional[SnapshotSaver]):
        """
        Initialize complete processing pipeline for a single camera.

        Args:
            camera_config: Camera configuration dictionary
            snapshot_saver: Shared snapshot saver instance (or None)
        """
        camera_id = camera_config.get('id', 'default')
        camera_name = camera_config.get('name', 'Default Camera')

        logger.info(f"Setting up camera: {camera_name} (ID: {camera_id})")

        # Create per-camera queues
        frame_queue, inference_queue = self._create_per_camera_queues()
        self.frame_queues.append(frame_queue)
        self.inference_queues.append(inference_queue)

        # Initialize stream capture
        stream_capture = self._create_stream_capture(camera_config, frame_queue)
        self.stream_captures.append(stream_capture)

        # Get detection config and merge camera-specific overrides
        detection_config = self.config.get('detection', {})
        camera_detection_config = self._merge_camera_detection_config(camera_config, detection_config)

        # Initialize two-stage pipeline if enabled
        use_two_stage = detection_config.get('use_two_stage', False)
        two_stage_pipeline = None
        if use_two_stage:
            two_stage_pipeline = self._initialize_two_stage_pipeline(camera_config, camera_detection_config)

        # Initialize inference engine
        inference_engine = self._create_inference_engine(
            camera_config, camera_detection_config,
            frame_queue, inference_queue, two_stage_pipeline
        )
        self.inference_engines.append(inference_engine)

        # Initialize detection processor
        detection_processor = self._create_detection_processor(
            camera_config, inference_queue, stream_capture, snapshot_saver
        )
        self.detection_processors.append(detection_processor)

    def _initialize_web_server(self):
        """Initialize web server for UI and API."""
        self.web_config = self.config.get('web', {})

        self.web_server = WebServer(
            detection_queue=self.detection_queue,
            frame_sources=self.stream_captures,
            host=self.web_config.get('host', '0.0.0.0'),
            port=self.web_config.get('port', 8000)
        )

        logger.info("Web server initialized")

    def initialize_components(self) -> bool:
        """
        Initialize all system components.

        Returns:
            True if all components initialized successfully, False otherwise
        """
        try:
            # Create shared detection queue
            self.detection_queue = self._create_shared_queue()
            logger.info("Shared detection queue created")

            # Get enabled cameras
            enabled_cameras = self._get_enabled_cameras()
            logger.info(f"Initializing {len(enabled_cameras)} camera(s)...")

            # Log detection pipeline information
            detection_config = self.config.get('detection', {})
            use_two_stage = detection_config.get('use_two_stage', False)

            if use_two_stage:
                logger.info("Per-camera two-stage detection pipeline enabled (YOLOX + iNaturalist)")
                logger.info("Each camera will have its own Stage 2 pipeline for thread-safe parallel processing")

            # Log YOLOX inference parameters
            input_size = detection_config.get('input_size', [640, 640])
            input_size_tuple = tuple(input_size)

            logger.info(f"Using YOLOX (Apache 2.0) for Stage 1 detection")
            if input_size_tuple == (640, 640):
                logger.info(f"  Expected inference time: 11-21ms (47x faster than GroundingDINO)")
            elif input_size_tuple == (1280, 1280):
                logger.info(f"  Expected inference time: 50-100ms (better for small/distant wildlife)")
            elif input_size_tuple == (1920, 1920):
                logger.info(f"  Expected inference time: 150-250ms (maximum detail for tiny IR wildlife)")
            else:
                logger.info(f"  Input size: {input_size_tuple}")

            # Initialize shared snapshot saver
            snapshot_saver = self._initialize_snapshot_saver()

            # Initialize camera pipelines
            for camera_config in enabled_cameras:
                self._initialize_camera_pipeline(camera_config, snapshot_saver)

            logger.info(f"All {len(enabled_cameras)} camera pipeline(s) initialized successfully")

            # Initialize web server
            self._initialize_web_server()

            return True

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False

    def _validate_active_cameras(self, active_cameras: list, component_name: str) -> bool:
        """
        Validate that at least one camera is active.

        Args:
            active_cameras: List of active camera indices
            component_name: Name of component for error message

        Returns:
            True if cameras are active, False if none remain
        """
        if not active_cameras:
            logger.error(f"No {component_name} started successfully - cannot continue")
            return False
        return True

    def start(self) -> bool:
        """
        Start all system components.
        Fault-tolerant: continues running even if some cameras fail.

        Returns:
            True if at least one camera started successfully, False otherwise
        """
        try:
            logger.info("Starting Backyard Computer Vision System...")

            # Track which cameras successfully start
            active_cameras = []

            # Start all stream captures (don't fail if one doesn't connect)
            for i, stream_capture in enumerate(self.stream_captures):
                camera_id = stream_capture.camera_id
                camera_name = stream_capture.camera_name

                logger.info(f"Starting stream capture for {camera_name} (ID: {camera_id})...")
                if stream_capture.start():
                    active_cameras.append(i)
                    logger.info(f"  [{camera_id}] ✓ Stream capture started")
                else:
                    logger.warning(f"  [{camera_id}] ✗ Failed to start stream capture - camera will be skipped")

            if not self._validate_active_cameras(active_cameras, "cameras"):
                return False

            logger.info(f"{len(active_cameras)}/{len(self.stream_captures)} camera(s) started successfully")

            # Start inference engines only for active cameras
            failed_inference = []
            for i in active_cameras:
                inference_engine = self.inference_engines[i]
                camera_id = self.stream_captures[i].camera_id

                if not inference_engine.start():
                    logger.warning(f"  [{camera_id}] ✗ Failed to start inference engine - camera disabled")
                    # Stop this camera's stream capture
                    self.stream_captures[i].stop()
                    failed_inference.append(i)

            # Remove failed cameras
            active_cameras = [i for i in active_cameras if i not in failed_inference]

            if not self._validate_active_cameras(active_cameras, "inference engines"):
                return False

            logger.info(f"{len(active_cameras)} inference engine(s) started successfully")

            # Start detection processors only for active cameras
            failed_processors = []
            for i in active_cameras:
                detection_processor = self.detection_processors[i]
                camera_id = self.stream_captures[i].camera_id

                if not detection_processor.start():
                    logger.warning(f"  [{camera_id}] ✗ Failed to start detection processor - camera disabled")
                    # Stop this camera's inference and stream
                    self.inference_engines[i].stop()
                    self.stream_captures[i].stop()
                    failed_processors.append(i)

            # Remove failed cameras
            active_cameras = [i for i in active_cameras if i not in failed_processors]

            if not self._validate_active_cameras(active_cameras, "detection processors"):
                return False

            logger.info(f"{len(active_cameras)} detection processor(s) started successfully")

            # Start web server (blocking)
            host = self.web_config.get('host', '0.0.0.0')
            port = self.web_config.get('port', 8000)
            logger.info(f"Starting web server on http://{host}:{port}")
            logger.info("=" * 80)
            logger.info("System is running!")
            logger.info(f"Open browser to: http://localhost:{port}")
            logger.info(f"Monitoring {len(active_cameras)} camera(s) (out of {len(self.stream_captures)} configured)")

            # List active cameras
            for i in active_cameras:
                cam_name = self.stream_captures[i].camera_name
                cam_id = self.stream_captures[i].camera_id
                logger.info(f"  ✓ {cam_name} (ID: {cam_id})")

            # List failed cameras
            failed_cameras = [idx for idx in range(len(self.stream_captures)) if idx not in active_cameras]
            if failed_cameras:
                logger.warning(f"{len(failed_cameras)} camera(s) failed to start:")
                for i in failed_cameras:
                    cam_name = self.stream_captures[i].camera_name
                    cam_id = self.stream_captures[i].camera_id
                    logger.warning(f"  ✗ {cam_name} (ID: {cam_id})")

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
        logger.info("Stopping Backyard Computer Vision System...")

        # Stop all detection processors
        for detection_processor in self.detection_processors:
            if detection_processor:
                detection_processor.stop()

        # Stop all inference engines
        for inference_engine in self.inference_engines:
            if inference_engine:
                inference_engine.stop()

        # Stop all stream captures
        for stream_capture in self.stream_captures:
            if stream_capture:
                stream_capture.stop()

        logger.info("System stopped")

    def print_stats(self):
        """Print system statistics."""
        logger.info("=" * 80)
        logger.info("System Statistics:")

        # Print stats for each camera
        for i, stream_capture in enumerate(self.stream_captures):
            if stream_capture:
                stats = stream_capture.get_stats()
                logger.info(f"  Camera {i} ({stream_capture.camera_name}):")
                logger.info(f"    Stream Capture:")
                logger.info(f"      - Connected: {stats['is_connected']}")
                logger.info(f"      - FPS: {stats['fps']:.1f}")
                logger.info(f"      - Dropped frames: {stats['dropped_frames']}")

        for i, inference_engine in enumerate(self.inference_engines):
            if inference_engine:
                stats = inference_engine.get_stats()
                logger.info(f"    Inference Engine:")
                logger.info(f"      - Device: {stats['device']}")
                logger.info(f"      - FPS: {stats['fps']:.1f}")
                logger.info(f"      - Avg inference time: {stats['avg_inference_time_ms']:.1f}ms")

        for i, detection_processor in enumerate(self.detection_processors):
            if detection_processor:
                stats = detection_processor.get_stats()
                logger.info(f"    Detection Processor:")
                logger.info(f"      - Processed: {stats['processed_count']}")
                logger.info(f"      - History size: {stats['history_size']}")

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

        # Validate configuration
        if not system.validate_config():
            logger.error("Configuration validation failed")
            sys.exit(1)

        # Validate model files
        if not system.validate_model_files():
            logger.error("Model file validation failed")
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
