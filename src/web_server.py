"""
FastAPI Web Server with WebSocket Support
Serves web interface and streams detection results in real-time.
"""

import asyncio
import json
import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from queue import Empty

logger = logging.getLogger(__name__)


class WebServer:
    """
    Web server for backyard computer vision system.
    Provides WebSocket for detection data and HTTP for video streaming.
    """

    def __init__(
        self,
        detection_queue: Optional[Any] = None,
        frame_sources: Optional[List[Any]] = None,
        inference_engines: Optional[List[Any]] = None,
        detection_processors: Optional[List[Any]] = None,
        health_monitor: Optional[Any] = None,
        restart_callback: Optional[Any] = None,
        reload_config_callback: Optional[Any] = None,
        config_getter: Optional[Any] = None,
        host: str = "0.0.0.0",
        port: int = 8000
    ):
        """
        Initialize web server.

        Args:
            detection_queue: Queue to receive detection results from
            frame_sources: List of frame sources (RTSPStreamCapture instances)
            inference_engines: List of inference engines (InferenceEngine instances)
            detection_processors: List of detection processors (DetectionProcessor instances)
            health_monitor: Camera health monitor instance
            restart_callback: Callback function to restart a camera (takes camera index)
            reload_config_callback: Callback function to reload configuration
            config_getter: Callback function to get current configuration
            host: Host to bind to
            port: Port to bind to
        """
        self.detection_queue = detection_queue
        self.frame_sources = frame_sources if frame_sources else []
        self.inference_engines = inference_engines if inference_engines else []
        self.detection_processors = detection_processors if detection_processors else []
        self.health_monitor = health_monitor
        self.restart_callback = restart_callback
        self.reload_config_callback = reload_config_callback
        self.config_getter = config_getter
        self.host = host
        self.port = port

        self.app = FastAPI(title="Backyard Computer Vision System")
        self.active_connections: list[WebSocket] = []

        # Latest detections per camera
        self.latest_detections: Dict[str, Dict[str, Any]] = {}  # camera_id -> detection_result

        # Track camera start times for uptime calculation
        self.camera_start_times: Dict[str, float] = {}  # camera_id -> timestamp

        self._setup_routes()
        self._mount_static_files()

    def _mount_static_files(self):
        """Mount static file directories."""
        web_dir = Path(__file__).parent.parent / "web"
        if web_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        # Mount clips directory for browsing saved snapshots
        clips_dir = Path(__file__).parent.parent / "clips"
        if clips_dir.exists():
            self.app.mount("/clips", StaticFiles(directory=str(clips_dir)), name="clips")

    def _setup_routes(self):
        """Set up FastAPI routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve the main page."""
            web_dir = Path(__file__).parent.parent / "web"
            index_file = web_dir / "index.html"

            if index_file.exists():
                return index_file.read_text()
            else:
                return """
                <html>
                    <head><title>Telescope Detection System</title></head>
                    <body>
                        <h1>Telescope Detection System</h1>
                        <p>Web interface files not found. Please check web/ directory.</p>
                    </body>
                </html>
                """

        @self.app.get("/app.js")
        async def serve_app_js():
            """Serve app.js directly."""
            web_dir = Path(__file__).parent.parent / "web"
            js_file = web_dir / "app.js"
            if js_file.exists():
                return HTMLResponse(content=js_file.read_text(), media_type="application/javascript")
            return HTMLResponse(content="console.error('app.js not found');", status_code=404)

        @self.app.get("/clips_browser")
        async def clips_browser():
            """Serve the clips browser page."""
            web_dir = Path(__file__).parent.parent / "web"
            clips_file = web_dir / "clips.html"
            if clips_file.exists():
                return HTMLResponse(content=clips_file.read_text())
            return HTMLResponse(content="<h1>Clips browser not found</h1>", status_code=404)

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "ok",
                "timestamp": time.time(),
                "active_connections": len(self.active_connections)
            }

        @self.app.get("/cameras")
        async def cameras():
            """Get list of available cameras."""
            camera_list = []
            for frame_source in self.frame_sources:
                camera_list.append({
                    "id": frame_source.camera_id,
                    "name": frame_source.camera_name,
                    "is_connected": frame_source.is_connected,
                    "fps": frame_source.fps
                })
            return {"cameras": camera_list}

        @self.app.get("/stats")
        async def stats():
            """Get system statistics."""
            return {
                "active_connections": len(self.active_connections),
                "num_cameras": len(self.frame_sources),
                "cameras_with_detections": len(self.latest_detections),
                "last_detection_times": {
                    cam_id: det.get("timestamp")
                    for cam_id, det in self.latest_detections.items()
                }
            }

        def _create_fallback_health_data(self, camera_id: str, frame_source) -> dict:
            """
            Create basic health data from frame source when health monitor is unavailable.

            Args:
                camera_id: Camera ID
                frame_source: RTSPStreamCapture instance

            Returns:
                Dictionary with basic health information
            """
            # Calculate uptime
            uptime_seconds = 0
            if camera_id in self.camera_start_times:
                uptime_seconds = time.time() - self.camera_start_times[camera_id]

            # Get stream capture stats
            stream_stats = frame_source.get_stats()

            # Get last frame time from latest_frame timestamp
            last_frame_time = None
            if hasattr(frame_source, 'last_fps_check'):
                last_frame_time = frame_source.last_fps_check

            return {
                "camera_id": camera_id,
                "name": frame_source.camera_name,
                "is_connected": frame_source.is_connected,
                "fps": round(frame_source.fps, 1),
                "dropped_frames": stream_stats.get('dropped_frames', 0),
                "total_frames": stream_stats.get('total_frames', 0),
                "last_frame_time": last_frame_time,
                "uptime_seconds": round(uptime_seconds, 1)
            }

        @self.app.get("/api/cameras/{camera_id}/health")
        async def camera_health(camera_id: str):
            """Get health status for a specific camera."""
            # Use health monitor if available
            if self.health_monitor:
                health_data = self.health_monitor.get_camera_health(camera_id)
                if health_data:
                    return health_data
                else:
                    raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

            # Fallback to basic stats if health monitor not available
            # Find camera index
            camera_idx = None
            for idx, fs in enumerate(self.frame_sources):
                if fs.camera_id == camera_id:
                    camera_idx = idx
                    break

            if camera_idx is None:
                raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

            frame_source = self.frame_sources[camera_idx]
            return _create_fallback_health_data(camera_id, frame_source)

        @self.app.get("/api/cameras/{camera_id}/stats")
        async def camera_stats(camera_id: str):
            """Get detailed statistics for a specific camera."""
            # Find camera index
            camera_idx = None
            for idx, fs in enumerate(self.frame_sources):
                if fs.camera_id == camera_id:
                    camera_idx = idx
                    break

            if camera_idx is None:
                raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

            # Gather stats from all components
            result = {"camera_id": camera_id}

            # Stream capture stats
            if camera_idx < len(self.frame_sources):
                stream_stats = self.frame_sources[camera_idx].get_stats()
                result["capture"] = {
                    "is_connected": stream_stats.get('is_connected', False),
                    "fps": round(stream_stats.get('fps', 0), 1),
                    "total_frames": stream_stats.get('total_frames', 0),
                    "dropped_frames": stream_stats.get('dropped_frames', 0),
                    "queue_size": stream_stats.get('queue_size', 0)
                }

            # Inference engine stats
            if camera_idx < len(self.inference_engines):
                inference_stats = self.inference_engines[camera_idx].get_stats()
                result["inference"] = {
                    "device": inference_stats.get('device', 'unknown'),
                    "fps": round(inference_stats.get('fps', 0), 1),
                    "avg_inference_time_ms": round(inference_stats.get('avg_inference_time_ms', 0), 1),
                    "total_inferences": inference_stats.get('total_inferences', 0),
                    "dropped_results": inference_stats.get('dropped_results', 0),
                    "drop_rate": round(inference_stats.get('drop_rate', 0), 3)
                }

            # Detection processor stats
            if camera_idx < len(self.detection_processors):
                processor_stats = self.detection_processors[camera_idx].get_stats()
                result["detections"] = {
                    "total_processed": processor_stats.get('processed_count', 0),
                    "history_size": processor_stats.get('history_size', 0),
                    "last_detection_time": processor_stats.get('last_detection_time'),
                    "dropped_results": processor_stats.get('dropped_results', 0),
                    "drop_rate": round(processor_stats.get('drop_rate', 0), 3)
                }

                # Get detection counts by class from latest detection
                if camera_id in self.latest_detections:
                    result["detections"]["by_class"] = self.latest_detections[camera_id].get("detection_counts", {})
                else:
                    result["detections"]["by_class"] = {}

                # Filter stats
                result["filters"] = {
                    "motion_filter_enabled": processor_stats.get('motion_filter_enabled', False),
                    "time_of_day_filter_enabled": processor_stats.get('time_of_day_filter_enabled', False)
                }

            return result

        @self.app.post("/api/cameras/{camera_id}/restart")
        async def restart_camera(camera_id: str):
            """Manually trigger camera restart."""
            if not self.restart_callback:
                raise HTTPException(status_code=503, detail="Camera restart not available")

            # Find camera index
            camera_idx = None
            for idx, fs in enumerate(self.frame_sources):
                if fs.camera_id == camera_id:
                    camera_idx = idx
                    break

            if camera_idx is None:
                raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

            # Attempt restart (run in executor to avoid blocking)
            try:
                loop = asyncio.get_running_loop()
                success = await loop.run_in_executor(
                    None,
                    lambda: self.restart_callback(camera_idx)
                )

                if success:
                    return {
                        "success": True,
                        "camera_id": camera_id,
                        "message": f"Camera '{camera_id}' restarted successfully"
                    }
                else:
                    return {
                        "success": False,
                        "camera_id": camera_id,
                        "message": f"Failed to restart camera '{camera_id}'"
                    }
            except Exception as e:
                logger.error(f"Error restarting camera {camera_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Error restarting camera: {str(e)}")

        @self.app.get("/api/cameras/health/summary")
        async def cameras_health_summary():
            """Get health summary for all cameras."""
            if self.health_monitor:
                return self.health_monitor.get_health_summary()

            # Fallback if health monitor not available
            cameras = []
            for fs in self.frame_sources:
                cameras.append({
                    'id': fs.camera_id,
                    'name': fs.camera_name,
                    'status': 'healthy' if fs.is_connected else 'failed',
                    'health_score': 100 if fs.is_connected else 0
                })

            return {
                'total_cameras': len(self.frame_sources),
                'healthy': sum(1 for c in cameras if c['status'] == 'healthy'),
                'degraded': 0,
                'failed': sum(1 for c in cameras if c['status'] == 'failed'),
                'restarting': 0,
                'cameras': cameras
            }

        @self.app.get("/api/system/stats")
        async def system_stats():
            """Get system-wide statistics including GPU memory and queue depths."""
            result = {
                "timestamp": time.time(),
                "cameras": {
                    "total": len(self.frame_sources),
                    "connected": sum(1 for fs in self.frame_sources if fs.is_connected),
                    "active_detections": len(self.latest_detections)
                },
                "websocket": {
                    "active_connections": len(self.active_connections)
                },
                "queues": {},
                "gpu": {},
                "performance": {}
            }

            # Aggregate queue depths
            total_frame_queue_depth = 0
            total_inference_queue_depth = 0

            for idx, fs in enumerate(self.frame_sources):
                stream_stats = fs.get_stats()
                total_frame_queue_depth += stream_stats.get('queue_size', 0)

            result["queues"]["frame_queue_total_depth"] = total_frame_queue_depth
            result["queues"]["detection_queue_depth"] = self.detection_queue.qsize() if self.detection_queue else 0

            # GPU memory stats (aggregate from all inference engines)
            if self.inference_engines:
                try:
                    # Get GPU stats from first inference engine
                    first_engine_stats = self.inference_engines[0].get_stats()
                    result["gpu"]["memory_allocated_mb"] = round(first_engine_stats.get('gpu_memory_allocated_mb', 0), 1)
                    result["gpu"]["memory_reserved_mb"] = round(first_engine_stats.get('gpu_memory_reserved_mb', 0), 1)
                    result["gpu"]["device"] = first_engine_stats.get('device', 'unknown')
                except Exception as e:
                    logger.error(f"Error getting GPU stats: {e}")
                    result["gpu"]["error"] = str(e)

            # Aggregate performance metrics
            total_fps = 0
            total_inference_time = 0
            total_inferences = 0

            for engine in self.inference_engines:
                engine_stats = engine.get_stats()
                total_fps += engine_stats.get('fps', 0)
                total_inference_time += engine_stats.get('avg_inference_time_ms', 0)
                total_inferences += engine_stats.get('total_inferences', 0)

            if self.inference_engines:
                result["performance"]["avg_fps"] = round(total_fps / len(self.inference_engines), 1)
                result["performance"]["avg_inference_time_ms"] = round(total_inference_time / len(self.inference_engines), 1)

            result["performance"]["total_inferences"] = total_inferences

            # Per-camera summary
            result["cameras"]["per_camera"] = []
            for idx, fs in enumerate(self.frame_sources):
                cam_summary = {
                    "id": fs.camera_id,
                    "name": fs.camera_name,
                    "is_connected": fs.is_connected,
                    "fps": round(fs.fps, 1)
                }

                # Add last detection time
                if fs.camera_id in self.latest_detections:
                    cam_summary["last_detection_time"] = self.latest_detections[fs.camera_id].get("timestamp")

                result["cameras"]["per_camera"].append(cam_summary)

            return result

        @self.app.get("/clips_list")
        async def clips_list():
            """Get list of saved clips."""
            clips_dir = Path(__file__).parent.parent / "clips"
            if not clips_dir.exists():
                return {"clips": []}

            clips = []
            # Only look at non-annotated files to avoid duplicates
            for file_path in sorted(clips_dir.glob("*.jpg"), reverse=True):
                # Skip annotated versions
                if "_annotated" in file_path.name:
                    continue

                json_path = file_path.with_suffix('.json')
                metadata = None
                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass

                # Check if annotated version exists
                base_name = file_path.stem
                annotated_path = clips_dir / f"{base_name}_annotated.jpg"
                has_annotated = annotated_path.exists()

                clips.append({
                    "filename": file_path.name,
                    "url": f"/clips/{file_path.name}",
                    "annotated_url": f"/clips/{base_name}_annotated.jpg" if has_annotated else None,
                    "timestamp": file_path.stat().st_mtime,
                    "metadata": metadata
                })

            return {"clips": clips}

        @self.app.post("/api/config/reload")
        async def reload_config():
            """
            Reload configuration from file and apply hot-reloadable changes.
            Settings that require restart will be identified but not applied.
            """
            if not self.reload_config_callback:
                raise HTTPException(status_code=503, detail="Config reload not available")

            try:
                # Run reload in executor to avoid blocking
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    self.reload_config_callback
                )

                return result
            except Exception as e:
                logger.error(f"Error reloading config: {e}")
                raise HTTPException(status_code=500, detail=f"Error reloading config: {str(e)}")

        @self.app.get("/api/config/current")
        async def get_current_config():
            """
            Get current active configuration.
            Useful for debugging and verifying configuration state.
            """
            if not self.config_getter:
                raise HTTPException(status_code=503, detail="Config getter not available")

            try:
                config = self.config_getter()
                # Deep copy to avoid exposing internal references
                import copy
                return {"config": copy.deepcopy(config)}
            except Exception as e:
                logger.error(f"Error getting config: {e}")
                raise HTTPException(status_code=500, detail=f"Error getting config: {str(e)}")

        @self.app.websocket("/ws/detections")
        async def websocket_detections(websocket: WebSocket):
            """
            WebSocket endpoint for streaming detection results.
            """
            await websocket.accept()
            self.active_connections.append(websocket)
            logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

            try:
                # Send initial connection message
                await websocket.send_json({
                    "type": "connection",
                    "message": "Connected to backyard computer vision system",
                    "timestamp": time.time()
                })

                # Stream detection results
                while True:
                    # Run blocking queue.get() in thread executor to avoid blocking event loop
                    try:
                        loop = asyncio.get_running_loop()
                        detection_result = await loop.run_in_executor(
                            None,
                            lambda: self.detection_queue.get(timeout=1.0)
                        )

                        # Store latest detection for this camera
                        camera_id = detection_result.get('camera_id', 'default')
                        self.latest_detections[camera_id] = detection_result

                        # Prepare WebSocket message
                        message = self._prepare_detection_message(detection_result)

                        # Send to client
                        await websocket.send_json(message)

                    except Empty:
                        # No detection available, send heartbeat to keep connection alive
                        await websocket.send_json({
                            "type": "heartbeat",
                            "timestamp": time.time()
                        })
                        # Rate limit heartbeats to avoid overwhelming clients
                        await asyncio.sleep(1.0)

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
                logger.info(f"WebSocket removed. Total connections: {len(self.active_connections)}")

        @self.app.get("/video/feed/{camera_id}")
        async def video_feed(camera_id: str):
            """
            MJPEG video stream endpoint for a specific camera.
            Streams frames with detection overlays.
            """
            # Find the frame source for this camera
            frame_source = None
            for fs in self.frame_sources:
                if fs.camera_id == camera_id:
                    frame_source = fs
                    break

            if not frame_source:
                return {"error": f"Camera {camera_id} not found"}

            return StreamingResponse(
                self._generate_video_stream(camera_id),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )

        @self.app.get("/video/feed")
        async def video_feed_default():
            """
            MJPEG video stream endpoint (default camera).
            Streams frames with detection overlays from the first camera.
            """
            if not self.frame_sources:
                return {"error": "No cameras available"}

            # Use first camera as default
            camera_id = self.frame_sources[0].camera_id
            return StreamingResponse(
                self._generate_video_stream(camera_id),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )

    def _prepare_detection_message(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare detection result for WebSocket transmission.

        Args:
            detection_result: Raw detection result

        Returns:
            Formatted message for WebSocket
        """
        message = {
            "type": "detections",
            "camera_id": detection_result.get("camera_id", "default"),
            "camera_name": detection_result.get("camera_name", "Default Camera"),
            "frame_id": detection_result.get("frame_id"),
            "timestamp": detection_result.get("timestamp"),
            "processing_timestamp": detection_result.get("processing_timestamp"),
            "latency_ms": detection_result.get("total_latency_ms", 0),
            "inference_time_ms": detection_result.get("inference_time_ms", 0),
            "detections": detection_result.get("detections", []),
            "detection_counts": detection_result.get("detection_counts", {}),
            "total_detections": detection_result.get("total_detections", 0)
        }

        return message

    async def _generate_video_stream(self, camera_id: str):
        """
        Generate MJPEG video stream with detection overlays for a specific camera.

        Args:
            camera_id: ID of the camera to stream
        """
        # Find the frame source for this camera
        frame_source = None
        for fs in self.frame_sources:
            if fs.camera_id == camera_id:
                frame_source = fs
                break

        if not frame_source:
            logger.error(f"Camera {camera_id} not found for video streaming")
            return

        while True:
            if frame_source and hasattr(frame_source, 'latest_frame'):
                # Get frame copy with thread safety
                with frame_source.frame_lock:
                    frame = frame_source.latest_frame.copy() if frame_source.latest_frame is not None else None

                if frame is not None:
                    # Draw detections on frame for this camera
                    if camera_id in self.latest_detections:
                        frame = self._draw_detections(frame, self.latest_detections[camera_id])

                    # Encode frame as JPEG
                    try:
                        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        else:
                            logger.error(f"Failed to encode frame for camera {camera_id} - frame may be corrupted or empty")
                            # Continue to next frame instead of breaking stream
                    except Exception as e:
                        logger.error(f"Exception encoding frame for camera {camera_id}: {e}")
                        # Continue streaming - don't let one bad frame kill the stream

            await asyncio.sleep(0.033)  # ~30 FPS

    def _draw_detections(self, frame: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.

        Args:
            frame: Input frame
            detection_result: Detection result with bounding boxes

        Returns:
            Frame with detections drawn
        """
        detections = detection_result.get('detections', [])

        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']

            # Get bounding box coordinates
            x1 = int(bbox['x1'])
            y1 = int(bbox['y1'])
            x2 = int(bbox['x2'])
            y2 = int(bbox['y2'])

            # Choose color based on class
            if class_name == 'person':
                color = (0, 0, 255)  # Red for people
            elif class_name in ['cat', 'dog', 'bird']:
                color = (0, 165, 255)  # Orange for animals
            else:
                color = (0, 255, 0)  # Green for others

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        # Draw latency info
        latency_ms = detection_result.get('total_latency_ms', 0)
        cv2.putText(
            frame,
            f"Latency: {latency_ms:.0f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        return frame

    def set_camera_start_time(self, camera_id: str, start_time: Optional[float] = None):
        """
        Set the start time for a camera (for uptime tracking).

        Args:
            camera_id: Camera ID
            start_time: Start timestamp (defaults to current time)
        """
        if start_time is None:
            start_time = time.time()
        self.camera_start_times[camera_id] = start_time

    def run(self):
        """Start the web server."""
        logger.info(f"Starting web server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast message to all connected WebSocket clients.

        Args:
            message: Message to broadcast
        """
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to connection: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)


if __name__ == "__main__":
    # Test the web server
    from queue import Queue

    logger.info("Testing Web Server")

    # Create a test detection queue
    detection_queue = Queue(maxsize=10)

    # Initialize server
    server = WebServer(
        detection_queue=detection_queue,
        host="0.0.0.0",
        port=8000
    )

    # Add test detection data
    test_detection = {
        'frame_id': 1,
        'timestamp': time.time(),
        'processing_timestamp': time.time(),
        'inference_time_ms': 15.0,
        'total_latency_ms': 50.0,
        'detections': [
            {
                'class_name': 'person',
                'confidence': 0.95,
                'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 300}
            }
        ],
        'detection_counts': {'person': 1},
        'total_detections': 1
    }

    detection_queue.put(test_detection)

    # Run server
    logger.info("Starting server on http://0.0.0.0:8000")
    logger.info("Press Ctrl+C to stop")
    server.run()
