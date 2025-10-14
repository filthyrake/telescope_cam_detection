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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

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
        host: str = "0.0.0.0",
        port: int = 8000
    ):
        """
        Initialize web server.

        Args:
            detection_queue: Queue to receive detection results from
            frame_sources: List of frame sources (RTSPStreamCapture instances)
            host: Host to bind to
            port: Port to bind to
        """
        self.detection_queue = detection_queue
        self.frame_sources = frame_sources if frame_sources else []
        self.host = host
        self.port = port

        self.app = FastAPI(title="Backyard Computer Vision System")
        self.active_connections: list[WebSocket] = []

        # Latest detections per camera
        self.latest_detections: Dict[str, Dict[str, Any]] = {}  # camera_id -> detection_result

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
                    if self.detection_queue and not self.detection_queue.empty():
                        detection_result = self.detection_queue.get()

                        # Store latest detection for this camera
                        camera_id = detection_result.get('camera_id', 'default')
                        self.latest_detections[camera_id] = detection_result

                        # Prepare WebSocket message
                        message = self._prepare_detection_message(detection_result)

                        # Send to client
                        await websocket.send_json(message)

                    else:
                        # Send heartbeat to keep connection alive
                        await asyncio.sleep(1.0)
                        await websocket.send_json({
                            "type": "heartbeat",
                            "timestamp": time.time()
                        })

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
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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
