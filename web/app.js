/**
 * Telescope Detection System - Frontend Application
 * Handles WebSocket connection and canvas rendering
 */

class DetectionApp {
    constructor() {
        this.ws = null;
        this.canvas = document.getElementById('videoCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.img = new Image();
        this.videoStream = null;

        // State
        this.latestDetections = null;
        this.isConnected = false;
        this.frameCount = 0;
        this.lastFpsUpdate = Date.now();
        this.fps = 0;

        // Session statistics (cumulative)
        this.sessionStats = {
            totalDetections: 0,
            peopleCount: 0,
            animalCount: 0,
            detectionsByClass: {}
        };

        // Animal class names for categorization
        this.animalClasses = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                              'elephant', 'bear', 'zebra', 'giraffe', 'coyote',
                              'rabbit', 'lizard', 'fox', 'deer'];

        this.init();
    }

    init() {
        console.log('Initializing Detection App');
        this.setupCanvas();
        this.connectWebSocket();
        this.startVideoStream();
    }

    setupCanvas() {
        // Set canvas size
        this.canvas.width = 1280;
        this.canvas.height = 720;
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/detections`;

        console.log('Connecting to WebSocket:', wsUrl);

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.updateConnectionStatus(true);
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus(false);

            // Attempt to reconnect after 3 seconds
            setTimeout(() => {
                console.log('Attempting to reconnect...');
                this.connectWebSocket();
            }, 3000);
        };
    }

    handleMessage(data) {
        if (data.type === 'connection') {
            console.log('Connection message:', data.message);
        } else if (data.type === 'detections') {
            this.latestDetections = data;
            this.updateUI(data);
            this.updateFPS();
        } else if (data.type === 'heartbeat') {
            // Handle heartbeat if needed
        }
    }

    startVideoStream() {
        // Load video stream from MJPEG endpoint
        const videoUrl = `${window.location.protocol}//${window.location.host}/video/feed`;

        this.img.onload = () => {
            this.drawFrame();
        };

        // Start loading the video stream
        this.img.src = videoUrl;

        // Alternative: Use fetch API for frame-by-frame streaming
        // this.streamFrames();
    }

    async streamFrames() {
        // Alternative implementation using fetch API
        const videoUrl = `${window.location.protocol}//${window.location.host}/video/feed`;

        try {
            const response = await fetch(videoUrl);
            const reader = response.body.getReader();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                // Process frame data
                // This would require parsing the multipart stream
            }
        } catch (e) {
            console.error('Failed to stream video:', e);
        }
    }

    drawFrame() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw video frame
        this.ctx.drawImage(this.img, 0, 0, this.canvas.width, this.canvas.height);

        // Draw detections overlay
        if (this.latestDetections && this.latestDetections.detections) {
            this.drawDetections(this.latestDetections.detections);
        }

        // Continue streaming
        requestAnimationFrame(() => this.drawFrame());
    }

    drawDetections(detections) {
        detections.forEach(detection => {
            const bbox = detection.bbox;
            const className = detection.class_name;
            const confidence = detection.confidence;

            // Choose color based on class
            let color;
            if (className === 'person') {
                color = '#ff6666'; // Red for people
            } else if (this.animalClasses.includes(className)) {
                color = '#ffaa00'; // Orange for animals
            } else {
                color = '#00ff88'; // Green for others
            }

            // Draw bounding box
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(
                bbox.x1,
                bbox.y1,
                bbox.x2 - bbox.x1,
                bbox.y2 - bbox.y1
            );

            // Draw label
            const label = `${className}: ${(confidence * 100).toFixed(0)}%`;
            this.ctx.font = 'bold 16px Arial';

            // Measure text for background
            const textMetrics = this.ctx.measureText(label);
            const textHeight = 20;

            // Draw label background
            this.ctx.fillStyle = color;
            this.ctx.fillRect(
                bbox.x1,
                bbox.y1 - textHeight - 5,
                textMetrics.width + 10,
                textHeight + 5
            );

            // Draw label text
            this.ctx.fillStyle = '#000000';
            this.ctx.fillText(label, bbox.x1 + 5, bbox.y1 - 8);
        });
    }

    updateUI(data) {
        // Update frame ID
        document.getElementById('frameId').textContent = data.frame_id || '-';

        // Update inference time
        const inferenceTime = data.inference_time_ms;
        if (inferenceTime !== undefined) {
            document.getElementById('inferenceTime').textContent = `${inferenceTime.toFixed(1)}ms`;
        }

        // Update total latency
        const totalLatency = data.latency_ms;
        if (totalLatency !== undefined) {
            document.getElementById('totalLatency').textContent = `${totalLatency.toFixed(1)}ms`;
            this.updateLatencyIndicator(totalLatency);
        }

        // Update session statistics (cumulative)
        const detections = data.detections || [];
        if (detections.length > 0) {
            // Count detections by class
            detections.forEach(det => {
                const className = det.class_name;

                // Increment total
                this.sessionStats.totalDetections++;

                // Track by class
                if (!this.sessionStats.detectionsByClass[className]) {
                    this.sessionStats.detectionsByClass[className] = 0;
                }
                this.sessionStats.detectionsByClass[className]++;

                // Count people
                if (className === 'person') {
                    this.sessionStats.peopleCount++;
                }

                // Count animals
                if (this.animalClasses.includes(className)) {
                    this.sessionStats.animalCount++;
                }
            });
        }

        // Display session statistics
        document.getElementById('totalDetections').textContent = this.sessionStats.totalDetections;
        document.getElementById('peopleCount').textContent = this.sessionStats.peopleCount;
        document.getElementById('animalCount').textContent = this.sessionStats.animalCount;

        // Update detections list
        this.updateDetectionsList(data.detections || []);
    }

    updateDetectionsList(detections) {
        const listElement = document.getElementById('detectionsList');

        if (detections.length === 0) {
            listElement.innerHTML = '<div class="no-detections">No detections</div>';
            return;
        }

        // Sort by confidence
        const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);

        // Display top 10
        const top = sorted.slice(0, 10);

        listElement.innerHTML = top.map(det => {
            let itemClass = 'detection-item';
            if (det.class_name === 'person') {
                itemClass += ' person';
            } else if (this.animalClasses.includes(det.class_name)) {
                itemClass += ' animal';
            }

            return `
                <div class="${itemClass}">
                    <div class="detection-class">${det.class_name}</div>
                    <div class="detection-confidence">Confidence: ${(det.confidence * 100).toFixed(1)}%</div>
                </div>
            `;
        }).join('');
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        if (connected) {
            statusElement.textContent = 'Connected';
            statusElement.className = 'status connected';
        } else {
            statusElement.textContent = 'Disconnected';
            statusElement.className = 'status disconnected';
        }
    }

    updateLatencyIndicator(latency) {
        const indicator = document.getElementById('latencyStatus');
        indicator.textContent = `Latency: ${latency.toFixed(0)}ms`;

        // Update color based on latency
        if (latency < 100) {
            indicator.className = 'latency-indicator latency-good';
        } else if (latency < 200) {
            indicator.className = 'latency-indicator latency-warning';
        } else {
            indicator.className = 'latency-indicator latency-bad';
        }
    }

    updateFPS() {
        this.frameCount++;
        const now = Date.now();
        const elapsed = now - this.lastFpsUpdate;

        if (elapsed >= 1000) {
            this.fps = Math.round((this.frameCount / elapsed) * 1000);
            document.getElementById('fps').textContent = this.fps;

            this.frameCount = 0;
            this.lastFpsUpdate = now;
        }
    }
}

// Initialize app when page loads
window.addEventListener('DOMContentLoaded', () => {
    console.log('Page loaded, starting Detection App');
    const app = new DetectionApp();
});
