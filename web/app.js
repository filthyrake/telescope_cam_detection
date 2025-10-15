/**
 * Backyard Computer Vision System - Frontend Application
 * Handles WebSocket connection and canvas rendering
 */

class DetectionApp {
    constructor() {
        this.ws = null;
        this.canvas = document.getElementById('videoCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.img = new Image();
        this.videoStream = null;

        // Multi-camera support
        this.cameras = [];
        this.currentCameraId = null;
        this.viewMode = 'single'; // 'single' or 'grid'
        this.gridImages = {}; // Store image elements for grid view
        this.gridLayout = 'auto'; // 'auto', '1x1', '2x1', '2x2', '3x2', '3x3'
        this.visibleCameras = new Set(); // Track which cameras are visible in grid
        this.cameraDetectionCounts = {}; // Track detection counts per camera
        this.cameraFps = {}; // Track FPS per camera

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

    async init() {
        console.log('Initializing Detection App');
        this.setupCanvas();
        this.setupFullscreen();
        await this.fetchCameras();
        this.setupCameraSelector();
        this.setupViewModeToggle();
        this.setupLayoutSelector();
        this.setupCameraTogglePanel();
        this.connectWebSocket();
        this.startVideoStream();
    }

    async fetchCameras() {
        try {
            const response = await fetch('/cameras');
            const data = await response.json();
            this.cameras = data.cameras || [];
            console.log('Available cameras:', this.cameras);

            // Set default camera to first one
            if (this.cameras.length > 0) {
                this.currentCameraId = this.cameras[0].id;
            }
        } catch (e) {
            console.error('Failed to fetch cameras:', e);
            // Fallback to default camera
            this.currentCameraId = 'cam1';
        }
    }

    setupCameraSelector() {
        const selector = document.getElementById('cameraSelect');

        // Clear existing options
        selector.innerHTML = '';

        // Populate with available cameras
        if (this.cameras.length === 0) {
            selector.innerHTML = '<option value="">No cameras available</option>';
            return;
        }

        this.cameras.forEach(camera => {
            const option = document.createElement('option');
            option.value = camera.id;
            option.textContent = `${camera.name} (${camera.is_connected ? 'Connected' : 'Disconnected'})`;
            selector.appendChild(option);
        });

        // Set initial selection
        selector.value = this.currentCameraId;

        // Handle camera change
        selector.addEventListener('change', (e) => {
            this.switchCamera(e.target.value);
        });
    }

    switchCamera(cameraId) {
        console.log('Switching to camera:', cameraId);
        this.currentCameraId = cameraId;

        // Update video stream
        const videoUrl = `${window.location.protocol}//${window.location.host}/video/feed/${cameraId}`;
        this.img.src = videoUrl;

        // Update camera name display
        const camera = this.cameras.find(c => c.id === cameraId);
        if (camera) {
            document.getElementById('cameraName').textContent = camera.name;
        }
    }

    setupViewModeToggle() {
        const viewModeBtn = document.getElementById('viewModeBtn');
        const cameraSelect = document.getElementById('cameraSelect');
        const gridControls = document.getElementById('gridControls');

        viewModeBtn.addEventListener('click', () => {
            if (this.viewMode === 'single') {
                this.switchViewMode('grid');
                viewModeBtn.textContent = '📹 Single View';
                cameraSelect.disabled = true;
                gridControls.style.display = 'flex';
            } else {
                this.switchViewMode('single');
                viewModeBtn.textContent = '📊 Grid View';
                cameraSelect.disabled = false;
                gridControls.style.display = 'none';
                // Hide camera toggle panel when switching to single view
                document.getElementById('cameraTogglePanel').style.display = 'none';
            }
        });

        // Hide grid view button if no cameras, disable if only one camera
        if (this.cameras.length === 0) {
            viewModeBtn.style.display = 'none';
        } else if (this.cameras.length === 1) {
            viewModeBtn.disabled = true;
            viewModeBtn.style.opacity = '0.5';
            viewModeBtn.style.cursor = 'not-allowed';
            viewModeBtn.title = 'Grid view requires multiple cameras';
        }
    }

    setupLayoutSelector() {
        const layoutSelect = document.getElementById('layoutSelect');

        layoutSelect.addEventListener('change', (e) => {
            this.gridLayout = e.target.value;
            console.log('Grid layout changed to:', this.gridLayout);
            this.updateGridLayout();
        });
    }

    setupCameraTogglePanel() {
        const toggleBtn = document.getElementById('cameraToggleBtn');
        const panel = document.getElementById('cameraTogglePanel');
        const togglesContainer = document.getElementById('cameraToggles');

        // Initialize all cameras as visible
        this.cameras.forEach(camera => {
            this.visibleCameras.add(camera.id);
        });

        // Toggle panel visibility
        toggleBtn.addEventListener('click', () => {
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        });

        // Create checkbox for each camera
        togglesContainer.innerHTML = '';
        this.cameras.forEach(camera => {
            const checkboxWrapper = document.createElement('div');
            checkboxWrapper.style.cssText = 'display: flex; align-items: center; gap: 8px; padding: 8px; background-color: #1a1a1a; border-radius: 5px;';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `toggle-${camera.id}`;
            checkbox.checked = true;
            checkbox.style.cssText = 'cursor: pointer; width: 18px; height: 18px;';

            const label = document.createElement('label');
            label.htmlFor = `toggle-${camera.id}`;
            label.textContent = camera.name;
            label.style.cssText = 'cursor: pointer; color: #00ff88; user-select: none;';

            checkbox.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.visibleCameras.add(camera.id);
                } else {
                    this.visibleCameras.delete(camera.id);
                }
                this.updateGridVisibility();
            });

            checkboxWrapper.appendChild(checkbox);
            checkboxWrapper.appendChild(label);
            togglesContainer.appendChild(checkboxWrapper);
        });
    }

    switchViewMode(mode) {
        console.log('Switching to view mode:', mode);
        this.viewMode = mode;

        const singleView = document.getElementById('singleView');
        const gridView = document.getElementById('gridView');

        if (mode === 'grid') {
            // Hide single view, show grid
            singleView.style.display = 'none';
            gridView.style.display = 'block';
            this.setupGridView();
        } else {
            // Hide grid, show single view
            gridView.style.display = 'none';
            singleView.style.display = 'grid';
        }
    }

    setupGridView() {
        const gridContainer = document.getElementById('cameraGrid');

        // Clear existing grid
        gridContainer.innerHTML = '';

        // Apply grid class based on layout mode
        this.updateGridLayout();

        // Create grid items for each camera
        this.cameras.forEach(camera => {
            const gridItem = document.createElement('div');
            gridItem.className = 'grid-camera-item';
            gridItem.id = `grid-${camera.id}`;
            gridItem.dataset.cameraId = camera.id;

            // Click to fullscreen
            gridItem.addEventListener('click', () => {
                this.fullscreenGridCamera(camera.id);
            });

            // Create image element for video stream
            const img = document.createElement('img');
            img.src = `${window.location.protocol}//${window.location.host}/video/feed/${camera.id}`;
            img.alt = camera.name;

            // Create status indicator
            const status = document.createElement('div');
            status.className = `grid-camera-status ${camera.is_connected ? '' : 'disconnected'}`;
            status.id = `status-${camera.id}`;

            // Create FPS counter
            const fpsCounter = document.createElement('div');
            fpsCounter.className = 'grid-camera-fps';
            fpsCounter.id = `fps-${camera.id}`;
            fpsCounter.textContent = 'FPS: --';

            // Create detection count badge
            const detectionBadge = document.createElement('div');
            detectionBadge.className = 'grid-camera-detection-badge';
            detectionBadge.id = `badge-${camera.id}`;
            detectionBadge.textContent = '0 detections';

            // Create label
            const label = document.createElement('div');
            label.className = 'grid-camera-label';
            label.textContent = camera.name;

            // Assemble grid item
            gridItem.appendChild(img);
            gridItem.appendChild(status);
            gridItem.appendChild(fpsCounter);
            gridItem.appendChild(detectionBadge);
            gridItem.appendChild(label);
            gridContainer.appendChild(gridItem);

            // Store image reference
            this.gridImages[camera.id] = img;

            // Initialize detection count
            this.cameraDetectionCounts[camera.id] = 0;
        });

        console.log(`Grid view created with ${this.cameras.length} cameras`);
    }

    updateGridLayout() {
        const gridContainer = document.getElementById('cameraGrid');

        if (this.gridLayout === 'auto') {
            // Auto layout based on number of visible cameras
            const visibleCount = this.visibleCameras.size || this.cameras.length;
            gridContainer.className = `camera-grid grid-${visibleCount}`;
        } else {
            // Manual layout
            gridContainer.className = `camera-grid layout-${this.gridLayout}`;
        }
    }

    updateGridVisibility() {
        this.cameras.forEach(camera => {
            const gridItem = document.getElementById(`grid-${camera.id}`);
            if (gridItem) {
                if (this.visibleCameras.has(camera.id)) {
                    gridItem.classList.remove('hidden');
                } else {
                    gridItem.classList.add('hidden');
                }
            }
        });

        // Update grid layout for new visible count
        this.updateGridLayout();
    }

    fullscreenGridCamera(cameraId) {
        // Switch to single view and select this camera
        this.currentCameraId = cameraId;
        document.getElementById('cameraSelect').value = cameraId;

        // Switch view mode
        const viewModeBtn = document.getElementById('viewModeBtn');
        viewModeBtn.textContent = '📊 Grid View';
        document.getElementById('cameraSelect').disabled = false;
        document.getElementById('gridControls').style.display = 'none';
        document.getElementById('cameraTogglePanel').style.display = 'none';

        this.switchViewMode('single');
        this.switchCamera(cameraId);
    }

    setupCanvas() {
        // Set canvas size
        this.canvas.width = 1280;
        this.canvas.height = 720;
    }

    setupFullscreen() {
        const fullscreenBtn = document.getElementById('fullscreenBtn');
        const videoContainer = document.getElementById('videoContainer');

        // Store original canvas size
        this.originalCanvasWidth = this.canvas.width;
        this.originalCanvasHeight = this.canvas.height;

        fullscreenBtn.addEventListener('click', () => {
            if (!document.fullscreenElement) {
                // Enter fullscreen
                videoContainer.requestFullscreen().catch(err => {
                    console.error('Error attempting to enable fullscreen:', err);
                });
            } else {
                // Exit fullscreen
                document.exitFullscreen();
            }
        });

        // Update button text and canvas size when fullscreen state changes
        document.addEventListener('fullscreenchange', () => {
            if (document.fullscreenElement) {
                fullscreenBtn.textContent = '⛶ Exit Fullscreen';
                this.resizeCanvasForFullscreen();
            } else {
                fullscreenBtn.textContent = '⛶ Fullscreen';
                this.restoreCanvasSize();
            }
        });

        // Handle window resize in fullscreen with throttling
        let resizeTimeout;
        window.addEventListener('resize', () => {
            if (document.fullscreenElement) {
                clearTimeout(resizeTimeout);
                resizeTimeout = setTimeout(() => {
                    this.resizeCanvasForFullscreen();
                }, 100);
            }
        });
    }

    resizeCanvasForFullscreen() {
        // Get viewport dimensions (not screen dimensions)
        const screenWidth = window.innerWidth;
        const screenHeight = window.innerHeight;

        // Calculate aspect ratio
        const videoAspect = this.originalCanvasWidth / this.originalCanvasHeight;
        const screenAspect = screenWidth / screenHeight;

        // Resize canvas to fill screen while maintaining aspect ratio
        if (screenAspect > videoAspect) {
            // Screen is wider - fit to height
            this.canvas.height = screenHeight;
            this.canvas.width = screenHeight * videoAspect;
        } else {
            // Screen is taller - fit to width
            this.canvas.width = screenWidth;
            this.canvas.height = screenWidth / videoAspect;
        }
    }

    restoreCanvasSize() {
        this.canvas.width = this.originalCanvasWidth;
        this.canvas.height = this.originalCanvasHeight;
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

            // Update camera name from detection if available
            if (data.camera_name) {
                document.getElementById('cameraName').textContent = data.camera_name;
            }

            this.updateUI(data);
            this.updateFPS();

            // Update grid view elements if in grid mode
            if (this.viewMode === 'grid') {
                this.updateGridCameraStats(data);
            }
        } else if (data.type === 'heartbeat') {
            // Handle heartbeat if needed
        }
    }

    updateGridCameraStats(data) {
        const cameraId = data.camera_id || 'default';

        // Update FPS counter
        const fpsElement = document.getElementById(`fps-${cameraId}`);
        if (fpsElement && this.cameraFps[cameraId] !== undefined) {
            fpsElement.textContent = `FPS: ${Math.round(this.cameraFps[cameraId])}`;
        }

        // Update detection count and badge
        const detections = data.detections || [];
        if (detections.length > 0) {
            // Increment detection count
            this.cameraDetectionCounts[cameraId] = (this.cameraDetectionCounts[cameraId] || 0) + detections.length;

            // Update badge
            const badgeElement = document.getElementById(`badge-${cameraId}`);
            if (badgeElement) {
                badgeElement.textContent = `${this.cameraDetectionCounts[cameraId]} detection${this.cameraDetectionCounts[cameraId] !== 1 ? 's' : ''}`;
                badgeElement.classList.add('visible');

                // Hide badge after 3 seconds
                setTimeout(() => {
                    badgeElement.classList.remove('visible');
                }, 3000);
            }

            // Trigger flash animation
            const gridItem = document.getElementById(`grid-${cameraId}`);
            if (gridItem) {
                gridItem.classList.add('flash');
                setTimeout(() => {
                    gridItem.classList.remove('flash');
                }, 500);
            }
        }

        // Update connection status
        const statusElement = document.getElementById(`status-${cameraId}`);
        if (statusElement) {
            const camera = this.cameras.find(c => c.id === cameraId);
            if (camera) {
                statusElement.className = `grid-camera-status ${camera.is_connected ? '' : 'disconnected'}`;
            }
        }
    }

    startVideoStream() {
        // Load video stream from MJPEG endpoint for current camera
        const videoUrl = this.currentCameraId
            ? `${window.location.protocol}//${window.location.host}/video/feed/${this.currentCameraId}`
            : `${window.location.protocol}//${window.location.host}/video/feed`;

        this.img.onload = () => {
            this.drawFrame();
        };

        // Start loading the video stream
        this.img.src = videoUrl;

        // Update camera name display
        if (this.currentCameraId) {
            const camera = this.cameras.find(c => c.id === this.currentCameraId);
            if (camera) {
                document.getElementById('cameraName').textContent = camera.name;
            }
        }

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

            // Update per-camera FPS if we have detection data
            if (this.latestDetections && this.latestDetections.camera_id) {
                const cameraId = this.latestDetections.camera_id;
                this.cameraFps[cameraId] = this.fps;
            }

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
