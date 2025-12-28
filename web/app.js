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
        this.lastProcessedFrames = {}; // Track last processed frame ID per camera to avoid overcounting
        this.badgeTimeouts = {}; // Store timeout IDs for detection badges to prevent flickering
        this.cameraFrameCounts = {}; // Track frame counts per camera for FPS calculation
        this.cameraLastFpsUpdate = {}; // Track last FPS update timestamp per camera

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

        // Memory monitoring (Issue #125)
        this.memoryStatsInterval = null;

        // Accessibility state tracking
        this._accessibilitySetup = false;
        this._previousTopDetection = null;
        this._hadDetections = false;

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
        this.setupAccessibility();
        this.connectWebSocket();
        this.startVideoStream();
        this.startMemoryStatsPolling();
    }

    /* Accessibility setup: ARIA attributes, keyboard handlers, live-region wiring */
    setupAccessibility() {
        // Prevent duplicate event listeners on hot reload
        if (this._accessibilitySetup) return;
        this._accessibilitySetup = true;

        // Controls
        const viewModeBtn = document.getElementById('viewModeBtn');
        const fullscreenBtn = document.getElementById('fullscreenBtn');
        const cameraToggleBtn = document.getElementById('cameraToggleBtn');
        const cameraSelect = document.getElementById('cameraSelect');
        const cameraTogglePanel = document.getElementById('cameraTogglePanel');
        const detectionsList = document.getElementById('detectionsList');
        const detectionStatus = document.getElementById('detection-status');

        if (viewModeBtn) {
            viewModeBtn.setAttribute('aria-label', 'Toggle view mode: switch between single and grid view');
            viewModeBtn.setAttribute('aria-pressed', this.viewMode === 'grid' ? 'true' : 'false');
        }

        if (fullscreenBtn) {
            fullscreenBtn.setAttribute('aria-label', 'Toggle fullscreen for video');
        }

        if (cameraToggleBtn) {
            cameraToggleBtn.setAttribute('aria-controls', 'cameraTogglePanel');
            cameraToggleBtn.setAttribute('aria-expanded', 'false');
            cameraToggleBtn.setAttribute('aria-label', 'Show or hide camera visibility controls');
        }

        if (cameraSelect) {
            cameraSelect.setAttribute('aria-label', 'Select camera');
        }

        // Global Escape handling: close panels and return focus
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                // Close camera toggle panel if open
                if (cameraTogglePanel && cameraTogglePanel.style.display === 'block') {
                    cameraTogglePanel.style.display = 'none';
                    if (cameraToggleBtn) {
                        cameraToggleBtn.focus();
                        cameraToggleBtn.setAttribute('aria-expanded', 'false');
                    }
                }

                // Close grid controls when visible
                const gridControls = document.getElementById('gridControls');
                if (gridControls && gridControls.style.display === 'flex') {
                    gridControls.style.display = 'none';
                    if (viewModeBtn) viewModeBtn.focus();
                }
            }
        });

        // Arrow navigation in detections list: let ArrowDown/ArrowUp move focus between items
        if (detectionsList) {
            detectionsList.addEventListener('keydown', (e) => {
                if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                    const focusable = Array.from(detectionsList.querySelectorAll('.detection-item[tabindex="0"]'));
                    if (focusable.length === 0) return;

                    const idx = focusable.indexOf(document.activeElement);
                    if (idx === -1 && focusable[0]) {
                        focusable[0].focus();
                        e.preventDefault();
                        return;
                    }

                    if (e.key === 'ArrowDown' && idx < focusable.length - 1) {
                        focusable[idx + 1].focus();
                        e.preventDefault();
                    } else if (e.key === 'ArrowUp' && idx > 0) {
                        focusable[idx - 1].focus();
                        e.preventDefault();
                    }
                }
            });
        }

        // Ensure live region exists
        if (!detectionStatus) {
            const region = document.createElement('div');
            region.id = 'detection-status';
            region.setAttribute('role', 'status');
            region.setAttribute('aria-live', 'polite');
            region.setAttribute('aria-atomic', 'true');
            region.className = 'visually-hidden';
            document.body.appendChild(region);
        }
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

        // Mark initial accessibility state
        viewModeBtn.setAttribute('aria-pressed', this.viewMode === 'grid' ? 'true' : 'false');

        viewModeBtn.addEventListener('click', () => {
            if (this.viewMode === 'single') {
                this.switchViewMode('grid');
                viewModeBtn.textContent = '📹 Single View';
                cameraSelect.disabled = true;
                cameraSelect.setAttribute('aria-disabled', 'true');
                gridControls.style.display = 'flex';
                viewModeBtn.setAttribute('aria-pressed', 'true');
            } else {
                this.switchViewMode('single');
                viewModeBtn.textContent = '📊 Grid View';
                cameraSelect.disabled = false;
                cameraSelect.removeAttribute('aria-disabled');
                gridControls.style.display = 'none';
                // Hide camera toggle panel when switching to single view
                document.getElementById('cameraTogglePanel').style.display = 'none';
                viewModeBtn.setAttribute('aria-pressed', 'false');
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
            viewModeBtn.setAttribute('aria-disabled', 'true');
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
        toggleBtn.setAttribute('aria-expanded', 'false');
        toggleBtn.addEventListener('click', () => {
            const isOpen = panel.style.display === 'block';
            panel.style.display = isOpen ? 'none' : 'block';
            toggleBtn.setAttribute('aria-expanded', isOpen ? 'false' : 'true');
            if (isOpen) {
                // Panel is being closed, return focus to the toggle button
                toggleBtn.focus();
            } else {
                // Panel is being opened, move focus into the panel for keyboard users
                const firstInput = panel.querySelector('input, button, select');
                if (firstInput) firstInput.focus();
            }
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

            // Make grid item keyboard accessible and clickable
            gridItem.tabIndex = 0;
            gridItem.setAttribute('role', 'button');
            gridItem.setAttribute('aria-label', `Open ${camera.name} in single view`);

            // Click to fullscreen
            gridItem.addEventListener('click', () => {
                this.fullscreenGridCamera(camera.id);
            });

            // Keyboard activation (Enter / Space)
            gridItem.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this.fullscreenGridCamera(camera.id);
                }
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

            // Update per-camera FPS based on message rate
            this.updateCameraFPS(data);

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

    updateCameraFPS(data) {
        const cameraId = data.camera_id || 'default';
        const now = Date.now();

        // Initialize tracking for this camera if needed
        if (!this.cameraFrameCounts[cameraId]) {
            this.cameraFrameCounts[cameraId] = 0;
            this.cameraLastFpsUpdate[cameraId] = now;
        }

        // Increment frame count for this camera
        this.cameraFrameCounts[cameraId]++;

        // Calculate FPS every second
        const elapsed = now - this.cameraLastFpsUpdate[cameraId];
        if (elapsed >= 1000) {
            const fps = Math.round((this.cameraFrameCounts[cameraId] / elapsed) * 1000);
            this.cameraFps[cameraId] = fps;

            // Reset counters
            this.cameraFrameCounts[cameraId] = 0;
            this.cameraLastFpsUpdate[cameraId] = now;
        }
    }

    updateGridCameraStats(data) {
        const cameraId = data.camera_id || 'default';
        const frameId = data.frame_id;

        // Update FPS counter
        const fpsElement = document.getElementById(`fps-${cameraId}`);
        if (fpsElement && this.cameraFps[cameraId] !== undefined) {
            fpsElement.textContent = `FPS: ${Math.round(this.cameraFps[cameraId])}`;
        }

        // Update detection count and badge - only if this is a new frame
        const detections = data.detections || [];
        if (detections.length > 0 && frameId !== this.lastProcessedFrames[cameraId]) {
            // Mark this frame as processed to avoid overcounting
            this.lastProcessedFrames[cameraId] = frameId;

            // Increment detection count by number of detections in this frame
            this.cameraDetectionCounts[cameraId] = (this.cameraDetectionCounts[cameraId] || 0) + detections.length;

            // Update badge
            const badgeElement = document.getElementById(`badge-${cameraId}`);
            if (badgeElement) {
                badgeElement.textContent = `${this.cameraDetectionCounts[cameraId]} detection${this.cameraDetectionCounts[cameraId] !== 1 ? 's' : ''}`;
                badgeElement.classList.add('visible');

                // Clear any existing timeout for this camera to prevent flickering
                if (this.badgeTimeouts[cameraId]) {
                    clearTimeout(this.badgeTimeouts[cameraId]);
                }

                // Hide badge after 3 seconds
                this.badgeTimeouts[cameraId] = setTimeout(() => {
                    badgeElement.classList.remove('visible');
                    delete this.badgeTimeouts[cameraId];
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
        const liveRegion = document.getElementById('detection-status');

        if (detections.length === 0) {
            listElement.innerHTML = '<div class="no-detections">No detections</div>';
            // Only announce "No detections" when transitioning from having detections
            if (liveRegion && this._hadDetections) {
                liveRegion.textContent = 'No detections';
            }
            this._hadDetections = false;
            this._previousTopDetection = null;
            return;
        }

        // Sort by confidence and show top 10
        const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
        const top = sorted.slice(0, 10);

        // Clear and build accessible list items
        listElement.innerHTML = '';
        top.forEach(det => {
            let itemClass = 'detection-item';
            if (det.class_name === 'person') {
                itemClass += ' person';
            } else if (this.animalClasses.includes(det.class_name)) {
                itemClass += ' animal';
            }

            const item = document.createElement('div');
            item.className = itemClass;
            item.setAttribute('role', 'listitem');
            item.tabIndex = 0;
            item.setAttribute('aria-label', `${det.class_name}, confidence ${(det.confidence * 100).toFixed(1)} percent`);

            const classDiv = document.createElement('div');
            classDiv.className = 'detection-class';
            classDiv.textContent = det.class_name;

            const confDiv = document.createElement('div');
            confDiv.className = 'detection-confidence';
            confDiv.textContent = `Confidence: ${(det.confidence * 100).toFixed(1)}%`;

            // Activation: announce selection in live region
            const announce = () => {
                if (liveRegion) {
                    liveRegion.textContent = `Selected detection: ${det.class_name} at ${(det.confidence * 100).toFixed(1)} percent`;
                }
            };

            item.addEventListener('click', announce);
            item.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    announce();
                }
            });

            item.appendChild(classDiv);
            item.appendChild(confDiv);
            listElement.appendChild(item);
        });

        // Only announce when top detection changes (different class or new detection)
        if (liveRegion && top.length > 0) {
            const topDet = top[0];
            const prevTop = this._previousTopDetection;
            const isNewDetection = !prevTop || prevTop.class_name !== topDet.class_name;

            if (isNewDetection) {
                liveRegion.textContent = `New detection: ${topDet.class_name} detected at ${(topDet.confidence * 100).toFixed(0)} percent confidence`;
            }

            this._previousTopDetection = { class_name: topDet.class_name, confidence: topDet.confidence };
        }
        this._hadDetections = true;
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
        // This tracks canvas drawing FPS for single view
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

    startMemoryStatsPolling() {
        // Fetch memory stats every 5 seconds (Issue #125)
        this.fetchMemoryStats();
        this.memoryStatsInterval = setInterval(() => {
            this.fetchMemoryStats();
        }, 5000);
    }

    async fetchMemoryStats() {
        try {
            const response = await fetch('/stats');
            const stats = await response.json();
            this.updateMemoryStats(stats);
        } catch (e) {
            console.error('Failed to fetch memory stats:', e);
        }
    }

    updateMemoryStats(stats) {
        // Update GPU memory gauge (Issue #125)
        const memory = stats.memory || {};
        const gpuMemorySection = document.getElementById('gpuMemorySection');
        const memoryPressureSection = document.getElementById('memoryPressureSection');
        const degradationSection = document.getElementById('degradationSection');

        if (memory.cuda_available) {
            // Show GPU memory stats
            gpuMemorySection.style.display = 'flex';

            const usagePercent = memory.usage_percent || 0;
            const allocatedGB = memory.allocated_gb || 0;
            const reservedGB = memory.reserved_gb || 0;
            const totalGB = memory.total_gb || 0;

            const gpuMemoryElement = document.getElementById('gpuMemory');
            if (gpuMemoryElement) {
                // Show both allocated and reserved memory for transparency
                // Reserved includes allocated + freeable cache
                gpuMemoryElement.textContent =
                    `${allocatedGB.toFixed(1)}GB alloc, ${reservedGB.toFixed(1)}GB rsv / ${totalGB.toFixed(1)}GB (${usagePercent.toFixed(0)}%)`;
            }

            // Update memory pressure indicator
            const pressure = memory.current_pressure || 'normal';
            memoryPressureSection.style.display = 'flex';
            const pressureElement = document.getElementById('memoryPressure');

            if (pressure === 'normal') {
                pressureElement.textContent = 'Normal';
                pressureElement.style.color = '#00ff88';
            } else if (pressure === 'high') {
                pressureElement.textContent = 'High';
                pressureElement.style.color = '#ffaa00';
            } else if (pressure === 'critical' || pressure === 'extreme') {
                pressureElement.textContent = pressure.charAt(0).toUpperCase() + pressure.slice(1);
                pressureElement.style.color = '#ff6666';
            }

            // Update degradation status
            const degradationActive = stats.degradation_active || false;
            const degradationLevel = memory.degradation_level || 0;
            const oomEvents = memory.oom_events || 0;

            if (degradationActive || degradationLevel > 0 || oomEvents > 0) {
                degradationSection.style.display = 'flex';
                const degradationText = [];

                if (degradationActive) {
                    degradationText.push('Active');
                }
                if (degradationLevel > 0) {
                    degradationText.push(`Level ${degradationLevel}`);
                }
                if (oomEvents > 0) {
                    degradationText.push(`${oomEvents} OOM`);
                }

                document.getElementById('degradationStatus').textContent = degradationText.join(', ');
            } else {
                degradationSection.style.display = 'none';
            }

            // Update memory alert banner
            this.updateMemoryAlert(pressure, usagePercent, degradationActive, reservedGB, totalGB);

        } else {
            // GPU not available - hide sections
            gpuMemorySection.style.display = 'none';
            memoryPressureSection.style.display = 'none';
            degradationSection.style.display = 'none';
        }
    }

    updateMemoryAlert(pressure, usagePercent, degradationActive, reservedGB, totalGB) {
        const alertElement = document.getElementById('memoryAlert');
        const titleElement = document.getElementById('memoryAlertTitle');
        const messageElement = document.getElementById('memoryAlertMessage');

        if (pressure === 'critical' || pressure === 'extreme') {
            // Show critical alert
            alertElement.classList.add('visible', 'critical');
            titleElement.textContent = '🚨 Critical GPU Memory Pressure';
            messageElement.textContent =
                `GPU reserved memory is at ${usagePercent.toFixed(0)}% (${reservedGB.toFixed(1)}GB / ${totalGB.toFixed(1)}GB). System is reducing quality to prevent crashes. ${degradationActive ? 'Degradation active.' : ''}`;
        } else if (pressure === 'high' || usagePercent > 80) {
            // Show warning alert
            alertElement.classList.add('visible');
            alertElement.classList.remove('critical');
            titleElement.textContent = '⚠️ High GPU Memory Usage';
            messageElement.textContent =
                `GPU reserved memory is at ${usagePercent.toFixed(0)}% (${reservedGB.toFixed(1)}GB / ${totalGB.toFixed(1)}GB). System may reduce quality if memory pressure increases.`;
        } else {
            // Hide alert
            alertElement.classList.remove('visible', 'critical');
        }
    }
}

// Initialize app when page loads
window.addEventListener('DOMContentLoaded', () => {
    console.log('Page loaded, starting Detection App');
    const app = new DetectionApp();
});
