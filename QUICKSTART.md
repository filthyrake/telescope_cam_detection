# Telescope Detection System - Quick Start

## Setup (One Time)

```bash
# 1. Install service
sudo ./service.sh install

# 2. Start service
sudo ./service.sh start

# 3. Access web interface
# Open browser: http://localhost:8000
```

## Daily Commands

```bash
# Check if running
./service.sh status

# View live logs
./service.sh logs -f

# Restart after config changes
sudo ./service.sh restart

# Stop service
sudo ./service.sh stop
```

## Configuration

Edit settings: `nano config/config.yaml`

After changes: `sudo ./service.sh restart`

## Troubleshooting

```bash
# View error logs
./service.sh logs | grep ERROR

# Test camera connection
python tests/test_camera_connection.py

# Check GPU
nvidia-smi

# Full service status
systemctl status telescope_detection.service -l
```

## File Locations

- **Config**: `config/config.yaml`
- **Saved clips**: `clips/`
- **Logs**: `journalctl -u telescope_detection.service`
- **Service file**: `/etc/systemd/system/telescope_detection.service`

## Web Interface

- **Local**: http://localhost:8000
- **Network**: http://YOUR_IP:8000

## Features

### Stage 1: YOLOX Detection
- Detects: person, bird, cat, dog, and 76 other COCO classes
- Wildlife-only filter available
- Real-time bounding boxes
- 11-21ms inference time

### Stage 2: iNaturalist Species ID (Optional)
- Fine-grained species classification
- 10,000 species database
- Hierarchical taxonomy fallback
- Example: "bird" → "Gambel's Quail" (species level)

## Common Tasks

### View Saved Wildlife Clips
```bash
ls -lh clips/
python scripts/view_snapshots.py
```

### Update Code
```bash
sudo ./service.sh stop
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo ./service.sh start
```

### Change Camera Settings
```bash
nano config/config.yaml
sudo ./service.sh restart
```

## Performance Metrics

- **Stage 1 only**: ~11-21ms per frame (YOLOX-S @ 1920x1920)
- **Stage 1 + 2**: ~30-50ms per frame (with species classification)
- **Expected FPS**: 25-30 sustained (real-time)

## Help

- **Full docs**: See [SERVICE_SETUP.md](SERVICE_SETUP.md)
- **Project info**: See [README.md](README.md)
- **Stage 2 setup**: See [STAGE2_SETUP.md](STAGE2_SETUP.md)
