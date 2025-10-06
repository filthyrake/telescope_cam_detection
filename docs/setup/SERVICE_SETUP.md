# Telescope Detection System - Service Setup

This guide explains how to run the telescope detection system as a systemd service that starts automatically on boot.

## Quick Start

### 1. Install the Service

```bash
# Install and enable the service
sudo ./service.sh install

# Start the service
sudo ./service.sh start

# Check status
./service.sh status
```

### 2. Access the Web Interface

Once the service is running, access the web interface at:
- **Local**: http://localhost:8000
- **Network**: http://YOUR_IP:8000

## Service Management Commands

### Installation & Setup

```bash
# Install service (first time setup)
sudo ./service.sh install

# Uninstall service (removes from systemd)
sudo ./service.sh uninstall
```

### Starting & Stopping

```bash
# Start the service
sudo ./service.sh start

# Stop the service
sudo ./service.sh stop

# Restart the service
sudo ./service.sh restart

# Check if service is running
./service.sh status
```

### Auto-Start on Boot

```bash
# Enable service to start on boot (done automatically by install)
sudo ./service.sh enable

# Disable auto-start on boot
sudo ./service.sh disable
```

### Viewing Logs

```bash
# View recent logs (last 100 lines)
./service.sh logs

# Follow logs in real-time (like tail -f)
./service.sh logs -f

# View logs with journalctl directly
journalctl -u telescope_detection.service -n 100

# Follow all logs since boot
journalctl -u telescope_detection.service -b -f
```

## Service Details

### Service Configuration

- **Service template**: `telescope_detection.service.template`
- **Generated file**: `telescope_detection.service` (auto-generated, not tracked in git)
- **Installed to**: `/etc/systemd/system/telescope_detection.service`
- **Runs as**: Automatically uses the user who runs `sudo ./service.sh install`
- **Working directory**: Automatically uses the current project directory
- **Python executable**: Uses venv at `./venv/bin/python`

The service file is generated from a template during installation, automatically configuring paths and user settings for your system.

### Restart Policy

The service is configured to automatically restart if it crashes:
- **Restart**: Always (on any exit)
- **RestartSec**: 10 seconds delay before restart

### Logging

All logs are sent to systemd journal:
- **Identifier**: `telescope-detection`
- **View logs**: `journalctl -u telescope_detection.service`

## Troubleshooting

### Service Won't Start

```bash
# Check detailed status
systemctl status telescope_detection.service -l

# View recent error logs
journalctl -u telescope_detection.service -n 50

# Check if camera is accessible
python tests/test_camera_connection.py

# Check GPU availability
nvidia-smi
```

### Camera Connection Issues

If the service starts but camera connection fails:

1. **Check camera IP/credentials** in `config/config.yaml`
2. **Test camera connection** manually:
   ```bash
   python tests/test_camera_connection.py
   ```
3. **Check network connectivity**:
   ```bash
   ping 192.168.1.100  # Your camera IP
   ```

### GPU Not Available

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check NVIDIA driver
nvidia-smi
```

### Service Keeps Restarting

Check logs for errors:
```bash
journalctl -u telescope_detection.service -f
```

Common issues:
- Camera not reachable (network/credentials)
- GPU not available (CUDA/driver issues)
- Configuration file errors
- Missing dependencies

### Modify Service Configuration

If you need to change service settings:

1. **Edit the service template**:
   ```bash
   nano telescope_detection.service.template
   ```

2. **Reinstall the service** (regenerates from template):
   ```bash
   sudo ./service.sh uninstall
   sudo ./service.sh install
   ```

The install script automatically fills in `{{USER}}` and `{{PROJECT_DIR}}` placeholders with your actual username and project path.

## Advanced Configuration

### Change User

The service automatically runs as the user who executes `sudo ./service.sh install`. If you need to run as a different user:

1. **Switch to that user** and run the install:
   ```bash
   su - otheruser
   cd /path/to/telescope_cam_detection
   sudo ./service.sh install
   ```

Or manually edit `telescope_detection.service.template` before installing:
```ini
User={{USER}}    # Will be replaced with actual username
Group={{USER}}
WorkingDirectory={{PROJECT_DIR}}  # Will be replaced with actual path
```

### Resource Limits

Uncomment and adjust in `telescope_detection.service.template`, then reinstall:
```ini
# Limit memory usage
MemoryMax=8G

# Limit CPU usage
CPUQuota=400%  # 4 cores max
```

### Environment Variables

Add environment variables in `telescope_detection.service.template`, then reinstall:
```ini
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="TZ=America/Phoenix"
```

### Custom Timezone

Set timezone for logs:
```bash
sudo timedatectl set-timezone America/Phoenix
```

Or add to service file:
```ini
Environment="TZ=America/Phoenix"
```

## Monitoring

### Real-time Monitoring

```bash
# Watch logs live
./service.sh logs -f

# Watch GPU usage
watch -n 1 nvidia-smi

# Watch system resources
htop
```

### Check Service Health

```bash
# Quick status check
./service.sh status

# Detailed status with recent logs
systemctl status telescope_detection.service -l

# Check if web interface is responding
curl http://localhost:8000/health
```

### Performance Monitoring

Access web interface metrics:
- **Main UI**: http://localhost:8000
- **Stats endpoint**: http://localhost:8000/stats

## Backup & Maintenance

### Backup Configuration

```bash
# Backup config file
cp config/config.yaml config/config.yaml.backup

# Backup saved clips
tar -czf clips_backup_$(date +%Y%m%d).tar.gz clips/
```

### Update Code

```bash
# Stop service
sudo ./service.sh stop

# Pull updates (if using git)
git pull

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Restart service
sudo ./service.sh start
```

### View Saved Wildlife Clips

```bash
# List saved clips
ls -lh clips/

# View clips with metadata
python scripts/view_snapshots.py
```

## Integration with Other Services

### Nginx Reverse Proxy

To expose the web interface through nginx:

```nginx
server {
    listen 80;
    server_name telescope.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Auto-restart on Config Changes

Use inotify to watch for config changes:

```bash
# Install inotify-tools
sudo apt install inotify-tools

# Watch config and auto-restart
while inotifywait -e modify config/config.yaml; do
    sudo systemctl restart telescope_detection.service
done
```

## Security Considerations

### Network Access

- Web interface binds to `0.0.0.0` (all interfaces)
- Consider using firewall to restrict access:
  ```bash
  sudo ufw allow from 192.168.1.0/24 to any port 8000
  ```

### Camera Credentials

- Camera password is stored in `config/config.yaml`
- Protect this file:
  ```bash
  chmod 600 config/config.yaml
  ```

## Getting Help

### Check Logs First

```bash
# Recent errors
./service.sh logs | grep ERROR

# Last 200 lines
./service.sh logs

# Follow live
./service.sh logs -f
```

### Useful Commands

```bash
# Full system status
./service.sh status

# GPU status
nvidia-smi

# Test camera
python tests/test_camera_connection.py

# Test inference
python tests/test_inference.py
```

### Common Log Messages

- `INFO: Stream capture started` - Camera connected successfully
- `INFO: Inference engine started` - GPU detection working
- `INFO: System is running!` - All systems operational
- `ERROR: Failed to connect to camera` - Check camera IP/credentials
- `ERROR: CUDA not available` - Check GPU/CUDA setup

## Service Status Examples

### Healthy Running Service

```
● telescope_detection.service - Telescope Wildlife Detection System
   Loaded: loaded (/etc/systemd/system/telescope_detection.service; enabled)
   Active: active (running) since Mon 2025-10-05 12:00:00 MST
   Main PID: 12345
   Status: "Running detection pipeline"
```

### Service with Issues

```
● telescope_detection.service - Telescope Wildlife Detection System
   Loaded: loaded (/etc/systemd/system/telescope_detection.service; enabled)
   Active: failed (Result: exit-code)

   Check logs with: journalctl -u telescope_detection.service
```

## Uninstalling

To completely remove the service:

```bash
# Stop and uninstall
sudo ./service.sh stop
sudo ./service.sh uninstall

# Optionally remove clips and models
# rm -rf clips/ models/
```

---

## Summary of Common Commands

```bash
# Setup (first time)
sudo ./service.sh install
sudo ./service.sh start

# Daily operations
./service.sh status        # Check if running
./service.sh logs          # View recent activity
./service.sh logs -f       # Watch live logs

# Maintenance
sudo ./service.sh restart  # After config changes
sudo ./service.sh stop     # Temporarily stop

# Troubleshooting
journalctl -u telescope_detection.service -n 100
systemctl status telescope_detection.service -l
```
