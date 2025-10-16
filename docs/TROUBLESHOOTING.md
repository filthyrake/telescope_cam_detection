# Troubleshooting Guide

Common issues and solutions for the Telescope Detection System.

## Camera Connection Issues

### Test Connection

```bash
python tests/test_camera_connection.py
```

### Common Problems

**Wrong IP address**
- Check camera IP in router's DHCP list
- Try pinging the camera: `ping 192.168.1.100`

**Wrong credentials**
- Verify `camera_credentials.yaml` has correct username/password
- Default Reolink credentials: username `admin`, password is what you set during camera setup

**Network issues**
- Ensure camera is on same subnet as server
- Check firewall isn't blocking port 554 (RTSP)
- Try accessing camera web UI in browser: `http://192.168.1.100`

**Stream not available**
- Some cameras don't support "sub" stream - use "main" instead
- Check camera settings to ensure RTSP is enabled

## GPU/CUDA Issues

### Verify GPU

```bash
# Check GPU is detected
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### CUDA Not Available

**Driver issues**
```bash
# Check NVIDIA driver version
nvidia-smi

# If driver missing or outdated, reinstall
sudo apt update
sudo apt install nvidia-driver-535
```

**PyTorch issues**
```bash
# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Performance Issues

### High Latency (>50ms)

Try these in order:

1. **Reduce model size**
   ```yaml
   # In config/config.yaml
   detection:
     model:
       name: "yolox-tiny"  # or yolox-nano for fastest
   ```

2. **Lower input size**
   ```yaml
   detection:
     input_size: [640, 640]  # down from [1920, 1920]
   ```

3. **Use sub stream**
   ```yaml
   cameras:
     - stream: "sub"  # lower resolution stream
   ```

4. **Disable Stage 2** (if enabled)
   ```yaml
   detection:
     use_two_stage: false
   ```

5. **Check network latency**
   ```bash
   ping 192.168.1.100  # Should be <5ms
   ```

### Low FPS

- Check GPU isn't thermal throttling: `nvidia-smi` (temp should be <80°C)
- Reduce number of cameras or use smaller model
- Ensure no other processes using GPU: `nvidia-smi`

### High Memory Usage

**RAM issues**
- Reduce buffer sizes in config
- Disable snapshot saving temporarily
- Close other applications

**VRAM issues**
- System has **automatic GPU OOM graceful degradation** (see [GPU OOM Graceful Degradation](features/OOM_GRACEFUL_DEGRADATION.md))
- Progressive degradation: cache clearing → batch reduction → input size reduction → CPU fallback
- Monitor GPU memory in web UI or via `/api/system/stats` endpoint
- Manual tuning:
  - Use smaller model (yolox-tiny instead of yolox-s)
  - Reduce input_size
  - Reduce number of cameras

## Web Interface Issues

### Can't Access Web UI

```bash
# Check if server is running
curl http://localhost:8000/health

# Check firewall
sudo ufw status

# Allow port if needed
sudo ufw allow 8000
```

### Video Feed Not Loading

- Check browser console for errors (F12)
- Try different browser (Chrome/Firefox recommended)
- Ensure cameras are connected: check logs with `./service.sh logs`

### WebSocket Disconnecting

- Check for network issues
- Increase timeouts in web server config
- Check browser console for specific error messages

## Service Issues

### Service Won't Start

```bash
# Check service status
./service.sh status

# View detailed logs
./service.sh logs -f

# Common causes:
# - Missing camera_credentials.yaml
# - Wrong Python path in service file
# - Permissions issue (try: sudo chown -R $USER:$USER /path/to/telescope_cam_detection)
```

### Service Crashes Repeatedly

```bash
# Check logs for errors
./service.sh logs

# Test running manually to see errors
source venv/bin/activate
python main.py
```

## Detection Issues

### Too Many False Positives

```yaml
# In config/config.yaml
detection:
  conf_threshold: 0.30  # Increase from 0.15
  min_box_area: 200     # Increase from 50

  # Or adjust per-class thresholds
  class_confidence_overrides:
    bird: 0.70  # Increase from 0.55
    person: 0.80
```

### Missing Detections

```yaml
# In config/config.yaml
detection:
  conf_threshold: 0.10  # Lower threshold
  input_size: [1920, 1920]  # Larger size for small objects

  # Make sure wildlife_only isn't filtering out what you want
  wildlife_only: false  # If you need all COCO classes
```

### Wrong Species Classification

Stage 2 species classification requires:
- Properly trained models in `models/species_classifiers/`
- Adequate cropped image size (adjust `crop_padding` in config)
- Good lighting conditions

See [STAGE2_SETUP.md](features/STAGE2_SETUP.md) for details.

## Snapshot Issues

### Snapshots Not Saving

```yaml
# Check config
snapshots:
  enabled: true
  trigger_classes: []  # Empty = save all detections
  min_confidence: 0.25
```

Check permissions:
```bash
ls -la clips/
# Should be writable by user running the service
```

### Too Many Snapshots

```yaml
snapshots:
  cooldown_seconds: 120  # Increase from 45
  trigger_classes: ["person", "dog"]  # Only save specific classes
  min_confidence: 0.50  # Raise threshold
```

### Disk Space Issues

```bash
# Check disk usage
df -h

# Clean old clips
rm clips/*  # Or use scripts/cleanup_old_clips.sh if it exists
```

## Log Analysis

### Finding Errors

```bash
# Follow logs in real-time
./service.sh logs -f

# Search for errors
./service.sh logs | grep -i error

# Check specific timeframe
journalctl -u telescope-detection --since "10 minutes ago"
```

### Understanding Log Messages

**Normal operation:**
- "Frame captured from cam1" - Camera working
- "Inference took 15.2ms" - GPU working
- "Detection: bird (0.65)" - Detection working

**Problems:**
- "Failed to connect to camera" - Check camera connection
- "CUDA out of memory" - System auto-recovers (see [OOM Graceful Degradation](features/OOM_GRACEFUL_DEGRADATION.md))
- "No frames in queue" - Camera connection lost

## Getting Help

If you're still stuck:

1. Check logs: `./service.sh logs`
2. Test components individually using scripts in `tests/`
3. Review configuration in `config/config.yaml`
4. Check [GitHub Issues](https://github.com/filthyrake/telescope_cam_detection/issues)
