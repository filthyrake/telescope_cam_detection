# Privacy-Preserving Face Masking

**Status:** Implemented ✅
**Issue:** [#87](https://github.com/filthyrake/telescope_cam_detection/issues/87)
**License:** Apache 2.0 (OpenCV Haar Cascades)

## Overview

The privacy-preserving face masking feature automatically detects and masks faces in both live video feeds and saved clips to protect individual privacy. The system uses a dual-storage approach: unmasked versions are stored on the backend for security investigation, while only masked versions are served to users through the web UI.

## Features

✅ **Automatic face detection** using OpenCV Haar Cascades (fast, CPU-friendly)
✅ **Multiple masking styles**: Gaussian blur, pixelation, black box, adaptive blur
✅ **Dual storage**: Unmasked (backend) and masked (web UI) versions
✅ **Live feed masking**: Real-time face masking in MJPEG video streams
✅ **Performance caching**: Detect faces every N frames to reduce CPU usage
✅ **Per-camera configuration**: Enable/disable masking per camera
✅ **Configurable masking strength**: Adjust blur intensity, pixelation level, etc.

## Quick Start

### 1. Enable Face Masking

Edit `config/config.yaml`:

```yaml
privacy:
  enable_face_masking: true  # Enable face masking
  mask_style: "gaussian_blur"  # Masking style
  detection_backend: "opencv_haar"  # Detection backend
```

### 2. Restart System

```bash
sudo ./service.sh restart
```

### 3. Verify Operation

- **Live feed**: Visit `http://<host>:8000` - faces should be blurred in real-time
- **Saved clips**: Check `clips/<camera_id>/masked/` directory
- **Unmasked clips** (restricted): Check `clips/<camera_id>/raw/` directory

## Configuration

### Global Privacy Settings

Located in `config/config.yaml` under the `privacy` section:

```yaml
privacy:
  # Master switch
  enable_face_masking: false  # Set to true to enable

  # Masking style
  mask_style: "gaussian_blur"  # Options: gaussian_blur, pixelate, black_box, adaptive_blur

  # Detection backend
  detection_backend: "opencv_haar"  # Options: opencv_haar, mediapipe (if installed)

  # Performance tuning
  live_feed_detection_interval: 5  # Detect faces every N frames (default: 5)
  min_face_size: 30  # Minimum face size in pixels (filters false positives)
  blur_strength: 25  # Gaussian blur kernel size (odd number, higher = more blur)
  pixelate_blocks: 10  # Pixel block size for pixelation effect

  # Haar Cascade parameters (advanced)
  scale_factor: 1.1  # Lower = more detections but slower (1.05-1.5 typical)
  min_neighbors: 5  # Higher = fewer false positives (3-6 typical)

  # Per-camera overrides (optional)
  camera_overrides: {}
```

### Per-Camera Overrides

Disable face masking for specific cameras or use different masking styles:

```yaml
privacy:
  enable_face_masking: true
  mask_style: "gaussian_blur"

  camera_overrides:
    cam1:
      enable_face_masking: false  # Disable for remote outdoor camera
    cam2:
      mask_style: "black_box"  # Use different masking style
      blur_strength: 35  # Stronger blur for this camera
```

### API Access Control

Control access to unmasked clips:

```yaml
api:
  allow_unmasked_access: false  # Require authentication for /api/clips/raw/
  unmasked_access_token_env: "TELESCOPE_UNMASKED_TOKEN"  # Environment variable for token
```

**Security Note:** Set the environment variable for production:

```bash
export TELESCOPE_UNMASKED_TOKEN="your-secure-random-token-here"
```

## Masking Styles

### 1. Gaussian Blur (Default)

**Description:** Smooth, natural-looking blur effect
**Performance:** Fast (~5-10ms per face)
**Use case:** General-purpose, least jarring for viewers

```yaml
privacy:
  mask_style: "gaussian_blur"
  blur_strength: 25  # Higher = more blur (odd numbers only)
```

**Example:** Face region is blurred with a Gaussian filter, making features indistinguishable while maintaining natural appearance.

### 2. Pixelation

**Description:** Retro mosaic/pixelated effect
**Performance:** Fast (~5-10ms per face)
**Use case:** When you want a more obvious privacy indicator

```yaml
privacy:
  mask_style: "pixelate"
  pixelate_blocks: 10  # Block size (lower = larger blocks)
```

**Example:** Face region is downsampled and upsampled to create pixelation effect.

### 3. Black Box

**Description:** Solid black rectangle over face
**Performance:** Very fast (~1-2ms per face)
**Use case:** Maximum privacy, most aggressive masking

```yaml
privacy:
  mask_style: "black_box"
```

**Example:** Face region is completely replaced with a black rectangle.

### 4. Adaptive Blur

**Description:** Blur strength adapts based on face size (larger faces get more blur)
**Performance:** Fast (~5-10ms per face)
**Use case:** When faces vary greatly in distance from camera

```yaml
privacy:
  mask_style: "adaptive_blur"
  blur_strength: 25  # Base blur strength (scales up for larger faces)
```

**Example:** Close-up faces get stronger blur, distant faces get lighter blur.

## Storage Structure

When face masking is enabled, clips are saved to a per-camera directory structure:

```
clips/
  cam1/
    raw/                    # Unmasked (restricted access)
      person_20250101_120000_conf0.95.jpg
      person_20250101_120000_conf0.95.json
    masked/                 # Masked (served by web UI)
      person_20250101_120000_conf0.95.jpg
      person_20250101_120000_conf0.95.json
    annotated/              # Masked + detection boxes
      person_20250101_120000_conf0.95.jpg

  cam2/
    raw/
    masked/
    annotated/
```

### Directory Purposes

- **raw/**: Unmasked frames for security investigation (not served by web UI)
- **masked/**: Masked frames without detection boxes (served by clips gallery)
- **annotated/**: Masked frames with detection boxes (served by live feed overlays)

### Metadata

Each clip includes a JSON metadata file:

```json
{
  "timestamp": 1704110400.123,
  "camera_id": "cam1",
  "camera_name": "Main Backyard View",
  "detections": [...],
  "detection_counts": {"person": 1},
  "face_masking_enabled": true,
  "faces_detected": 1
}
```

## Performance Optimization

### Face Detection Caching

To avoid detecting faces in every frame (CPU-intensive), the system caches face positions:

```yaml
privacy:
  live_feed_detection_interval: 5  # Detect faces every 5 frames
```

**How it works:**
1. Detect faces in frame 1 → cache positions
2. Frames 2-5 → reuse cached positions
3. Frame 6 → detect faces again, update cache

**Trade-offs:**
- Lower value (1-3): More accurate, higher CPU usage
- Higher value (10+): Less accurate (faces may move), lower CPU usage
- Recommended: 5 frames (~150ms at 30 FPS)

### Minimum Face Size

Filter out false positives (small shadows, distant objects):

```yaml
privacy:
  min_face_size: 30  # Minimum face size in pixels
```

**Guidelines:**
- **1920x1080 streams:** 30-50 pixels
- **640x480 streams:** 20-30 pixels
- **4K streams:** 50-80 pixels

## Detection Backends

### OpenCV Haar Cascades (Default)

**Pros:**
- ✅ Fast (~5-10ms per frame)
- ✅ No GPU required
- ✅ Already included in OpenCV dependency
- ✅ Works well for frontal faces

**Cons:**
- ❌ Lower accuracy than deep learning methods
- ❌ May miss profile/rotated faces
- ❌ More false positives

**Use case:** Default choice for real-time masking with minimal CPU impact

**Configuration:**
```yaml
privacy:
  detection_backend: "opencv_haar"
  scale_factor: 1.1  # Detection sensitivity (lower = more sensitive)
  min_neighbors: 5   # False positive filtering (higher = fewer false positives)
```

### MediaPipe Face Detection (Optional)

**Pros:**
- ✅ Better accuracy than Haar cascades
- ✅ Still fast (~15-20ms per frame)
- ✅ Detects rotated/profile faces

**Cons:**
- ❌ Requires additional dependency (`pip install mediapipe`)
- ❌ Slightly slower than Haar cascades

**Installation:**
```bash
source venv/bin/activate
pip install mediapipe
```

**Configuration:**
```yaml
privacy:
  detection_backend: "mediapipe"
```

### YOLOX-based (Future)

**Status:** Not yet implemented (placeholder in code)

**Pros:**
- ✅ Very accurate
- ✅ Leverages existing YOLOX infrastructure

**Cons:**
- ❌ GPU bottleneck (YOLOX already running for wildlife detection)
- ❌ Would need to add face detection class to YOLOX model

## Testing

Run face masking tests:

```bash
python tests/test_face_masking.py
```

**Test coverage:**
- ✅ Face detection (Haar Cascades, MediaPipe)
- ✅ All masking styles (blur, pixelate, black box, adaptive)
- ✅ Caching mechanism
- ✅ Per-camera cache isolation
- ✅ Mask style overrides

## Security Considerations

### Unmasked Clip Access

**Default:** Unmasked clips are stored in `clips/<camera_id>/raw/` but **not served** by the web server.

**Accessing unmasked clips:**

1. **Direct filesystem access** (requires SSH/server access):
   ```bash
   ls clips/cam1/raw/
   ```

2. **API endpoint** (future enhancement - not yet implemented):
   ```bash
   curl -H "Authorization: Bearer $TELESCOPE_UNMASKED_TOKEN" \
        http://<host>:8000/api/clips/raw/cam1/person_20250101_120000.jpg
   ```

### Token Management

**Option 1: Environment Variable (Recommended)**
```bash
export TELESCOPE_UNMASKED_TOKEN="your-secure-random-token"
```

**Option 2: Generate Secure Token**
```bash
openssl rand -hex 32  # Generates a 64-character hex token
```

### Filesystem Permissions

Ensure `clips/*/raw/` directories are not served by any web server:

```bash
# Check that raw/ directories are not accessible via web
curl http://<host>:8000/clips/cam1/raw/test.jpg  # Should return 404
```

## Troubleshooting

### Issue: No faces detected

**Possible causes:**
1. Faces are too small (below `min_face_size`)
2. Faces are rotated/profile (Haar Cascades only detect frontal faces)
3. Poor lighting/image quality
4. `scale_factor` or `min_neighbors` too restrictive

**Solutions:**
- Lower `min_face_size` (e.g., from 30 to 20)
- Lower `scale_factor` (e.g., from 1.1 to 1.05) for more sensitive detection
- Lower `min_neighbors` (e.g., from 5 to 3) to reduce filtering
- Switch to MediaPipe backend for better rotation handling

### Issue: Too many false positives

**Possible causes:**
1. `scale_factor` too low (overly sensitive)
2. `min_neighbors` too low (insufficient filtering)
3. Low-quality camera feed (noise, compression artifacts)

**Solutions:**
- Increase `min_neighbors` (e.g., from 5 to 6 or 7)
- Increase `scale_factor` (e.g., from 1.1 to 1.2)
- Increase `min_face_size` to filter small detections

### Issue: High CPU usage

**Possible causes:**
1. `live_feed_detection_interval` too low (detecting every frame)
2. Multiple high-resolution cameras

**Solutions:**
- Increase `live_feed_detection_interval` (e.g., from 5 to 10)
- Disable face masking for remote cameras using `camera_overrides`
- Consider disabling face masking on live feed, enable only for saved clips

### Issue: Faces not masked in live feed

**Possible causes:**
1. Face masking enabled but not reloaded
2. Error in face detection (check logs)

**Solutions:**
- Restart system: `sudo ./service.sh restart`
- Check logs: `./service.sh logs -f | grep -i face`
- Verify configuration: `curl http://<host>:8000/api/config/current | jq .config.privacy`

## Limitations

### Current Limitations

1. **Frontal faces only** (Haar Cascades): Profile and rotated faces may not be detected
2. **No access control API** (yet): Unmasked clips require direct filesystem access
3. **No batch masking tool** (yet): Existing clips must be manually reprocessed
4. **MediaPipe optional**: Requires manual installation

### Planned Enhancements (Future)

- [ ] API endpoint for unmasked clip access with authentication
- [ ] Batch masking script for existing clips
- [ ] YOLOX-based face detection (leverage existing GPU pipeline)
- [ ] Region-based masking (only mask faces in specific zones)
- [ ] Configurable retention policy for unmasked clips
- [ ] Person detection masking (blur entire body, not just face)

## Best Practices

### Privacy Compliance

1. **Inform users**: Post signs indicating camera surveillance with face masking
2. **Retention policy**: Delete unmasked clips after investigation period (e.g., 7 days)
3. **Access logs**: Log access to unmasked clips for compliance
4. **Regular audits**: Verify masked clips are being served (not unmasked)

### Performance

1. **Start with default settings**: `live_feed_detection_interval: 5`, `min_face_size: 30`
2. **Monitor CPU usage**: Use `top` or `htop` to check CPU load
3. **Adjust per-camera**: Disable for remote outdoor cameras where privacy is not a concern
4. **Test masking styles**: Different styles have different performance characteristics

### Security

1. **Restrict filesystem access**: Ensure only authorized users can access `clips/*/raw/`
2. **Use strong tokens**: Generate cryptographically secure tokens (see Token Management)
3. **HTTPS in production**: Serve web UI over HTTPS to prevent token interception
4. **Regular updates**: Keep OpenCV and dependencies updated for security patches

## Examples

### Example 1: Enable for all cameras

```yaml
privacy:
  enable_face_masking: true
  mask_style: "gaussian_blur"
  blur_strength: 25
```

### Example 2: Enable for front door only

```yaml
privacy:
  enable_face_masking: false  # Disabled globally

  camera_overrides:
    front_door_cam:
      enable_face_masking: true  # Enable for this camera only
      mask_style: "adaptive_blur"
```

### Example 3: Aggressive masking for high-traffic area

```yaml
privacy:
  enable_face_masking: true
  mask_style: "black_box"  # Maximum privacy
  min_face_size: 20  # Detect smaller faces
  live_feed_detection_interval: 3  # More frequent detection
```

## Support

For issues, questions, or feature requests:

- **GitHub Issues:** https://github.com/filthyrake/telescope_cam_detection/issues
- **Documentation:** `docs/features/PRIVACY_MASKING.md` (this file)
- **Configuration Reference:** `docs/setup/CONFIG_REFERENCE.md`

## License

This feature uses OpenCV Haar Cascades (Apache 2.0 license) for face detection, which is compatible with commercial use.

**Optional MediaPipe backend:** Apache 2.0 license (commercial-friendly)

---

**Last Updated:** 2025-10-14
**Feature Status:** ✅ Implemented (MVP)
**Related Issue:** [#87](https://github.com/filthyrake/telescope_cam_detection/issues/87)
