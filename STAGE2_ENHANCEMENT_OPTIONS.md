# Stage 2 Image Enhancement Options

**Context:** Stage 2 species classification doesn't need to be real-time (up to 60s acceptable). This allows us to apply sophisticated image enhancement to improve iNaturalist classification accuracy, especially for IR/night vision grayscale images.

**Problem:** iNaturalist trained on color photos. IR/night vision produces grayscale with poor contrast, noise, and small distant subjects.

**Solution:** Pre-process bounding box crops before Stage 2 classification.

---

## Enhancement Strategy Tiers

### **Tier 1: Traditional Methods** (5-15ms) ⭐ IMPLEMENTED

**CLAHE + Bilateral Denoising**
- **Speed:** 5-10ms per crop
- **Size:** 0MB (OpenCV built-in)
- **Pros:**
  - ✅ Fast, no model loading
  - ✅ Proven to help IR images
  - ✅ Works well for low-contrast night vision
  - ✅ Research-backed: "Simple grayscale methods improve over naive identity method for infrared"
- **Cons:**
  - ⚠️ Limited improvement on very dark images
  - ⚠️ Doesn't help with small/distant subjects
- **Implementation:** OpenCV `cv2.createCLAHE()` + `cv2.bilateralFilter()`
- **Research:** Multiple 2024 papers confirm preprocessing helps downstream classification

---

### **Tier 2: Lightweight Deep Learning** (50-200ms)

#### **Zero-DCE (Zero-Reference Deep Curve Estimation)**
- **Speed:** 50-100ms on GPU
- **Size:** ~1MB model
- **Pros:**
  - ✅ Very small model
  - ✅ Zero-shot (no training data needed)
  - ✅ Designed specifically for low-light enhancement
- **Cons:**
  - ⚠️ Research: "suffers from overexposure" in some cases
  - ⚠️ Moderate quality vs newer methods
- **Repository:** https://github.com/Li-Chongyi/Zero-DCE
- **License:** Check repository

#### **SCI (Self-Calibrated Illumination)**
- **Speed:** ~100-150ms
- **Size:** Small
- **Pros:**
  - ✅ Lightweight
  - ✅ Unsupervised approach
- **Cons:**
  - ⚠️ Research notes: "poor enhancement effects" compared to newer methods
- **Use Case:** Fallback if Zero-DCE doesn't work well

---

### **Tier 3: SOTA Deep Learning** (200-500ms)

#### **SNR-Aware (Signal-to-Noise Ratio Aware)**
- **Speed:** 200-400ms on A30
- **Size:** Medium (~50-100MB)
- **Pros:**
  - ✅ 2024 research: "Superior performance in dark light enhancement"
  - ✅ Excellent for noisy IR images
  - ✅ Specifically handles low SNR (common in night vision)
  - ✅ Best supervised method per recent reviews
- **Cons:**
  - ⚠️ Slower than lightweight methods
  - ⚠️ Larger model size
- **Use Case:** Maximum quality low-light enhancement
- **Research:** NTIRE 2024 Challenge winner class

#### **Bilateral Enhancement Network (2024)**
- **Speed:** ~300ms
- **Size:** Medium
- **Pros:**
  - ✅ Very recent (2024 Nature Scientific Reports)
  - ✅ SNR fusion specifically for low-light
  - ✅ Generalizable (works on unseen data)
  - ✅ Lightweight for a deep learning method
- **Cons:**
  - ⚠️ May need implementation work
- **Research:** https://www.nature.com/articles/s41598-024-81706-2

---

### **Tier 4: Super-Resolution** (500ms-2s) ⭐ IMPLEMENTED

#### **Real-ESRGAN (4x upscaling)**
- **Speed:** 500ms-1s for 640x640 → 2560x2560 on A30
- **Size:** 64MB (RealESRGAN_x4plus.pth)
- **Pros:**
  - ✅ Excellent for small/distant animals
  - ✅ Makes tiny details visible for classification
  - ✅ PyTorch implementation ready
  - ✅ Pretrained models available
  - ✅ Well-maintained official repository
  - ✅ Proven technology (used in photography, medical imaging)
- **Cons:**
  - ⚠️ Slower than lightweight methods
  - ⚠️ Large model size
  - ⚠️ May introduce artifacts on heavily compressed images
- **Models:**
  - `RealESRGAN_x4plus.pth` - General purpose (recommended)
  - `RealESRGAN_x4plus_anime_6B.pth` - Sharper edges
  - `RealESRNet_x4plus.pth` - No GAN, more stable
- **Repository:** https://github.com/xinntao/Real-ESRGAN
- **License:** BSD-3-Clause
- **Use Case:** Primary enhancement for small/distant wildlife
- **Configuration:**
  - Tile processing for large images
  - FP16 option for 2x speedup
  - Configurable output scale (2x, 3x, 4x)

---

### **Tier 5: Advanced Multi-Model** (1-3s)

#### **Ensemble Approach: Real-ESRGAN + SNR-Aware + CLAHE**
- **Pipeline:**
  1. Real-ESRGAN 4x upscale (1s)
  2. SNR-Aware low-light enhance (300ms)
  3. CLAHE final polish (10ms)
  4. iNaturalist classify (30ms)
- **Total:** ~1.3s per detection
- **Pros:**
  - ✅ Maximum possible quality
  - ✅ Handles small subjects + low light + noise
  - ✅ Each step addresses different problem
- **Cons:**
  - ⚠️ Complex implementation
  - ⚠️ Multiple model dependencies
  - ⚠️ Highest computational cost
- **Use Case:** When absolute maximum accuracy needed

---

### **Tier 6: Experimental** (5-30s+)

#### **Zero-Shot Colorization**
- **Speed:** 5-10s with CLIP-based models
- **Size:** Large (hundreds of MB)
- **Pros:**
  - ✅ Adds realistic colors to grayscale
  - ✅ iNaturalist trained on color photos
  - ✅ Could help species that rely on color patterns
- **Cons:**
  - ⚠️ Very slow
  - ⚠️ Research: "pseudo-colorization often doesn't help" (but zero-shot wasn't tested)
  - ⚠️ Synthetic colors might confuse classifier
  - ⚠️ Uncertain benefit
- **Models:**
  - Colorization with OpenCV + Deep Learning
  - CLIP-guided colorization
- **Use Case:** Experimental - test if colorization helps iNaturalist accuracy
- **Status:** NOT RECOMMENDED until proven beneficial

#### **WildCLIP / BioCLIP (Alternative Classifier)**
- **Speed:** ~50ms inference
- **Size:** Similar to iNaturalist
- **Pros:**
  - ✅ Trained specifically on wildlife/biology
  - ✅ Better on African/European fauna
  - ✅ Zero-shot classification capability
  - ✅ May handle grayscale better
- **Cons:**
  - ⚠️ Different taxonomy than iNaturalist
  - ⚠️ May have fewer species coverage
- **Use Case:** Replace iNaturalist entirely, not enhance
- **Research:** 2025 paper shows efficient zero-shot animal behavior classification

---

## Recommended Implementations

### **Option A: Fast & Effective** (20-50ms total)
```python
1. CLAHE (5ms)
2. Bilateral denoise (10ms)
3. iNaturalist (30ms)
```
**Status:** ✅ Minimal enhancement, proven effective
**Best For:** Real-time constraints, visible wildlife
**Implementation Complexity:** ⭐ Easy

---

### **Option B: Balanced (RECOMMENDED)** (500ms-1s total) ⭐ CURRENT
```python
1. Real-ESRGAN 4x upscale (500ms-1s) - makes tiny animals visible
2. CLAHE + bilateral (15ms) - polish the result
3. iNaturalist (30ms)
```
**Status:** ⭐ IMPLEMENTED
**Best For:** Small/distant animals, general use
**Implementation Complexity:** ⭐⭐ Medium
**Why:** Sweet spot of performance vs complexity, huge win for distant wildlife

---

### **Option C: Maximum Quality** (1.5-2s total)
```python
1. Real-ESRGAN 4x upscale (1s)
2. SNR-Aware low-light enhance (400ms)
3. CLAHE polish (5ms)
4. iNaturalist (30ms)
```
**Status:** 🔄 Future enhancement
**Best For:** Night vision, maximum accuracy needed
**Implementation Complexity:** ⭐⭐⭐ Harder
**Requirements:** SNR-Aware model implementation + weights

---

### **Option D: Experimental** (10-30s)
```python
1. Zero-shot colorization (5-10s)
2. Real-ESRGAN upscale (1s)
3. CLAHE (5ms)
4. iNaturalist (30ms)
```
**Status:** ❓ Experimental - needs validation
**Best For:** Species with distinct color patterns
**Implementation Complexity:** ⭐⭐⭐⭐ Hardest
**Warning:** May not improve accuracy, could hurt it

---

## Implementation Priority

1. ✅ **Tier 1 (Traditional)** - DONE
2. ⭐ **Tier 4 (Real-ESRGAN)** - IN PROGRESS (Option B)
3. 🔄 **Tier 3 (SNR-Aware)** - Future (Option C)
4. 🔄 **Tier 2 (Zero-DCE)** - Future fallback option
5. ❓ **Tier 6 (Experimental)** - Research needed

---

## Performance Targets

| Enhancement | Time Budget | Target FPS | Use Case |
|-------------|-------------|------------|----------|
| None | 30ms | 33 FPS | Daytime, clear |
| CLAHE only | 50ms | 20 FPS | Night, quick |
| Real-ESRGAN | 1s | 1 FPS | Small animals |
| Full pipeline | 2s | 0.5 FPS | Maximum quality |

**Note:** Stage 2 runs per-detection, not per-frame. Multiple detections in one frame processed sequentially.

---

## Research References

1. **Low-light Enhancement Survey (2024)**: https://www.sciencedirect.com/science/article/pii/S1319157824003239
2. **NTIRE 2024 Challenge**: https://arxiv.org/html/2404.14248v1
3. **SNR-Aware Enhancement (2024)**: https://www.nature.com/articles/s41598-024-81706-2
4. **Real-ESRGAN**: https://github.com/xinntao/Real-ESRGAN
5. **Zero-Shot Wildlife Classification (2025)**: https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.70059
6. **IR to RGB Transfer Learning**: Meta-Learning for Color-to-Infrared Cross-Modal Style Transfer

---

## Configuration

Enhancement settings stored in `config/config.yaml`:

```yaml
species_classification:
  enhancement:
    enabled: true
    method: "realesrgan"  # Options: "none", "clahe", "realesrgan", "ensemble"
    realesrgan:
      model_path: "models/enhancement/RealESRGAN_x4plus.pth"
      scale: 4
      tile: 512  # Process in tiles for large images
      tile_pad: 10
      pre_pad: 0
      half: false  # FP16 for 2x speedup
    clahe:
      clip_limit: 2.0
      tile_grid_size: [8, 8]
    bilateral:
      d: 9
      sigma_color: 75
      sigma_space: 75
```

---

## Future Enhancements

- [ ] A/B testing framework to compare methods
- [ ] Automatic enhancement selection based on image characteristics
- [ ] Time-based enhancement (more aggressive at night)
- [ ] Ensemble voting from multiple enhancement methods
- [ ] Custom lightweight model trained on our wildlife data
- [ ] GPU batching for multiple detections
- [ ] Caching enhancement results for recurring patterns

---

**Last Updated:** 2025-10-05
**Status:** Option B (Real-ESRGAN) in active development
