# Stage 2 Species Classification Setup Guide

## Current Status

✅ **Stage 1 (YOLOX Detection)**: ACTIVE and working
- Detects 80 COCO classes (person, bird, cat, dog, etc.)
- Wildlife-only filter available
- 11-21ms inference with YOLOX-S @ 1920x1920

✅ **Stage 2 (iNaturalist Classification)**: FULLY IMPLEMENTED and working
- iNaturalist EVA02-Large model (10,000 species)
- Hierarchical taxonomy fallback (species → genus → family → order → class)
- Geographic filtering for Mojave Desert species
- Time-of-day aware re-ranking (birds at night → owls, not quail)
- 20-30ms per detection

## What is Stage 2?

Stage 2 adds **fine-grained species identification** on top of YOLOX detection:

1. **Stage 1**: YOLOX detects "bird" (class-level) and creates bounding box
2. **Stage 2**: iNaturalist classifier crops that box and identifies "Gambel's Quail" (species-level)

**Performance**: Adds ~20-30ms per detection, total pipeline ~30-50ms (still real-time!)

## What's Already Built

✅ **COMPLETE IMPLEMENTATION:**
- `src/species_classifier.py` - iNaturalist EVA02 classifier
- `src/two_stage_pipeline_yolox.py` - Two-stage orchestration
- `src/species_activity_patterns.py` - 128+ species activity database
- Inference engine fully integrated
- Config structure complete
- Hierarchical taxonomy with fallback
- Geographic filtering
- Time-of-day filtering and re-ranking

## Current Configuration in `config/config.yaml`:

```yaml
detection:
  use_two_stage: true                    # Enable Stage 2
  enable_species_classification: true

species_classification:
  enabled: true
  confidence_threshold: 0.5              # Species-level minimum confidence

  inat_classifier:
    # Model automatically downloaded from HuggingFace
    model_name: "timm/eva02_large_patch14_clip_336.merged2b_ft_inat21"
    taxonomy_file: "models/inat2021_taxonomy.json"
    input_size: 336
    use_hierarchical: true               # Fallback: species → genus → family

    # Geographic filtering (optional)
    enable_geographic_filter: false
    allowed_species_file: "config/mojave_desert_species.txt"

  # Preprocessing options (per-camera overrides available)
  crop_padding_percent: 20               # Add 20% context around bbox
  min_crop_size: 50                      # Skip Stage 2 for tiny crops

  # Time-of-day filtering (integrated with time_of_day_filter)
  time_of_day_top_k: 5                   # Re-rank top-5 species
  time_of_day_penalty: 0.3               # Penalize unlikely species (70% reduction)

  # Rejected taxonomic levels (too vague)
  rejected_taxonomic_levels:
    - "order"
    - "class"
```

## How to Use Stage 2

### Enable Stage 2 (if disabled)

1. **Edit `config/config.yaml`:**
```yaml
detection:
  use_two_stage: true
  enable_species_classification: true

species_classification:
  enabled: true
```

2. **Restart the system:**
```bash
sudo ./service.sh restart
```

3. **Verify in logs:**
```bash
./service.sh logs | grep "Stage 2"
```

You should see:
```
INFO: Two-stage detection pipeline initialized (YOLOX + iNaturalist)
INFO: Added species classifier for category: bird
INFO: Added species classifier for category: mammal
INFO: ✓ Image enhancer loaded (if configured)
```

### Download iNaturalist Taxonomy (First Time)

The taxonomy file should already exist in `models/inat2021_taxonomy.json`. If it's missing:

```bash
python scripts/download_inat_taxonomy.py
```

This downloads the full iNaturalist 2021 taxonomy with:
- 10,000 species
- Hierarchical structure (kingdom → phylum → class → order → family → genus → species)
- Common names and scientific names

### Optional: Enable Geographic Filtering

To filter to Mojave Desert species only:

1. **Create species whitelist** (`models/mojave_desert_species.txt`):
```
Gambel's Quail
Greater Roadrunner
Cactus Wren
Desert Cottontail
Black-tailed Jackrabbit
Coyote
Bobcat
Kit Fox
Gila Monster
```

2. **Enable in config:**
```yaml
species_classification:
  inat_classifier:
    enable_geographic_filter: true
    allowed_species_file: "models/mojave_desert_species.txt"
```

3. **Restart:**
```bash
sudo ./service.sh restart
```

### Optional: Enable Image Enhancement

For better classification of distant/blurry animals:

```yaml
species_classification:
  enhancement:
    enabled: true
    method: "clahe"  # Fast, or "realesrgan" for quality

    clahe:
      clip_limit: 2.0
      tile_grid_size: [8, 8]

    # Or use Real-ESRGAN (slower but better quality)
    # method: "realesrgan"
    # realesrgan:
    #   model_name: "RealESRGAN_x4plus"
    #   model_path: "models/enhancement/RealESRGAN_x4plus.pth"
    #   scale: 4
```

**Note**: Real-ESRGAN adds ~1000ms per detection. Only use for critical species ID needs.
**Recommendation**: Use CLAHE (adds ~5ms) for general purpose.
- Generic training data

**Time:** 1-2 hours to download and configure

### Option 2: Train Custom Models LATER
**Pros:**
- Perfect for your specific location
- Only your target species
- Highest accuracy for your animals

**Cons:**
- Need to collect training data
- Takes several days

**Time:** 1-2 weeks (data collection + training)

### Option 3: Skip Stage 2 for Now
**Pros:**
- YOLOX already working great
- Simpler system

**Cons:**
- No fine-grained species ID
- "bird" instead of "Gambel's Quail"

## Recommended Approach

**Phase 1 (NOW)**: Keep using YOLOX only
- Collect more wildlife detections
- Build training dataset from clips
- Verify YOLOX fixes rabbit/bird confusion

**Phase 2 (In 2-4 weeks)**: Add Stage 2 when you have data
- Download HuggingFace bird classifier for immediate use
- Train custom classifiers on your collected clips
- Enable two-stage pipeline

## Expected Results with Stage 2

**Before Stage 2 (Stage 1 only):**
```json
{
  "class_name": "bird",
  "confidence": 0.85,
  "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
}
```

**After Stage 2 (YOLOX + iNaturalist):**
```json
{
  "class_name": "bird",
  "confidence": 0.85,
  "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
  "species": "Gambel's Quail",
  "species_confidence": 0.92,
  "taxonomic_level": "species",
  "stage2_category": "bird"
}
```

**With Hierarchical Fallback (low confidence):**
```json
{
  "class_name": "bird",
  "confidence": 0.75,
  "bbox": {...},
  "species": "Odontophoridae",     # Family level (quail family)
  "species_confidence": 0.45,
  "taxonomic_level": "family",
  "stage2_category": "bird"
}
```

## Performance Impact

- **Stage 1 only**: ~11-21ms per frame (YOLOX-S @ 1920x1920)
- **Stage 1 + Stage 2**: ~30-50ms per frame (+ 20-30ms per detection)
- **Difference**: +20-30ms per detection
- **Still real-time**: 25-30 FPS sustained (excellent for monitoring)

## Troubleshooting

### Stage 2 Not Working

**Check logs:**
```bash
./service.sh logs | grep -i "stage 2"
```

**Common issues:**
1. `use_two_stage: false` in config → Set to `true`
2. Missing taxonomy file → Run `python scripts/download_inat_taxonomy.py`
3. Model download failed → Check internet connection, HuggingFace availability

### Low Species Confidence

**Solutions:**
1. **Enable image enhancement:** Add CLAHE or Real-ESRGAN preprocessing
2. **Increase crop padding:** `crop_padding_percent: 40` for more context
3. **Use hierarchical fallback:** Gets genus/family when species uncertain
4. **Enable geographic filtering:** Limits to local species (higher accuracy)

### False Species IDs at Night

**Enable time-of-day filtering:**
```yaml
species_classification:
  time_of_day_top_k: 5
  time_of_day_penalty: 0.3
```

This automatically re-ranks species based on activity patterns (e.g., favors owls over quail at night).

## Advanced Configuration

### Per-Camera Stage 2 Settings

Override preprocessing for specific cameras:

```yaml
cameras:
  - id: "cam1"
    name: "Distant View"
    stage2_preprocessing:
      crop_padding_percent: 40    # More context for distant animals
      min_crop_size: 100          # Skip tiny distant crops

  - id: "cam2"
    name: "Close View"
    stage2_preprocessing:
      crop_padding_percent: 20    # Less padding needed
      min_crop_size: 50           # Classify smaller crops
```

## Questions?

- **"Do I need Stage 2?"** - Yes, if you want species-level identification. No, if class-level ("bird") is sufficient.
- **"Will Stage 2 slow things down?"** - Yes (+20-30ms per detection), but still real-time (25-30 FPS).
- **"How accurate is it?"** - 92% top-1 for iNaturalist validation set. Real-world: 80-90% for common species.
- **"Can I disable it?"** - Yes, set `use_two_stage: false` in config.
- **"Does it work at night?"** - Yes, with time-of-day filtering it favors nocturnal species.

---

**Current Status: Stage 2 (iNaturalist) is FULLY IMPLEMENTED and ready to use. Enable it in config to get species-level identification!**
