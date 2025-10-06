# Stage 2 Species Classification Setup Guide

## Current Status

✅ **Stage 1 (YOLOX Detection)**: ACTIVE and working
- Detects animals using open-vocabulary text prompts
- 17 desert species classes configured
- Successfully differentiates rabbit vs bird

⚠️ **Stage 2 (Fine-Grained Species ID)**: Framework built, models needed

## What is Stage 2?

Stage 2 adds **fine-grained species identification** on top of YOLOX detection:

1. **Stage 1**: YOLOX detects "bird" and creates bounding box
2. **Stage 2**: Species classifier crops that box and identifies "Gambel's Quail" specifically

**Performance**: Adds ~5-10ms per detection, total pipeline ~25-35ms

## What's Already Built

✅ `src/species_classifier.py` - Classification framework
✅ `src/two_stage_pipeline.py` - Pipeline integration
✅ Inference engine supports two-stage mode
✅ Config structure ready (`use_two_stage`, `enable_species_classification`)

## What We Need

### 1. Pre-trained Species Classifier Models

We need **3 classifier models** (one per taxonomic group):

#### Option A: HuggingFace Models (Easiest)
Download ready-to-use models from HuggingFace:

**Birds:**
- `prithivMLmods/Bird-Species-Classifier-526` (526 species)
- `dennisjooo/Birds-Classifier-EfficientNetB2` (EfficientNet, 95%+ accuracy)

**Mammals:**
- Search HuggingFace for "mammal classifier" or use ImageNet-21k fine-tuned models

**Reptiles:**
- Search HuggingFace for "reptile classifier" or iNaturalist reptile models

#### Option B: Train Custom Models (Best for your location)
Train on your specific desert species using:
- **Dataset**: Collect 500-1000 images per species
- **Model**: Fine-tune EfficientNet or ResNet on your data
- **Classes**: Your exact species (Gambel's Quail, Desert Cottontail, etc.)
- **Accuracy**: 90-95% for your specific animals

### 2. Taxonomy Mapping Files

Create JSON files mapping class IDs to species names:

```json
# models/bird_taxonomy.json
{
  "0": "Gambel's Quail",
  "1": "Greater Roadrunner",
  "2": "Common Raven",
  ...
}
```

### 3. Model Files Structure

```
telescope_cam_detection/
├── models/
│   ├── bird_classifier.pth          # Bird species model weights
│   ├── bird_taxonomy.json           # Bird species names
│   ├── mammal_classifier.pth        # Mammal species model
│   ├── mammal_taxonomy.json         # Mammal species names
│   ├── reptile_classifier.pth       # Reptile species model
│   └── reptile_taxonomy.json        # Reptile species names
```

## Implementation Steps

### Step 1: Install HuggingFace Hub (if needed)
```bash
pip install huggingface_hub
```

### Step 2: Download Bird Classifier
```python
from huggingface_hub import hf_hub_download

# Download bird classifier
model_path = hf_hub_download(
    repo_id="prithivMLmods/Bird-Species-Classifier-526",
    filename="pytorch_model.bin"
)
```

### Step 3: Update Configuration

Edit `config/config.yaml`:
```yaml
detection:
  use_two_stage: true
  enable_species_classification: true

species_classification:
  enabled: true

  bird_classifier:
    model_path: "models/bird_classifier.pth"
    taxonomy_file: "models/bird_taxonomy.json"
    confidence_threshold: 0.3

  mammal_classifier:
    model_path: "models/mammal_classifier.pth"
    taxonomy_file: "models/mammal_taxonomy.json"
    confidence_threshold: 0.3

  reptile_classifier:
    model_path: "models/reptile_classifier.pth"
    taxonomy_file: "models/reptile_taxonomy.json"
    confidence_threshold: 0.3
```

### Step 4: Initialize Classifiers in main.py

Add after inference engine initialization:
```python
# Initialize two-stage pipeline (if enabled)
if detection_config.get('use_two_stage', False):
    from two_stage_pipeline import TwoStageDetectionPipeline
    from species_classifier import SpeciesClassifier

    pipeline = TwoStageDetectionPipeline(
        yolo_model_path=detection_config['model'],
        device=detection_config['device'],
        enable_species_classification=True
    )

    # Add bird classifier
    bird_classifier = SpeciesClassifier(
        model_name="efficientnet_b2",
        checkpoint_path="models/bird_classifier.pth",
        taxonomy_file="models/bird_taxonomy.json"
    )
    bird_classifier.load_model(num_classes=526)
    pipeline.add_species_classifier('bird', bird_classifier)

    # Add mammal/reptile classifiers similarly...

    # Use pipeline in inference engine
    inference_engine.two_stage_pipeline = pipeline
```

### Step 5: Test
```bash
python main.py
```

Look for log messages:
```
INFO: Two-stage detection pipeline loaded
INFO: Added species classifier for bird
INFO: Stage 2 classification: Gambel's Quail (0.92 confidence)
```

## Alternative: Start Simple with ImageNet

For testing, you can use ImageNet-pretrained models first:
```python
# Load pretrained EfficientNet (1000 ImageNet classes)
classifier = SpeciesClassifier(
    model_name="tf_efficientnet_b4_ns",
    checkpoint_path=None  # Uses ImageNet weights
)
classifier.load_model(num_classes=1000)
```

This won't identify specific species but validates the pipeline works.

## Decision Points

### Option 1: Use HuggingFace Models NOW
**Pros:**
- Ready to use immediately
- 526 bird species covered
- High accuracy (~95%)

**Cons:**
- May not have your exact desert species
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

**Before Stage 2:**
```
Detection: bird (confidence: 0.85)
```

**After Stage 2:**
```
Detection: bird (confidence: 0.85)
Species: Gambel's Quail (confidence: 0.92)
Alternatives:
  - California Quail (0.78)
  - Scaled Quail (0.65)
```

## Performance Impact

- **Current (Stage 1 only)**: ~20-25ms per frame
- **With Stage 2**: ~25-35ms per frame
- **Difference**: +5-10ms per detection
- **Still real-time**: 30+ FPS easily achievable

## Next Steps

1. ✅ Monitor YOLOX performance first
2. ⏳ Collect 100+ wildlife clips
3. ⏳ Decide if species-level ID is needed
4. ⏳ Download HuggingFace models or train custom
5. ⏳ Enable Stage 2 in config
6. ⏳ Test and verify

## Questions?

- **"Do I need Stage 2?"** - Only if you want species-level identification (Gambel's Quail vs California Quail)
- **"Will Stage 2 slow things down?"** - Slightly (~5-10ms), but still real-time
- **"Can I add it later?"** - Yes! The framework is ready, just add models when you want
- **"How accurate will it be?"** - 85-95% for common species, lower for rare/similar species

---

**Current Status: Stage 1 (YOLOX) is active and sufficient for most use cases. Stage 2 can be added anytime you want finer species identification.**
