# Complete Training Data Workflow

## Current Status: 1,050 Images Captured ✅

### Existing Dataset Breakdown:
- **200 images**: Covered, dark conditions
- **150 images**: Covered, afternoon lighting
- **150 images**: Uncovered, daylight (telescope 1)
- **150 images**: Uncovered, daylight (both telescopes)
- **150 images**: Mixed state (one covered, one uncovered)
- **150 images**: Covered, daytime
- **100 images**: Covered, late afternoon

**These are VALUABLE and should be used!** Especially the covered telescope images across different lighting conditions.

## Phase 1: Continue Static Captures (Ongoing)

### As Day Progresses - Keep Capturing Covered States
Capture more as lighting changes:

```bash
# Evening lighting (golden hour)
python training/scripts/capture_training_images_headless.py \
  --num 100 --interval 2 --desc covered_evening

# Dusk/twilight
python training/scripts/capture_training_images_headless.py \
  --num 100 --interval 2 --desc covered_dusk

# Night with artificial lighting
python training/scripts/capture_training_images_headless.py \
  --num 100 --interval 2 --desc covered_night
```

**Goal:** ~1,500-2,000 total static images with good lighting diversity

## Phase 2: Dynamic Video Capture (When Ready to Slew)

### Extract Frames from Video Stream
When you're ready to move the telescopes around:

```bash
# 5-10 minute recording session
python training/scripts/extract_frames_from_stream.py \
  --duration 600 \
  --interval 1.0 \
  --desc dynamic_slewing
```

**What to do during recording:**
- Slew Telescope 1 through full altitude range (0-90°)
- Rotate full 360° azimuth
- Repeat for Telescope 2
- Move both simultaneously
- Try parking positions, meridian flips
- Just operate naturally!

**Expected Output:** 600 frames over 10 minutes = **massive diversity**

### Multiple Sessions Recommended:
```bash
# Session 1: Uncovered telescopes
python training/scripts/extract_frames_from_stream.py \
  --duration 600 --interval 1.0 --desc dynamic_uncovered

# Session 2: Covered telescopes
python training/scripts/extract_frames_from_stream.py \
  --duration 600 --interval 1.0 --desc dynamic_covered

# Session 3: Mixed states
python training/scripts/extract_frames_from_stream.py \
  --duration 300 --interval 1.0 --desc dynamic_mixed
```

**Goal:** ~1,500-2,000 dynamic orientation images

## Phase 3: Collision Scenarios (Deliberate Setup)

### Guided Collision Capture
Use the guided script that walks you through each scenario:

```bash
python training/scripts/capture_collision_scenarios.py
```

This captures 12 critical collision scenarios (~470 images total):
- Tripod leg collisions (multiple types)
- Optical tube collisions
- Counterweight collisions
- Mount head collisions
- Finder scope collisions
- Extreme positions
- Covered collisions

**Or manually for each scenario:**
```bash
python training/scripts/capture_training_images_headless.py \
  --num 40 --interval 2 --desc collision_tripod_touching
```

See [COLLISION_SCENARIOS.md](COLLISION_SCENARIOS.md) for full list.

**Goal:** ~400-500 collision scenario images

## Target Final Dataset

| Category | Images | Status |
|----------|--------|--------|
| Static covered/uncovered (existing) | 1,050 | ✅ Complete |
| Additional static (lighting variations) | 500 | 🔄 Ongoing |
| Dynamic orientations (video extract) | 1,500 | ⏳ Pending |
| Collision scenarios | 470 | ⏳ Pending |
| **TOTAL** | **~3,500** | |

## Data Quality Checklist

Before annotation, review and clean:

✅ **Delete blurry images** (motion blur from fast slewing)
✅ **Remove redundant frames** (too similar)
✅ **Check exposure** (not over/under exposed)
✅ **Verify focus** (especially for distant parts)
✅ **Organize by type** (makes annotation easier)

## Annotation Phase

### Tools:
- **LabelImg** (simple, Python-based)
- **Roboflow** (web-based, team-friendly)
- **CVAT** (advanced, supports tracking)

### Classes to Label:
```yaml
0: telescope_ota        # Optical Tube Assembly
1: telescope_mount      # Mount head
2: tripod_leg          # Each tripod leg (label ALL 3!)
3: counterweight       # Counterweight bar/weights
4: telescope_cover     # Covers/caps
5: finder_scope        # Finder scope
6: focuser             # Focusing mechanism
```

### Annotation Tips:
- **Tripod legs are CRITICAL** - label all 3 precisely
- Draw tight bounding boxes
- Be consistent with box placement
- Label even partially visible parts
- For collision scenarios, label overlapping parts separately

### Time Estimate:
- ~30 seconds per image (with practice)
- 3,500 images × 30s = ~30 hours total
- Can split across multiple sessions

## Training Phase

### Prepare Dataset:
```bash
python training/scripts/prepare_dataset.py
```

This splits images 80/20 (train/val).

### Train Model:
```bash
python training/scripts/train_custom_model.py \
  --epochs 200 \
  --batch 16 \
  --base-model yolov8x.pt
```

**Expected Results:**
- Training time: 1-2 hours on A30
- Target mAP50: >0.85
- Output: `models/telescope_custom.pt`

### Evaluate:
```bash
# Test on validation set
python training/scripts/evaluate_model.py --mode images

# Test on live camera
python training/scripts/evaluate_model.py --mode live
```

## Deployment

### Update Config:
```yaml
detection:
  model: "models/telescope_custom.pt"
  use_yolo_world: false  # Switch to custom model
```

### Restart Service:
```bash
sudo ./service.sh restart
```

## Timeline Estimate

| Phase | Time | Can Do Now? |
|-------|------|-------------|
| Additional static captures | 1-2 hours | ✅ Yes (as day progresses) |
| Video extraction sessions | 30-60 min | ⏳ When ready to slew |
| Collision scenarios | 2-3 hours | ⏳ Dedicated session |
| Review & cleanup | 2-3 hours | After all captures |
| Annotation | 20-30 hours | Can split up |
| Training | 1-2 hours | After annotation |
| **TOTAL** | **~30-40 hours** | Spread over days/weeks |

## Phased Approach (Recommended)

### Week 1: Static Captures (Can Do Now!)
- Continue capturing covered telescopes as lighting changes
- Goal: Add 500 more images with lighting diversity
- Keep existing 1,050 images!

### Week 2-3: Dynamic Captures
- 3-4 video extraction sessions (10 min each)
- Cover different telescope states
- Goal: 1,500 dynamic images

### Week 3-4: Collision Scenarios
- Dedicated 2-3 hour session
- Go through all 12 scenarios methodically
- Goal: 470 collision images

### Week 4-5: Annotation
- Use LabelImg or Roboflow
- ~2-3 hours per day
- Can do while watching TV!

### Week 6: Training & Deployment
- Train model
- Evaluate results
- Deploy to production

## Quick Command Reference

```bash
# Static capture (as lighting changes)
python training/scripts/capture_training_images_headless.py \
  --num 100 --interval 2 --desc covered_[time_of_day]

# Video frame extraction
python training/scripts/extract_frames_from_stream.py \
  --duration 600 --interval 1.0 --desc dynamic_[state]

# Collision scenarios (guided)
python training/scripts/capture_collision_scenarios.py

# Collision scenarios (manual)
python training/scripts/capture_training_images_headless.py \
  --num 40 --interval 2 --desc collision_[scenario]

# Prepare dataset
python training/scripts/prepare_dataset.py

# Train model
python training/scripts/train_custom_model.py --epochs 200 --batch 16

# Evaluate
python training/scripts/evaluate_model.py --mode live
```

## Notes

- **Don't delete existing 1,050 images!** They're valuable baseline data
- **Covered images are important** - model needs to detect covered scopes too
- **Lighting diversity matters** - capture at different times of day
- **Collision scenarios are critical** - can't be captured randomly
- **Quality > Quantity** - 2,000 good images > 5,000 mediocre ones
- **Annotation is the bottleneck** - plan accordingly

---

**Current Priority:** Continue capturing covered telescopes as day progresses (evening/dusk/night)!
