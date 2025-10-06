# Wildlife & Telescope Annotation Guide

## Quick Start

You need to annotate your captured images with correct labels before training a custom model.

### Option 1: Roboflow (Web-based, Easiest)

1. **Sign up**: https://roboflow.com (free tier available)
2. **Create new project**: "Desert Wildlife Detection"
3. **Upload images**:
   - From `clips/` directory (wildlife captures)
   - From `training/datasets/telescope_equipment/images/raw/` (telescope images)
4. **Add classes**:
   ```
   - rabbit
   - coyote
   - iguana
   - tortoise
   - lizard
   - ground_squirrel
   - roadrunner
   - quail
   - bird_other
   - person
   - telescope_ota
   - telescope_mount
   - tripod_leg
   ```
5. **Draw boxes & label**: Click and drag to draw bounding boxes
6. **Export**: YOLOX format → download
7. **Place in**: `training/datasets/combined_wildlife_telescope/`

### Option 2: LabelImg (Desktop, More Control)

**Install:**
```bash
pip install labelImg
```

**Run:**
```bash
labelImg clips/
```

**Usage:**
1. `W` - Draw bounding box
2. Type class name (e.g., "rabbit")
3. `D` - Next image
4. `A` - Previous image
5. Saves `.txt` files in YOLO format

**Classes file** (create `classes.txt`):
```
rabbit
coyote
iguana
tortoise
lizard
ground_squirrel
roadrunner
quail
bird_other
person
telescope_ota
telescope_mount
tripod_leg
counterweight
telescope_cover
finder_scope
focuser
```

## Current Captures

As of now, you have:
- **4 wildlife snapshots** (including 1 rabbit misclassified as "bird")
- **350 telescope images** (covered, day and night conditions)

## How Many Images Do You Need?

**Minimum for training:**
- 50-100 images per class (bare minimum)
- 100-200 images per class (good)
- 200-500 images per class (excellent)

**Your strategy:**
1. Let system collect wildlife for 3-7 days (aim for 50+ per species)
2. Already have 350 telescope images (plenty for telescope parts!)
3. Annotate all in one batch
4. Train combined model

## Training Timeline

**Current progress:**
- ✅ 350 telescope images (covered) collected
- 🔄 4 wildlife images collected (need 50+ per species)
- ⏳ 3-7 days of collection time needed

**When ready to train:**
1. Annotate all images: 2-4 hours
2. Train model: 30-60 minutes on A30 GPU
3. Deploy: 5 minutes
4. **Result**: Correctly classified wildlife + telescope detection!

## Tips

**For wildlife:**
- Annotate tightly around the animal
- Label partially visible animals too
- Include different poses, distances, lighting

**For telescopes:**
- Label each part separately (OTA, mount, legs)
- Even when covered, try to identify parts under cover
- Label tripod legs individually if visible

**Quality > Quantity:**
- Better to have 100 well-labeled images than 500 poorly labeled ones
- Consistent bounding box sizes matter
- Skip extremely blurry/unclear images

## What Happens After Training?

Your custom model will:
- ✅ Correctly identify rabbits (not as "bird")
- ✅ Identify coyotes (not as "dog")
- ✅ Identify iguanas/lizards (not as "cat")
- ✅ Identify all desert wildlife correctly
- ✅ Identify telescope parts when uncovered
- ✅ Work in day and night conditions

**Model accuracy:**
- Expect 80-95% mAP with 100+ images per class
- 90-98% mAP with 200+ images per class

## Ready to Start?

**Option A**: Wait a week, collect more wildlife, annotate all at once
**Option B**: Start annotating now, add more as you capture them

Both work! Option A is easier (one annotation session), Option B lets you train sooner.
