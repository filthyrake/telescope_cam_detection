# 🎉 Custom Training Infrastructure - Ready to Use!

All the infrastructure for custom telescope detection training is now set up and ready for tomorrow!

## 📁 What's Been Created

```
training/
├── TRAINING_GUIDE.md          # Complete step-by-step guide
├── datasets/
│   └── telescope_equipment/
│       ├── classes.yaml        # Dataset configuration with 7 classes
│       ├── images/
│       │   ├── train/          # Training images go here
│       │   ├── val/            # Validation images go here
│       │   └── raw/            # Raw captured images
│       └── labels/
│           ├── train/          # Training annotations go here
│           └── val/            # Validation annotations go here
├── scripts/
│   ├── capture_training_images.py   # Auto-capture from camera
│   ├── prepare_dataset.py           # Split train/val sets
│   ├── train_custom_model.py        # Train YOLOv8 model
│   └── evaluate_model.py            # Test trained model
└── runs/                       # Training outputs will go here
```

## 🚀 Tomorrow's Workflow

### Step 1: Capture Images (15-30 minutes)
```bash
source venv/bin/activate
python training/scripts/capture_training_images.py --num 150
```

**What to do:**
- Uncover your telescopes
- Walk around them while script captures
- Move closer/farther
- Different angles
- Maybe move telescope if on motorized mount

### Step 2: Prepare Dataset (1 minute)
```bash
python training/scripts/prepare_dataset.py
```

Splits 150 images → 120 train, 30 validation

### Step 3: Annotate Images (1-2 hours)

**Option A: Use Roboflow (Recommended - Easier)**
1. Go to https://roboflow.com
2. Upload images
3. Draw boxes around telescope parts
4. Export in YOLOv8 format
5. Download labels

**Option B: Use LabelImg (Desktop)**
```bash
pip install labelImg
labelImg training/datasets/telescope_equipment/images/train
```

**What to label:**
- `telescope_ota` - Main telescope tube
- `telescope_mount` - Mount head
- `tripod_leg` - Each tripod leg (IMPORTANT!)
- `counterweight` - Counterweight bar
- `telescope_cover` - Any covers
- `finder_scope` - Finder scope
- `focuser` - Focuser

### Step 4: Train Model (15-30 minutes)
```bash
python training/scripts/train_custom_model.py
```

Training will run automatically and save best model to `models/telescope_custom.pt`

### Step 5: Test Model
```bash
# Test on live camera
python training/scripts/evaluate_model.py --mode live

# Or test on validation images
python training/scripts/evaluate_model.py --mode images
```

### Step 6: Deploy
Edit `config/config.yaml`:
```yaml
detection:
  model: "models/telescope_custom.pt"
  confidence: 0.4
```

Restart system: `./start.sh`

## 🎯 Custom Classes Defined

Your model will detect these 7 classes:
1. **telescope_ota** - Optical tube (the main telescope body)
2. **telescope_mount** - Mount head that holds telescope
3. **tripod_leg** - Individual tripod legs ⚠️ Critical for collision detection
4. **counterweight** - Counterweight bar and weights
5. **telescope_cover** - Protective covers
6. **finder_scope** - Small finder scope
7. **focuser** - Focusing mechanism

## 📊 Expected Results

With 100-150 well-annotated images:
- **Training time**: 15-30 minutes on A30
- **Accuracy (mAP50)**: 85-95% expected
- **Inference speed**: 10-15ms per frame
- **Benefits**:
  - ✅ No more false "person" detections on covered telescopes
  - ✅ Accurate telescope part recognition
  - ✅ Foundation for collision detection
  - ✅ Track movement and positioning

## 💡 Tips for Success

**Image Collection:**
- Take 100-200 images minimum
- Vary distance (2-20 feet)
- Multiple angles (360° around)
- Both covered and uncovered
- Different lighting conditions

**Annotation:**
- Be consistent with box placement
- Draw tight boxes around objects
- Include partially visible objects
- **CRITICAL**: Annotate ALL three tripod legs precisely
- Don't skip any visible parts

**Training:**
- Start with yolov8n (fastest)
- 100 epochs minimum
- Watch for mAP50 > 0.85
- If poor results, collect more diverse images

## 🔧 Troubleshooting

**"Out of memory during training"**
```bash
python training/scripts/train_custom_model.py --batch 8
```

**"Not detecting well"**
- Lower confidence in config (try 0.3)
- Train more epochs (--epochs 200)
- Collect more diverse training data

**"Too slow"**
- Use yolov8n instead of yolov8x
- Reduce image size (--imgsz 320)

## 📚 Full Documentation

See `training/TRAINING_GUIDE.md` for complete detailed instructions!

---

**Everything is ready! Tomorrow you can train your custom telescope detector in just a few hours.** 🔭✨
