# Custom Telescope Detection Training Guide

Complete guide for training a custom YOLOX model to detect your telescope equipment.

## 📋 Overview

This guide will walk you through:
1. Collecting training images
2. Annotating images with bounding boxes
3. Training a custom model
4. Evaluating the model
5. Deploying to your detection system

## 🎯 Goal

Train a model to detect:
- `telescope_ota` - Optical Tube Assembly (main telescope body)
- `telescope_mount` - Mount head
- `tripod_leg` - Tripod legs (**critical for collision detection!**)
- `counterweight` - Counterweight bar and weights
- `telescope_cover` - Telescope covers
- `finder_scope` - Finder scope
- `focuser` - Focuser assembly

---

## Step 1: Collect Training Images (100-200 images)

### Using the Automated Capture Script:

```bash
source venv/bin/activate

# Capture 100 images (one every 2 seconds)
python training/scripts/capture_training_images.py \
    --num 100 \
    --interval 2
```

**Important Tips:**
- Walk around your telescopes during capture
- Move closer and farther away
- Capture different angles (side, front, 45°)
- Include both covered and uncovered states
- Vary lighting conditions (daytime, evening, with lights)
- Press **'s'** to manually save interesting frames
- Press **'q'** to stop early

### Diversity is Key:
- ✅ Different distances (1-20 feet)
- ✅ Different angles
- ✅ Covered and uncovered
- ✅ Different lighting
- ✅ Telescope in different positions (if on mount that moves)

---

## Step 2: Prepare Dataset

Split images into training and validation sets:

```bash
python training/scripts/prepare_dataset.py
```

This creates:
```
training/datasets/telescope_equipment/
├── images/
│   ├── train/  (80% of images)
│   └── val/    (20% of images)
└── labels/
    ├── train/  (annotations go here)
    └── val/    (annotations go here)
```

---

## Step 3: Annotate Images

You need to draw bounding boxes around each telescope part in every image.

### Option A: LabelImg (Free, Desktop)

**Install:**
```bash
pip install labelImg
```

**Run:**
```bash
labelImg training/datasets/telescope_equipment/images/train
```

**How to use:**
1. Click "Open Dir" → select `images/train`
2. Click "Change Save Dir" → select `labels/train`
3. Select "YOLO" format (bottom left)
4. Press 'W' to draw a box
5. Label the object (telescope_ota, tripod_leg, etc.)
6. Press 'D' to go to next image
7. Repeat for all training images
8. Then repeat for validation images!

### Option B: Roboflow (Free, Web-based, Easier)

1. Go to https://roboflow.com/
2. Create free account
3. Create new project: "Telescope Detection"
4. Upload images from `images/train` and `images/val`
5. Draw bounding boxes using their interface
6. Export in "YOLOX" format
7. Download and extract labels to `labels/train` and `labels/val`

### Annotation Tips:
- **Be consistent**: Draw tight boxes around objects
- **Include partial objects**: If telescope is partially visible, annotate it
- **tripod_leg is critical**: Make sure to annotate ALL three legs precisely
- **Label everything visible**: Don't skip objects, even if small

---

## Step 4: Train the Model

Once you have annotated all images:

```bash
source venv/bin/activate

# Basic training (recommended to start)
python training/scripts/train_custom_model.py

# Or with custom parameters
python training/scripts/train_custom_model.py \
    --base-model yolov8n.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 640
```

**Training parameters:**
- `--base-model`: Start from yolov8n (fast), yolov8s, yolov8m, or yolov8x (accurate)
- `--epochs`: 100-300 epochs typical (more is usually better)
- `--batch`: Reduce if you get OOM errors (try 8 or 4)
- `--imgsz`: 640 is good balance (larger = more accurate but slower)

**Training time:**
- On NVIDIA A30: ~10-30 minutes for 100 epochs with yolov8n
- Watch for the mAP50 metric - higher is better!

**What to expect:**
```
Epoch 1/100: 100%|██████████| 10/10 [00:15<00:00]
  metrics/mAP50: 0.342
Epoch 50/100:
  metrics/mAP50: 0.852
Epoch 100/100:
  metrics/mAP50: 0.925  ← Goal: >0.85 is excellent!
```

**Output:**
- Training charts: `training/runs/telescope_detection/results.png`
- Best model: `training/runs/telescope_detection/weights/best.pt`
- Exported model: `models/telescope_custom.pt` (ready to use!)

---

## Step 5: Evaluate the Model

### Test on validation images:
```bash
python training/scripts/evaluate_model.py --mode images
```

### Test on live camera feed:
```bash
python training/scripts/evaluate_model.py --mode live
```

**What to look for:**
- ✅ Detects all telescope parts correctly
- ✅ Tripod legs detected with high confidence
- ✅ Few false positives
- ✅ Works at different distances and angles

**If results are poor:**
- Collect more diverse images
- Add more annotations
- Train for more epochs
- Try larger base model (yolov8s or yolov8m)

---

## Step 6: Deploy to Detection System

### Update configuration:

Edit `config/config.yaml`:

```yaml
detection:
  model: "models/telescope_custom.pt"  # Use your custom model
  confidence: 0.4  # Lower threshold for your specific objects
  device: "cuda:0"

  # Optional: Only detect telescope-specific classes
  target_classes:
    - "telescope_ota"
    - "telescope_mount"
    - "tripod_leg"
    - "counterweight"
```

### Restart the detection system:

```bash
# Stop current system (Ctrl+C)
# Then restart:
./start.sh
```

### Verify it works:
- Open browser to http://localhost:8000
- You should now see accurate detections of your telescope parts!
- No more false "person" detections on covered telescopes 🎉

---

## 🔍 Troubleshooting

### "No detections found"
- Lower confidence threshold in config (try 0.3 or 0.25)
- Train for more epochs
- Collect more diverse training data

### "Too many false positives"
- Raise confidence threshold (try 0.5 or 0.6)
- Add negative examples (images without telescopes)
- Train for more epochs

### "Out of memory during training"
- Reduce batch size: `--batch 8` or `--batch 4`
- Use smaller model: `--base-model yolov8n.pt`
- Reduce image size: `--imgsz 320`

### "Training takes too long"
- Use smaller model (yolov8n instead of yolov8x)
- Reduce epochs (but 100 minimum recommended)
- Reduce image size

---

## 📊 Expected Performance

With 100-200 well-annotated images:
- **mAP50**: >0.85 (excellent)
- **mAP50-95**: >0.60 (good)
- **Inference time**: 10-15ms (yolov8n), 30-50ms (yolov8x)
- **Accuracy**: Should correctly identify telescope parts 90%+ of the time

---

## 🚀 Next Steps After Training

1. **Implement Collision Detection** - Use tripod_leg positions to detect proximity warnings
2. **Add Snapshot Saving** - Save images when specific objects detected
3. **Zone-Based Alerts** - Define danger zones around tripod legs
4. **Track Movement** - Monitor telescope slewing and predict collisions

---

## 💡 Pro Tips

- **Start small**: Get 50 images working first, then expand
- **Quality > Quantity**: 100 good annotations better than 500 bad ones
- **Iterate**: Train → Test → Collect more data → Retrain
- **Use data augmentation**: Training script already includes rotation, flipping, etc.
- **Save your work**: Keep your annotated dataset backed up!

---

## 📚 Additional Resources

- YOLOX Docs: https://github.com/Megvii-BaseDetection/YOLOX
- LabelImg: https://github.com/HumanSignal/labelImg
- Roboflow: https://roboflow.com/
- YOLO Format: https://github.com/Megvii-BaseDetection/YOLOX/tree/main/datasets

---

**Questions or issues?** Check the logs in `training/runs/` or refer back to this guide!
