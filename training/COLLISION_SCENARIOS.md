# Collision Scenario Training Data Guide

This guide walks you through capturing critical collision scenarios for training the telescope collision detection model.

## Why Collision Scenarios Matter

Your model needs to **recognize dangerous positions** that could lead to:
- Tripod legs colliding
- Optical tubes hitting each other
- Counterweights striking tripod legs
- Mount heads colliding
- Finder scopes hitting other equipment

These scenarios are **rare in normal operation** (and should be!), so you must deliberately stage them for training data.

## Safety Notes ⚠️

- **Move slowly** when setting up collision scenarios
- **Stop before actual contact** (get close, don't damage equipment!)
- **Have a spotter** if possible
- **Know your equipment limits**

## Collision Scenarios to Capture

### 1. Tripod Leg Collisions (CRITICAL)

**Why:** Most common collision risk. Tripod legs must be tracked precisely.

#### Scenario 1A: Adjacent Tripod Legs Touching
- Position tripod legs of both scopes 1-2 inches apart
- Capture: 30-50 images
- Label: Both tripod legs with "collision zone" annotation

#### Scenario 1B: Tripod Legs Crossing
- Position scopes so tripod legs would cross if moved
- Capture: 30-50 images
- Label: Tripod legs + danger zone

#### Scenario 1C: Near Miss (6-12 inches apart)
- Position tripod legs close but not touching
- Capture: 30-50 images
- Label: Tripod legs with "warning zone"

**Capture Command:**
```bash
# After positioning tripod legs in collision
python training/scripts/capture_training_images_headless.py \
  --num 40 --interval 2 --desc collision_tripod_legs
```

### 2. Optical Tube Collisions

**Why:** Tubes can collide when pointed at each other or when one slews into the other's path.

#### Scenario 2A: Tubes Pointed at Each Other
- Point both telescope tubes directly at each other
- Tubes should be 6-24 inches apart
- Capture from multiple angles by rotating slightly
- Capture: 40-60 images

#### Scenario 2B: One Tube in Another's Slew Path
- Position Telescope 1 at 45° altitude
- Position Telescope 2 aimed where Telescope 1 would slew through
- Capture: 30-50 images

**Capture Command:**
```bash
python training/scripts/capture_training_images_headless.py \
  --num 50 --interval 2 --desc collision_tubes_pointed
```

### 3. Counterweight Collisions

**Why:** Counterweights swing in arcs and can hit tripod legs or other equipment.

#### Scenario 3A: Counterweight Near Tripod Leg
- Slew scope so counterweight bar comes within 6 inches of other scope's tripod leg
- Capture at multiple points along the arc
- Capture: 40-60 images

#### Scenario 3B: Counterweights Crossing Paths
- Position both scopes so counterweights would collide mid-slew
- Capture: 30-40 images

**Capture Command:**
```bash
python training/scripts/capture_training_images_headless.py \
  --num 50 --interval 2 --desc collision_counterweight
```

### 4. Mount Head Collisions

**Why:** Mount heads can collide when scopes are pointed at extreme angles.

#### Scenario 4A: Mount Heads Too Close
- Position both scopes at high altitude (70-85°)
- Rotate so mount heads are 6-12 inches apart
- Capture: 30-40 images

**Capture Command:**
```bash
python training/scripts/capture_training_images_headless.py \
  --num 35 --interval 2 --desc collision_mount_heads
```

### 5. Finder Scope / Accessory Collisions

**Why:** Finder scopes stick out and can hit other equipment unexpectedly.

#### Scenario 5A: Finder Scope Near Other Tube
- Position so finder scope points toward other scope's tube
- Get within 4-8 inches
- Capture: 20-30 images

**Capture Command:**
```bash
python training/scripts/capture_training_images_headless.py \
  --num 25 --interval 2 --desc collision_finder_scope
```

### 6. Extreme Dangerous Positions

**Why:** Edge cases that are especially risky.

#### Scenario 6A: Both Scopes at Zenith (Collision Risk High)
- Position both scopes pointing near zenith (85-90°)
- Tripod legs and counterweights in close proximity
- Capture: 40-50 images

#### Scenario 6B: Opposite Directions with Overlap
- Scope 1 at 30° altitude, 0° azimuth
- Scope 2 at 30° altitude, 180° azimuth
- Counterweights/legs overlapping in middle
- Capture: 30-40 images

**Capture Command:**
```bash
python training/scripts/capture_training_images_headless.py \
  --num 45 --interval 2 --desc collision_extreme_positions
```

### 7. Covered Telescope Collisions

**Why:** Model must detect collisions even with covers on.

#### Scenario 7A: Covered Scopes in Collision Position
- Cover both telescopes
- Set up tripod leg collision scenario
- Capture: 30-40 images

**Capture Command:**
```bash
python training/scripts/capture_training_images_headless.py \
  --num 35 --interval 2 --desc collision_covered
```

## Capture Summary

| Scenario | Images | Priority | Total |
|----------|--------|----------|-------|
| Tripod legs | 40×3 | ⭐⭐⭐ | 120 |
| Optical tubes | 50+40 | ⭐⭐⭐ | 90 |
| Counterweights | 50+35 | ⭐⭐ | 85 |
| Mount heads | 35 | ⭐⭐ | 35 |
| Finder scopes | 25 | ⭐ | 25 |
| Extreme positions | 45+35 | ⭐⭐ | 80 |
| Covered collisions | 35 | ⭐⭐ | 35 |
| **TOTAL** | | | **~470** |

## Annotation Guidelines

When labeling collision scenarios:

### Standard Labels:
- `telescope_ota` - Optical Tube Assembly
- `telescope_mount` - Mount head
- `tripod_leg` - Each tripod leg (label all 3!)
- `counterweight` - Counterweight bar/weights
- `finder_scope` - Finder scope

### Collision-Specific Labels:
- Draw bounding boxes **precisely** around parts in collision
- If parts are touching/overlapping, draw separate boxes for each
- Label **all visible tripod legs** (most important!)
- Include counterweights even if small in frame

### What Makes a "Collision" Image:
- Parts within 6 inches = collision risk
- Parts overlapping in image = definite collision
- Trajectory shows collision path = collision risk

## Using the Guided Capture Script

I've created a script that walks you through each scenario:

```bash
python training/scripts/capture_collision_scenarios.py
```

This will:
1. Display instructions for each scenario
2. Wait for you to position telescopes
3. Capture images automatically
4. Track progress through all scenarios
5. Save metadata about each scenario

## Quick Reference Commands

```bash
# Start collision scenario captures
python training/scripts/capture_collision_scenarios.py

# Or manually for each scenario:
python training/scripts/capture_training_images_headless.py \
  --num 40 --interval 2 --desc collision_[scenario_name]
```

## Timeline Estimate

- Setup + capture per scenario: ~5-10 minutes
- Total for all scenarios: ~2-3 hours
- Can be done over multiple sessions

## After Capturing Collision Scenarios

1. **Review all images** - delete blurry or duplicate frames
2. **Combine with dynamic video frames**
3. **Total dataset should be: ~2000-3000 images**
   - ~1500-2000 from video extraction (normal operation)
   - ~400-500 from collision scenarios
4. **Annotate everything** with LabelImg or Roboflow
5. **Train the model**

## Tips for Success

✅ **Go slow** - Take time to position accurately
✅ **Get close** - But stop before actual contact
✅ **Multiple angles** - Rotate camera view slightly between captures
✅ **Document** - Take notes on what collision risk each represents
✅ **Be thorough** - Tripod legs are the most critical to get right

---

**Remember:** The quality and diversity of collision scenarios directly impacts how well your model prevents real accidents!
