# Anatomical Landmark-Based OCT Registration Plan

## Overview

This approach uses anatomical structures (blood vessels, optic nerve, retinal layers) as landmarks for accurate registration, rather than generic image features. This is medically grounded and should provide much better accuracy than ORB/SURF.

---

## Two-Phase Registration Strategy

### Phase 1: XY-Plane Alignment (En-Face Registration)
**Goal**: Find lateral offset (X, Y) between volumes

**Method**: Surface projection + anatomical structure segmentation

### Phase 2: Depth (Z) Alignment
**Goal**: Find depth offset and ensure retinal layers align

**Method**: Cross-sectional (B-scan) layer segmentation

---

## Phase 1: XY-Plane Alignment (Top-Down View)

### Step 1.1: Extract En-Face Surface Projection

**Approach**: Create 2D projection showing retinal surface

**Options**:

#### Option A: Maximum Intensity Projection (MIP)
```python
# Simple and fast
enface = np.max(volume, axis=1)  # Max along Y (depth) axis
```
**Pros**: Fast, shows blood vessels clearly
**Cons**: May include noise

#### Option B: Surface Detection + Projection
```python
# More sophisticated - detect actual retinal surface
# Project tissue starting from detected surface
```
**Pros**: More accurate anatomical representation
**Cons**: Requires surface detection algorithm

**Recommendation**: Start with Option A (MIP), upgrade to B if needed

---

### Step 1.2: Segment Anatomical Structures

**Goal**: Identify blood vessels, optic disc, fovea on en-face images

**Method**: Deep Learning - U-Net Segmentation

#### Pre-trained Models Available:

##### 1. **OCT-A Vessel Segmentation Models**
- **OCTA-500 Dataset Models**: Pre-trained on 500 OCTA images
- **ROSE Challenge Models**: Retinal OCT vessel segmentation
- GitHub repositories with pre-trained weights available

##### 2. **OCT Retinal Layer Segmentation**
- **ReLayNet**: Deep learning for retinal layer segmentation
- **Retina U-Net**: Specialized for OCT layer detection

##### 3. **Optic Disc/Cup Segmentation**
- Multiple models available from REFUGE challenge
- Can detect optic nerve head as landmark

#### U-Net Architecture Options:

**Option A: Use Pre-trained Model (RECOMMENDED)**
```python
# Libraries with pre-trained OCT models:
# 1. MONAI (Medical Open Network for AI)
# 2. nnU-Net (self-configuring U-Net)
# 3. DeepLabV3+ with medical imaging weights
```

**Option B: Train Custom U-Net**
```python
# Using segmentation_models_pytorch or TensorFlow
# Would need annotated training data
```

**Recommendation**: Use MONAI or find pre-trained weights from research papers

---

### Step 1.3: Register Based on Segmented Structures

**Method**: Feature-based registration on binary masks

**Algorithms**:

#### Option A: Vessel Centerline Matching
```python
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist

# Extract vessel centerlines
skeleton1 = skeletonize(vessel_mask1)
skeleton2 = skeletonize(vessel_mask2)

# Find bifurcation points (strong landmarks)
bifurcations1 = detect_bifurcations(skeleton1)
bifurcations2 = detect_bifurcations(skeleton2)

# Match bifurcation patterns
offset_x, offset_y = match_bifurcations(bifurcations1, bifurcations2)
```

#### Option B: Optic Disc Registration
```python
# If optic disc visible in both volumes
disc_center1 = find_centroid(optic_disc_mask1)
disc_center2 = find_centroid(optic_disc_mask2)

offset_x = disc_center2[0] - disc_center1[0]
offset_y = disc_center2[1] - disc_center1[1]
```

#### Option C: Multi-Structure Registration (BEST)
```python
# Combine multiple landmarks for robustness
# - Vessel bifurcations (high confidence)
# - Optic disc center (if visible)
# - Fovea location (if visible)

# Use weighted combination or RANSAC
final_offset = robust_multi_landmark_registration(
    vessels1, vessels2,
    optic_disc1, optic_disc2,
    fovea1, fovea2
)
```

**Recommendation**: Option C for highest accuracy

---

## Phase 2: Depth (Z) Alignment

### Step 2.1: Extract Cross-Sectional View (YZ Plane)

**Approach**: Take middle B-scan from each volume

```python
# Extract central B-scan
bscan1 = volume1[:, :, width // 2]  # Middle slice along Z
bscan2 = volume2[:, :, width // 2]
```

---

### Step 2.2: Segment Retinal Layers

**Goal**: Detect ILM (Inner Limiting Membrane) and RPE (Retinal Pigment Epithelium)

**Methods**:

#### Option A: Graph-Based Layer Segmentation (Classical)
```python
# Dijkstra's algorithm or graph cuts
# Fast, deterministic, but needs tuning
from oct_layer_segmentation import segment_layers

layers = segment_layers(bscan, num_layers=9)
ilm = layers[0]  # Inner limiting membrane
rpe = layers[-1]  # Retinal pigment epithelium
```

Libraries:
- **OCT-Converter**: Has layer segmentation
- **eyepy**: Python library for OCT analysis

#### Option B: Deep Learning Layer Segmentation (RECOMMENDED)
```python
# Pre-trained models available:
# 1. ReLayNet - 9 retinal layers
# 2. OCT-NET - Multi-layer segmentation
# 3. DeepLabV3+ trained on OCT layers

model = load_pretrained_oct_layer_model()
layer_masks = model.predict(bscan)

ilm_position = layer_masks['ILM']
rpe_position = layer_masks['RPE']
```

**Recommendation**: Deep learning (Option B) for robustness

---

### Step 2.3: Align Based on Layer Positions

**Method**: Match layer heights between volumes

```python
# Find mean height of ILM in overlap region
ilm_height1 = np.mean(ilm_layer1[overlap_region])
ilm_height2 = np.mean(ilm_layer2[overlap_region])

# Calculate depth offset
depth_offset = ilm_height2 - ilm_height1

# Apply offset to align layers
volume2_aligned = shift_volume_depth(volume2, depth_offset)
```

**Additional check**: Verify RPE also aligns after correction

---

## Recommended Technology Stack

### Deep Learning Frameworks

#### 1. **MONAI** (Medical Open Network for AI)
```bash
pip install monai
```
**Pros**:
- Purpose-built for medical imaging
- Pre-trained models available
- Excellent for 3D volumes
- Active community

**Use for**: U-Net segmentation, data augmentation

#### 2. **nnU-Net** (Self-configuring U-Net)
```bash
pip install nnunet
```
**Pros**:
- State-of-the-art segmentation
- Self-configuring (minimal tuning)
- Excellent for medical images

**Use for**: Retinal layer segmentation

#### 3. **Segmentation Models PyTorch**
```bash
pip install segmentation-models-pytorch
```
**Pros**:
- Many architectures (U-Net, DeepLabV3+, etc.)
- Pre-trained ImageNet weights
- Easy to fine-tune

**Use for**: En-face vessel segmentation

---

### Image Processing Libraries

#### 1. **scikit-image**
```bash
pip install scikit-image
```
**Use for**: Skeletonization, morphology, bifurcation detection

#### 2. **SimpleITK**
```bash
pip install SimpleITK
```
**Use for**: Medical image registration, resampling, transformations

#### 3. **eyepy** (OCT-specific)
```bash
pip install eyepy
```
**Use for**: OCT-specific operations, layer segmentation

---

## Pre-trained Models to Investigate

### For Vessel Segmentation (En-Face)

1. **OCTA-500 Models**
   - Dataset: 500 OCTA volumes with vessel annotations
   - GitHub: Look for "OCTA-500" implementations

2. **ROSE Challenge Winners**
   - Retinal OCT Segmentation Challenge
   - Multiple winning solutions on GitHub

3. **DRIVE/STARE Models Adapted for OCT**
   - Originally for fundus images
   - Can be fine-tuned for OCT en-face

### For Retinal Layer Segmentation (B-Scans)

1. **ReLayNet**
   - Paper: "ReLayNet: Retinal Layer and Fluid Segmentation"
   - Pre-trained weights available
   - 9 retinal layers + fluid regions

2. **OCT-NET**
   - Multi-layer segmentation
   - Works on multiple OCT device types

3. **DeepLabV3+ OCT**
   - Adapted DeepLabV3+ for OCT
   - Check medical imaging repositories

---

## Implementation Phases

### Phase 1: Proof of Concept (1-2 days)
**Goal**: Validate approach with simple methods

**Tasks**:
1. Extract en-face projections (MIP)
2. Manual landmark identification (click points)
3. Simple offset calculation
4. Test alignment quality

**Deliverable**: Working XY alignment with manual landmarks

---

### Phase 2: Automated Vessel Segmentation (2-3 days)
**Goal**: Automate en-face registration

**Tasks**:
1. Find/train U-Net for vessel segmentation
2. Implement vessel centerline extraction
3. Implement bifurcation detection
4. Automatic XY offset calculation

**Deliverable**: Automated en-face registration

---

### Phase 3: Layer Segmentation (2-3 days)
**Goal**: Automate depth alignment

**Tasks**:
1. Find/train layer segmentation model
2. Extract ILM and RPE boundaries
3. Calculate depth offset
4. Apply 3D transformation

**Deliverable**: Full 3D automated registration

---

### Phase 4: Optimization & Validation (1-2 days)
**Goal**: Refine and test at scale

**Tasks**:
1. Multi-landmark registration (vessels + disc + layers)
2. Confidence scoring
3. Test on all 8 volumes
4. Quality metrics (Dice score, surface distance)

**Deliverable**: Production-ready system

---

## Quick Start: Minimum Viable Product

### Simplest Approach (No Deep Learning Initially)

```python
# Step 1: En-face projection
enface1 = np.max(vol1, axis=1)  # Max intensity projection
enface2 = np.max(vol2, axis=1)

# Step 2: Enhance vessels
from skimage.filters import frangi
vessels1 = frangi(enface1, sigmas=range(1, 5))
vessels2 = frangi(enface2, sigmas=range(1, 5))

# Step 3: Phase correlation on vessel-enhanced images
from scipy.signal import correlate2d
correlation = correlate2d(vessels1, vessels2, mode='same')
offset_y, offset_x = np.unravel_index(np.argmax(correlation), correlation.shape)

# Step 4: For depth, detect bright RPE line
bscan1 = vol1[:, :, vol1.shape[2] // 2]
bscan2 = vol2[:, :, vol2.shape[2] // 2]

# Find brightest horizontal line (RPE approximation)
rpe_y1 = find_brightest_line(bscan1)
rpe_y2 = find_brightest_line(bscan2)
depth_offset = rpe_y2 - rpe_y1
```

**Pros**: Can implement today, no ML required
**Cons**: Less robust than deep learning

---

## Performance Expectations

### XY Alignment Accuracy:
- **Simple phase correlation**: ±5-10 pixels
- **Vessel-enhanced correlation**: ±2-5 pixels
- **Bifurcation matching**: ±1-2 pixels ✅ **BEST**

### Depth Alignment Accuracy:
- **Simple brightest line**: ±3-5 pixels
- **Layer segmentation**: ±1-2 pixels ✅ **BEST**

---

## Recommended Path Forward

### Option A: Fast Implementation (1-2 days)
1. Start with Frangi filter for vessels (no ML)
2. Phase correlation on vessel-enhanced images
3. Simple RPE detection for depth
4. Test on 2 volumes → 8 volumes

### Option B: Production Quality (1 week)
1. Research and download pre-trained models
2. Implement U-Net vessel segmentation
3. Implement layer segmentation
4. Multi-landmark registration
5. Full validation

**My Recommendation**: Start with Option A for quick results, then upgrade to Option B if accuracy isn't sufficient.

---

## Next Steps - Your Decision

1. **Should we start with simple approach (no ML) to see if it works?**
2. **Or research/download pre-trained models first?**
3. **Do you have any annotated OCT data for training?**
4. **What's your priority: speed vs. accuracy?**

Let me know which direction you prefer, and I'll implement it!
