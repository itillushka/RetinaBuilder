# OCT Retinal Curvature Estimation Pipeline
## Comprehensive Technical Documentation for Diploma Thesis

---

## 1. Project Overview

**Objective:** Develop an automated pipeline for estimating horizontal retinal curvature from multiple overlapping OCT (Optical Coherence Tomography) volumes by aligning and stitching them into a wide-field panoramic reconstruction.

**Problem Statement:** Standard OCT imaging captures a limited 6x6mm field of view, insufficient for accurate curvature estimation. By acquiring multiple overlapping volumes and computationally stitching them, a wider field can be reconstructed to enable reliable curvature measurements.

---

## 2. Technologies Used

### Programming Language
- **Python 3.x** - Primary development language

### Core Libraries
| Library | Purpose |
|---------|---------|
| **NumPy** | N-dimensional array operations, mathematical computations |
| **SciPy** | Image transformations (ndimage), signal processing, optimization |
| **OpenCV (cv2)** | Image processing, denoising, morphological operations, feature detection |
| **scikit-image** | Frangi vessel filter, image registration, CLAHE enhancement |
| **Matplotlib** | Visualization, plotting, figure generation |
| **Pillow (PIL)** | Image I/O, BMP file loading |
| **joblib** | Parallel processing (Parallel, delayed) |

### Optional Dependencies
| Library | Purpose |
|---------|---------|
| **ANTsPy** | Medical image registration (alternative method) |
| **SimpleElastix** | Elastix registration wrapper (if available) |

---

## 3. Input/Output Specifications

### Input Data
- **Format:** Directory containing BMP B-scan images
- **B-scan dimensions:** 1536 (width) x 360 (height) pixels
- **Volume size:** 360 B-scans per volume
- **Scan area:** 6mm x 6mm
- **Data type:** 8-bit grayscale

```
input_volume/
├── scan_001.bmp
├── scan_002.bmp
├── ...
└── scan_360.bmp
```

### Output Data
| Output | Format | Description |
|--------|--------|-------------|
| Aligned volumes | .npy | 3D NumPy arrays with applied transformations |
| Panoramic reconstruction | .npy, .png | Merged wide-field volume |
| Transform parameters | .json | dx, dy, dz offsets, rotation angles |
| Curvature analysis | .png | Visualization with fitted curve and measurements |
| Intermediate visualizations | .png | Step-by-step alignment results |

---

## 4. Pipeline Architecture

### 4.1 High-Level Flow

```
Input: 5 OCT Volumes (cross pattern: center + 4 neighbors)
                            ↓
┌─────────────────────────────────────────────────────┐
│           PHASE 1: Pairwise Alignment               │
│  ┌─────────────────────────────────────────────┐    │
│  │ Step 1: XZ Alignment (En-face registration) │    │
│  │ Step 2: Y Alignment (Depth alignment)       │    │
│  │ Step 3: Z-Rotation (Layer tilt correction)  │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────┐
│           PHASE 2: Multi-Volume Merging             │
│  Chain 1: V1 ← V2 ← V3 (right extension)           │
│  Chain 2: V1 ← V4 ← V5 (left extension)            │
│  Final: Merge both chains into panorama             │
└─────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────┐
│           PHASE 3: Curvature Analysis               │
│  - Extract averaged central B-scan from panorama    │
│  - Detect ILM/RPE boundary points                   │
│  - Fit circular arc using least-squares             │
│  - Calculate radius and curvature                   │
└─────────────────────────────────────────────────────┘
                            ↓
Output: Curvature radius (mm), panoramic visualization
```

### 4.2 Alignment Steps Detail

#### Step 1: XZ Plane Alignment (Lateral Registration)
- **Input:** Two 3D volumes (reference, moving)
- **Method:**
  1. Generate vessel-enhanced MIP (Maximum Intensity Projection)
  2. Apply Frangi filter to enhance blood vessels
  3. Register using multi-scale phase correlation
- **Output:** offset_x, offset_z (lateral shifts in pixels)

#### Step 2: Y-Axis Alignment (Depth Registration)
- **Input:** Overlap regions from Step 1
- **Method:**
  1. Extract central B-scan from overlap
  2. Detect retinal surface using contour method
  3. Calculate vertical offset from surface positions
- **Output:** y_shift (depth offset in pixels)

#### Step 3: Z-Rotation Alignment (Tilt Correction)
- **Input:** Y-aligned volumes
- **Method:**
  1. Coarse search: -15° to +15° in 1° steps
  2. Fine search: ±3° around best in 0.5° steps
  3. Score each angle using NCC on detected surfaces
- **Output:** rotation_angle (degrees)

---

## 5. Algorithms Used

### 5.1 Vessel Enhancement - Frangi Filter
**Purpose:** Enhance tubular structures (blood vessels) for robust registration

**Mathematical basis:**
- Computes Hessian matrix eigenvalues at multiple scales
- Vessel response: high when λ1 ≈ 0 and |λ2| >> 0
- Multi-scale: σ = [1, 3, 5, 7, 9] pixels

```
Response = exp(-R_B²/2β²) × (1 - exp(-S²/2c²))
where R_B = λ1/λ2 (blobness), S = √(λ1² + λ2²) (structure)
```

### 5.2 Phase Correlation Registration
**Purpose:** Fast, robust translation estimation

**Algorithm:**
1. Compute FFT of both images: F1, F2
2. Cross-power spectrum: R = (F1 × F2*) / |F1 × F2*|
3. Inverse FFT → correlation surface
4. Peak location = translation offset

**Complexity:** O(n log n) vs O(n²) for spatial correlation

### 5.3 Contour-Based Surface Detection
**Purpose:** Detect retinal layer boundaries

**Algorithm:**
1. Apply bilateral denoising (edge-preserving)
2. Threshold to identify tissue regions
3. Extract contours using gradient magnitude
4. Select topmost contour as ILM surface
5. Smooth using median filtering

### 5.4 Coarse-to-Fine Rotation Search
**Purpose:** Find optimal rotation angle for layer alignment

**Algorithm:**
1. **Coarse phase:** Test 31 angles (-15° to +15°, step 1°)
2. **Fine phase:** Test 13 angles (±3° around best, step 0.5°)
3. **Scoring:** Normalized Cross-Correlation (NCC) between surfaces

```
NCC = Σ[(A - μA)(B - μB)] / (σA × σB × N)
```

### 5.5 Circle Fitting (Curvature Estimation)
**Purpose:** Estimate retinal curvature from detected surface points

**Algorithm:** Least-squares circle fitting
1. Detect 9-11 surface points across panorama width
2. Minimize: Σ(√((xi - cx)² + (yi - cy)²) - r)²
3. Output: center (cx, cy), radius r

**Curvature:** κ = 1/r (mm⁻¹)

---

## 6. Key Components

### 6.1 Main Scripts
| Script | Purpose |
|--------|---------|
| `five_volume_alignment.py` | Master orchestrator for 5-volume alignment |
| `retina_curvature_analysis.py` | Curvature estimation from panorama |
| `visualize_bscan_panorama.py` | Generate panoramic visualization |

### 6.2 Pipeline Steps
| Script | Purpose |
|--------|---------|
| `steps/step1_xz_alignment.py` | Lateral (XZ) registration |
| `steps/step2_y_alignment.py` | Depth (Y) alignment |
| `steps/step3_rotation_z.py` | Rotation correction |

### 6.3 Helper Modules
| Module | Purpose |
|--------|---------|
| `helpers/oct_loader.py` | Load and preprocess OCT volumes |
| `helpers/mip_generation.py` | Vessel MIP creation and registration |
| `helpers/rotation_alignment.py` | Surface detection and rotation search |
| `helpers/rotation_alignment_parallel.py` | Parallel rotation/shift operations |
| `helpers/volume_transforms.py` | Apply transformations to volumes |
| `helpers/canvas_merger.py` | Merge multiple volumes into panorama |

---

## 7. Image Processing Pipeline

### 7.1 Denoising Pipeline
```
Input B-scan
    ↓
Non-local Means Denoising (h=30)
    ↓
Bilateral Filter (σ_color=180, σ_space=180)
    ↓
Median Filter (kernel=19)
    ↓
Otsu Thresholding (85% of threshold)
    ↓
CLAHE Enhancement (clipLimit=3.6)
    ↓
Output: Denoised B-scan
```

### 7.2 Vessel Enhancement Pipeline
```
Input Volume (Y, X, Z)
    ↓
MIP along Y-axis → En-face image (X, Z)
    ↓
Normalize to [0, 1]
    ↓
Frangi Filter (scales: 1,3,5,7,9) [PARALLEL]
    ↓
Max across scales
    ↓
Bilateral Denoising (optional)
    ↓
Output: Vessel-enhanced MIP
```

---

## 8. Coordinate System

### Volume Coordinates (Y, X, Z)
- **Y-axis (dim 0):** Tissue depth (0 = anterior, 360 = posterior)
- **X-axis (dim 1):** Lateral/width (0 = left, 1536 = right)
- **Z-axis (dim 2):** Scanning direction (0-360 B-scans)

### Transformation Conventions
- **dx:** Positive = shift right
- **dy:** Positive = shift down (deeper)
- **dz:** Positive = shift forward in Z
- **rotation:** Positive = counter-clockwise

---

## 9. Performance Optimizations

| Optimization | Speedup | Implementation |
|--------------|---------|----------------|
| Parallel Frangi filter | ~4x | joblib across scales |
| OpenCV rotation | ~3-4x | SIMD-optimized warpAffine |
| FFT phase correlation | O(n log n) | vs O(n²) spatial |
| Z-stride subsampling | 50% memory | Load every 2nd B-scan |
| Parallel rotation search | ~4x | joblib across angles |

---

## 10. Validation Metrics

| Metric | Purpose | Expected Range |
|--------|---------|----------------|
| NCC (registration) | Alignment quality | > 0.7 good, > 0.85 excellent |
| Confidence score | Registration reliability | > 0.5 acceptable |
| Curvature radius | Anatomical validation | 20-35 mm (human eye) |
| Surface smoothness | Detection quality | Low standard deviation |

---

## 11. Results Summary

**Measured Parameters (Example):**
- Horizontal curvature radius: 31.61 mm
- Curvature: 0.0316 mm⁻¹
- Consistent with human eye anatomy (~24 mm axial length)

**Pipeline Capabilities:**
- Aligns 5 overlapping OCT volumes
- Creates wide-field panoramic reconstruction
- Estimates horizontal retinal curvature
- Automated processing with minimal user intervention

---

## 12. Limitations and Future Work

**Limitations:**
- Only horizontal curvature estimation (sparse B-scan spacing limits vertical)
- Requires sufficient overlap between volumes
- Accuracy depends on image quality and visible retinal features

**Future Directions:**
- Modified scanning protocol with vertically-oriented B-scans
- GPU acceleration for real-time processing
- Extension to pathological retinas with layer disruption
