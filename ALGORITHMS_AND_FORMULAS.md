# OCT Multi-Volume Alignment Pipeline: Algorithms and Formulas

## Overview
This document describes the mathematical algorithms and formulas used in the modular OCT panoramic stitching pipeline. The pipeline aligns 9 OCT volumes in a cross/plus pattern using a multi-step registration approach.

---

## Pipeline Architecture

The alignment pipeline consists of 4 main steps executed sequentially for each volume pair:

1. **Step 1**: XZ-Plane Alignment (Phase Correlation)
2. **Step 2**: Y-Axis Alignment (Surface Detection + NCC Search)
3. **Step 3**: Z-Rotation Alignment (ECC-Style Correlation)
4. **Step 3.5**: X-Rotation Alignment (Sagittal Plane) [Optional]

Additional components:
- Directional constraints for cross-pattern geometry
- Progressive canvas merging with weighted averaging

---

## Step 1: XZ-Plane Alignment

### 1.1 Vessel Enhancement (Frangi Filter)

**Purpose**: Enhance tubular structures (blood vessels) for better registration accuracy.

**Algorithm**: Frangi vesselness filter (Hessian-based multi-scale filtering)

**Process**:
1. Create en-face projection (mean along Y-axis):
   ```
   I_enface(x,z) = mean(V(y,x,z)) for all y
   ```

2. Normalize intensity:
   ```
   I_norm = (I_enface - min(I_enface)) / (max(I_enface) - min(I_enface))
   ```

3. Apply Frangi filter at multiple scales σ ∈ {1, 3, 5, 7, 9}:
   ```
   For each scale σ:
     - Compute Hessian matrix H at each pixel
     - Calculate eigenvalues λ₁, λ₂ (|λ₁| ≤ |λ₂|)
     - Vesselness measure V(σ) = 0 if λ₂ > 0, else:

       V(σ) = exp(-Rᵦ²/2β²) × (1 - exp(-S²/2c²))

       where:
       Rᵦ = |λ₁| / |λ₂|  (blob ratio)
       S = √(λ₁² + λ₂²)  (structure strength)
       β = 0.5, c = half of max Hessian norm
   ```

4. Take maximum response across scales:
   ```
   I_vessel(x,z) = max(V(σ)) for all σ
   ```

5. Scale to 0-255:
   ```
   I_final = ((I_vessel - min(I_vessel)) / (max(I_vessel) - min(I_vessel))) × 255
   ```

### 1.2 Phase Correlation Registration

**Purpose**: Find optimal XZ translation between vessel-enhanced MIPs.

**Algorithm**: Normalized 2D Cross-Correlation

**Formulas**:

1. Normalize images (zero mean, unit variance):
   ```
   Ī₁ = (I₁ - μ₁) / σ₁
   Ī₂ = (I₂ - μ₂) / σ₂

   where:
   μ = mean intensity
   σ = standard deviation
   ```

2. Compute 2D cross-correlation using FFT:
   ```
   C(Δx, Δz) = Ī₁ ⊗ Ī₂ = ∑∑ Ī₁(x,z) × Ī₂(x+Δx, z+Δz)
   ```

3. Find peak location:
   ```
   (Δx*, Δz*) = argmax(C(Δx, Δz))
   ```

4. Convert to offset from center:
   ```
   offset_x = peak_x - center_x
   offset_z = peak_z - center_z
   ```

5. Confidence score:
   ```
   confidence = max(C) / σ_C

   where σ_C is the standard deviation of C
   ```

6. Apply spatial transformation:
   ```
   V₁_aligned(y,x,z) = V₁(y, x-offset_x, z-offset_z)
   ```

### 1.3 Directional Constraints

**Purpose**: Enforce cross-pattern geometry by constraining offsets to expected directions.

**Algorithm**: Vector projection with tolerance

**Formulas**:

1. Define direction vectors (in XZ plane):
   ```
   d_right = [1, 0]    (positive X)
   d_left = [-1, 0]    (negative X)
   d_up = [0, 1]       (positive Z)
   d_down = [0, -1]    (negative Z)
   ```

2. Decompose offset into parallel and perpendicular components:
   ```
   offset = [Δx, Δz]

   parallel_magnitude = offset · d  (dot product)

   parallel_component = parallel_magnitude × d

   perpendicular_component = offset - parallel_component
   ```

3. Apply tolerance constraint:
   ```
   max_perpendicular = |parallel_magnitude| × tolerance

   If ||perpendicular_component|| > max_perpendicular:
     perpendicular_component *= max_perpendicular / ||perpendicular_component||
   ```

4. Reconstruct constrained offset:
   ```
   offset_constrained = parallel_component + perpendicular_component
   ```

---

## Step 2: Y-Axis Alignment

### 2.1 OCT Preprocessing for Surface Detection

**Purpose**: Denoise and enhance retinal layer structures.

**Algorithm**: Multi-stage denoising pipeline

**Process**:

1. Normalize to 0-255:
   ```
   I_norm = ((I - min(I)) / (max(I) - min(I))) × 255
   ```

2. Non-Local Means Denoising:
   ```
   NLM(I)(x,y) = ∑ w(x,y,x',y') × I(x',y')

   where:
   w(x,y,x',y') = (1/Z) × exp(-||N(x,y) - N(x',y')||²/(h²))

   N(x,y) = neighborhood patch around (x,y)
   h = 25 (filtering strength)
   Z = normalization constant
   ```

3. Bilateral Filter (edge-preserving smoothing):
   ```
   BF(I)(x,y) = (1/W) × ∑ G_σ_s(||p-q||) × G_σ_r(|I(p)-I(q)|) × I(q)

   where:
   G_σ_s = spatial Gaussian kernel (σ_s = 150)
   G_σ_r = range Gaussian kernel (σ_r = 150)
   W = normalization factor
   ```

4. Median Filter:
   ```
   I_median(x,y) = median({I(x',y') : (x',y') ∈ W_15×15(x,y)})

   where W_15×15 is a 15×15 window
   ```

5. Otsu Thresholding (50% threshold to preserve layers):
   ```
   threshold = 0.5 × threshold_otsu(I)

   I_thresh(x,y) = { I(x,y)  if I(x,y) ≥ threshold
                   { 0       otherwise
   ```

6. CLAHE (Contrast Limited Adaptive Histogram Equalization):
   ```
   I_clahe = CLAHE(I_thresh, clipLimit=3.0, tileSize=8×8)
   ```

### 2.2 Contour-Based Surface Detection

**Purpose**: Detect retinal surface boundary for alignment.

**Algorithm**: Top-boundary detection from denoised B-scan

**Formulas**:

1. Apply threshold to create binary mask:
   ```
   threshold = percentile_70(I_denoised)

   B(x,y) = { 1  if I(x,y) > threshold
            { 0  otherwise
   ```

2. For each column x, find first white pixel from top:
   ```
   surface(x) = min{y : B(x,y) = 1}
   ```

3. Calculate median Y-offset between surfaces:
   ```
   surface_diff(x) = surface_ref(x) - surface_mov(x)

   y_offset = median(surface_diff)
   ```

### 2.3 NCC Search for Y-Offset Validation

**Purpose**: Validate surface-based offset using correlation search.

**Algorithm**: Exhaustive search with NCC scoring

**Formulas**:

1. For each offset Δy ∈ [-50, +50]:

   a. Determine overlap region after shift:
   ```
   If Δy ≥ 0 (shift down):
     I₁_crop = I₁[0:Y-Δy, :]
     I₂_crop = I₂[Δy:Y, :]

   If Δy < 0 (shift up):
     I₁_crop = I₁[-Δy:Y, :]
     I₂_crop = I₂[0:Y+Δy, :]
   ```

   b. Calculate NCC:
   ```
   NCC(Δy) = mean((I₁_crop - μ₁) × (I₂_crop - μ₂)) / (σ₁ × σ₂)
   ```

2. Find optimal offset:
   ```
   Δy* = argmax(NCC(Δy))
   ```

3. Apply Y-shift:
   ```
   V_aligned(y,x,z) = V(y-Δy*, x, z)
   ```

---

## Step 3: Z-Rotation Alignment (In-Plane XY Rotation)

### 3.1 ECC-Style Correlation Metric

**Purpose**: Measure alignment quality between preprocessed B-scans.

**Algorithm**: Normalized correlation on aggressively denoised images

**Formulas**:

1. Preprocess both B-scans (see Step 2.1)

2. Create overlap mask:
   ```
   M = (I₁ > threshold₁) ∧ (I₂ > threshold₂)

   where threshold = percentile_10(I[I > 0])
   ```

3. Calculate normalized correlation within mask:
   ```
   I₁_masked = {I₁(x,y) : M(x,y) = 1}
   I₂_masked = {I₂(x,y) : M(x,y) = 1}

   I₁_norm = (I₁_masked - mean(I₁_masked)) / std(I₁_masked)
   I₂_norm = (I₂_masked - mean(I₂_masked)) / std(I₂_masked)

   correlation = mean(I₁_norm × I₂_norm)
   ```

### 3.2 Rotation Optimization (Coarse-to-Fine Search)

**Purpose**: Find optimal rotation angle to align retinal layers.

**Algorithm**: Two-stage grid search

**Coarse Search**:
```
For θ ∈ [-15°, -14°, ..., +14°, +15°]:  (step = 1°)
  1. Rotate B-scan:
     I₂_rotated = Rotate(I₂, θ, axes=(0,1))

  2. Calculate correlation:
     score(θ) = correlation(I₁, I₂_rotated)

  3. Track best:
     θ_coarse* = argmax(score(θ))
```

**Fine Search**:
```
For θ ∈ [θ_coarse*-3°, ..., θ_coarse*+3°]:  (step = 0.5°)
  1. Rotate B-scan:
     I₂_rotated = Rotate(I₂, θ, axes=(0,1))

  2. Calculate correlation:
     score(θ) = correlation(I₁, I₂_rotated)

  3. Track best:
     θ_fine* = argmax(score(θ))
```

**Rotation Application**:
```
V_rotated(y,x,z) = Rotate(V(y,x,z), θ_fine*, axes=(0,1))

Using bilinear interpolation with zero-padding for out-of-bounds regions.
```

### 3.3 Row-Wise Correlation Profile (RCP)

**Purpose**: Alternative metric that rewards parallel layer alignment.

**Algorithm**: Per-row correlation with uniformity penalty

**Formulas**:

1. For each row y:
   ```
   row₁ = I₁[y, :]
   row₂ = I₂[y, :]

   row₁_norm = (row₁ - mean(row₁)) / std(row₁)
   row₂_norm = (row₂ - mean(row₂)) / std(row₂)

   corr_row(y) = mean(row₁_norm × row₂_norm)
   ```

2. Calculate RCP score with variance penalty:
   ```
   RCP = mean(corr_row) × (1 - std(corr_row))

   Interpretation:
   - High mean correlation = overall good alignment
   - Low std correlation = uniform alignment across all rows
   - Product rewards both properties
   ```

### 3.4 Post-Rotation Y-Alignment Refinement

**Purpose**: Fine-tune Y-alignment after rotation (layers may shift slightly).

**Algorithm**: Central B-scan NCC search (narrow range)

**Formulas**:

1. Extract central B-scan from rotated volume:
   ```
   B₁ = V₁[:, :, Z/2]
   B₂_rotated = V₂_rotated[:, :, Z/2]
   ```

2. Search Y-offsets in range [-20, +20] with step=1:
   ```
   For each Δy:
     score(Δy) = NCC(B₁, shift(B₂_rotated, Δy))

   Δy_correction = argmax(score(Δy))
   ```

3. Apply correction if significant (|Δy| > 0.5 pixels):
   ```
   V_final(y,x,z) = V_rotated(y-Δy_correction, x, z)
   ```

---

## Step 3.5: X-Rotation Alignment (Sagittal Plane)

### Purpose
Correct pitch/tilt visible in the Y-Z plane (sagittal/coronal view).

### Algorithm
Same as Step 3, but applied to sagittal slices with rotation around X-axis.

**Key Differences**:
- Uses central sagittal slice: `V[:, X/2, :]` instead of central B-scan
- Rotation axes: `(0, 2)` instead of `(0, 1)`
- Rotates Y-Z plane instead of Y-X plane

**Formulas**: Identical to Step 3, with axis substitution:
```
V_rotated(y,x,z) = Rotate(V(y,x,z), θ*, axes=(0,2))
```

---

## Supporting Algorithms

### Normalized Cross-Correlation (NCC)

**3D Volume NCC**:
```
NCC_3D(V₁, V₂) = mean((V₁_norm × V₂_norm)[mask])

where:
mask = (V₁ > 0) ∧ (V₂ > 0)  (exclude background)

V₁_norm = (V₁[mask] - mean(V₁[mask])) / std(V₁[mask])
V₂_norm = (V₂[mask] - mean(V₂[mask])) / std(V₂[mask])
```

**2D Image NCC**:
```
NCC_2D(I₁, I₂) = mean(I₁_norm × I₂_norm)

where:
I₁_norm = (I₁ - mean(I₁)) / std(I₁)
I₂_norm = (I₂ - mean(I₂)) / std(I₂)
```

### Position-Aware Rotation Axis Selection

**Purpose**: Select appropriate rotation axis based on volume position in cross pattern.

**Rules**:
```
Horizontal volumes (left, right, far_left, far_right):
  rotation_axes = (0, 1)  → Z-rotation (YX plane)

Vertical volumes (up, down, far_up, far_down):
  rotation_axes = (0, 2)  → X-rotation (YZ plane)

Rationale:
- Horizontal volumes need YX rotation to align layers within B-scans
- Vertical volumes need YZ rotation to align layers across B-scans
```

---

## Canvas Merging

### Weighted Average Merging

**Purpose**: Combine aligned volumes into single panoramic canvas.

**Algorithm**: Progressive addition with overlap handling

**Formulas**:

1. Initialize canvas:
   ```
   Canvas(y,x,z) = 0
   Weight(y,x,z) = 0
   ```

2. For each volume V_i at offset (Δy_i, Δx_i, Δz_i):
   ```
   For each voxel (y,x,z) in V_i:
     canvas_pos = (y+Δy_i, x+Δx_i, z+Δz_i)

     Canvas[canvas_pos] += V_i[y,x,z]
     Weight[canvas_pos] += 1
   ```

3. Normalize by overlap count:
   ```
   Canvas_final(y,x,z) = Canvas(y,x,z) / max(Weight(y,x,z), 1)
   ```

4. Calculate coverage:
   ```
   coverage = (number of voxels where Weight > 0) / (total canvas voxels)
   ```

---

## Coordinate Transformations

### Global Coordinate System

Each volume has a 6-DOF transformation:

**Transform Parameters**:
```
T = {Δx, Δy, Δz, θ_z, θ_x, confidence}

where:
Δx, Δy, Δz = translation in pixels
θ_z = rotation around Z-axis (degrees)
θ_x = rotation around X-axis (degrees)
confidence = registration quality [0,1]
```

**Volume Transformation Chain**:
```
1. Translate in XZ (Step 1):
   V' = shift(V, (0, Δx, Δz))

2. Translate in Y (Step 2):
   V'' = shift(V', (Δy, 0, 0))

3. Rotate around Z (Step 3):
   V''' = rotate(V'', θ_z, axes=(0,1))

4. Rotate around X (Step 3.5, optional):
   V_final = rotate(V''', θ_x, axes=(0,2))
```

**Canvas Placement**:
```
For volume V_i with transform T_i = {Δx_i, Δy_i, Δz_i, θ_z_i, θ_x_i}:

1. Apply rotations to volume data
2. Calculate required canvas size considering all offsets
3. Place at global offset: (Y_base + Δy_i, X_base + Δx_i, Z_base + Δz_i)
```

---

## Key Parameters and Thresholds

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Frangi σ scales | {1,3,5,7,9} | Multi-scale vessel detection |
| NLM filtering strength h | 25 | Harsh denoising for OCT speckle |
| Bilateral σ_color | 150 | Edge-preserving smoothing |
| Bilateral σ_space | 150 | Spatial smoothing range |
| Otsu threshold fraction | 0.5 | Preserve tissue layers |
| CLAHE clip limit | 3.0 | Contrast enhancement |
| Y-offset search range | ±50 pixels | NCC validation search |
| Coarse rotation range | ±15° | Initial rotation search |
| Coarse rotation step | 1° | Angular resolution |
| Fine rotation range | ±3° | Refinement around coarse peak |
| Fine rotation step | 0.5° | Fine angular resolution |
| Rotation significance | 0.5° | Minimum angle to apply |
| Y-shift correction range | ±20 pixels | Post-rotation refinement |
| Direction tolerance | 0.1 (10%) | Cross-pattern constraint |
| Z-stride (subsampling) | 2 | Memory optimization |

---

## Algorithm Complexity

### Time Complexity (per volume pair)

| Step | Operation | Complexity | Typical Time |
|------|-----------|------------|--------------|
| Step 1 | Frangi filter | O(N² × K) | 2-3 min |
| Step 1 | Phase correlation | O(N² log N) | <1 sec |
| Step 2 | Preprocessing | O(N²) | 1-2 sec |
| Step 2 | NCC search | O(R × N²) | 2-5 sec |
| Step 3 | Coarse search | O(A₁ × N²) | 10-30 sec |
| Step 3 | Fine search | O(A₂ × N²) | 3-10 sec |
| Step 3.5 | X-rotation | O((A₁+A₂) × N²) | 10-30 sec |

Where:
- N = image dimensions (~1000×1000)
- K = number of Frangi scales (5)
- R = Y-search range (100)
- A₁ = coarse angles (30)
- A₂ = fine angles (12)

### Space Complexity

```
Per volume: O(Y × X × Z) ≈ 4 GB for full resolution
With Z-stride=2: O(Y × X × Z/2) ≈ 2 GB

Total for 9 volumes: ~18 GB with subsampling
Final canvas: ~25-30 GB (varies with overlap)
```

---

## Quality Metrics

### Confidence Scoring

**Phase Correlation Confidence**:
```
confidence = peak_value / std(correlation_map)

Good registration: confidence > 10
Marginal: 5 < confidence < 10
Poor: confidence < 5
```

**NCC Quality Ranges**:
```
Excellent: NCC > 0.8
Good: 0.6 < NCC < 0.8
Fair: 0.4 < NCC < 0.6
Poor: NCC < 0.4
```

### Coverage Metric

```
coverage = (voxels with data) / (total canvas voxels)

Typical values:
- 9-volume cross pattern: 35-45%
- With 2× overlap zones: 15-20% double coverage
```

---

## References

1. **Frangi Filter**: Frangi et al. (1998) "Multiscale vessel enhancement filtering"
2. **Phase Correlation**: Kuglin & Hines (1975) "The phase correlation image alignment method"
3. **Non-Local Means**: Buades et al. (2005) "A non-local algorithm for image denoising"
4. **Bilateral Filter**: Tomasi & Manduchi (1998) "Bilateral filtering for gray and color images"
5. **Otsu Thresholding**: Otsu (1979) "A threshold selection method from gray-level histograms"
6. **CLAHE**: Zuiderveld (1994) "Contrast Limited Adaptive Histogram Equalization"
7. **ECC Alignment**: Evangelidis & Psarakis (2008) "Parametric image alignment using enhanced correlation coefficient maximization"

---

## Implementation Notes

1. All spatial transformations use **bilinear interpolation** (order=1) to preserve smooth intensity transitions
2. Out-of-bounds regions are filled with **zeros** (mode='constant', cval=0)
3. Rotations use **reshape=False** to maintain consistent volume dimensions
4. All angle measurements are in **degrees** (converted internally to radians for scipy)
5. Coordinate system: **(Y, X, Z)** where Y=depth, X=lateral, Z=B-scan index
6. Confidence scores are logged for quality monitoring and debugging

---

*Document generated from scripts_modular pipeline analysis*
*Last updated: 2025-11-18*
