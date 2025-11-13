# OCT Registration Pipeline - Notebooks

## 🚀 New Phase Correlation Pipeline (Use These!)

Run in this order:

### 1️⃣ Phase 1: Retinal Surface Detection
**Notebook:** `01_retinal_surface_detection.ipynb`
- Loads raw OCT volumes
- Removes text from B-scans (100px top, 50px bottom)
- Detects retinal surfaces
- Creates MIP (Maximum Intensity Projection)
- **Output:** `data/enface_mip_volume*.npy`, `data/surface_peaks_volume*.npy`
- **Time:** ~2 minutes

---

### 2️⃣ Phase 2: Vessel Segmentation (OPTIONAL - SKIP FOR REGISTRATION!)
**Notebook:** `02_vessel_segmentation_simplified.ipynb`
- ⚠️ **NOT NEEDED** for registration pipeline
- Only run if you need vessel density analysis or morphology metrics
- Phase 3 uses MIP directly from Phase 1

---

### 3️⃣ Phase 3: XY Registration (Phase Correlation)
**Notebook:** `03_xy_registration_phase_correlation.ipynb`
- Uses phase correlation on MIP (no bifurcation detection!)
- Finds lateral (X) and B-scan (Z) offsets
- **Output:** `data/xy_registration_params.npy`
- **Time:** ~2 seconds ⚡
- **Success:** Confidence > 5.0, Improvement > 15%

---

### 4️⃣ Phase 4: Z-Axis Alignment
**Notebook:** `04_z_axis_alignment.ipynb`
- Refines Z-axis offset using coronal planes
- **Output:** `data/xyz_registration_params.npy`
- **Time:** ~5 seconds
- **Success:** Additional offset < 10 B-scans, Confidence > 3.0

---

### 5️⃣ Phase 5: Y-Axis Depth Alignment
**Notebook:** `05_depth_alignment.ipynb`
- Aligns volumes in depth using retinal surfaces
- **Output:** `data/registration_3d_params.npy`
- **Time:** ~3 seconds
- **Success:** Surface difference < 10 pixels

---

### 6️⃣ Phase 6: Visualization & Expanded Merging
**Notebook:** `06_visualize_results.ipynb`
- Applies full 3D transformation
- **EXPANDED MERGING:** Volume grows to fit both inputs (0% data loss!)
- Generates 3D visualizations
- **Output:** `data/merged_volume_expanded.npy`
- **Time:** ~30 seconds
- **Success:** Expanded size > original (942×1536×360)

---

## 📊 Total Pipeline Time

**Registration only:** Phases 1 → 3 → 4 → 5 → 6 = **~3 minutes**

**With vessel analysis:** Phases 1 → 2 → 3 → 4 → 5 → 6 = **~3.5 minutes**

---

## 🗂️ Old Notebooks (Archived)

Old bifurcation-based pipeline moved to: `old_bifurcation_notebooks/`

**Why archived:**
- 88% false positive rate in bifurcation detection
- Registration failed (0,0 offset)
- 15x slower (~30 seconds vs ~2 seconds)
- 10% data loss in volume merging

---

## 🔍 Quick Testing Checklist

### Phase 1
- [ ] B-scans have no text at top or bottom
- [ ] Surface detection looks smooth
- [ ] MIP shows clear vessel patterns

### Phase 3 (Critical!)
- [ ] Confidence score > 5.0
- [ ] Correlation map has single sharp peak
- [ ] Overlay shows aligned vessels (Red + Green = Yellow)
- [ ] Improvement > 15%

### Phase 6 (Critical!)
- [ ] Merged volume is LARGER than original (not 942×1536×360!)
- [ ] Data loss = 0%
- [ ] 3D scatter shows continuous vessel tree

---

## 📖 Full Documentation

See: `../docs/OCT_REGISTRATION_PIPELINE.md`
