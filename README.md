# RetinaBuilder

**Multi-Volume OCT Registration and Stitching Pipeline**

*Diploma Thesis Project*

---

## Overview

RetinaBuilder is a comprehensive pipeline for registering and stitching multiple Optical Coherence Tomography (OCT) volumes to create extended field-of-view retinal images. The system aligns up to 5 overlapping OCT volumes into a single unified 3D volume and generates panoramic B-scan visualizations with retinal curvature analysis.

## Features

- **Multi-volume alignment**: Aligns 5 OCT volumes using a two-phase approach around a central anchor volume
- **3D registration**: XZ lateral alignment, Y-axis depth alignment, and Z-axis rotation correction
- **B-scan panorama generation**: Creates averaged B-scan panoramas from aligned volumes
- **Retinal curvature analysis**: Estimates posterior pole curvature by detecting RPE layer and fitting circles
- **Parallel processing**: Optimized for performance with multi-core support

## Pipeline Architecture

```
Volume spatial order: V5 <- V4 <- V1 -> V2 -> V3
                            (center anchor)

Phase 1 (Right Chain): V2->V1, V3->V2 -> merged_right (V1+V2+V3)
Phase 2 (Left Chain):  V4->V1, V5->V4 -> merged_left (V1+V4+V5)
Phase 3 (Final Merge): Combine both chains with V1 as static anchor
```

## Directory Structure

```
RetinaBuilder/
├── scripts_modular_parallel/     # Main pipeline code
│   ├── five_volume_alignment.py  # Primary alignment pipeline
│   ├── visualize_bscan_panorama.py  # B-scan panorama generation
│   ├── retina_curvature_analysis.py # Curvature estimation
│   ├── visualize_surfaces.py     # Surface visualization
│   │
│   ├── helpers/                  # Core modules
│   │   ├── rotation_alignment.py # Rotation alignment algorithms
│   │   ├── volume_transforms.py  # 3D transformation utilities
│   │   ├── mip_generation.py     # Maximum intensity projections
│   │   ├── oct_loader.py         # OCT data loading
│   │   ├── visualization_3d.py   # 3D rendering
│   │   └── parallel_executor.py  # Parallel processing
│   │
│   ├── steps/                    # Alignment step modules
│   │   ├── step1_xz_alignment.py # XZ lateral alignment
│   │   ├── step2_y_alignment.py  # Y depth alignment
│   │   └── step3_rotation_z.py   # Z-axis rotation correction
│   │
│   └── averaged_bscan_alignment/ # B-scan alignment submodule
│       ├── averaged_bscan_pipeline.py
│       ├── bscan_averaging.py
│       └── averaged_alignment_steps.py
│
├── oct_data/                     # OCT scan data (not tracked)
├── notebooks/                    # Jupyter notebooks for exploration
└── venv/                         # Python virtual environment
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- OpenCV (cv2)
- Matplotlib

## Installation

```bash
# Clone or download the repository
cd RetinaBuilder

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install numpy scipy opencv-python matplotlib
```

## Usage

### 1. Five-Volume Alignment

Align 5 OCT volumes from a patient:

```bash
cd scripts_modular_parallel
python five_volume_alignment.py --patient EM005
```

Options:
- `--patient`: Patient ID (folder name in oct_data/)
- `--visual`: Generate 3D visualizations (optional)

Output files saved to `results_5vol_<patient>/`:
- `final_merged_5volumes.npy` - Merged 3D volume
- `alignment_transforms.json` - Transformation parameters
- `3d_final_5volumes_multiangle.png` - 3D visualization

### 2. B-Scan Panorama Visualization

Generate averaged B-scan panorama from alignment results:

```bash
python visualize_bscan_panorama.py results_5vol_em005/alignment_transforms.json
```

Options:
- `--n-bscans`: Number of central B-scans to average (default: 30)
- `--output`: Custom output path

Output:
- `averaged_middle_bscans_panorama.png` - Panorama image
- `averaged_middle_bscans_panorama.npy` - Panorama data for analysis

### 3. Retinal Curvature Analysis

Estimate retinal curvature from B-scan panorama:

```bash
python retina_curvature_analysis.py --data_dir results_5vol_em005 --input panorama.npy
```

Options:
- `--data_dir`: Output directory
- `--input`: Input .npy file (panorama or averaged B-scan)
- `--pixel_size`: Pixel size in mm (default: 0.003906 for 6mm/1536)

Output:
- `retina_curvature_analysis.png` - Visualization with detected RPE points and fitted circle
- `retina_curvature_results.npy` - Curvature measurements (radius in mm, curvature in mm^-1)

## OCT Data Format

Expected input format:
- Directory of TIFF images per volume
- Each image is one B-scan (Y x X)
- Images ordered by Z-index (B-scan number)
- Typical dimensions: 1536 x 1536 pixels per B-scan, 360 B-scans per volume
- Scale: 6mm x 6mm x 6mm scan region

## Alignment Algorithm

### Step 1: XZ Lateral Alignment
- Generates Maximum Intensity Projections (MIP) for both volumes
- Uses Normalized Cross-Correlation (NCC) to find optimal X and Z offsets
- Provides sub-pixel accuracy through interpolation

### Step 2: Y Depth Alignment
- Detects retinal surface using contour-based method
- Computes surface height differences across overlapping region
- Uses median of differences for robust Y-shift estimation

### Step 3: Z-Axis Rotation Correction
- Extracts central B-scans from both volumes
- Detects tissue contours using edge detection
- Tests rotation angles and finds optimal alignment via contour matching
- Applies rotation around volume center

## Curvature Analysis

The retinal curvature analysis detects the RPE (Retinal Pigment Epithelium) layer at 9 points across the B-scan panorama and fits a circle using least squares:

- Points detected at: edges, 1/8, 1/4, 3/8, center, 5/8, 3/4, 7/8 positions
- Uses peak detection to find brightest, deepest layer (RPE)
- Reports radius in mm and curvature in mm^-1
- Clinical reference: Normal posterior pole radius ~11-13 mm

## Author

Illia Pastushok - Diploma Thesis

## License

This project is part of a diploma thesis. All rights reserved.
