# RetinaBuilder - OCT Volume Stitching Project

Clean, production-ready OCT volume stitching and visualization system.

## 📁 Project Structure

```
RetinaBuilder/
├── src/
│   ├── oct_volumetric_viewer.py    # PyVista-based 3D viewer with proper voxel spacing
│   ├── oct_grid_stitcher.py        # 4×2 grid stitcher for 8 OCT volumes
│   └── requirements.txt             # Python dependencies
├── oct_data/                        # 8 OCT volumes (6mm × 6mm each)
├── venv/                            # Python virtual environment
├── stitched_2vol.npz               # Example stitched output
└── CLAUDE.md                        # Development configuration

## 🚀 Quick Start

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 2. View Single Volume
```bash
python src/oct_volumetric_viewer.py \
  --data-dir oct_data/F001_IP_20250604_175814_Retina_3D_L_6mm_1536x360_2 \
  --mode volume
```

### 3. Stitch Volumes

**Test with 2 volumes:**
```bash
python src/oct_grid_stitcher.py \
  --data-dir oct_data \
  --num-volumes 2 \
  --output stitched_2vol.npz
```

**Stitch full 4×2 grid (8 volumes):**
```bash
python src/oct_grid_stitcher.py \
  --data-dir oct_data \
  --num-volumes 8 \
  --output stitched_full.npz \
  --visualize
```

## 📊 Volume Grid Layout

```
Row 1: [Vol 4] [Vol 5] [Vol 6] [Vol 7]
Row 0: [Vol 0] [Vol 1] [Vol 2] [Vol 3]
       X=0     X=1     X=2     X=3
```

Each volume: 6mm × 6mm scan area
Expected overlap: ~12.5% between adjacent volumes

## 🔧 Key Features

### Volumetric Viewer
- Proper physical voxel spacing (3.9μm × 3.9μm × 16.7μm)
- Multiple visualization modes (volume, slices, surface)
- PyVista-based (no browser limitations)
- VTK export support

### Grid Stitcher
- Optimized registration (25% overlap regions only)
- Phase correlation for alignment
- Smooth blending in overlap regions
- Row-by-row stitching strategy

## 📝 Technical Specs

- **Input**: 942 × 1536 × 360 per volume (after preprocessing)
- **Preprocessing**: 250px left sidebar removal, 50px top crop
- **Registration**: Phase correlation on overlap regions
- **Output**: Full stitched volume with proper physical dimensions

## ⏱️ Performance

- 2-volume stitch: ~10 minutes
- Full 8-volume grid: ~40-50 minutes (estimated)
- Memory usage: ~6GB per volume loaded

## 🛠️ Dependencies

See `src/requirements.txt`:
- numpy
- scipy
- pyvista
- pillow
- matplotlib

## 📖 Usage Examples

### Visualize Different Views
```bash
# Volume rendering
python src/oct_volumetric_viewer.py --data-dir oct_data/[volume] --mode volume

# Orthogonal slices
python src/oct_volumetric_viewer.py --data-dir oct_data/[volume] --mode slices

# Surface extraction
python src/oct_volumetric_viewer.py --data-dir oct_data/[volume] --mode surface
```

### Export to VTK
```bash
python src/oct_volumetric_viewer.py \
  --data-dir oct_data/[volume] \
  --mode volume \
  --export output.vtk
```

## 🔬 Algorithm Details

**Registration Optimization:**
- Uses only 25% overlap regions (75% data reduction)
- Single middle slice for fastest alignment
- Median filtering for robustness

**Stitching Strategy:**
1. Stitch Row 0: Vol0 → Vol1 → Vol2 → Vol3
2. Stitch Row 1: Vol4 → Vol5 → Vol6 → Vol7
3. Merge rows vertically

## 📄 License

Research/Educational Use
