# Multi-Volume Panoramic OCT Stitching System

## Overview

This system extends the 2-volume alignment pipeline to support **9-volume panoramic stitching** with the following features:

- **Cross/Plus Pattern Layout**: 1 center + 4 neighbors + 4 extensions
- **B-scan Subsampling**: Load every 2nd B-scan (z_stride=2) for ~50% memory reduction
- **Parallel Processing**: Process independent volumes concurrently
- **Progressive Canvas Merging**: Build panorama incrementally
- **Global Coordinate Tracking**: Maintain spatial consistency across all volumes

## Architecture

### Component Overview

```
scripts_modular/
├── multi_volume_stitcher.py       # Main orchestrator (entry point)
├── volume_graph.py                 # Dependency graph manager
├── volume_layout.json              # Configuration file
└── helpers/
    ├── coordinate_system.py        # Global coordinate tracking
    ├── parallel_executor.py        # Parallel processing utilities
    ├── canvas_merger.py            # Progressive canvas merging
    └── oct_loader.py               # Volume loading with subsampling (UPDATED)
```

### Volume Layout (Cross/Plus Pattern)

```
        7
        |
        6
        |
5 - 4 - 1 - 2 - 3
        |
        8
        |
        9
```

- **Volume 1**: Center reference (all neighbors stitch to this)
- **Volumes 2, 4, 6, 8**: Direct neighbors (overlap with center)
- **Volumes 3, 5, 7, 9**: Extensions (overlap with neighbors only)

### Processing Levels

**Level 0**: Volume 1 (reference, no processing)

**Level 1** (Parallel):
- Volume 2 → Volume 1 (right neighbor)
- Volume 4 → Volume 1 (left neighbor)
- Volume 6 → Volume 1 (upper neighbor)
- Volume 8 → Volume 1 (lower neighbor)

**Level 2** (Parallel):
- Volume 3 → Volume 2 (far right extension)
- Volume 5 → Volume 4 (far left extension)
- Volume 7 → Volume 6 (far upper extension)
- Volume 9 → Volume 8 (far lower extension)

## Usage

### Quick Start

```bash
# Navigate to scripts_modular directory
cd scripts_modular

# Run with default settings (parallel, z_stride=2)
python multi_volume_stitcher.py \
    --oct-data ../OCT_DATA \
    --output ./results
```

### Data Organization

Organize your OCT data as follows:

```
OCT_DATA/
├── folder_1/          # Center volume
│   ├── Image_001.bmp
│   ├── Image_002.bmp
│   └── ...
├── folder_2/          # Right neighbor
│   └── ...
├── folder_3/          # Far right extension
│   └── ...
├── folder_4/          # Left neighbor
│   └── ...
├── folder_5/          # Far left extension
│   └── ...
├── folder_6/          # Upper neighbor
│   └── ...
├── folder_7/          # Far upper extension
│   └── ...
├── folder_8/          # Lower neighbor
│   └── ...
└── folder_9/          # Far lower extension
    └── ...
```

### Command-Line Options

```bash
python multi_volume_stitcher.py [OPTIONS]

Required:
  --oct-data PATH       Root directory containing folder_1, folder_2, etc.

Optional:
  --output PATH         Output directory for results (default: ./results)
  --config PATH         Path to volume_layout.json (default: auto-detect)
  --no-parallel         Disable parallel processing (sequential mode)
  --workers N           Maximum worker processes (default: auto, max 4)
```

### Examples

```bash
# Run sequentially (for debugging)
python multi_volume_stitcher.py \
    --oct-data ../OCT_DATA \
    --output ./results \
    --no-parallel

# Use custom number of workers
python multi_volume_stitcher.py \
    --oct-data ../OCT_DATA \
    --output ./results \
    --workers 2

# Use custom configuration file
python multi_volume_stitcher.py \
    --oct-data ../OCT_DATA \
    --output ./results \
    --config ./custom_layout.json
```

## Configuration

### volume_layout.json

The configuration file defines the volume layout, dependencies, and processing parameters:

```json
{
  "subsampling": {
    "enabled": true,
    "z_stride": 2,
    "description": "Load every 2nd B-scan for optimization"
  },
  "volumes": {
    "1": {
      "position": "center",
      "folder": "folder_1",
      "role": "reference"
    },
    "2": {
      "position": "right",
      "folder": "folder_2",
      "neighbor": 1,
      "level": 1
    },
    ...
  }
}
```

### Customization

To modify the layout or processing:

1. Edit `volume_layout.json` to change volume positions or dependencies
2. Adjust `z_stride` for different subsampling factors (1=no subsampling, 2=every 2nd, 3=every 3rd, etc.)
3. Change merge strategy in `ProgressiveCanvasMerger` ('average', 'max', 'additive')

## Output

### Generated Files

```
results/
├── panorama_merged_9volumes.npy   # Final merged panoramic volume
├── volume_graph_summary.txt       # Dependency graph visualization
└── alignment_logs/                # Individual alignment logs (if enabled)
```

### Output Volume Format

The merged panorama is saved as a NumPy array (.npy) with:
- **Dtype**: float32
- **Shape**: (height, width, depth) - automatically sized to fit all volumes
- **Units**: Same as input (typically grayscale intensity)

Load with:
```python
import numpy as np
panorama = np.load('results/panorama_merged_9volumes.npy')
```

## Module Details

### 1. VolumeGraph (volume_graph.py)

**Purpose**: Manages dependency graph and execution order

**Key Methods**:
- `get_execution_levels()`: Returns volumes grouped by processing level
- `get_processing_pairs(level)`: Returns (reference, moving) pairs for a level
- `get_topological_order()`: Returns valid execution order

**Example**:
```python
from volume_graph import VolumeGraph

graph = VolumeGraph()
levels = graph.get_execution_levels()  # [[1], [2,4,6,8], [3,5,7,9]]
```

### 2. GlobalCoordinateSystem (helpers/coordinate_system.py)

**Purpose**: Tracks cumulative transformations and global coordinates

**Key Methods**:
- `register_volume(id, shape, ...)`: Register a volume
- `set_transform(id, transform, ref_id)`: Set transformation
- `get_global_canvas_size()`: Calculate required canvas size
- `get_volume_offset_in_canvas(id)`: Get placement offset

**Example**:
```python
from helpers.coordinate_system import GlobalCoordinateSystem, Transform3D

coord_sys = GlobalCoordinateSystem(z_stride=2)
coord_sys.register_volume(1, shape=(500, 1000, 180))

transform = Transform3D(dx=50, dy=10, dz=0, rotation_z=5.0)
coord_sys.set_transform(2, transform, reference_id=1)

canvas_size = coord_sys.get_global_canvas_size()  # (500, 1050, 180)
```

### 3. ParallelAlignmentExecutor (helpers/parallel_executor.py)

**Purpose**: Manages parallel execution of alignment tasks

**Key Methods**:
- `execute_level(tasks, alignment_func)`: Execute tasks in parallel
- Returns `AlignmentResult` objects with transforms and aligned volumes

**Example**:
```python
from helpers.parallel_executor import ParallelAlignmentExecutor, AlignmentTask

executor = ParallelAlignmentExecutor(max_workers=4)

tasks = [
    AlignmentTask(ref_id=1, mov_id=2, ref_vol=vol1, mov_vol=vol2, level=1),
    AlignmentTask(ref_id=1, mov_id=4, ref_vol=vol1, mov_vol=vol4, level=1),
]

results = executor.execute_level(tasks, alignment_function)
```

### 4. ProgressiveCanvasMerger (helpers/canvas_merger.py)

**Purpose**: Incrementally builds panoramic canvas

**Key Methods**:
- `add_volume(volume, offset, volume_id)`: Add volume to canvas
- `finalize()`: Compute statistics and return final canvas
- `get_source_labels()`: Get volume source labels for visualization

**Example**:
```python
from helpers.canvas_merger import ProgressiveCanvasMerger

merger = ProgressiveCanvasMerger(
    initial_shape=(500, 1000, 200),
    merge_strategy='average',
    track_sources=True
)

merger.add_volume(vol1, offset=(0, 0, 0), volume_id=1)
merger.add_volume(vol2, offset=(0, 50, 0), volume_id=2)

canvas, metadata = merger.finalize()
```

## Performance Considerations

### Memory Usage

With z_stride=2:
- **Single volume**: ~500 MB (500×1000×180 float32)
- **9 volumes loaded**: ~4.5 GB
- **Final panorama**: ~2-3 GB (depending on overlap)
- **Peak usage**: ~8-10 GB (with intermediate results)

**Recommendation**: 16 GB RAM minimum for 9-volume stitching

### Processing Time

Typical execution time (4-core CPU):

- **Loading (9 volumes)**: ~2-3 minutes
- **Level 1 (4 pairs, parallel)**: ~5-8 minutes
- **Level 2 (4 pairs, parallel)**: ~5-8 minutes
- **Merging**: ~30 seconds
- **Total**: ~15-20 minutes

**With sequential mode**: ~40-60 minutes

### Optimization Tips

1. **Use parallel mode** (default) for 3-4× speedup
2. **Increase z_stride** (e.g., z_stride=3) for more memory savings
3. **Reduce max_workers** if running out of memory
4. **Disable X-rotation alignment** (Step 3.5) for faster processing

## Troubleshooting

### Common Issues

**1. Out of Memory Error**

```
MemoryError: Unable to allocate array
```

**Solutions**:
- Increase z_stride (2 → 3 or 4)
- Reduce max_workers (4 → 2)
- Close other applications
- Use a machine with more RAM

**2. Alignment Failed**

```
RuntimeError: Alignment failed for volume X
```

**Solutions**:
- Check that volumes actually overlap
- Verify BMP files are not corrupted
- Try sequential mode (`--no-parallel`) to see detailed error messages
- Check alignment parameters in step functions

**3. Volumes Don't Overlap Correctly**

**Solutions**:
- Verify volume layout configuration matches actual capture pattern
- Check that neighbor assignments in volume_layout.json are correct
- Inspect individual alignment results with visualization enabled

## Integration with Existing Pipeline

The multi-volume stitcher **reuses** the existing alignment steps:
- `steps/step1_xz_alignment.py`
- `steps/step2_y_alignment.py`
- `steps/step3_rotation_z.py`
- `steps/step3_5_rotation_x.py`

No changes to step functions are required. The multi-volume system orchestrates these steps for multiple volume pairs.

## Future Enhancements

Potential improvements:
- [ ] GPU acceleration for alignment steps
- [ ] Real-time quality metrics and validation
- [ ] Adaptive parameter selection per volume pair
- [ ] Non-rigid deformation correction
- [ ] Interactive visualization with PyVista
- [ ] Support for >9 volumes (arbitrary grid patterns)
- [ ] Incremental stitching (add volumes without reprocessing all)

## References

- Original 2-volume pipeline: `alignment_pipeline.py`
- Technical documentation: `TECHNICAL_SUMMARY.md`
- Executive summary: `EXECUTIVE_SUMMARY.md`

## License

Part of the OCT Volume Alignment project.

---

**Version**: 1.0
**Date**: 2025-01-15
**Author**: OCT Panoramic Stitching System
