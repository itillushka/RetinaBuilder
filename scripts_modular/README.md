# OCT Volume Alignment Pipeline - Modular Architecture

This is a refactored version of the alignment pipeline with improved modularity and organization.

## Directory Structure

```
scripts_modular/
├── alignment_pipeline.py      # Main orchestrator
├── steps/                      # Individual alignment steps
│   ├── __init__.py
│   ├── step1_xz_alignment.py  # XZ plane alignment (vessel-enhanced)
│   ├── step2_y_alignment.py   # Y-axis alignment (center of mass)
│   ├── step3_rotation_z.py    # Z-axis rotation alignment
│   └── step3_5_rotation_x.py  # X-axis rotation alignment
├── helpers/                    # Utility modules
│   ├── __init__.py
│   ├── volume_transforms.py   # Transformation functions
│   ├── mip_generation.py      # MIP generation and registration
│   └── visualization.py       # 3D visualization generation
└── README.md                   # This file
```

## Architecture Benefits

### 1. **Separation of Concerns**
   - Each step is self-contained in its own module
   - Helper functions grouped by functionality
   - Clear dependencies between modules

### 2. **Maintainability**
   - Easier to locate and fix bugs
   - Changes to one step don't affect others
   - Clear interfaces between modules

### 3. **Testability**
   - Each module can be tested independently
   - Easier to write unit tests
   - Isolated integration points

### 4. **Reusability**
   - Helper functions can be imported anywhere
   - Steps can be run individually or combined
   - Easy to create alternative pipelines

## Module Descriptions

### Main Pipeline (`alignment_pipeline.py`)
- Orchestrates all alignment steps
- Handles command-line arguments
- Loads OCT data
- Saves and loads step results
- Manages visualization generation

### Steps Modules

#### `step1_xz_alignment.py`
- **Purpose**: Align volumes in XZ plane using vessel-enhanced MIPs
- **Input**: volume_0, volume_1, data_dir
- **Output**: offset_x, offset_z, overlap regions, confidence
- **Method**: Phase correlation on Frangi-filtered vessel maps

#### `step2_y_alignment.py`
- **Purpose**: Align volumes along Y-axis
- **Input**: step1_results, data_dir
- **Output**: y_shift, aligned overlap regions
- **Method**: Center of mass matching with zero-cropping

#### `step3_rotation_z.py`
- **Purpose**: Rotate around Z-axis to align retinal layers
- **Input**: step1_results, step2_results, data_dir
- **Output**: rotation_angle, Y-shift correction, rotated volume
- **Method**: ECC correlation with Y-axis fine-tuning

#### `step3_5_rotation_x.py`
- **Purpose**: Rotate around X-axis for sagittal plane alignment
- **Input**: step1_results, step2_results, step3_results, data_dir
- **Output**: rotation_angle_x, fully rotated volume
- **Method**: ECC correlation on sagittal slices

### Helper Modules

#### `volume_transforms.py`
- `rotate_volume_around_point()`: Rotate volume around arbitrary center point
- `apply_all_transformations_to_volume()`: Apply all transformations sequentially to original volume

#### `mip_generation.py`
- `create_vessel_enhanced_mip()`: Generate Frangi-filtered vessel MIP
- `register_mip_phase_correlation()`: Phase correlation registration
- `find_y_center()`: Calculate center of mass along Y-axis

#### `visualization.py`
- `generate_3d_visualizations()`: Create merged volumes and 3D projections
- Generates multiple views and comparison images

## Usage

### Run All Steps
```bash
python alignment_pipeline.py --all --visual
```

### Run Specific Steps
```bash
# Run Step 1 only
python alignment_pipeline.py --step 1

# Run Steps 1, 2, and 3
python alignment_pipeline.py --steps 1 2 3
```

### Visualization Only
```bash
# Regenerate visualizations from saved results
python alignment_pipeline.py --visual-only
```

## Key Design Patterns

### 1. **Step Results Pattern**
Each step returns a dictionary containing:
- Transformation parameters (offsets, angles, etc.)
- Transformed volumes (for next step's input)
- Quality metrics (NCC scores, confidence, etc.)
- Metadata (bounds, intermediate results, etc.)

### 2. **Separation of Calculation and Application**
- Steps calculate transformations on overlap regions
- `apply_all_transformations_to_volume()` applies all transforms to original volume_1
- This avoids accumulation of interpolation errors

### 3. **Lazy Loading**
- Step results are saved to .npy files
- Subsequent steps load previous results if not in memory
- Allows resuming pipeline from any step

## Migration from Original Pipeline

The modular version maintains full compatibility with the original pipeline:

1. **Same algorithms**: All core algorithms are identical
2. **Same parameters**: Rotation ranges, NCC calculations, etc. are unchanged
3. **Same results**: Output .npy files have the same structure
4. **Same visualization**: 3D visualizations are generated identically

To use the modular version:
```bash
# Instead of:
python scripts/alignment_pipeline.py --all --visual

# Use:
python scripts_modular/alignment_pipeline.py --all --visual
```

## Future Enhancements

Possible improvements with this architecture:

1. **Configuration files**: YAML/JSON configs for parameters
2. **Plugin system**: Add new steps without modifying core code
3. **Parallel processing**: Run independent steps concurrently
4. **Web interface**: REST API for remote execution
5. **Batch processing**: Process multiple volume pairs automatically
6. **Quality reports**: Generate comprehensive alignment quality reports
7. **Alternative algorithms**: Easy to swap different registration methods

## Dependencies

The modular pipeline depends on the original scripts:
- `scripts/oct_loader.py`: OCT volume loading
- `scripts/rotation_alignment.py`: Rotation calculation functions
- `scripts/visualization_3d.py`: 3D visualization rendering

These dependencies are imported at runtime using path manipulation.
