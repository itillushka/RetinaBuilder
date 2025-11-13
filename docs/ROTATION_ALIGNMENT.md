# Z-Axis Rotation Alignment for OCT Volumes

## Overview

The rotation alignment module adds **Z-axis rotation correction** to the OCT volume registration pipeline. This corrects in-plane (XY) rotational misalignment between overlapping retinal scans.

## Why Rotation Correction?

After translation alignment (Steps 1-2), volumes may still have rotational misalignment:
- **Patient head rotation** between scans
- **Eye rotation** (torsion)
- **Operator positioning** differences

Z-axis rotation (in the XZ plane / en-face view) is the most common correction needed for OCT registration.

## Algorithm

### Coarse-to-Fine Search

1. **Coarse Search** (default: ±30° in 2° steps)
   - Tests 31 rotation angles
   - Fast initial estimate
   - Uses full overlap volume

2. **Fine Search** (default: ±3° in 0.5° steps)
   - Refines around coarse optimum
   - Tests 13 angles
   - Sub-degree precision

### Optimization Metric

**NCC (Normalized Cross-Correlation):**
- Range: -1 to 1 (1 = perfect match)
- Robust to intensity variations
- Standard for medical image registration

```
NCC = Σ[(I₁ - μ₁)(I₂ - μ₂)] / (σ₁ · σ₂ · N)
```

## Usage

### Standalone Script

```bash
# Run Step 3 after completing Steps 1-2
cd scripts
python step3_rotation.py

# Custom search parameters
python step3_rotation.py --coarse-range 20 --coarse-step 1

# Skip visualizations
python step3_rotation.py --no-vis
```

### As Python Module

```python
from rotation_alignment import find_optimal_rotation_z, apply_rotation_z

# Find optimal angle
angle, metrics = find_optimal_rotation_z(
    overlap_v0,
    overlap_v1,
    coarse_range=30,
    coarse_step=2,
    fine_range=3,
    fine_step=0.5
)

# Apply rotation
rotated_volume = apply_rotation_z(volume, angle, axes=(1, 2))
```

## Module Functions

### Core Functions

#### `find_optimal_rotation_z(overlap_v0, overlap_v1, ...)`
- **Purpose:** Find optimal rotation angle using coarse-to-fine search
- **Returns:** `(optimal_angle, metrics_dict)`
- **Time:** ~30 seconds for default parameters

#### `apply_rotation_z(volume, angle, axes=(1, 2))`
- **Purpose:** Apply Z-axis rotation to volume
- **Returns:** Rotated volume (same shape)
- **Method:** Bilinear interpolation

#### `calculate_ncc_3d(volume1, volume2, mask=None)`
- **Purpose:** Calculate NCC between two volumes
- **Returns:** NCC score [-1, 1]
- **Features:** Automatic background masking

### Visualization Functions

#### `visualize_rotation_search(coarse_results, fine_results, output_path)`
- Shows NCC vs rotation angle for both search stages

#### `visualize_rotation_comparison(vol_before, vol_after, ...)`
- Side-by-side before/after comparison
- En-face MIPs and B-scan overlays

## Input/Output Files

### Required Inputs
- `notebooks/data/step1_results.npy` (XZ alignment)
- `notebooks/data/step2_results.npy` (Y alignment)

### Generated Outputs
- `step3_results.npy` - Combined results from all steps
- `rotation_params.npy` - Rotation parameters only
- `step3_rotation_search.png` - Angle search plot (2 panels)
- `step3_rotation_comparison.png` - Before/after visualization (8 panels)

## Performance

| Parameter | Value |
|-----------|-------|
| Coarse search angles | 31 (default) |
| Fine search angles | 13 (default) |
| Total runtime | ~30 seconds |
| Memory usage | ~2 GB |
| Typical NCC improvement | +2-5% |

## Integration with Pipeline

### Method 1: Standalone (Current)

```bash
python alignment_pipeline.py --steps 1 2  # Run XZ + Y
python step3_rotation.py                   # Add rotation
```

### Method 2: Integrated (Future)

```bash
python alignment_pipeline.py --all  # Will include Step 3
```

**To integrate:** Add `step3_rotation_z()` function to `alignment_pipeline.py`:

```python
elif step_num == 3:
    if step1_results is None:
        step1_results = np.load(data_dir / 'step1_results.npy', allow_pickle=True).item()
    if step2_results is None:
        step2_results = np.load(data_dir / 'step2_results.npy', allow_pickle=True).item()

    step3_results = step3_rotation_z(step1_results, step2_results, data_dir)
    visualize_step3(step1_results, step2_results, step3_results, data_dir)

    np.save(data_dir / 'step3_results.npy', {**step1_results, **step2_results, **step3_results}, allow_pickle=True)
```

## Expected Results

### Typical Scenarios

**Scenario 1: Minimal Rotation**
- Rotation angle: -1.5° to +1.5°
- NCC improvement: +1-2%
- Visual: Minor vessel alignment improvement

**Scenario 2: Moderate Rotation**
- Rotation angle: ±3° to ±10°
- NCC improvement: +3-5%
- Visual: Clear vessel pattern alignment

**Scenario 3: Large Rotation**
- Rotation angle: ±10° to ±20°
- NCC improvement: +5-10%
- Visual: Dramatic vessel pattern correction

## Parameters

### `find_optimal_rotation_z()`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coarse_range` | 30 | Max angle for coarse search (°) |
| `coarse_step` | 2 | Step size for coarse search (°) |
| `fine_range` | 3 | Range around coarse optimum (°) |
| `fine_step` | 0.5 | Step size for fine search (°) |
| `verbose` | True | Print progress messages |

### When to Adjust

**Increase `coarse_range`** if:
- Expected rotation > 30°
- Previous scans show large variations

**Decrease `coarse_step`** if:
- Need higher initial precision
- Rotation signal is weak

**Increase `fine_range`** if:
- Coarse search finds edge of range
- Bimodal NCC landscape

## Troubleshooting

### Issue: NCC improvement < 1%
**Possible causes:**
- Volumes already well-aligned (no rotation needed)
- Low overlap region quality
- Insufficient vessel contrast

**Solutions:**
- Check visual comparison (may still have small improvement)
- Verify Step 1-2 alignment quality
- Use vessel-enhanced MIPs

### Issue: Unexpected rotation angle
**Possible causes:**
- Local NCC maximum (not global)
- Symmetric vessel patterns

**Solutions:**
- Increase `coarse_step` to 1° for finer coarse search
- Visually inspect search plot for multiple peaks
- Compare multiple metrics (add SSIM)

### Issue: Slow execution
**Possible causes:**
- Large overlap volume
- Too many search angles

**Solutions:**
- Reduce `coarse_step` (test fewer angles)
- Downsample overlap volumes by 2x
- Reduce `fine_range` to ±2°

## Future Enhancements

### Planned Features
1. **X-axis rotation** (pitch/tilt in YZ plane)
2. **Multi-metric optimization** (NCC + SSIM consensus)
3. **Adaptive search ranges** (auto-detect based on coarse results)
4. **GPU acceleration** (for large volumes)

### X-Axis Rotation (Coming Soon)
- Corrects B-scan tilt
- Applied after Z-axis rotation
- Uses depth-dependent correlation

## References

### Related Notebooks
- `03_xy_registration_phase_correlation.ipynb` - Phase correlation basics
- `rotation_metrics_analysis.py` - Metric comparison study

### Academic Papers
- Lewis, J.P. (1995). "Fast Normalized Cross-Correlation"
- Penney, G.P. et al. (1998). "A Comparison of Similarity Measures for Use in 2-D-3-D Medical Image Registration"

## Contact

For questions or issues:
- Check `/notebooks/README.md` for pipeline overview
- Review `/docs/OCT_REGISTRATION_PIPELINE.md` for full system documentation
- Examine output visualizations for diagnostic information

---

**Last Updated:** 2025-11-06
**Module Version:** 1.0.0
**Compatible with:** RetinaBuilder v2.0 (phase correlation pipeline)
