#!/usr/bin/env python3
"""
Visualization-Only Script

Loads existing alignment results and regenerates 3D visualizations
without re-running the alignment pipeline.
"""

import numpy as np
from pathlib import Path
from scipy import ndimage
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from oct_volumetric_viewer import OCTVolumeLoader, OCTImageProcessor
from rotation_alignment import apply_rotation_z, apply_rotation_x, apply_windowed_y_alignment
from alignment_pipeline import generate_3d_visualizations

def main():
    print("="*70)
    print("VISUALIZATION-ONLY MODE")
    print("="*70)

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'notebooks' / 'data'
    oct_data_dir = project_root / 'oct_data'

    # Load existing results
    print("\n1. Loading existing results...")
    step1_results = np.load(data_dir / 'step1_results.npy', allow_pickle=True).item()
    step2_results = np.load(data_dir / 'step2_results.npy', allow_pickle=True).item()
    step3_results = np.load(data_dir / 'step3_results.npy', allow_pickle=True).item()

    print(f"  ✓ Step 1: XZ alignment")
    print(f"  ✓ Step 2: Y-shift = {step2_results['y_shift']:+.2f}px")
    print(f"  ✓ Step 3: Z-rotation = {step3_results['rotation_angle']:+.2f}°")

    if 'rotation_angle_x' in step3_results:
        print(f"  ✓ Step 3.5: X-rotation = {step3_results['rotation_angle_x']:+.2f}°")

    if 'y_offsets_interpolated' in step3_results:
        print(f"  ✓ Step 4: Windowed Y-alignment ({len(step3_results['y_offsets_interpolated'])} B-scans)")

    if 'ncc_after_bspline' in step3_results:
        print(f"  ✓ Step 5: B-spline FFD (NCC: {step3_results['ncc_after_bspline']:.4f})")
        print(f"    Note: Step 5 applied to overlap region; visualization uses Step 4 for full volume")

    # Load volumes
    print("\n2. Loading volumes...")
    processor = OCTImageProcessor(sidebar_width=250, crop_top=100, crop_bottom=50)
    loader = OCTVolumeLoader(processor)

    bmp_dirs = []
    for bmp_file in oct_data_dir.rglob('*.bmp'):
        vol_dir = bmp_file.parent
        if vol_dir not in bmp_dirs:
            bmp_dirs.append(vol_dir)

    f001_vols = sorted([v for v in bmp_dirs if 'F001_IP' in str(v)])

    print(f"  Loading Volume 0: {f001_vols[0].name}")
    volume_0 = loader.load_volume_from_directory(str(f001_vols[0]))

    # Apply transformations
    print("\n3. Applying transformations...")
    volume_1_xz_aligned = step1_results['volume_1_xz_aligned']
    y_shift = step2_results['y_shift']
    rotation_angle_z = step3_results['rotation_angle']
    rotation_angle_x = step3_results.get('rotation_angle_x')
    y_offsets = step3_results.get('y_offsets_interpolated')

    # Y-shift
    print(f"  Applying Y-shift: {y_shift:+.2f}px")
    volume_1_aligned = ndimage.shift(
        volume_1_xz_aligned,
        shift=(y_shift, 0, 0),
        order=1, mode='constant', cval=0
    )

    # Z-axis rotation
    print(f"  Applying Z-rotation: {rotation_angle_z:+.2f}°")
    volume_1_aligned = apply_rotation_z(volume_1_aligned, rotation_angle_z, axes=(0, 1))

    # X-axis rotation
    if rotation_angle_x is not None:
        print(f"  Applying X-rotation: {rotation_angle_x:+.2f}°")
        volume_1_aligned = apply_rotation_x(volume_1_aligned, rotation_angle_x, axes=(0, 2))

    # Windowed Y-alignment
    if y_offsets is not None:
        print(f"  Applying windowed Y-offsets...")
        # Extend offsets to match full volume
        if volume_1_aligned.shape[2] > len(y_offsets):
            pad_size = volume_1_aligned.shape[2] - len(y_offsets)
            y_offsets_full = np.pad(y_offsets, (0, pad_size), mode='edge')
            print(f"    Extended {len(y_offsets)} → {len(y_offsets_full)} B-scans")
        else:
            y_offsets_full = y_offsets

        volume_1_aligned = apply_windowed_y_alignment(
            volume_1_aligned,
            y_offsets_full,
            verbose=False
        )

    # Generate visualizations
    print("\n4. Generating 3D visualizations...")
    generate_3d_visualizations(
        volume_0,
        step1_results,
        step2_results,
        data_dir,
        volume_1_aligned=volume_1_aligned
    )

    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nOutput files in: {data_dir}")
    print("  - 3d_merged_multiangle.png")
    print("  - 3d_comparison_sidebyside.png")

if __name__ == '__main__':
    main()
