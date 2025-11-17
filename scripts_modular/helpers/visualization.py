"""
3D Visualization Generation

Functions for creating 3D visualizations of merged OCT volumes.
"""

import numpy as np
from scipy import ndimage
from pathlib import Path

# Import visualization functions from local module
from .visualization_3d import (
    create_expanded_merged_volume,
    visualize_3d_multiangle,
    visualize_3d_comparison
)


def generate_3d_visualizations(volume_0, step1_results, step2_results, data_dir, step3_results=None, volume_1_aligned=None):
    """
    Generate 3D visualizations after all steps complete.

    Creates merged volume and multi-angle 3D projections.

    Args:
        volume_0: Reference volume
        step1_results: Results from Step 1
        step2_results: Results from Step 2
        data_dir: Output directory
        step3_results: Results from Step 3 (optional, for Y-correction)
        volume_1_aligned: Pre-aligned volume (optional, if None will be reconstructed)
    """
    print("\n" + "="*70)
    print("GENERATING 3D VISUALIZATIONS")
    print("="*70)

    # Calculate TOTAL Y-offset including correction from Step 3
    y_shift = step2_results['y_shift']
    if step3_results and 'y_shift_correction' in step3_results:
        y_shift_correction = step3_results['y_shift_correction']
        total_y_shift = -(y_shift + y_shift_correction) * 2.0  # INVERTED and scaled 2.0x
        print(f"  Y-offset calculation:")
        print(f"    Base Y-shift (Step 2): {y_shift:.2f}")
        print(f"    Y-correction (Step 3): {y_shift_correction:.2f}")
        print(f"    Total Y-offset (INVERTED, 2.0x): {total_y_shift:.2f}")
    else:
        total_y_shift = -y_shift * 2.0  # INVERTED and scaled 2.0x
        print(f"  Y-offset (INVERTED, 2.0x): {total_y_shift:.2f} (no Step 3 correction)")

    # Use provided aligned volume or reconstruct it
    if volume_1_aligned is None:
        # Get XZ-aligned volume (NO Y-shift applied to data)
        volume_1_xz_aligned = step1_results['volume_1_xz_aligned']

        # DO NOT shift volume data - only use offset for placement!
        # This preserves the original data without interpolation artifacts
        print("\n1. Using XZ-aligned volume (Y-offset for placement only)...")
        volume_1_aligned = volume_1_xz_aligned  # No Y-shift to data!
        print(f"  [OK] Volume 1 (XZ-aligned): {volume_1_aligned.shape}")

        # Y-offset is used ONLY for placement in merge function, NOT applied to data
        # This prevents interpolation artifacts and shape distortion
        transform_3d = {
            'dy': float(total_y_shift),  # For PLACEMENT only
            'dx': float(step1_results['offset_x']),
            'dz': float(step1_results['offset_z'])
        }
        print(f"  [INFO] Y-offset {total_y_shift:.2f}px will be used for placement (not applied to data)")

    else:
        print("\n1. Using pre-aligned volume (with ALL transformations applied)...")
        print(f"  [OK] Volume 1 fully aligned: {volume_1_aligned.shape}")

        # volume_1_aligned has transformations applied (rotations), but merge function
        # still needs to know the spatial offsets for proper placement in expanded canvas
        transform_3d = {
            'dy': float(total_y_shift),  # Use TOTAL Y-offset including correction
            'dx': float(step1_results['offset_x']),
            'dz': float(step1_results['offset_z'])
        }
        print(f"  [INFO] Using offsets for spatial placement: dy={total_y_shift:.2f}, dx={step1_results['offset_x']}, dz={step1_results['offset_z']}")

    # Create merged volume
    print("\n2. Creating expanded merged volume...")
    merged_volume, merge_metadata = create_expanded_merged_volume(
        volume_0, volume_1_aligned, transform_3d
    )

    # Extract source labels from metadata
    source_labels = merge_metadata.get('source_labels', None)

    print(f"\n[OK] Merged volume created: {merged_volume.shape}")
    print(f"  Total voxels: {merge_metadata['total_voxels']:,}")
    print(f"  Data loss: {merge_metadata['data_loss']}% ✅")

    # Generate visualizations
    print("\n3. Generating 3D projections...")

    # COMPARISON: Create merged volume WITHOUT Y-alignment - SKIPPED FOR FASTER TESTING
    # print("\n  [Comparison] Creating merged volume WITHOUT Y-alignment...")
    # transform_3d_no_y = {
    #     'dy': 0,  # NO Y-alignment for comparison
    #     'dx': float(step1_results['offset_x']),
    #     'dz': float(step1_results['offset_z'])
    # }
    #
    # # Get volume with only XZ alignment (no Y, no rotation)
    # volume_1_xz_only = step1_results.get('volume_1_xz_aligned', None)
    # if volume_1_xz_only is not None:
    #     merged_no_y, merge_metadata_no_y = create_expanded_merged_volume(
    #         volume_0, volume_1_xz_only, transform_3d_no_y
    #     )
    #     source_labels_no_y = merge_metadata_no_y.get('source_labels', None)
    #
    #     visualize_3d_multiangle(
    #         merged_no_y,
    #         title="WITHOUT Y-Alignment (XZ only) - For Comparison",
    #         output_path=data_dir / '3d_merged_NO_Y_alignment.png',
    #         subsample=8,
    #         percentile=75,
    #         source_labels=source_labels_no_y
    #     )
    #     print(f"  ✓ Saved comparison (NO Y-alignment): 3d_merged_NO_Y_alignment.png")

    # Multi-angle merged volume (full) - SKIPPED FOR FASTER TESTING
    # visualize_3d_multiangle(
    #     merged_volume,
    #     title="WITH Full Alignment (XZ + Y + Rotation) - Final Result",
    #     output_path=data_dir / '3d_merged_multiangle.png',
    #     subsample=8,  # Every 8th voxel for good quality/speed balance
    #     percentile=75,  # Show more tissue (less aggressive filtering)
    #     source_labels=source_labels  # Enable color coding
    # )

    # Multi-angle merged volume (with back 80 B-scans removed for clearer view) - WITH COLOR CODING - KEEP THIS ONE
    visualize_3d_multiangle(
        merged_volume,
        title="Merged Volume: Multi-Angle 3D Projections (Back 80 B-scans Removed, Color-Coded)",
        output_path=data_dir / '3d_merged_multiangle_cropped.png',
        subsample=8,
        percentile=75,  # Show more tissue
        z_crop_front=80,  # Remove back 80 B-scans (parameter name is "front" but crops from back)
        source_labels=source_labels  # Enable color coding
    )

    # Side-by-side comparison - WITH 2.0x multiplier
    visualize_3d_comparison(
        volume_0,
        volume_1_aligned,
        merged_volume,
        transform_3d,
        output_path=data_dir / '3d_comparison_sidebyside.png',
        subsample=8,
        percentile=75,
        z_crop_front=100,
        z_crop_back=100
    )

    # Save merged volume
    # print("\n4. Saving merged volume...")
    # np.save(data_dir / 'merged_volume_3d.npy', merged_volume)
    # print(f"  [OK] Saved: {data_dir / 'merged_volume_3d.npy'}")

    print("\n" + "="*70)
    print("✅ 3D VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - 3d_merged_multiangle.png (4 angle views)")
    print(f"  - 3d_comparison_sidebyside.png (side-by-side)")
