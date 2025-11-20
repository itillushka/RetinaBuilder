"""
Step 3.5: X-Axis Rotation Alignment

Aligns volumes by rotating around X-axis (Y-Z plane / sagittal view alignment).
Corrects pitch/tilt misalignment visible in the Y-axis view.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from helpers.rotation_alignment import (
        calculate_ncc_3d,
        find_optimal_rotation_x,
        apply_rotation_x
    )
    ROTATION_AVAILABLE = True
except ImportError:
    ROTATION_AVAILABLE = False
    print("⚠️  Warning: rotation_alignment module not available")


def perform_x_rotation_alignment(ref_volume, mov_volume, visualize=False):
    """
    X-rotation alignment wrapper for multi-volume stitcher compatibility.

    Simplified version that works directly on full volumes.

    Args:
        ref_volume: Reference volume (Y, X, Z)
        mov_volume: Volume to align (Y, X, Z)
        visualize: Whether to generate visualizations (default: False)

    Returns:
        dict containing:
            - 'volume_1_rotated': Rotated moving volume (or None if rotation not significant)
            - 'rotation_angle': Optimal rotation angle (degrees)
            - 'ncc_after': NCC score after rotation
    """
    if not ROTATION_AVAILABLE:
        return {
            'volume_1_rotated': None,
            'rotation_angle': 0.0,
            'ncc_after': 0.0
        }

    # Find optimal X-axis rotation angle using full volumes
    rotation_angle_x, rotation_metrics_x = find_optimal_rotation_x(
        ref_volume,
        mov_volume,
        coarse_range=15,
        coarse_step=1,
        fine_range=3,
        fine_step=0.5,
        verbose=False,
        visualize_masks=False
    )

    # Apply rotation if significant (> 0.5 degrees)
    if abs(rotation_angle_x) > 0.5:
        volume_1_rotated = apply_rotation_x(
            mov_volume,
            rotation_angle_x,
            axes=(0, 2)
        )
        ncc_after = calculate_ncc_3d(ref_volume, volume_1_rotated)
    else:
        volume_1_rotated = None
        ncc_after = calculate_ncc_3d(ref_volume, mov_volume)

    return {
        'volume_1_rotated': volume_1_rotated,
        'rotation_angle': float(rotation_angle_x),
        'ncc_after': float(ncc_after)
    }


def step3_5_rotation_x(step1_results, step2_results, step3_results, data_dir):
    """
    Step 3.5: X-axis rotation alignment (Y-Z plane / sagittal view alignment).

    Corrects pitch/tilt misalignment visible in the Y-axis view (coronal/sagittal plane).
    Uses ECC-style correlation on sagittal slices to find optimal rotation angle.

    Args:
        step1_results: Dictionary from step1_xz_alignment
        step2_results: Dictionary from step2_y_alignment
        step3_results: Dictionary from step3_rotation_z (Z-axis rotation)
        data_dir: Path to data directory for saving results

    Returns:
        results: Dictionary containing:
            - rotation_angle_x: Optimal X-axis rotation angle (degrees)
            - rotation_correlation_x: Correlation score at optimal rotation
            - ncc_before_x/ncc_after_x: Alignment quality before/after X-rotation
            - overlap_v1_fully_rotated: Volume with both Z and X rotations applied
    """
    if not ROTATION_AVAILABLE:
        print("\n⚠️  Rotation module not available. Skipping Step 3.5.")
        return None

    print("\n" + "="*70)
    print("STEP 3.5: X-AXIS ROTATION ALIGNMENT (Y-Z PLANE)")
    print("="*70)
    print("  Goal: Align retinal layers visible in Y-axis view (sagittal/coronal plane)")
    print("  Method: ECC-style correlation on central sagittal slice")
    print("="*70)

    # Use Y-cropped overlap_v0 from Step 2 (not the original from Step 1)
    overlap_v0 = step2_results.get('overlap_v0', step1_results['overlap_v0'])
    overlap_v1_z_rotated = step3_results['overlap_v1_rotated']

    # Calculate NCC before X-rotation (after Z-rotation)
    print("\n1. Calculating baseline NCC (after Z-rotation, before X-rotation)...")
    ncc_before_x = calculate_ncc_3d(overlap_v0, overlap_v1_z_rotated)
    print(f"  Baseline NCC: {ncc_before_x:.4f}")

    # Find optimal X-axis rotation angle
    print("\n2. Finding optimal X-axis rotation angle...")
    rotation_angle_x, rotation_metrics_x = find_optimal_rotation_x(
        overlap_v0,
        overlap_v1_z_rotated,
        coarse_range=15,
        coarse_step=1,
        fine_range=3,
        fine_step=0.5,
        verbose=True,
        visualize_masks=True,
        mask_vis_path=data_dir / 'step3_5_mask_verification.png'
    )

    correlation_optimal_x = rotation_metrics_x['optimal_correlation']

    # Apply X-axis rotation
    print(f"\n3. Applying X-axis rotation: {rotation_angle_x:+.2f}°...")
    overlap_v1_fully_rotated = apply_rotation_x(
        overlap_v1_z_rotated,
        rotation_angle_x,
        axes=(0, 2)
    )

    # Verify final NCC
    print(f"\n4. Final verification...")
    ncc_after_x = calculate_ncc_3d(overlap_v0, overlap_v1_fully_rotated)
    print(f"  NCC after X-rotation: {ncc_after_x:.4f}")

    # Calculate improvement from X-rotation
    improvement_x = (ncc_after_x - ncc_before_x) * 100
    print(f"\n  X-rotation NCC improvement: {ncc_before_x:.4f} → {ncc_after_x:.4f} ({improvement_x:+.2f}%)")
    print(f"  Optimal X-axis rotation: {rotation_angle_x:+.2f}°")
    print(f"  Correlation at optimal: {correlation_optimal_x:.4f}")

    # Calculate total improvement (from Step 3 baseline)
    ncc_step3_before = step3_results['ncc_before']
    total_improvement = (ncc_after_x - ncc_step3_before) * 100
    print(f"\n  Total improvement (Steps 3 + 3.5): {ncc_step3_before:.4f} → {ncc_after_x:.4f} ({total_improvement:+.2f}%)")

    results = {
        # X-axis rotation (Step 3.5)
        'rotation_angle_x': float(rotation_angle_x),
        'rotation_correlation_x': float(correlation_optimal_x),
        'coarse_results_x': rotation_metrics_x['coarse_results'],
        'fine_results_x': rotation_metrics_x['fine_results'],
        # NCC metrics
        'ncc_before_x': float(ncc_before_x),  # After Z-rotation
        'ncc_after_x': float(ncc_after_x),    # After both rotations
        'ncc_improvement_x_percent': float(improvement_x),
        'ncc_total_improvement_percent': float(total_improvement),
        # Final volume with both rotations
        'overlap_v1_fully_rotated': overlap_v1_fully_rotated
    }

    print("\n[OK] Step 3.5 complete (X-axis rotation for Y-Z plane alignment)!")
    print("="*70)

    return results
