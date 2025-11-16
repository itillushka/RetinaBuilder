"""
Step 3: Z-Axis Rotation Alignment

Aligns volumes by rotating around Z-axis (in-plane XY rotation) to match retinal layers.
Includes Y-axis fine-tuning after rotation.
"""

import numpy as np
from scipy import ndimage
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from helpers.rotation_alignment import (
        calculate_ncc_3d,
        find_optimal_rotation_z,
        apply_rotation_z,
        find_optimal_y_shift_central_bscan,
        visualize_contour_y_alignment
    )
    ROTATION_AVAILABLE = True
except ImportError:
    ROTATION_AVAILABLE = False
    print("⚠️  Warning: rotation_alignment module not available")


def step3_rotation_z(step1_results, step2_results, data_dir):
    """
    Step 3: Z-axis rotation alignment (in-plane XY rotation) + Y-axis re-alignment.

    Step 3: Uses ECC-style correlation with aggressive denoising to find rotation angle.
    Step 3.1: Re-aligns Y-axis on central B-scan after rotation using NCC search.

    Args:
        step1_results: Dictionary from step1_xz_alignment
        step2_results: Dictionary from step2_y_alignment
        data_dir: Path to data directory for saving results

    Returns:
        results: Dictionary containing:
            - rotation_angle: Optimal rotation angle (degrees)
            - rotation_correlation: Correlation score at optimal rotation
            - y_shift_correction: Y-axis correction after rotation (pixels)
            - y_shift_ncc: NCC score for Y-shift
            - ncc_before/ncc_final: Overall alignment quality
            - overlap_v1_rotated: Final aligned volume
    """
    if not ROTATION_AVAILABLE:
        print("\n⚠️  Rotation module not available. Skipping Step 3.")
        return None

    print("\n" + "="*70)
    print("STEP 3: ROTATION ALIGNMENT + Y-AXIS RE-ALIGNMENT")
    print("="*70)
    print("  Step 3:   Rotates around Z-axis → Layer structure alignment (ECC correlation)")
    print("  Step 3.1: Re-aligns Y-axis on central B-scan (NCC search)")
    print("="*70)

    # Use Y-cropped overlap_v0 from Step 2 (not the original from Step 1)
    overlap_v0 = step2_results.get('overlap_v0', step1_results['overlap_v0'])
    overlap_v1_y_aligned = step2_results['overlap_v1_y_aligned']

    # Calculate NCC before rotation
    print("\n1. Calculating baseline NCC (before rotation)...")
    ncc_before = calculate_ncc_3d(overlap_v0, overlap_v1_y_aligned)
    print(f"  Baseline NCC: {ncc_before:.4f}")

    # Find optimal rotation angle
    print("\n2. Finding optimal rotation angle...")
    rotation_angle, rotation_metrics = find_optimal_rotation_z(
        overlap_v0,
        overlap_v1_y_aligned,
        coarse_range=15,
        coarse_step=1,
        fine_range=3,
        fine_step=0.5,
        verbose=True,
        visualize_masks=True,
        mask_vis_path=data_dir / 'step3_mask_verification.png'
    )

    correlation_optimal = rotation_metrics['optimal_correlation']

    # Apply rotation
    print(f"\n3. Applying rotation: {rotation_angle:+.2f}°...")
    overlap_v1_rotated = apply_rotation_z(
        overlap_v1_y_aligned,
        rotation_angle,
        axes=(0, 1)
    )

    # Step 3.1: Y-axis re-alignment on central B-scan (fine-tuning)
    print(f"\n3.1. Y-axis fine-tuning after rotation...")
    y_shift_correction, y_shift_ncc, y_shift_results = find_optimal_y_shift_central_bscan(
        overlap_v0,
        overlap_v1_rotated,
        search_range=20,
        step=1,
        verbose=True
    )

    # Create visualization for Step 3.1
    print(f"\n  Creating Step 3.1 visualization...")
    vis_data = y_shift_results[0]  # Get visualization data from results
    visualize_contour_y_alignment(
        bscan_v0=vis_data['bscan_v0'],
        bscan_v1=vis_data['bscan_v1'],
        bscan_v0_denoised=vis_data['bscan_v0_denoised'],
        bscan_v1_denoised=vis_data['bscan_v1_denoised'],
        surface_v0=vis_data['surface_v0'],
        surface_v1=vis_data['surface_v1'],
        y_shift=y_shift_correction,
        ncc_score=y_shift_ncc,
        confidence=vis_data['confidence'],
        output_path=data_dir / 'step3_1_contour_y_alignment.png'
    )

    # Apply Y-shift correction if significant
    if abs(y_shift_correction) > 0.5:
        print(f"\n  Applying Y correction: {y_shift_correction:+.1f} px")
        overlap_v1_rotated = ndimage.shift(
            overlap_v1_rotated,
            shift=(y_shift_correction, 0, 0),
            order=1,
            mode='constant',
            cval=0
        )
    else:
        print(f"\n  No significant Y correction needed ({y_shift_correction:+.1f} px < 0.5 px threshold)")

    # Verify final NCC
    print(f"\n4. Final verification...")
    ncc_final = calculate_ncc_3d(overlap_v0, overlap_v1_rotated)
    print(f"  Final NCC: {ncc_final:.4f}")

    # Calculate overall improvement
    improvement = (ncc_final - ncc_before) * 100
    print(f"\n  Overall NCC improvement: {ncc_before:.4f} → {ncc_final:.4f} ({improvement:+.2f}%)")
    print(f"  Optimal rotation: {rotation_angle:+.2f}°")
    print(f"  Correlation at optimal: {correlation_optimal:.4f}")
    print(f"  Y-shift fine-tuning: {y_shift_correction:+.1f} px (NCC={y_shift_ncc:.4f})")

    results = {
        # Rotation (Step 3)
        'rotation_angle': float(rotation_angle),
        'rotation_correlation': float(correlation_optimal),
        'coarse_results': rotation_metrics['coarse_results'],
        'fine_results': rotation_metrics['fine_results'],
        # Y-shift correction (Step 3.1)
        'y_shift_correction': float(y_shift_correction),
        'y_shift_ncc': float(y_shift_ncc),
        'y_shift_results': y_shift_results,
        # NCC metrics
        'ncc_before': float(ncc_before),
        'ncc_final': float(ncc_final),
        'ncc_improvement_percent': float(improvement),
        # Final volume
        'overlap_v1_rotated': overlap_v1_rotated
    }

    print("\n[OK] Step 3 complete (including Step 3.1 Y-axis re-alignment)!")
    print("="*70)

    return results
