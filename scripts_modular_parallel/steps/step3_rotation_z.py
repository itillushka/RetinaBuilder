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
        find_optimal_rotation_z_contour,  # NEW: Contour-based rotation
        visualize_rotation_angle_with_contours,  # NEW: Contour visualization
        create_rotation_search_summary,  # NEW: Summary visualization
        apply_rotation_z,
        find_optimal_y_shift_central_bscan,
        visualize_contour_y_alignment
    )
    from helpers.rotation_alignment_parallel import (
        apply_rotation_z_parallel,
        shift_volume_y_parallel
    )
    ROTATION_AVAILABLE = True
except ImportError:
    ROTATION_AVAILABLE = False
    print("⚠️  Warning: rotation_alignment module not available")


def perform_z_rotation_alignment(ref_volume, mov_volume, visualize=False, position=None, output_dir=None, vis_interval=1):
    """
    Position-aware rotation alignment wrapper using CONTOUR-BASED method.

    Uses surface contour variance minimization instead of NCC correlation.
    Rotates around appropriate axis based on volume position:
    - Horizontal volumes (left/right): Z-rotation (YX plane)
    - Vertical volumes (up/down): X-rotation (YZ plane)

    Args:
        ref_volume: Reference volume (Y, X, Z)
        mov_volume: Volume to align (Y, X, Z)
        visualize: Whether to generate visualizations (default: False)
        position: Volume position ('right', 'left', 'up', 'down', etc.)
                 Used to select appropriate rotation axis
        output_dir: Directory to save visualizations (required if visualize=True)
        vis_interval: Visualize every N angles (default: 1 = all angles for debugging)

    Returns:
        dict containing:
            - 'volume_1_rotated': Rotated moving volume (or None if rotation not significant)
            - 'rotation_angle': Optimal rotation angle (degrees)
            - 'ncc_after': NCC score after rotation
            - 'rotation_axes': The axes used for rotation
            - 'variance': Surface difference variance at optimal angle
    """
    if not ROTATION_AVAILABLE:
        return {
            'volume_1_rotated': None,
            'rotation_angle': 0.0,
            'ncc_after': 0.0,
            'rotation_axes': (0, 1),
            'variance': np.inf
        }

    # Get rotation axes based on position
    from helpers.direction_constraints import get_rotation_axes_for_position
    rotation_axes = get_rotation_axes_for_position(position)

    print(f"  Position '{position}': Using rotation axes {rotation_axes}")
    print(f"  Method: NCC correlation-based alignment")

    # Find optimal rotation angle using NCC method (original)
    rotation_angle, rotation_metrics = find_optimal_rotation_z(
        ref_volume,
        mov_volume,
        coarse_range=15,  # Test ±15° as requested
        coarse_step=1,
        fine_range=3,
        fine_step=0.5,
        verbose=True  # Enable verbose for debugging
    )

    # Extract central B-scan for visualization
    Z = ref_volume.shape[2]
    z_mid = Z // 2
    bscan_v0 = ref_volume[:, :, z_mid]
    bscan_v1 = mov_volume[:, :, z_mid]

    # Note: NCC method visualization can be added later if needed
    # For now, NCC method works as before without detailed per-angle visualization

    # Apply rotation if significant (> 0.5 degrees)
    # Apply rotation directly (NO inversion)
    if abs(rotation_angle) > 0.5:
        # Apply rotation with position-specific axes
        volume_1_rotated = ndimage.rotate(
            mov_volume,
            angle=-rotation_angle,  # Direct angle, no inversion
            axes=rotation_axes,
            reshape=False,
            order=1,
            mode='constant',
            cval=0
        )
        ncc_after = calculate_ncc_3d(ref_volume, volume_1_rotated)
        print(f"  Applied {rotation_angle:.2f}° rotation around axes {rotation_axes}")
        print(f"  NCC correlation at optimal: {rotation_metrics.get('optimal_correlation', 'N/A')}")
    else:
        volume_1_rotated = None
        ncc_after = calculate_ncc_3d(ref_volume, mov_volume)
        print(f"  Rotation angle {rotation_angle:.2f}° too small, skipped")

    return {
        'volume_1_rotated': volume_1_rotated,
        'rotation_angle': float(rotation_angle),
        'ncc_after': float(ncc_after),
        'rotation_axes': rotation_axes
    }


def step3_rotation_z(step1_results, step2_results, data_dir, visualize=False):
    """
    Step 3: Z-axis rotation alignment (in-plane XY rotation) + Y-axis re-alignment.

    Step 3: Uses ECC-style correlation with aggressive denoising to find rotation angle.
    Step 3.1: Re-aligns Y-axis on central B-scan after rotation using NCC search.

    Args:
        step1_results: Dictionary from step1_xz_alignment
        step2_results: Dictionary from step2_y_alignment
        data_dir: Path to data directory for saving results
        visualize: Whether to generate visualizations (default: False)

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

    # Only generate mask visualization if visualize flag is set
    mask_vis_path = data_dir / 'step3_mask_verification.png' if visualize else None

    rotation_angle, rotation_metrics = find_optimal_rotation_z(
        overlap_v0,
        overlap_v1_y_aligned,
        coarse_range=15,
        coarse_step=1,
        fine_range=3,
        fine_step=0.5,
        verbose=True,
        visualize_masks=visualize,
        mask_vis_path=mask_vis_path
    )

    correlation_optimal = rotation_metrics['optimal_correlation']

    # Apply rotation using PARALLEL OpenCV-based method
    print(f"\n3. Applying rotation: {rotation_angle:+.2f}° (PARALLEL)...")
    overlap_v1_rotated = apply_rotation_z_parallel(
        overlap_v1_y_aligned,
        rotation_angle,
        axes=(0, 1),
        n_jobs=-1  # Use all CPU cores
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

    # Create visualization for Step 3.1 only if requested
    if visualize:
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

    # Apply Y-shift correction if significant using PARALLEL method
    if abs(y_shift_correction) > 0.5:
        print(f"\n  Applying Y correction: {y_shift_correction:+.1f} px (PARALLEL)")
        overlap_v1_rotated = shift_volume_y_parallel(
            overlap_v1_rotated,
            y_shift_correction,
            n_jobs=-1  # Use all CPU cores
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
