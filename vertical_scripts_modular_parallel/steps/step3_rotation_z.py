"""
Step 3: Z-Axis Rotation Alignment (VERTICAL VOLUMES)

Aligns volumes by rotating around Z-axis (in-plane XY rotation) to match retinal layers.
Includes Y-axis fine-tuning after rotation.

VERTICAL VERSION: Uses X-axis cross-sections (Y, Z) instead of B-scans (Y, X).
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
        visualize_contour_y_alignment,
        preprocess_oct_for_visualization,
        detect_contour_surface,
        calculate_rotation_edge_margin
    )
    from helpers.rotation_alignment_parallel import (
        apply_rotation_z_parallel,
        shift_volume_y_parallel
    )
    ROTATION_AVAILABLE = True
except ImportError:
    ROTATION_AVAILABLE = False
    print("⚠️  Warning: rotation_alignment module not available")


def visualize_step3_rotation(bscan_ref_denoised, bscan_mov_denoised, bscan_mov_rotated_denoised,
                              surface_ref, surface_mov, surface_mov_rotated,
                              rotation_angle, edge_margin, output_path, prefix=""):
    """
    Create visualization for Step 3 rotation alignment.

    Shows:
    - Reference B-scan with detected surface
    - Moving B-scan BEFORE rotation with surface
    - Moving B-scan AFTER rotation with surface
    - Surface comparison before/after rotation

    Args:
        bscan_ref_denoised: Denoised reference B-scan (Y, X)
        bscan_mov_denoised: Denoised moving B-scan BEFORE rotation (Y, X)
        bscan_mov_rotated_denoised: Denoised moving B-scan AFTER rotation (Y, X)
        surface_ref: Detected surface for reference (X,)
        surface_mov: Detected surface for moving before rotation (X,)
        surface_mov_rotated: Detected surface for moving after rotation (X,)
        rotation_angle: Applied rotation angle (degrees)
        edge_margin: Edge margin used for surface detection
        output_path: Path to save visualization
        prefix: Prefix for title
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    X = len(surface_ref)
    x_coords = np.arange(X)

    # Top-left: Reference DENOISED B-scan with surface
    axes[0, 0].imshow(bscan_ref_denoised, cmap='gray', aspect='auto')
    axes[0, 0].plot(x_coords, surface_ref, 'g-', linewidth=2, label='Detected surface')
    # Show edge margin regions
    if edge_margin > 0:
        axes[0, 0].axvspan(0, edge_margin, alpha=0.3, color='yellow', label=f'Edge margin ({edge_margin}px)')
        axes[0, 0].axvspan(X - edge_margin, X, alpha=0.3, color='yellow')
    axes[0, 0].set_title(f'{prefix} Reference B-scan (denoised, overlap region)', fontsize=12)
    axes[0, 0].set_xlabel('X (pixels)')
    axes[0, 0].set_ylabel('Y (pixels)')
    axes[0, 0].legend(loc='upper right')

    # Top-right: Moving DENOISED B-scan BEFORE rotation with surface
    axes[0, 1].imshow(bscan_mov_denoised, cmap='gray', aspect='auto')
    axes[0, 1].plot(x_coords, surface_mov, 'r-', linewidth=2, label='Detected surface')
    if edge_margin > 0:
        axes[0, 1].axvspan(0, edge_margin, alpha=0.3, color='yellow', label=f'Edge margin ({edge_margin}px)')
        axes[0, 1].axvspan(X - edge_margin, X, alpha=0.3, color='yellow')
    axes[0, 1].set_title(f'{prefix} Moving B-scan BEFORE rotation (denoised)', fontsize=12)
    axes[0, 1].set_xlabel('X (pixels)')
    axes[0, 1].set_ylabel('Y (pixels)')
    axes[0, 1].legend(loc='upper right')

    # Bottom-left: Moving DENOISED B-scan AFTER rotation with surface
    axes[1, 0].imshow(bscan_mov_rotated_denoised, cmap='gray', aspect='auto')
    axes[1, 0].plot(x_coords, surface_mov_rotated, 'b-', linewidth=2, label='Detected surface (after rotation)')
    if edge_margin > 0:
        axes[1, 0].axvspan(0, edge_margin, alpha=0.3, color='yellow', label=f'Edge margin ({edge_margin}px)')
        axes[1, 0].axvspan(X - edge_margin, X, alpha=0.3, color='yellow')
    axes[1, 0].set_title(f'{prefix} Moving B-scan AFTER rotation ({rotation_angle:+.2f}°)', fontsize=12)
    axes[1, 0].set_xlabel('X (pixels)')
    axes[1, 0].set_ylabel('Y (pixels)')
    axes[1, 0].legend(loc='upper right')

    # Bottom-right: Surface comparison (before and after rotation)
    axes[1, 1].plot(x_coords, surface_ref, 'g-', linewidth=2, label='Reference surface')
    axes[1, 1].plot(x_coords, surface_mov, 'r--', linewidth=2, alpha=0.7, label='Moving (before rotation)')
    axes[1, 1].plot(x_coords, surface_mov_rotated, 'b-', linewidth=2, label=f'Moving (after {rotation_angle:+.2f}°)')
    if edge_margin > 0:
        axes[1, 1].axvspan(0, edge_margin, alpha=0.2, color='yellow', label=f'Edge margin ({edge_margin}px)')
        axes[1, 1].axvspan(X - edge_margin, X, alpha=0.2, color='yellow')
    axes[1, 1].set_title(f'Surface comparison (rotation={rotation_angle:+.2f}°)', fontsize=12)
    axes[1, 1].set_xlabel('X (pixels)')
    axes[1, 1].set_ylabel('Y (pixels)')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {output_path.name}")


def perform_z_rotation_alignment(ref_volume, mov_volume, visualize=False, position=None, output_dir=None, vis_interval=1, offset_z=0):
    """
    Position-aware rotation alignment wrapper using CONTOUR-BASED method.

    VERTICAL VERSION: Uses Z offset for overlap cropping instead of X offset.

    Uses CROPPED overlap region for calculating rotation angle, then applies
    rotation to the full volume. This ensures accurate alignment at seams.

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
        offset_z: Z offset from step 1 (VERTICAL: Z is the unlimited axis)

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

    # CROP volumes to overlap region for CALCULATING rotation angle
    # VERTICAL: crop on Z axis instead of X axis
    Y, X_ref, Z_ref = ref_volume.shape
    _, X_mov, Z_mov = mov_volume.shape
    offset_z_int = int(round(offset_z))

    if position == 'right':
        # V2 is to the right of V1 (in Z direction for vertical)
        # offset_z is NEGATIVE (e.g., -462 means V2 shifts left 462 slices to align)
        overlap_depth = min(Z_ref, Z_mov) - abs(offset_z_int)
        if overlap_depth > 0:
            ref_cropped = ref_volume[:, :, :overlap_depth]   # V1's FIRST N slices
            mov_cropped = mov_volume[:, :, :overlap_depth]   # V2's FIRST N slices
            print(f"  [Overlap] position='right', offset_z={offset_z_int}, overlap_depth={overlap_depth}")
            print(f"  [Overlap] Cropped V1 to FIRST {overlap_depth} slices, V2 to FIRST {overlap_depth} slices")
        else:
            ref_cropped = ref_volume
            mov_cropped = mov_volume
            print(f"  [Overlap] No overlap (offset_z={offset_z_int}), using full volumes")

    elif position == 'left':
        # V2 is to the left of V1 (in Z direction for vertical)
        # offset_z is POSITIVE (e.g., +462 means V2 shifts right 462 slices to align)
        overlap_depth = min(Z_ref, Z_mov) - abs(offset_z_int)
        if overlap_depth > 0:
            ref_cropped = ref_volume[:, :, -overlap_depth:]  # V1's LAST N slices
            mov_cropped = mov_volume[:, :, -overlap_depth:]  # V2's LAST N slices
            print(f"  [Overlap] position='left', offset_z={offset_z_int}, overlap_depth={overlap_depth}")
            print(f"  [Overlap] Cropped V1 to LAST {overlap_depth} slices, V2 to LAST {overlap_depth} slices")
        else:
            ref_cropped = ref_volume
            mov_cropped = mov_volume
            print(f"  [Overlap] No overlap (offset_z={offset_z_int}), using full volumes")

    else:
        # Fallback: unknown position
        ref_cropped = ref_volume
        mov_cropped = mov_volume
        print(f"  [Overlap] Unknown position='{position}', using full volumes")

    print(f"  [Overlap] Volume shapes for calculation: ref={ref_cropped.shape}, mov={mov_cropped.shape}")

    # Find optimal rotation angle using NCC method on CROPPED volumes
    rotation_angle, rotation_metrics = find_optimal_rotation_z(
        ref_cropped,
        mov_cropped,
        coarse_range=15,  # Test ±15°
        coarse_step=1,
        fine_range=3,
        fine_step=0.5,
        verbose=True  # Enable verbose for debugging
    )

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

    # Generate visualization if requested
    if visualize and output_dir is not None:
        output_dir = Path(output_dir)
        print(f"  Creating Step 3 visualization...")

        # Extract central X-sections from CROPPED overlap region - VERTICAL: (Y, Z)
        x_mid = ref_cropped.shape[1] // 2  # VERTICAL: use X center instead of Z center
        bscan_ref = ref_cropped[:, x_mid, :]  # (Y, Z)
        bscan_mov = mov_cropped[:, x_mid, :]  # (Y, Z)

        # Rotate the cropped moving X-section for visualization
        # Use +rotation_angle for X-sections (direct, same as search)
        # Note: 3D volumes use -rotation_angle because they are vertically inverted
        bscan_mov_rotated = ndimage.rotate(
            bscan_mov,
            angle=rotation_angle,  # Direct rotation for X-sections
            axes=(0, 1),
            reshape=False,
            order=1,
            mode='constant',
            cval=0
        )

        # Denoise all B-scans
        bscan_ref_denoised = preprocess_oct_for_visualization(bscan_ref)
        bscan_mov_denoised = preprocess_oct_for_visualization(bscan_mov)
        bscan_mov_rotated_denoised = preprocess_oct_for_visualization(bscan_mov_rotated)

        # Calculate edge margin for rotation (same as used in rotation search)
        H, W = bscan_mov.shape
        edge_margin = calculate_rotation_edge_margin(H, W, rotation_angle)

        # Detect surfaces with edge margin
        surface_ref = detect_contour_surface(bscan_ref_denoised, edge_margin=edge_margin)
        surface_mov = detect_contour_surface(bscan_mov_denoised, edge_margin=0)  # No edge margin for original
        surface_mov_rotated = detect_contour_surface(bscan_mov_rotated_denoised, edge_margin=edge_margin)

        # Save surface coordinates to CSV for verification
        import csv
        csv_filename = output_dir / "step3_surfaces.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'surface_ref', 'surface_mov_before', 'surface_mov_after', 'edge_margin_px', 'rotation_angle_deg'])
            for x in range(len(surface_ref)):
                writer.writerow([x, surface_ref[x], surface_mov[x], surface_mov_rotated[x], edge_margin, rotation_angle])
        print(f"  [SAVED] step3_surfaces.csv ({len(surface_ref)} points)")

        # Create visualization
        vis_filename = f"step3_rotation_alignment.png"
        visualize_step3_rotation(
            bscan_ref_denoised, bscan_mov_denoised, bscan_mov_rotated_denoised,
            surface_ref, surface_mov, surface_mov_rotated,
            rotation_angle, edge_margin,
            output_dir / vis_filename,
            prefix=position.upper() if position else ""
        )

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
    # VERTICAL volumes use axes=(0,2) for X-axis rotation (YZ plane)
    # This is set in direction_constraints.py for 'right'/'left' positions
    rotation_axes = (0, 2)  # VERTICAL: X-axis rotation (YZ plane for X-sections)
    print(f"\n3. Applying rotation: {rotation_angle:+.2f}° around axes {rotation_axes} (PARALLEL)...")
    overlap_v1_rotated = apply_rotation_z_parallel(
        overlap_v1_y_aligned,
        rotation_angle,
        axes=rotation_axes,
        n_jobs=-1  # Use all CPU cores
    )

    # Step 3.1: Y-axis re-alignment on central B-scan (fine-tuning)
    print(f"\n3.1. Y-axis fine-tuning after rotation...")
    y_shift_correction, y_shift_ncc, y_shift_results = find_optimal_y_shift_central_bscan(
        overlap_v0,
        overlap_v1_rotated,
        search_range=20,
        step=1,
        verbose=True,
        rotation_angle=rotation_angle  # Pass rotation angle to calculate edge margin
    )

    # Create visualization for Step 3.1 only if requested
    if visualize:
        print(f"\n  Creating Step 3.1 visualization...")
        vis_data = y_shift_results[0]  # Get visualization data from results

        # Save surface coordinates to CSV for verification
        import csv
        csv_filename = data_dir / "step3_1_surfaces.csv"
        surface_v0 = vis_data['surface_v0']
        surface_v1 = vis_data['surface_v1']
        edge_margin_31 = vis_data.get('edge_margin', 0)
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'surface_ref', 'surface_mov_rotated', 'diff', 'y_shift_correction', 'edge_margin_px'])
            for x in range(len(surface_v0)):
                diff = surface_v0[x] - surface_v1[x]
                writer.writerow([x, surface_v0[x], surface_v1[x], diff, y_shift_correction, edge_margin_31])
        print(f"  [SAVED] step3_1_surfaces.csv ({len(surface_v0)} points)")

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
            output_path=data_dir / 'step3_1_contour_y_alignment.png',
            edge_margin=vis_data.get('edge_margin', 0)
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
