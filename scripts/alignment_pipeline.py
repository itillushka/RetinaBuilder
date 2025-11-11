#!/usr/bin/env python3
"""
OCT Volume Alignment Pipeline - Step by Step

Modular pipeline that can run individual steps or full pipeline.

Steps:
  1. XZ alignment (vessel-enhanced phase correlation)
     - Includes surface visualization (X/Y slices)
  2. Y alignment (center of mass matching)
  3. Rotation alignment (RCP-based for parallel layers)
     3.1: Y-axis re-alignment on central B-scan (NCC search)

Visualizations:
  - Step 1: Automatically generates surface X/Y slice views
  - --visual flag: Generates 3D multi-angle projections and merged volume

Usage:
    # Run specific step
    python alignment_pipeline.py --step 1
    python alignment_pipeline.py --step 2
    python alignment_pipeline.py --step 3  # Includes 3.1 automatically

    # Run all steps
    python alignment_pipeline.py --all

    # Run steps 1-3
    python alignment_pipeline.py --steps 1 2 3
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage, signal
from skimage import filters

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from oct_volumetric_viewer import OCTImageProcessor, OCTVolumeLoader

# Import visualization modules
try:
    from visualization_3d import (
        create_expanded_merged_volume,
        visualize_3d_multiangle,
        visualize_3d_comparison
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  Warning: 3D visualization module not available")

try:
    from surface_visualization import (
        load_or_detect_surface,
        apply_xz_alignment_to_surface,
        visualize_surface_xz_alignment,
        visualize_surface_slices_detailed
    )
    SURFACE_VIS_AVAILABLE = True
except ImportError:
    SURFACE_VIS_AVAILABLE = False
    print("⚠️  Warning: Surface visualization module not available")

try:
    from rotation_alignment import (
        calculate_ncc_3d,
        find_optimal_rotation_z,
        apply_rotation_z,
        find_optimal_y_shift_central_bscan,
        visualize_rotation_search,
        visualize_rotation_comparison
    )
    ROTATION_AVAILABLE = True
except ImportError:
    ROTATION_AVAILABLE = False
    print("⚠️  Warning: Rotation alignment module not available")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_vessel_enhanced_mip(volume):
    """
    Create Vessel-Enhanced MIP using Frangi filter.

    This enhances tubular structures (blood vessels) for better registration.
    """
    # Average projection (better baseline than max)
    enface = np.mean(volume, axis=0)

    # Normalize
    enface_norm = (enface - enface.min()) / (enface.max() - enface.min() + 1e-8)

    # Enhance vessels using Frangi filter
    # Multiple scales to capture vessels of different sizes
    vessels_enhanced = filters.frangi(enface_norm, sigmas=range(1, 10, 2), black_ridges=False)

    # Normalize to 0-255
    vessels_final = ((vessels_enhanced - vessels_enhanced.min()) /
                     (vessels_enhanced.max() - vessels_enhanced.min() + 1e-8) * 255).astype(np.uint8)

    return vessels_final


def register_mip_phase_correlation(mip1, mip2):
    """
    Register two MIP en-face images using phase correlation.

    This is the EXACT algorithm from the working notebook.

    Args:
        mip1: Reference MIP from Volume 0
        mip2: MIP to align from Volume 1

    Returns:
        (offset_x, offset_z): Translation offset (lateral X, B-scan Z)
        confidence: Match quality score
        correlation: Full correlation map
    """
    # Normalize images (remove mean and scale by std)
    mip1_norm = (mip1 - mip1.mean()) / (mip1.std() + 1e-8)
    mip2_norm = (mip2 - mip2.mean()) / (mip2.std() + 1e-8)

    # Compute 2D correlation
    correlation = signal.correlate2d(mip1_norm, mip2_norm, mode='same')

    # Find peak (strongest match position)
    peak_x, peak_z = np.unravel_index(np.argmax(correlation), correlation.shape)
    center_x, center_z = np.array(correlation.shape) // 2

    # Calculate offset from center
    offset_x = peak_x - center_x
    offset_z = peak_z - center_z

    # Confidence = peak strength relative to noise
    confidence = correlation.max() / (correlation.std() + 1e-8)

    return (offset_x, offset_z), confidence, correlation


def find_y_center(volume):
    """
    Центр маси по Y осі.

    EXACT implementation from notebook Phase 5.

    Returns: center_y (float)
    """
    y_profile = volume.sum(axis=(1, 2))
    y_coords = np.arange(len(y_profile))
    center = np.average(y_coords, weights=y_profile + 1e-8)
    return center


# ============================================================================
# STEP 1: XZ ALIGNMENT
# ============================================================================

def step1_xz_alignment(volume_0, volume_1, data_dir):
    """
    Step 1: XZ plane alignment using Vessel-Enhanced phase correlation.

    Loads pre-computed vessel MIPs if available, otherwise creates them.

    Returns:
        volume_1_xz_aligned, offset_x, offset_z, confidence
    """
    print("\n" + "="*70)
    print("STEP 1: XZ ALIGNMENT (VESSEL-ENHANCED)")
    print("="*70)

    # Try to load pre-computed MIPs, fall back to creating them
    mip_v0_path = data_dir / 'enface_mip_vessels_volume0.npy'
    mip_v1_path = data_dir / 'enface_mip_vessels_volume1.npy'

    if mip_v0_path.exists() and mip_v1_path.exists():
        print("  Loading pre-computed Vessel-Enhanced MIPs...")
        mip_v0 = np.load(mip_v0_path)
        mip_v1 = np.load(mip_v1_path)
    else:
        print("  Pre-computed MIPs not found. Creating Vessel-Enhanced MIPs (Frangi filter)...")
        print("  This may take 2-3 minutes per volume...")
        mip_v0 = create_vessel_enhanced_mip(volume_0)
        mip_v1 = create_vessel_enhanced_mip(volume_1)

        # Save for future use
        print("  Saving MIPs for future use...")
        np.save(mip_v0_path, mip_v0)
        np.save(mip_v1_path, mip_v1)
        print(f"  ✓ Saved: {mip_v0_path.name}")
        print(f"  ✓ Saved: {mip_v1_path.name}")

    print(f"  Vessel MIP V0: shape={mip_v0.shape}, mean={mip_v0.mean():.1f}, std={mip_v0.std():.1f}")
    print(f"  Vessel MIP V1: shape={mip_v1.shape}, mean={mip_v1.mean():.1f}, std={mip_v1.std():.1f}")

    print("  Running phase correlation on vessel maps...")
    (offset_x, offset_z), confidence, correlation_map = register_mip_phase_correlation(mip_v0, mip_v1)

    print(f"\n  Results:")
    print(f"    Offset X: {offset_x} pixels")
    print(f"    Offset Z: {offset_z} pixels")
    print(f"    Confidence: {confidence:.2f}")

    # Apply shift
    print("  Applying XZ shift...")
    volume_1_xz_aligned = ndimage.shift(
        volume_1, shift=(0, offset_x, offset_z),
        order=1, mode='constant', cval=0
    )

    # Calculate overlap bounds
    # IMPORTANT: After shifting, both volumes are in the SAME coordinate system!
    # We extract the SAME region from both, avoiding zero-padding from the shift.
    Y, X, Z = volume_0.shape

    # Determine valid region (where both volumes have non-padded data)
    if offset_x >= 0:
        # Volume_1 shifted RIGHT -> zeros on left side of volume_1_xz_aligned
        x_start, x_end = offset_x, X
    else:
        # Volume_1 shifted LEFT -> zeros on right side of volume_1_xz_aligned
        x_start, x_end = 0, X + offset_x

    if offset_z >= 0:
        # Volume_1 shifted FORWARD -> zeros on front of volume_1_xz_aligned
        z_start, z_end = offset_z, Z
    else:
        # Volume_1 shifted BACKWARD -> zeros on back of volume_1_xz_aligned
        z_start, z_end = 0, Z + offset_z

    # Extract SAME region from both volumes (both are now in same coordinate system!)
    overlap_bounds = {
        'x': (x_start, x_end),
        'z': (z_start, z_end),
        'size': (Y, x_end - x_start, z_end - z_start)
    }

    print(f"\n  Overlap region (same for both volumes):")
    print(f"    X[{x_start}:{x_end}], Z[{z_start}:{z_end}]")
    print(f"    Size: {overlap_bounds['size']}")

    # Extract overlap regions - SAME indices for both!
    overlap_v0 = volume_0[:, x_start:x_end, z_start:z_end].copy()
    overlap_v1 = volume_1_xz_aligned[:, x_start:x_end, z_start:z_end].copy()

    # Save
    results = {
        'volume_1_xz_aligned': volume_1_xz_aligned,
        'overlap_v0': overlap_v0,
        'overlap_v1': overlap_v1,
        'offset_x': offset_x,
        'offset_z': offset_z,
        'confidence': confidence,
        'overlap_bounds': overlap_bounds,
        'vessel_mip_v0': mip_v0,
        'vessel_mip_v1': mip_v1
    }

    print("\n✓ Step 1 complete!")
    print("="*70)

    return results


# ============================================================================
# STEP 2: Y ALIGNMENT (CENTER OF MASS)
# ============================================================================

def step2_y_alignment(step1_results, data_dir):
    """
    Step 2: Y alignment by matching center of mass.

    EXACT implementation from notebook Phase 5 - center-based approach.

    Args:
        step1_results: Dictionary from step1_xz_alignment

    Returns:
        overlap_v1_y_aligned, y_shift, center_v0, center_v1
    """
    print("\n" + "="*70)
    print("STEP 2: Y ALIGNMENT (CENTER-BASED)")
    print("="*70)

    overlap_v0 = step1_results['overlap_v0']
    overlap_v1 = step1_results['overlap_v1']

    # Find center of mass (EXACT as notebook)
    center_y_v0 = find_y_center(overlap_v0)
    center_y_v1 = find_y_center(overlap_v1)
    y_shift = center_y_v0 - center_y_v1

    print("="*70)
    print("Y-AXIS CENTER POINTS")
    print("="*70)
    print(f"  V0 center Y: {center_y_v0:.2f}")
    print(f"  V1 center Y: {center_y_v1:.2f}")
    print(f"  Y shift needed: {y_shift:+.2f} px")
    print("="*70)

    # Apply Y shift
    overlap_v1_y_aligned = ndimage.shift(
        overlap_v1, shift=(y_shift, 0, 0),
        order=1, mode='constant', cval=0
    )

    print(f"\n✓ Applied Y shift: {y_shift:+.2f} px")
    print(f"  New center Y: {find_y_center(overlap_v1_y_aligned):.2f}")

    results = {
        'overlap_v1_y_aligned': overlap_v1_y_aligned,
        'y_shift': float(y_shift),
        'center_y_v0': float(center_y_v0),
        'center_y_v1': float(center_y_v1)
    }

    print("\n✓ Step 2 complete!")
    print("="*70)

    return results


# ============================================================================
# STEP 3: Z-AXIS ROTATION
# ============================================================================

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
        Dictionary with rotation results including:
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

    overlap_v0 = step1_results['overlap_v0']
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
        coarse_range=10,
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

    print("\n✓ Step 3 complete (including Step 3.1 Y-axis re-alignment)!")
    print("="*70)

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_step1(volume_0, volume_1, step1_results, data_dir):
    """Visualize XZ alignment results with vessel MIPs."""
    print("\nCreating Step 1 visualization...")

    volume_1_xz = step1_results['volume_1_xz_aligned']
    offset_x = step1_results['offset_x']
    offset_z = step1_results['offset_z']
    confidence = step1_results['confidence']
    vessel_mip_v0 = step1_results['vessel_mip_v0']
    vessel_mip_v1 = step1_results['vessel_mip_v1']

    z_mid = volume_0.shape[2] // 2

    fig = plt.figure(figsize=(24, 12))

    # Row 1: Vessel MIPs
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(vessel_mip_v0.T, cmap='gray', origin='lower')
    ax1.set_title(f'V0 Vessel MIP\nmean={vessel_mip_v0.mean():.1f}', fontweight='bold')
    ax1.set_xlabel('X (lateral)')
    ax1.set_ylabel('Z (B-scans)')

    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(vessel_mip_v1.T, cmap='gray', origin='lower')
    ax2.set_title(f'V1 Vessel MIP\nmean={vessel_mip_v1.mean():.1f}', fontweight='bold')
    ax2.set_xlabel('X (lateral)')

    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(vessel_mip_v0.T, cmap='Reds', alpha=0.5, origin='lower')
    ax3.imshow(vessel_mip_v1.T, cmap='Greens', alpha=0.5, origin='lower')
    ax3.set_title(f'Vessel MIPs Overlay\nConf={confidence:.2f}', fontweight='bold')
    ax3.set_xlabel('X (lateral)')

    ax4 = plt.subplot(2, 4, 4)
    ax4.text(0.5, 0.5, f'XZ Alignment\n\nΔX = {offset_x} px\nΔZ = {offset_z} px\n\nConfidence: {confidence:.2f}',
             ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')

    # Row 2: B-scan results
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(volume_0[:, :, z_mid], cmap='Reds', alpha=0.5, aspect='auto')
    ax5.imshow(volume_1[:, :, z_mid], cmap='Greens', alpha=0.5, aspect='auto')
    ax5.set_title(f'B-scan BEFORE (Z={z_mid})', fontweight='bold')
    ax5.set_ylabel('Y (depth)')
    ax5.set_xlabel('X (lateral)')

    ax6 = plt.subplot(2, 4, 6)
    ax6.imshow(volume_0[:, :, z_mid], cmap='Reds', alpha=0.5, aspect='auto')
    ax6.imshow(volume_1_xz[:, :, z_mid], cmap='Greens', alpha=0.5, aspect='auto')
    ax6.set_title(f'B-scan AFTER (Z={z_mid})', fontweight='bold')
    ax6.set_xlabel('X (lateral)')

    # Difference maps
    diff_before = np.abs(volume_0[:, :, z_mid].astype(float) - volume_1[:, :, z_mid].astype(float))
    ax7 = plt.subplot(2, 4, 7)
    im1 = ax7.imshow(diff_before, cmap='hot', aspect='auto')
    ax7.set_title(f'Diff BEFORE\n{diff_before.mean():.2f}', fontweight='bold')
    ax7.set_xlabel('X (lateral)')
    plt.colorbar(im1, ax=ax7)

    diff_after = np.abs(volume_0[:, :, z_mid].astype(float) - volume_1_xz[:, :, z_mid].astype(float))
    improvement = 100 * (1 - diff_after.mean() / diff_before.mean())
    ax8 = plt.subplot(2, 4, 8)
    im2 = ax8.imshow(diff_after, cmap='hot', aspect='auto')
    ax8.set_title(f'Diff AFTER\n{diff_after.mean():.2f} ({improvement:+.1f}%)', fontweight='bold')
    ax8.set_xlabel('X (lateral)')
    plt.colorbar(im2, ax=ax8)

    plt.suptitle(f'Step 1: XZ Alignment with Vessel Enhancement (Frangi Filter)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = data_dir / 'step1_xz_alignment.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def visualize_step2(step1_results, step2_results, data_dir):
    """Visualize Y alignment with center of mass."""
    print("\nCreating Step 2 visualization...")

    overlap_v0 = step1_results['overlap_v0']
    overlap_v1 = step1_results['overlap_v1']
    overlap_v1_y_aligned = step2_results['overlap_v1_y_aligned']

    center_y_v0 = step2_results['center_y_v0']
    center_y_v1 = step2_results['center_y_v1']
    y_shift = step2_results['y_shift']

    # Use center slices
    x_center = overlap_v0.shape[1] // 2
    z_center = overlap_v0.shape[2] // 2
    y_center_v0_int = int(center_y_v0)
    y_center_v1_int = int(center_y_v1)

    # Get slices at center positions
    # B-scans (X-slice): YZ plane at X position
    bscan_v0 = overlap_v0[:, x_center, :].copy()
    bscan_v1_before = overlap_v1[:, x_center, :].copy()
    bscan_v1_after = overlap_v1_y_aligned[:, x_center, :].copy()

    # Z-slices (cross-sectional): YX plane at Z position
    zslice_v0 = overlap_v0[:, :, z_center].copy()
    zslice_v1_before = overlap_v1[:, :, z_center].copy()
    zslice_v1_after = overlap_v1_y_aligned[:, :, z_center].copy()

    fig = plt.figure(figsize=(24, 16))

    # Row 1: B-scans BEFORE Y alignment
    ax1 = plt.subplot(4, 3, 1)
    ax1.imshow(bscan_v0, cmap='Reds', aspect='auto', vmin=0, vmax=255)
    ax1.axhline(y=center_y_v0, color='yellow', linestyle='--', linewidth=2)
    ax1.set_title(f'V0 B-scan @ X={x_center}\nCenter Y={center_y_v0:.1f}', fontweight='bold')
    ax1.set_ylabel('Y (depth)')

    ax2 = plt.subplot(4, 3, 2)
    ax2.imshow(bscan_v1_before, cmap='Greens', aspect='auto', vmin=0, vmax=255)
    ax2.axhline(y=center_y_v1, color='yellow', linestyle='--', linewidth=2)
    ax2.set_title(f'V1 B-scan @ X={x_center}\nCenter Y={center_y_v1:.1f}', fontweight='bold')

    ax3 = plt.subplot(4, 3, 3)
    ax3.imshow(bscan_v0, cmap='Reds', alpha=0.5, aspect='auto')
    ax3.imshow(bscan_v1_before, cmap='Greens', alpha=0.5, aspect='auto')
    ax3.axhline(y=center_y_v0, color='red', linestyle='--', linewidth=2, label=f'V0 Y={center_y_v0:.1f}')
    ax3.axhline(y=center_y_v1, color='green', linestyle='--', linewidth=2, label=f'V1 Y={center_y_v1:.1f}')
    ax3.set_title(f'Overlay BEFORE\nΔY={y_shift:+.1f}px', fontweight='bold')
    ax3.legend()

    # Row 2: B-scans AFTER Y alignment
    ax4 = plt.subplot(4, 3, 4)
    ax4.imshow(bscan_v0, cmap='Reds', aspect='auto', vmin=0, vmax=255)
    ax4.axhline(y=center_y_v0, color='yellow', linestyle='--', linewidth=2)
    ax4.set_title(f'V0 B-scan @ X={x_center}', fontweight='bold')
    ax4.set_ylabel('Y (depth)')

    ax5 = plt.subplot(4, 3, 5)
    ax5.imshow(bscan_v1_after, cmap='Greens', aspect='auto', vmin=0, vmax=255)
    ax5.axhline(y=center_y_v0, color='yellow', linestyle='--', linewidth=2)
    ax5.set_title(f'V1 B-scan AFTER Y shift', fontweight='bold')

    ax6 = plt.subplot(4, 3, 6)
    ax6.imshow(bscan_v0, cmap='Reds', alpha=0.5, aspect='auto')
    ax6.imshow(bscan_v1_after, cmap='Greens', alpha=0.5, aspect='auto')
    ax6.axhline(y=center_y_v0, color='yellow', linestyle='--', linewidth=2)
    ax6.set_title('Overlay AFTER\nCenters aligned!', fontweight='bold')

    # Row 3: Z-slices BEFORE (cross-sectional YX plane)
    ax7 = plt.subplot(4, 3, 7)
    ax7.imshow(zslice_v0, cmap='Reds', aspect='auto', vmin=0, vmax=255)
    ax7.set_title(f'V0 Z-slice @ Z={z_center}\nCross-section', fontweight='bold')
    ax7.set_ylabel('Y (depth)')
    ax7.set_xlabel('X (lateral)')

    ax8 = plt.subplot(4, 3, 8)
    ax8.imshow(zslice_v1_before, cmap='Greens', aspect='auto', vmin=0, vmax=255)
    ax8.set_title(f'V1 Z-slice @ Z={z_center}\nCross-section', fontweight='bold')
    ax8.set_xlabel('X (lateral)')

    ax9 = plt.subplot(4, 3, 9)
    ax9.imshow(zslice_v0, cmap='Reds', alpha=0.5, aspect='auto')
    ax9.imshow(zslice_v1_before, cmap='Greens', alpha=0.5, aspect='auto')
    ax9.axhline(y=center_y_v0, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'V0 center')
    ax9.axhline(y=center_y_v1, color='green', linestyle='--', linewidth=1, alpha=0.7, label=f'V1 center')
    ax9.set_title('Overlay BEFORE', fontweight='bold')
    ax9.set_xlabel('X (lateral)')
    ax9.legend(fontsize=8)

    # Row 4: Z-slices AFTER (cross-sectional YX plane)
    ax10 = plt.subplot(4, 3, 10)
    ax10.imshow(zslice_v0, cmap='Reds', aspect='auto', vmin=0, vmax=255)
    ax10.set_title(f'V0 Z-slice @ Z={z_center}', fontweight='bold')
    ax10.set_ylabel('Y (depth)')
    ax10.set_xlabel('X (lateral)')

    ax11 = plt.subplot(4, 3, 11)
    ax11.imshow(zslice_v1_after, cmap='Greens', aspect='auto', vmin=0, vmax=255)
    ax11.set_title('V1 Z-slice AFTER Y shift', fontweight='bold')
    ax11.set_xlabel('X (lateral)')

    ax12 = plt.subplot(4, 3, 12)
    ax12.imshow(zslice_v0, cmap='Reds', alpha=0.5, aspect='auto')
    ax12.imshow(zslice_v1_after, cmap='Greens', alpha=0.5, aspect='auto')
    ax12.axhline(y=center_y_v0, color='yellow', linestyle='--', linewidth=2)
    ax12.set_title('Overlay AFTER\nVertically aligned!', fontweight='bold')
    ax12.set_xlabel('X (lateral)')

    plt.suptitle(f'Step 2: Y Alignment (ΔY={y_shift:+.0f}px)', fontsize=18, fontweight='bold')
    plt.tight_layout()

    output_path = data_dir / 'step2_y_alignment.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def visualize_step3(step1_results, step2_results, step3_results, data_dir):
    """Visualize Z-axis rotation alignment."""
    if not ROTATION_AVAILABLE:
        print("\n⚠️  Rotation visualization not available.")
        return

    print("\nCreating Step 3 visualizations...")

    overlap_v0 = step1_results['overlap_v0']
    overlap_v1_y_aligned = step2_results['overlap_v1_y_aligned']
    overlap_v1_rotated = step3_results['overlap_v1_rotated']

    rotation_angle = step3_results['rotation_angle']
    ncc_before = step3_results['ncc_before']
    ncc_final = step3_results['ncc_final']
    y_shift_correction = step3_results.get('y_shift_correction', 0.0)

    # Rotation search plot
    visualize_rotation_search(
        step3_results['coarse_results'],
        step3_results['fine_results'],
        output_path=data_dir / 'step3_rotation_search.png'
    )

    # Before/after comparison
    visualize_rotation_comparison(
        overlap_v0,
        overlap_v1_y_aligned,
        overlap_v1_rotated,
        rotation_angle,
        ncc_before,
        ncc_final,
        output_path=data_dir / 'step3_rotation_comparison.png'
    )

    print("✓ Step 3 visualizations complete!")


# ============================================================================
# SURFACE VISUALIZATION
# ============================================================================

def visualize_surfaces_after_xz(volume_0, volume_1, volume_1_xz_aligned, step1_results, data_dir):
    """
    Visualize retinal surfaces after XZ alignment.

    Creates X-slices (B-scan view) and Y-slices (coronal view) of surfaces.
    """
    if not SURFACE_VIS_AVAILABLE:
        print("\n⚠️  Surface visualization module not available. Skipping surface visualization.")
        return

    print("\n" + "="*70)
    print("VISUALIZING SURFACES AFTER XZ ALIGNMENT")
    print("="*70)

    offset_x = step1_results['offset_x']
    offset_z = step1_results['offset_z']

    # Detect or load surfaces
    print("\n1. Detecting retinal surfaces...")
    surface_v0 = load_or_detect_surface(volume_0, method='peak')
    surface_v1 = load_or_detect_surface(volume_1, method='peak')
    surface_v1_xz = load_or_detect_surface(volume_1_xz_aligned, method='peak')

    print(f"  ✓ Surface V0: {surface_v0.shape}")
    print(f"  ✓ Surface V1 (before): {surface_v1.shape}")
    print(f"  ✓ Surface V1 (after XZ): {surface_v1_xz.shape}")

    # Create visualizations
    print("\n2. Creating surface alignment visualization...")
    visualize_surface_xz_alignment(
        surface_v0,
        surface_v1,
        surface_v1_xz,
        offset_x,
        offset_z,
        output_path=data_dir / 'surface_xz_alignment.png'
    )

    print("\n3. Creating detailed surface slices...")
    visualize_surface_slices_detailed(
        surface_v0,
        surface_v1_xz,
        offset_x,
        offset_z,
        output_path=data_dir / 'surface_slices_detailed.png',
        num_slices=5
    )

    print("\n" + "="*70)
    print("✅ SURFACE VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - surface_xz_alignment.png (overview + X/Y slices)")
    print(f"  - surface_slices_detailed.png (multiple slices)")


# ============================================================================
# 3D VISUALIZATION
# ============================================================================

def generate_3d_visualizations(volume_0, step1_results, step2_results, data_dir):
    """
    Generate 3D visualizations after all steps complete.

    Creates merged volume and multi-angle 3D projections.
    """
    if not VISUALIZATION_AVAILABLE:
        print("\n⚠️  3D visualization module not available. Skipping visualizations.")
        return

    print("\n" + "="*70)
    print("GENERATING 3D VISUALIZATIONS")
    print("="*70)

    # Get aligned volume
    volume_1_xz_aligned = step1_results['volume_1_xz_aligned']
    y_shift = step2_results['y_shift']

    # Apply Y shift to full volume 1
    print("\n1. Applying Y shift to full volume...")
    volume_1_aligned = ndimage.shift(
        volume_1_xz_aligned, shift=(y_shift, 0, 0),
        order=1, mode='constant', cval=0
    )
    print(f"  ✓ Volume 1 fully aligned: {volume_1_aligned.shape}")

    # Build transform dict
    transform_3d = {
        'dy': float(y_shift),
        'dx': float(step1_results['offset_x']),
        'dz': float(step1_results['offset_z'])
    }

    # Create merged volume
    print("\n2. Creating expanded merged volume...")
    merged_volume, merge_metadata = create_expanded_merged_volume(
        volume_0, volume_1_aligned, transform_3d
    )

    print(f"\n✓ Merged volume created: {merged_volume.shape}")
    print(f"  Total voxels: {merge_metadata['total_voxels']:,}")
    print(f"  Data loss: {merge_metadata['data_loss']}% ✅")

    # Generate visualizations
    print("\n3. Generating 3D projections...")

    # Multi-angle merged volume
    visualize_3d_multiangle(
        merged_volume,
        title="Merged Volume: Multi-Angle 3D Projections",
        output_path=data_dir / '3d_merged_multiangle.png',
        subsample=4,
        percentile=70
    )

    # Side-by-side comparison
    visualize_3d_comparison(
        volume_0,
        volume_1_aligned,
        merged_volume,
        transform_3d,
        output_path=data_dir / '3d_comparison_sidebyside.png',
        subsample=4,
        percentile=70
    )

    # Save merged volume
    print("\n4. Saving merged volume...")
    np.save(data_dir / 'merged_volume_3d.npy', merged_volume)
    print(f"  ✓ Saved: {data_dir / 'merged_volume_3d.npy'}")

    print("\n" + "="*70)
    print("✅ 3D VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - 3d_merged_multiangle.png (4 angle views)")
    print(f"  - 3d_comparison_sidebyside.png (side-by-side)")
    print(f"  - merged_volume_3d.npy ({merged_volume.nbytes / 1024**2:.1f} MB)")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='OCT Volume Alignment Pipeline - Step by Step',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run specific step
  python alignment_pipeline.py --step 1
  python alignment_pipeline.py --step 2
  python alignment_pipeline.py --step 3

  # Run multiple steps
  python alignment_pipeline.py --steps 1 2 3

  # Run all steps (XZ, Y, Rotation)
  python alignment_pipeline.py --all

  # Run all steps with 3D visualization
  python alignment_pipeline.py --all --visual
        """
    )
    parser.add_argument('--step', type=int, help='Run specific step (1=XZ, 2=Y, 3=Rotation)')
    parser.add_argument('--steps', type=int, nargs='+', help='Run multiple steps')
    parser.add_argument('--all', action='store_true', help='Run all steps (1+2+3)')
    parser.add_argument('--visual', action='store_true', help='Generate 3D visualizations after alignment')

    args = parser.parse_args()

    # Determine which steps to run
    if args.all:
        steps_to_run = [1, 2, 3]
    elif args.steps:
        steps_to_run = sorted(args.steps)
    elif args.step:
        steps_to_run = [args.step]
    else:
        print("Error: Must specify --step, --steps, or --all")
        parser.print_help()
        return

    print("="*70)
    print("OCT VOLUME ALIGNMENT PIPELINE")
    print("="*70)
    print(f"Steps to run: {steps_to_run}")

    # Setup
    data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'
    oct_data_dir = Path(__file__).parent.parent / 'oct_data'

    # Storage for results
    step1_results = None
    step2_results = None
    step3_results = None

    # Check for existing results
    step1_path = data_dir / 'step1_results.npy'
    step2_path = data_dir / 'step2_results.npy'
    step3_path = data_dir / 'step3_results.npy'

    # Determine which steps need volumes to be loaded
    need_volumes = False
    if 1 in steps_to_run:
        need_volumes = True
    elif (2 in steps_to_run or 3 in steps_to_run) and not step1_path.exists():
        # Need step 1 results but they don't exist
        print("\n⚠️  Step 1 results not found. Will run Step 1 first.")
        steps_to_run = [1] + [s for s in steps_to_run if s != 1]
        need_volumes = True
    elif args.visual:
        # Visualization requires original volumes
        need_volumes = True

    # Try to load existing results if not running those steps
    if 1 not in steps_to_run and step1_path.exists():
        print(f"\n✓ Loading existing Step 1 results from {step1_path.name}")
        step1_results = np.load(step1_path, allow_pickle=True).item()

    if 2 not in steps_to_run and step2_path.exists():
        print(f"✓ Loading existing Step 2 results from {step2_path.name}")
        step2_results = np.load(step2_path, allow_pickle=True).item()

    if 3 not in steps_to_run and step3_path.exists():
        print(f"✓ Loading existing Step 3 results from {step3_path.name}")
        step3_results = np.load(step3_path, allow_pickle=True).item()

    # Load volumes only if needed
    volume_0 = None
    volume_1 = None

    if need_volumes:
        processor = OCTImageProcessor(sidebar_width=250, crop_top=100, crop_bottom=50)
        loader = OCTVolumeLoader(processor)

        print("\nLoading volumes...")
        bmp_dirs = []
        for bmp_file in oct_data_dir.rglob('*.bmp'):
            vol_dir = bmp_file.parent
            if vol_dir not in bmp_dirs:
                bmp_dirs.append(vol_dir)

        f001_vols = sorted([v for v in bmp_dirs if 'F001_IP' in str(v)])

        if len(f001_vols) < 2:
            raise ValueError("Need at least 2 F001 volumes")

        print(f"  Volume 0: {f001_vols[0].name}")
        print(f"  Volume 1: {f001_vols[1].name}")

        volume_0 = loader.load_volume_from_directory(str(f001_vols[0]))
        volume_1 = loader.load_volume_from_directory(str(f001_vols[1]))

        print(f"  V0 shape: {volume_0.shape}")
        print(f"  V1 shape: {volume_1.shape}")

    # Execute steps
    for step_num in steps_to_run:
        if step_num == 1:
            step1_results = step1_xz_alignment(volume_0, volume_1, data_dir)
            visualize_step1(volume_0, volume_1, step1_results, data_dir)

            # Visualize surfaces after XZ alignment
            visualize_surfaces_after_xz(
                volume_0,
                volume_1,
                step1_results['volume_1_xz_aligned'],
                step1_results,
                data_dir
            )

            # Save
            np.save(data_dir / 'step1_results.npy', step1_results, allow_pickle=True)

        elif step_num == 2:
            # Load step1 if not already done
            if step1_results is None:
                print("\nLoading Step 1 results...")
                step1_results = np.load(data_dir / 'step1_results.npy', allow_pickle=True).item()

            step2_results = step2_y_alignment(step1_results, data_dir)
            visualize_step2(step1_results, step2_results, data_dir)

            # Save
            combined_results = {**step1_results, **step2_results}
            np.save(data_dir / 'step2_results.npy', combined_results, allow_pickle=True)

        elif step_num == 3:
            # Load step1 and step2 if not already done
            if step1_results is None:
                print("\nLoading Step 1 results...")
                step1_results = np.load(data_dir / 'step1_results.npy', allow_pickle=True).item()

            if step2_results is None:
                print("\nLoading Step 2 results...")
                step2_results = np.load(data_dir / 'step2_results.npy', allow_pickle=True).item()

            step3_results = step3_rotation_z(step1_results, step2_results, data_dir)

            if step3_results is not None:
                visualize_step3(step1_results, step2_results, step3_results, data_dir)

                # Save
                combined_results = {**step1_results, **step2_results, **step3_results}
                np.save(data_dir / 'step3_results.npy', combined_results, allow_pickle=True)

        else:
            print(f"\n⚠️  Step {step_num} not implemented yet")

    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)

    if step1_results:
        print(f"\nStep 1 (XZ): ΔX={step1_results['offset_x']}, ΔZ={step1_results['offset_z']}")
    if step2_results:
        print(f"Step 2 (Y): ΔY={step2_results['y_shift']:+.2f}")
        print(f"  Centers: V0={step2_results['center_y_v0']:.2f}, V1={step2_results['center_y_v1']:.2f}")
    if step3_results:
        print(f"Step 3 (Rotation): θ={step3_results['rotation_angle']:+.2f}° (NCC: {step3_results['ncc_before']:.4f}→{step3_results['ncc_final']:.4f})")

    # Generate 3D visualizations if requested
    if args.visual:
        if step1_results and step2_results:
            # Use rotated volume if available, otherwise Y-aligned
            if step3_results:
                print("\nℹ️  Using Step 3 (rotated) volume for 3D visualization")
                # Apply rotation to full volume for merging
                volume_1_xz_aligned = step1_results['volume_1_xz_aligned']
                y_shift = step2_results['y_shift']
                rotation_angle = step3_results['rotation_angle']

                # Apply Y shift + rotation to full volume
                volume_1_aligned = ndimage.shift(
                    volume_1_xz_aligned, shift=(y_shift, 0, 0),
                    order=1, mode='constant', cval=0
                )
                volume_1_aligned = apply_rotation_z(volume_1_aligned, rotation_angle, axes=(0, 1))

                # Update step2_results for visualization
                step2_results_with_rotation = step2_results.copy()
                step2_results_with_rotation['y_shift'] = y_shift

            generate_3d_visualizations(volume_0, step1_results, step2_results, data_dir)
        else:
            print("\n⚠️  Cannot generate 3D visualizations: Both steps 1 and 2 must be run")
            print("     Use: --all --visual  or  --steps 1 2 --visual")


if __name__ == '__main__':
    main()
