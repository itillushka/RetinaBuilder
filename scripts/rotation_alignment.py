#!/usr/bin/env python3
"""
Rotation Alignment Module for OCT Volume Registration

Implements Z-axis (in-plane XY) rotation correction using NCC optimization.

This module provides:
  - Z-axis rotation detection (coarse-to-fine search)
  - NCC-based metric optimization
  - Rotation visualization functions

Usage:
    from rotation_alignment import find_optimal_rotation_z, apply_rotation_z

    angle, metrics = find_optimal_rotation_z(overlap_v0, overlap_v1)
    volume_rotated = apply_rotation_z(volume, angle)
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import rotate
import matplotlib.pyplot as plt


# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def calculate_ncc(img1, img2, mask=None):
    """
    Normalized Cross-Correlation - standard metric for image registration.

    NCC = sum((img1 - mean1) * (img2 - mean2)) / (std1 * std2 * N)

    Args:
        img1: Reference image
        img2: Image to compare
        mask: Optional binary mask to focus on valid regions

    Returns:
        NCC score: Range [-1, 1], where 1 = perfect match
    """
    if mask is not None:
        img1 = img1[mask]
        img2 = img2[mask]

    # Normalize images (remove mean, scale by std)
    img1_norm = (img1 - img1.mean()) / (img1.std() + 1e-8)
    img2_norm = (img2 - img2.mean()) / (img2.std() + 1e-8)

    # Calculate correlation
    ncc = np.mean(img1_norm * img2_norm)

    return float(ncc)


def calculate_ncc_3d(volume1, volume2, mask=None):
    """
    Calculate NCC for 3D volumes.

    Args:
        volume1: Reference volume (Y, X, Z)
        volume2: Volume to compare (Y, X, Z)
        mask: Optional binary mask (Y, X, Z)

    Returns:
        NCC score for the entire volume
    """
    if mask is not None:
        volume1 = volume1[mask]
        volume2 = volume2[mask]
    else:
        volume1 = volume1.ravel()
        volume2 = volume2.ravel()

    # Remove zero regions (background)
    valid = (volume1 > 0) & (volume2 > 0)
    if valid.sum() < 100:  # Need at least 100 pixels
        return -1.0

    volume1 = volume1[valid]
    volume2 = volume2[valid]

    # Normalize
    v1_norm = (volume1 - volume1.mean()) / (volume1.std() + 1e-8)
    v2_norm = (volume2 - volume2.mean()) / (volume2.std() + 1e-8)

    ncc = np.mean(v1_norm * v2_norm)

    return float(ncc)


# ============================================================================
# ROTATION FUNCTIONS
# ============================================================================

def apply_rotation_z(volume, angle_degrees, axes=(1, 2), reshape=False):
    """
    Apply Z-axis rotation to volume (in-plane XY rotation).

    Z-axis rotation rotates in the XZ plane (en-face view).
    This is the most common correction needed for OCT registration.

    Args:
        volume: 3D numpy array (Y, X, Z)
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)
        axes: Tuple of axes to rotate (default: (1, 2) = X and Z)
        reshape: If True, expand array to fit rotated volume

    Returns:
        Rotated volume with same shape (or expanded if reshape=True)
    """
    volume_rotated = ndimage.rotate(
        volume,
        angle=angle_degrees,
        axes=axes,
        reshape=reshape,
        order=1,  # Bilinear interpolation
        mode='constant',
        cval=0
    )

    return volume_rotated


def create_overlap_mask(volume, percentile=10):
    """
    Create a binary mask for valid (non-zero) regions in volume.

    Useful for focusing NCC calculation on tissue regions only.

    Args:
        volume: 3D volume
        percentile: Threshold percentile for valid pixels

    Returns:
        Binary mask (True = valid tissue)
    """
    threshold = np.percentile(volume[volume > 0], percentile) if (volume > 0).any() else 0
    mask = volume > threshold
    return mask


# ============================================================================
# ROTATION SEARCH FUNCTIONS
# ============================================================================

def find_optimal_rotation_z_coarse(overlap_v0, overlap_v1,
                                     angle_range=30, step=2,
                                     verbose=True):
    """
    Coarse search for optimal Z-axis rotation angle.

    Searches rotation angles from -angle_range to +angle_range with given step.
    Uses NCC as optimization metric.

    Args:
        overlap_v0: Reference volume (Y, X, Z)
        overlap_v1: Volume to rotate (Y, X, Z)
        angle_range: Maximum angle to search (degrees)
        step: Angle step size (degrees)
        verbose: Print progress

    Returns:
        best_angle: Optimal rotation angle (degrees)
        best_ncc: NCC score at optimal angle
        results: List of dicts with all tested angles and their scores
    """
    angles_to_test = np.arange(-angle_range, angle_range + step, step)

    if verbose:
        print(f"\n{'='*70}")
        print(f"COARSE Z-AXIS ROTATION SEARCH")
        print(f"{'='*70}")
        print(f"  Angle range: ±{angle_range}°")
        print(f"  Step size: {step}°")
        print(f"  Total angles: {len(angles_to_test)}")
        print(f"  Overlap region: {overlap_v0.shape}")

    results = []
    best_ncc = -1.0
    best_angle = 0.0

    # Create mask for valid regions
    mask_v0 = create_overlap_mask(overlap_v0)

    for i, angle in enumerate(angles_to_test):
        if verbose and i % 5 == 0:
            print(f"  Progress: {i}/{len(angles_to_test)} angles tested... (current best: {best_angle}° @ NCC={best_ncc:.4f})")

        # Rotate volume 1
        overlap_v1_rotated = apply_rotation_z(overlap_v1, angle, axes=(1, 2))

        # Create mask for rotated volume
        mask_v1 = create_overlap_mask(overlap_v1_rotated)

        # Combined mask (both volumes have tissue)
        mask_combined = mask_v0 & mask_v1

        # Calculate NCC
        ncc = calculate_ncc_3d(overlap_v0, overlap_v1_rotated, mask=mask_combined)

        results.append({
            'angle': float(angle),
            'ncc': ncc,
            'valid_voxels': int(mask_combined.sum())
        })

        # Track best
        if ncc > best_ncc:
            best_ncc = ncc
            best_angle = float(angle)

    if verbose:
        print(f"\n  ✓ Coarse search complete!")
        print(f"    Best angle: {best_angle}°")
        print(f"    Best NCC: {best_ncc:.4f}")

    return best_angle, best_ncc, results


def find_optimal_rotation_z_fine(overlap_v0, overlap_v1,
                                   coarse_angle, angle_range=3, step=0.5,
                                   verbose=True):
    """
    Fine search for optimal Z-axis rotation angle around coarse estimate.

    Refines rotation angle with smaller step size.

    Args:
        overlap_v0: Reference volume (Y, X, Z)
        overlap_v1: Volume to rotate (Y, X, Z)
        coarse_angle: Starting angle from coarse search (degrees)
        angle_range: Search range around coarse_angle (degrees)
        step: Angle step size (degrees)
        verbose: Print progress

    Returns:
        best_angle: Refined rotation angle (degrees)
        best_ncc: NCC score at refined angle
        results: List of dicts with all tested angles and their scores
    """
    angles_to_test = np.arange(coarse_angle - angle_range,
                                coarse_angle + angle_range + step,
                                step)

    if verbose:
        print(f"\n{'='*70}")
        print(f"FINE Z-AXIS ROTATION SEARCH")
        print(f"{'='*70}")
        print(f"  Center angle: {coarse_angle}°")
        print(f"  Search range: ±{angle_range}°")
        print(f"  Step size: {step}°")
        print(f"  Total angles: {len(angles_to_test)}")

    results = []
    best_ncc = -1.0
    best_angle = coarse_angle

    # Create mask for valid regions
    mask_v0 = create_overlap_mask(overlap_v0)

    for i, angle in enumerate(angles_to_test):
        # Rotate volume 1
        overlap_v1_rotated = apply_rotation_z(overlap_v1, angle, axes=(1, 2))

        # Create mask for rotated volume
        mask_v1 = create_overlap_mask(overlap_v1_rotated)

        # Combined mask
        mask_combined = mask_v0 & mask_v1

        # Calculate NCC
        ncc = calculate_ncc_3d(overlap_v0, overlap_v1_rotated, mask=mask_combined)

        results.append({
            'angle': float(angle),
            'ncc': ncc,
            'valid_voxels': int(mask_combined.sum())
        })

        # Track best
        if ncc > best_ncc:
            best_ncc = ncc
            best_angle = float(angle)

    if verbose:
        print(f"\n  ✓ Fine search complete!")
        print(f"    Refined angle: {best_angle}°")
        print(f"    Refined NCC: {best_ncc:.4f}")

    return best_angle, best_ncc, results


def find_optimal_rotation_z(overlap_v0, overlap_v1,
                              coarse_range=30, coarse_step=2,
                              fine_range=3, fine_step=0.5,
                              verbose=True):
    """
    Find optimal Z-axis rotation using coarse-to-fine search.

    Two-stage approach:
      1. Coarse search: ±30° with 2° steps (16 angles)
      2. Fine search: ±3° with 0.5° steps around coarse optimum (13 angles)

    Args:
        overlap_v0: Reference overlap volume (Y, X, Z)
        overlap_v1: Volume to rotate (Y, X, Z)
        coarse_range: Coarse search angle range (degrees)
        coarse_step: Coarse search step size (degrees)
        fine_range: Fine search angle range (degrees)
        fine_step: Fine search step size (degrees)
        verbose: Print progress

    Returns:
        optimal_angle: Best rotation angle (degrees)
        metrics: Dictionary with NCC scores and search results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Z-AXIS ROTATION ALIGNMENT (COARSE-TO-FINE)")
        print(f"{'='*70}")

    # Stage 1: Coarse search
    coarse_angle, coarse_ncc, coarse_results = find_optimal_rotation_z_coarse(
        overlap_v0, overlap_v1,
        angle_range=coarse_range,
        step=coarse_step,
        verbose=verbose
    )

    # Stage 2: Fine search
    fine_angle, fine_ncc, fine_results = find_optimal_rotation_z_fine(
        overlap_v0, overlap_v1,
        coarse_angle=coarse_angle,
        angle_range=fine_range,
        step=fine_step,
        verbose=verbose
    )

    metrics = {
        'optimal_angle': fine_angle,
        'optimal_ncc': fine_ncc,
        'coarse_angle': coarse_angle,
        'coarse_ncc': coarse_ncc,
        'coarse_results': coarse_results,
        'fine_results': fine_results
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"✓ Z-AXIS ROTATION COMPLETE")
        print(f"{'='*70}")
        print(f"  Optimal angle: {fine_angle:.2f}°")
        print(f"  NCC improvement: {coarse_ncc:.4f} → {fine_ncc:.4f}")
        print(f"  Total angles tested: {len(coarse_results) + len(fine_results)}")

    return fine_angle, metrics


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_rotation_search(coarse_results, fine_results, output_path=None):
    """
    Visualize rotation angle search results.

    Creates plot showing NCC vs rotation angle for both coarse and fine searches.

    Args:
        coarse_results: List of dicts from coarse search
        fine_results: List of dicts from fine search
        output_path: Path to save figure (if None, display only)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Coarse search
    coarse_angles = [r['angle'] for r in coarse_results]
    coarse_ncc = [r['ncc'] for r in coarse_results]
    best_coarse_idx = np.argmax(coarse_ncc)

    axes[0].plot(coarse_angles, coarse_ncc, 'b-o', linewidth=2, markersize=6)
    axes[0].plot(coarse_angles[best_coarse_idx], coarse_ncc[best_coarse_idx],
                 'r*', markersize=20, label=f'Best: {coarse_angles[best_coarse_idx]:.1f}°')
    axes[0].set_xlabel('Rotation Angle (degrees)', fontweight='bold')
    axes[0].set_ylabel('NCC Score', fontweight='bold')
    axes[0].set_title('Coarse Search: Z-Axis Rotation', fontweight='bold', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Plot 2: Fine search
    fine_angles = [r['angle'] for r in fine_results]
    fine_ncc = [r['ncc'] for r in fine_results]
    best_fine_idx = np.argmax(fine_ncc)

    axes[1].plot(fine_angles, fine_ncc, 'g-o', linewidth=2, markersize=6)
    axes[1].plot(fine_angles[best_fine_idx], fine_ncc[best_fine_idx],
                 'r*', markersize=20, label=f'Best: {fine_angles[best_fine_idx]:.2f}°')
    axes[1].set_xlabel('Rotation Angle (degrees)', fontweight='bold')
    axes[1].set_ylabel('NCC Score', fontweight='bold')
    axes[1].set_title('Fine Search: Z-Axis Rotation', fontweight='bold', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.suptitle('Z-Axis Rotation Optimization (NCC Metric)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved rotation search plot: {output_path}")

    plt.close()


def visualize_rotation_comparison(overlap_v0, overlap_v1_before, overlap_v1_after,
                                    angle, ncc_before, ncc_after,
                                    output_path=None):
    """
    Visualize before/after rotation alignment.

    Shows en-face MIP and central B-scan comparisons.

    Args:
        overlap_v0: Reference volume
        overlap_v1_before: Volume before rotation
        overlap_v1_after: Volume after rotation
        angle: Applied rotation angle (degrees)
        ncc_before: NCC before rotation
        ncc_after: NCC after rotation
        output_path: Path to save figure
    """
    fig = plt.figure(figsize=(24, 12))

    # Create MIPs (average projection)
    mip_v0 = np.mean(overlap_v0, axis=0)
    mip_v1_before = np.mean(overlap_v1_before, axis=0)
    mip_v1_after = np.mean(overlap_v1_after, axis=0)

    # Get central B-scan
    z_center = overlap_v0.shape[2] // 2
    bscan_v0 = overlap_v0[:, :, z_center]
    bscan_v1_before = overlap_v1_before[:, :, z_center]
    bscan_v1_after = overlap_v1_after[:, :, z_center]

    # Row 1: En-face MIPs - BEFORE
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(mip_v0.T, cmap='Reds', alpha=0.7, origin='lower', aspect='auto')
    ax1.set_title('V0 MIP (Reference)', fontweight='bold')
    ax1.set_xlabel('X'); ax1.set_ylabel('Z')

    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(mip_v1_before.T, cmap='Greens', alpha=0.7, origin='lower', aspect='auto')
    ax2.set_title('V1 MIP (BEFORE rotation)', fontweight='bold')
    ax2.set_xlabel('X')

    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(mip_v0.T, cmap='Reds', alpha=0.5, origin='lower', aspect='auto')
    ax3.imshow(mip_v1_before.T, cmap='Greens', alpha=0.5, origin='lower', aspect='auto')
    ax3.set_title(f'Overlay BEFORE\nNCC={ncc_before:.4f}', fontweight='bold')
    ax3.set_xlabel('X')

    # B-scan before
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(bscan_v0, cmap='Reds', alpha=0.5, aspect='auto')
    ax4.imshow(bscan_v1_before, cmap='Greens', alpha=0.5, aspect='auto')
    ax4.set_title(f'B-scan BEFORE (Z={z_center})', fontweight='bold')
    ax4.set_xlabel('X'); ax4.set_ylabel('Y')

    # Row 2: En-face MIPs - AFTER
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(mip_v0.T, cmap='Reds', alpha=0.7, origin='lower', aspect='auto')
    ax5.set_title('V0 MIP (Reference)', fontweight='bold')
    ax5.set_xlabel('X'); ax5.set_ylabel('Z')

    ax6 = plt.subplot(2, 4, 6)
    ax6.imshow(mip_v1_after.T, cmap='Greens', alpha=0.7, origin='lower', aspect='auto')
    ax6.set_title(f'V1 MIP (AFTER {angle:.2f}° rotation)', fontweight='bold')
    ax6.set_xlabel('X')

    ax7 = plt.subplot(2, 4, 7)
    ax7.imshow(mip_v0.T, cmap='Reds', alpha=0.5, origin='lower', aspect='auto')
    ax7.imshow(mip_v1_after.T, cmap='Greens', alpha=0.5, origin='lower', aspect='auto')
    ax7.set_title(f'Overlay AFTER\nNCC={ncc_after:.4f} ({(ncc_after-ncc_before)*100:+.2f}%)',
                  fontweight='bold')
    ax7.set_xlabel('X')

    # B-scan after
    ax8 = plt.subplot(2, 4, 8)
    ax8.imshow(bscan_v0, cmap='Reds', alpha=0.5, aspect='auto')
    ax8.imshow(bscan_v1_after, cmap='Greens', alpha=0.5, aspect='auto')
    ax8.set_title(f'B-scan AFTER (Z={z_center})', fontweight='bold')
    ax8.set_xlabel('X'); ax8.set_ylabel('Y')

    plt.suptitle(f'Z-Axis Rotation: {angle:.2f}° | NCC: {ncc_before:.4f} → {ncc_after:.4f}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved rotation comparison: {output_path}")

    plt.close()


if __name__ == "__main__":
    print("Rotation Alignment Module")
    print("=" * 70)
    print("Functions:")
    print("  - find_optimal_rotation_z(): Coarse-to-fine rotation search")
    print("  - apply_rotation_z(): Apply rotation to volume")
    print("  - calculate_ncc_3d(): Calculate NCC metric")
    print("  - visualize_rotation_search(): Plot angle search results")
    print("  - visualize_rotation_comparison(): Plot before/after comparison")
