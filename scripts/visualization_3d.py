#!/usr/bin/env python3
"""
3D Visualization Module for OCT Volume Alignment

This module provides 3D visualization functions for OCT volumes,
including multi-angle projections and merged volume visualization.

Based on notebook 06_visualize_results.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import shift as nd_shift


def create_expanded_merged_volume(volume_0, volume_1, transform_3d):
    """
    Create expanded volume that fits both volumes without data loss.

    EXACT implementation from notebook 06.

    Args:
        volume_0: Reference volume (h, w, d)
        volume_1: Volume to merge (h, w, d)
        transform_3d: {'dy': Y_offset, 'dx': X_offset, 'dz': Z_offset}

    Returns:
        expanded_volume: Merged volume with NO data loss
        metadata: Information about the merge
    """
    h, w, d = volume_0.shape
    dy, dx, dz = int(transform_3d['dy']), int(transform_3d['dx']), int(transform_3d['dz'])

    print(f"Original volume shape: {volume_0.shape}")
    print(f"Translation: dy={dy}, dx={dx}, dz={dz}")

    # Calculate expanded dimensions to fit both volumes
    new_h = h + abs(dy)
    new_w = w + abs(dx)
    new_d = d + abs(dz)

    print(f"Expanded volume shape: ({new_h}, {new_w}, {new_d})")
    print(f"  Height increase: {abs(dy)} pixels")
    print(f"  Width increase: {abs(dx)} pixels")
    print(f"  Depth increase: {abs(dz)} B-scans")

    # Create expanded canvas
    expanded = np.zeros((new_h, new_w, new_d), dtype=volume_0.dtype)

    # Determine placement coordinates
    y0_start = max(0, -dy)
    x0_start = max(0, -dx)
    z0_start = max(0, -dz)

    y1_start = max(0, dy)
    x1_start = max(0, dx)
    z1_start = max(0, dz)

    print(f"\nPlacement:")
    print(f"  Volume 0: y={y0_start}:{y0_start+h}, x={x0_start}:{x0_start+w}, z={z0_start}:{z0_start+d}")
    print(f"  Volume 1: y={y1_start}:{y1_start+h}, x={x1_start}:{x1_start+w}, z={z1_start}:{z1_start+d}")

    # Place volume 0
    expanded[y0_start:y0_start+h, x0_start:x0_start+w, z0_start:z0_start+d] = volume_0

    # Calculate overlap region
    y_overlap_start = max(y0_start, y1_start)
    y_overlap_end = min(y0_start+h, y1_start+h)
    x_overlap_start = max(x0_start, x1_start)
    x_overlap_end = min(x0_start+w, x1_start+w)
    z_overlap_start = max(z0_start, z1_start)
    z_overlap_end = min(z0_start+d, z1_start+d)

    has_overlap = (y_overlap_end > y_overlap_start and
                   x_overlap_end > x_overlap_start and
                   z_overlap_end > z_overlap_start)

    if has_overlap:
        print(f"\nOverlap region:")
        print(f"  y: {y_overlap_start}:{y_overlap_end} ({y_overlap_end-y_overlap_start} pixels)")
        print(f"  x: {x_overlap_start}:{x_overlap_end} ({x_overlap_end-x_overlap_start} pixels)")
        print(f"  z: {z_overlap_start}:{z_overlap_end} ({z_overlap_end-z_overlap_start} B-scans)")

        # Extract overlap regions from both volumes
        v0_y = slice(y_overlap_start-y0_start, y_overlap_end-y0_start)
        v0_x = slice(x_overlap_start-x0_start, x_overlap_end-x0_start)
        v0_z = slice(z_overlap_start-z0_start, z_overlap_end-z0_start)

        v1_y = slice(y_overlap_start-y1_start, y_overlap_end-y1_start)
        v1_x = slice(x_overlap_start-x1_start, x_overlap_end-x1_start)
        v1_z = slice(z_overlap_start-z1_start, z_overlap_end-z1_start)

        overlap_v0 = volume_0[v0_y, v0_x, v0_z]
        overlap_v1 = volume_1[v1_y, v1_x, v1_z]

        # Blend 50/50
        blended_overlap = 0.5 * overlap_v0 + 0.5 * overlap_v1

        # Place blended overlap in expanded volume
        expanded[y_overlap_start:y_overlap_end,
                x_overlap_start:x_overlap_end,
                z_overlap_start:z_overlap_end] = blended_overlap

        overlap_voxels = blended_overlap.size
    else:
        print("\n⚠️  No overlap region (volumes completely separated)")
        overlap_voxels = 0

    # Place non-overlap regions of volume 1
    mask_v1 = np.ones((new_h, new_w, new_d), dtype=bool)
    mask_v1[y0_start:y0_start+h, x0_start:x0_start+w, z0_start:z0_start+d] = False

    v1_placed = np.zeros_like(expanded)
    v1_placed[y1_start:y1_start+h, x1_start:x1_start+w, z1_start:z1_start+d] = volume_1

    if has_overlap:
        v1_mask = (v1_placed > 0) & (expanded == 0)
        expanded[v1_mask] = v1_placed[v1_mask]
    else:
        expanded[y1_start:y1_start+h, x1_start:x1_start+w, z1_start:z1_start+d] = volume_1

    total_voxels = (expanded > 0).sum()

    metadata = {
        'original_shape': (h, w, d),
        'expanded_shape': (new_h, new_w, new_d),
        'translation': (dy, dx, dz),
        'expansion': (abs(dy), abs(dx), abs(dz)),
        'has_overlap': has_overlap,
        'overlap_voxels': overlap_voxels,
        'total_voxels': total_voxels,
        'data_loss': 0
    }

    return expanded, metadata


def visualize_3d_multiangle(volume, title, output_path, subsample=4, percentile=70):
    """
    Create 3D volume projections from multiple angles.

    EXACT implementation from notebook 06.

    Shows 4 views: X-axis (side), Y-axis (front), Z-axis (top), 45° angle
    """
    print(f"\nCreating 3D multi-angle visualization: {title}")

    fig = plt.figure(figsize=(20, 16))

    # Subsample volume for visualization
    vol_sub = volume[::subsample, ::subsample, ::subsample]
    print(f"  Subsampled to: {vol_sub.shape}")

    # Threshold to show only high-intensity voxels
    threshold = np.percentile(vol_sub[vol_sub > 0], percentile)
    print(f"  Intensity threshold: {threshold:.1f}")

    # Get coordinates of voxels above threshold
    z, y, x = np.where(vol_sub > threshold)
    colors = vol_sub[z, y, x]
    colors_norm = (colors - colors.min()) / (colors.max() - colors.min())

    print(f"  Rendering {len(x):,} voxels...")

    # View 1: X-axis (sagittal/side view)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(y, x, z, c=colors_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax1.set_xlabel('Height (Y)')
    ax1.set_ylabel('Width (X)')
    ax1.set_zlabel('Depth (Z)')
    ax1.set_title('X-Axis View (Side/Sagittal)', fontsize=12, fontweight='bold')
    ax1.view_init(elev=0, azim=0)

    # View 2: Y-axis (coronal/front view)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.scatter(y, x, z, c=colors_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax2.set_xlabel('Height (Y)')
    ax2.set_ylabel('Width (X)')
    ax2.set_zlabel('Depth (Z)')
    ax2.set_title('Y-Axis View (Front/Coronal)', fontsize=12, fontweight='bold')
    ax2.view_init(elev=0, azim=90)

    # View 3: Z-axis (axial/top view)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.scatter(y, x, z, c=colors_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax3.set_xlabel('Height (Y)')
    ax3.set_ylabel('Width (X)')
    ax3.set_zlabel('Depth (Z)')
    ax3.set_title('Z-Axis View (Top/Axial)', fontsize=12, fontweight='bold')
    ax3.view_init(elev=90, azim=-90)

    # View 4: 45° angle
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.scatter(y, x, z, c=colors_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax4.set_xlabel('Height (Y)')
    ax4.set_ylabel('Width (X)')
    ax4.set_zlabel('Depth (Z)')
    ax4.set_title('45° Angle View (Isometric)', fontsize=12, fontweight='bold')
    ax4.view_init(elev=30, azim=45)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def visualize_3d_comparison(volume_0, volume_1_aligned, merged_volume, transform, output_path, subsample=4, percentile=70):
    """
    Create side-by-side 3D volume comparison.

    EXACT implementation from notebook 06.

    Shows: Volume 0, Volume 1 (aligned), Merged volume
    From 4 angles: X-axis, Y-axis, Z-axis, 45°
    """
    print("\nCreating side-by-side 3D comparison...")

    fig = plt.figure(figsize=(20, 24))

    # Subsample all volumes
    vol0_sub = volume_0[::subsample, ::subsample, ::subsample]
    vol1_sub = volume_1_aligned[::subsample, ::subsample, ::subsample]
    vol_merged_sub = merged_volume[::subsample, ::subsample, ::subsample]

    # Get thresholds
    threshold0 = np.percentile(vol0_sub[vol0_sub > 0], percentile)
    threshold1 = np.percentile(vol1_sub[vol1_sub > 0], percentile)
    threshold_merged = np.percentile(vol_merged_sub[vol_merged_sub > 0], percentile)

    print(f"  Thresholds: Vol0={threshold0:.1f}, Vol1={threshold1:.1f}, Merged={threshold_merged:.1f}")

    # Get coordinates for each volume
    z0, y0, x0 = np.where(vol0_sub > threshold0)
    colors0 = vol0_sub[z0, y0, x0]
    colors0_norm = (colors0 - colors0.min()) / (colors0.max() - colors0.min())

    z1, y1, x1 = np.where(vol1_sub > threshold1)
    colors1 = vol1_sub[z1, y1, x1]
    colors1_norm = (colors1 - colors1.min()) / (colors1.max() - colors1.min())

    zm, ym, xm = np.where(vol_merged_sub > threshold_merged)
    colors_merged = vol_merged_sub[zm, ym, xm]
    colors_merged_norm = (colors_merged - colors_merged.min()) / (colors_merged.max() - colors_merged.min())

    print(f"  Voxel counts: Vol0={len(x0):,}, Vol1={len(x1):,}, Merged={len(xm):,}")

    # Row 1: X-axis view (side)
    ax1 = fig.add_subplot(4, 3, 1, projection='3d')
    ax1.scatter(y0, x0, z0, c=colors0_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax1.set_title('Volume 0 (Reference)\nX-axis view', fontweight='bold')
    ax1.view_init(elev=0, azim=0)
    ax1.set_xlabel('Y'); ax1.set_ylabel('X'); ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(4, 3, 2, projection='3d')
    ax2.scatter(y1, x1, z1, c=colors1_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax2.set_title(f'Volume 1 (Aligned, Y{transform["dy"]:+.0f}px)\nX-axis view', fontweight='bold')
    ax2.view_init(elev=0, azim=0)
    ax2.set_xlabel('Y'); ax2.set_ylabel('X'); ax2.set_zlabel('Z')

    ax3 = fig.add_subplot(4, 3, 3, projection='3d')
    ax3.scatter(ym, xm, zm, c=colors_merged_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax3.set_title('Merged Volume\nX-axis view', fontweight='bold')
    ax3.view_init(elev=0, azim=0)
    ax3.set_xlabel('Y'); ax3.set_ylabel('X'); ax3.set_zlabel('Z')

    # Row 2: Y-axis view (front)
    ax4 = fig.add_subplot(4, 3, 4, projection='3d')
    ax4.scatter(y0, x0, z0, c=colors0_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax4.set_title('Y-axis view', fontweight='bold')
    ax4.view_init(elev=0, azim=90)
    ax4.set_xlabel('Y'); ax4.set_ylabel('X'); ax4.set_zlabel('Z')

    ax5 = fig.add_subplot(4, 3, 5, projection='3d')
    ax5.scatter(y1, x1, z1, c=colors1_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax5.set_title('Y-axis view', fontweight='bold')
    ax5.view_init(elev=0, azim=90)
    ax5.set_xlabel('Y'); ax5.set_ylabel('X'); ax5.set_zlabel('Z')

    ax6 = fig.add_subplot(4, 3, 6, projection='3d')
    ax6.scatter(ym, xm, zm, c=colors_merged_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax6.set_title('Y-axis view', fontweight='bold')
    ax6.view_init(elev=0, azim=90)
    ax6.set_xlabel('Y'); ax6.set_ylabel('X'); ax6.set_zlabel('Z')

    # Row 3: Z-axis view (top)
    ax7 = fig.add_subplot(4, 3, 7, projection='3d')
    ax7.scatter(y0, x0, z0, c=colors0_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax7.set_title('Z-axis view', fontweight='bold')
    ax7.view_init(elev=90, azim=-90)
    ax7.set_xlabel('Y'); ax7.set_ylabel('X'); ax7.set_zlabel('Z')

    ax8 = fig.add_subplot(4, 3, 8, projection='3d')
    ax8.scatter(y1, x1, z1, c=colors1_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax8.set_title('Z-axis view', fontweight='bold')
    ax8.view_init(elev=90, azim=-90)
    ax8.set_xlabel('Y'); ax8.set_ylabel('X'); ax8.set_zlabel('Z')

    ax9 = fig.add_subplot(4, 3, 9, projection='3d')
    ax9.scatter(ym, xm, zm, c=colors_merged_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax9.set_title('Z-axis view', fontweight='bold')
    ax9.view_init(elev=90, azim=-90)
    ax9.set_xlabel('Y'); ax9.set_ylabel('X'); ax9.set_zlabel('Z')

    # Row 4: 45° angle view
    ax10 = fig.add_subplot(4, 3, 10, projection='3d')
    ax10.scatter(y0, x0, z0, c=colors0_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax10.set_title('45° angle view', fontweight='bold')
    ax10.view_init(elev=30, azim=45)
    ax10.set_xlabel('Y'); ax10.set_ylabel('X'); ax10.set_zlabel('Z')

    ax11 = fig.add_subplot(4, 3, 11, projection='3d')
    ax11.scatter(y1, x1, z1, c=colors1_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax11.set_title('45° angle view', fontweight='bold')
    ax11.view_init(elev=30, azim=45)
    ax11.set_xlabel('Y'); ax11.set_ylabel('X'); ax11.set_zlabel('Z')

    ax12 = fig.add_subplot(4, 3, 12, projection='3d')
    ax12.scatter(ym, xm, zm, c=colors_merged_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax12.set_title('45° angle view', fontweight='bold')
    ax12.view_init(elev=30, azim=45)
    ax12.set_xlabel('Y'); ax12.set_ylabel('X'); ax12.set_zlabel('Z')

    plt.suptitle('Side-by-Side 3D Volume Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
