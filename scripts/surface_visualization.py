#!/usr/bin/env python3
"""
Surface Visualization Module for OCT Volume Alignment

This module provides visualization of retinal surfaces after XZ alignment,
showing X-slices (B-scan view) and Y-slices (coronal view).

Visualizes the common surface overlap region after XZ alignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2


def denoise_bscan_for_surface_detection(bscan):
    """
    Apply harsh denoising to B-scan for robust surface detection.

    Uses the same aggressive preprocessing as rotation alignment (Steps 3, 3.1, 3.5).

    Args:
        bscan: 2D B-scan array (Y, X)

    Returns:
        Denoised B-scan (Y, X) as uint8
    """
    # Normalize to 0-255
    img_norm = ((bscan - bscan.min()) / (bscan.max() - bscan.min() + 1e-8) * 255).astype(np.uint8)

    # Step 1: Non-local means denoising (HARSH)
    denoised = cv2.fastNlMeansDenoising(img_norm, h=25, templateWindowSize=7, searchWindowSize=21)

    # Step 2: Bilateral filtering (HARSH)
    denoised = cv2.bilateralFilter(denoised, d=11, sigmaColor=150, sigmaSpace=150)

    # Step 3: Median filter (HARSH)
    denoised = cv2.medianBlur(denoised, 15)

    # Step 4: Threshold (50% of Otsu - preserves tissue layers)
    thresh_val = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_val = int(thresh_val * 0.5)
    denoised[denoised < thresh_val] = 0

    # Step 5: CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    denoised = clahe.apply(denoised)

    return denoised


def load_or_detect_surface(volume, method='peak', sampled_positions=None):
    """
    Detect retinal surface from volume with harsh denoising preprocessing.

    PERFORMANCE OPTIMIZATION: If sampled_positions is provided, only denoises
    those B-scans. Otherwise denoises all B-scans (slow for full volumes).

    Args:
        volume: 3D OCT volume (Y, X, Z)
        method: Detection method ('peak' for maximum intensity)
        sampled_positions: Optional array of Z-indices to process (for Step 4 windowed alignment)
                          If None, processes all Z slices (slower)

    Returns:
        surface: 2D array (X, Z_sampled) with Y positions of surface
                 If sampled_positions provided: Z_sampled = len(sampled_positions)
                 Otherwise: Z_sampled = Z (full volume)
    """
    Y, X, Z = volume.shape

    # Determine which Z slices to process
    if sampled_positions is not None:
        z_indices = sampled_positions
    else:
        z_indices = np.arange(Z)

    surface = np.zeros((X, len(z_indices)))

    if method == 'peak':
        # Process only the specified B-scans with harsh denoising
        for i, z in enumerate(z_indices):
            bscan = volume[:, :, z]  # (Y, X)

            # Apply harsh denoising
            bscan_denoised = denoise_bscan_for_surface_detection(bscan)

            # Find Y position of maximum intensity for each X
            surface[:, i] = np.argmax(bscan_denoised, axis=0)  # (X,)
    else:
        raise ValueError(f"Unknown method: {method}")

    return surface


def visualize_surface_xz_alignment(surface_v0, surface_v1, surface_v1_aligned,
                                   offset_x, offset_z, output_path):
    """
    Visualize surface alignment in XZ plane.

    Shows:
    - Before alignment: surfaces overlaid
    - After alignment: surfaces overlaid
    - X-slice (B-scan view) at center
    - Y-slice (coronal view) at center

    Args:
        surface_v0: Reference surface (X, Z)
        surface_v1: Volume 1 surface before alignment (X, Z)
        surface_v1_aligned: Volume 1 surface after XZ alignment (X, Z)
        offset_x: X translation offset
        offset_z: Z translation offset
        output_path: Where to save visualization
    """
    print(f"\nCreating surface XZ alignment visualization...")
    print(f"  Surface shapes: V0={surface_v0.shape}, V1={surface_v1.shape}")
    print(f"  Offset: X={offset_x}, Z={offset_z}")

    fig = plt.figure(figsize=(20, 12))

    # Calculate overlap region after alignment
    X, Z = surface_v0.shape

    # X overlap
    if offset_x >= 0:
        x0_start, x0_end = offset_x, X
        x1_start, x1_end = 0, X - offset_x
    else:
        x0_start, x0_end = 0, X + offset_x
        x1_start, x1_end = -offset_x, X

    # Z overlap
    if offset_z >= 0:
        z0_start, z0_end = offset_z, Z
        z1_start, z1_end = 0, Z - offset_z
    else:
        z0_start, z0_end = 0, Z + offset_z
        z1_start, z1_end = -offset_z, Z

    # Extract overlap regions
    surf_v0_overlap = surface_v0[x0_start:x0_end, z0_start:z0_end]
    surf_v1_overlap = surface_v1_aligned[x1_start:x1_end, z1_start:z1_end]

    print(f"  Overlap region: X[{x0_start}:{x0_end}] x Z[{z0_start}:{z0_end}]")
    print(f"  Overlap shape: {surf_v0_overlap.shape}")

    # Get center slices
    x_center = surf_v0_overlap.shape[0] // 2
    z_center = surf_v0_overlap.shape[1] // 2

    # Row 1: Full surfaces (before/after alignment)
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(surface_v0.T, aspect='auto', cmap='viridis', vmin=0, vmax=surface_v0.max())
    ax1.set_title('Volume 0: Reference Surface', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (lateral)')
    ax1.set_ylabel('Z (B-scan)')
    plt.colorbar(im1, ax=ax1, label='Y position (depth)')

    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(surface_v1.T, aspect='auto', cmap='viridis', vmin=0, vmax=surface_v1.max())
    ax2.set_title('Volume 1: Surface (Before Alignment)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (lateral)')
    ax2.set_ylabel('Z (B-scan)')
    plt.colorbar(im2, ax=ax2, label='Y position (depth)')

    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(surface_v1_aligned.T, aspect='auto', cmap='viridis', vmin=0, vmax=surface_v1_aligned.max())
    ax3.set_title('Volume 1: Surface (After XZ Alignment)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (lateral)')
    ax3.set_ylabel('Z (B-scan)')
    plt.colorbar(im3, ax=ax3, label='Y position (depth)')

    # Row 2: Overlap region slices
    # X-slice (B-scan view) - vertical slice showing depth profile
    ax4 = plt.subplot(2, 3, 4)
    x_slice_v0 = surf_v0_overlap[x_center, :]  # Shape: (Z,)
    x_slice_v1 = surf_v1_overlap[x_center, :]
    z_coords = np.arange(len(x_slice_v0))
    ax4.plot(z_coords, x_slice_v0, 'r-', linewidth=2, label='Vol 0 (ref)')
    ax4.plot(z_coords, x_slice_v1, 'g-', linewidth=2, label='Vol 1 (aligned)')
    ax4.fill_between(z_coords, x_slice_v0, x_slice_v1, alpha=0.3, color='yellow')
    ax4.set_xlabel('Z (B-scan index)')
    ax4.set_ylabel('Y (depth, pixels)')
    ax4.set_title(f'X-Slice @ X={x_center} (Overlap Region)\nB-scan View', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()  # Depth increases downward

    # Y-slice (coronal view) - horizontal slice
    ax5 = plt.subplot(2, 3, 5)
    y_slice_v0 = surf_v0_overlap[:, z_center]  # Shape: (X,)
    y_slice_v1 = surf_v1_overlap[:, z_center]
    x_coords = np.arange(len(y_slice_v0))
    ax5.plot(x_coords, y_slice_v0, 'r-', linewidth=2, label='Vol 0 (ref)')
    ax5.plot(x_coords, y_slice_v1, 'g-', linewidth=2, label='Vol 1 (aligned)')
    ax5.fill_between(x_coords, y_slice_v0, y_slice_v1, alpha=0.3, color='yellow')
    ax5.set_xlabel('X (lateral position)')
    ax5.set_ylabel('Y (depth, pixels)')
    ax5.set_title(f'Y-Slice @ Z={z_center} (Overlap Region)\nCoronal View', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.invert_yaxis()  # Depth increases downward

    # Difference map
    ax6 = plt.subplot(2, 3, 6)
    diff = np.abs(surf_v0_overlap - surf_v1_overlap)
    im6 = ax6.imshow(diff.T, aspect='auto', cmap='hot', vmin=0, vmax=np.percentile(diff, 95))
    ax6.axvline(x=x_center, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
    ax6.axhline(y=z_center, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
    ax6.set_title(f'Surface Height Difference\nMean: {diff.mean():.2f} px', fontsize=12, fontweight='bold')
    ax6.set_xlabel('X (lateral)')
    ax6.set_ylabel('Z (B-scan)')
    plt.colorbar(im6, ax=ax6, label='|ΔY| (pixels)')

    plt.suptitle(f'Surface Alignment after XZ Translation (ΔX={offset_x}, ΔZ={offset_z})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")

    # Print statistics
    print(f"\n  Surface alignment quality:")
    print(f"    Mean difference: {diff.mean():.2f} ± {diff.std():.2f} pixels")
    print(f"    Max difference: {diff.max():.2f} pixels")
    print(f"    Overlap area: {diff.size:,} pixels")


def visualize_surface_slices_detailed(surface_v0, surface_v1_aligned,
                                      offset_x, offset_z, output_path,
                                      num_slices=5):
    """
    Create detailed visualization with multiple X and Y slices.

    Shows several slices across the overlap region to give comprehensive view.

    Args:
        surface_v0: Reference surface (X, Z)
        surface_v1_aligned: Aligned surface (X, Z)
        offset_x: X translation
        offset_z: Z translation
        output_path: Where to save
        num_slices: Number of slices to show for each direction
    """
    print(f"\nCreating detailed surface slices visualization...")

    # Calculate overlap region
    X, Z = surface_v0.shape

    if offset_x >= 0:
        x0_start, x0_end = offset_x, X
        x1_start, x1_end = 0, X - offset_x
    else:
        x0_start, x0_end = 0, X + offset_x
        x1_start, x1_end = -offset_x, X

    if offset_z >= 0:
        z0_start, z0_end = offset_z, Z
        z1_start, z1_end = 0, Z - offset_z
    else:
        z0_start, z0_end = 0, Z + offset_z
        z1_start, z1_end = -offset_z, Z

    # Extract overlap
    surf_v0_overlap = surface_v0[x0_start:x0_end, z0_start:z0_end]
    surf_v1_overlap = surface_v1_aligned[x1_start:x1_end, z1_start:z1_end]

    overlap_x, overlap_z = surf_v0_overlap.shape

    # Create figure with multiple slices
    fig = plt.figure(figsize=(20, 12))

    # X-slices (B-scan view) - top row
    x_positions = np.linspace(0, overlap_x-1, num_slices, dtype=int)
    for i, x_pos in enumerate(x_positions):
        ax = plt.subplot(2, num_slices, i+1)

        x_slice_v0 = surf_v0_overlap[x_pos, :]
        x_slice_v1 = surf_v1_overlap[x_pos, :]
        z_coords = np.arange(len(x_slice_v0))

        ax.plot(z_coords, x_slice_v0, 'r-', linewidth=2, label='Vol 0' if i == 0 else '')
        ax.plot(z_coords, x_slice_v1, 'g-', linewidth=2, label='Vol 1' if i == 0 else '')
        ax.fill_between(z_coords, x_slice_v0, x_slice_v1, alpha=0.3, color='yellow')

        ax.set_xlabel('Z (B-scan)' if i == num_slices//2 else '')
        ax.set_ylabel('Y (depth)' if i == 0 else '')
        ax.set_title(f'X-Slice @ X={x_pos}\n({100*x_pos/overlap_x:.0f}%)', fontsize=10, fontweight='bold')
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

    # Y-slices (coronal view) - bottom row
    z_positions = np.linspace(0, overlap_z-1, num_slices, dtype=int)
    for i, z_pos in enumerate(z_positions):
        ax = plt.subplot(2, num_slices, num_slices + i + 1)

        y_slice_v0 = surf_v0_overlap[:, z_pos]
        y_slice_v1 = surf_v1_overlap[:, z_pos]
        x_coords = np.arange(len(y_slice_v0))

        ax.plot(x_coords, y_slice_v0, 'r-', linewidth=2, label='Vol 0' if i == 0 else '')
        ax.plot(x_coords, y_slice_v1, 'g-', linewidth=2, label='Vol 1' if i == 0 else '')
        ax.fill_between(x_coords, y_slice_v0, y_slice_v1, alpha=0.3, color='yellow')

        ax.set_xlabel('X (lateral)' if i == num_slices//2 else '')
        ax.set_ylabel('Y (depth)' if i == 0 else '')
        ax.set_title(f'Y-Slice @ Z={z_pos}\n({100*z_pos/overlap_z:.0f}%)', fontsize=10, fontweight='bold')
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

    plt.suptitle(f'Multiple Surface Slices - Overlap Region\n'
                 f'X: {overlap_x} pixels, Z: {overlap_z} B-scans',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def apply_xz_alignment_to_surface(surface, offset_x, offset_z):
    """
    Apply XZ translation to surface.

    Args:
        surface: 2D surface array (X, Z)
        offset_x: X translation
        offset_z: Z translation

    Returns:
        aligned_surface: Translated surface
    """
    # Apply 2D shift (no Y shift for surface, only XZ translation)
    aligned_surface = ndimage.shift(
        surface, shift=(offset_x, offset_z),
        order=1, mode='constant', cval=0
    )
    return aligned_surface
