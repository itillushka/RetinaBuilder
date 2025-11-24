"""
Helper functions for extracting, averaging, and manipulating 2D B-scans.

This module provides utilities for the averaged B-scan alignment pipeline,
which is a computationally lighter alternative to full volume alignment.
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def extract_averaged_central_bscan(volume, n_bscans=30):
    """
    Extract and average central B-scans from a volume.

    Args:
        volume: OCT volume (Y, X, Z) where Z is the B-scan index
        n_bscans: Number of central B-scans to average (default: 30)

    Returns:
        averaged_bscan: 2D averaged B-scan (Y, X)
        z_range: Tuple (z_start, z_end) of extracted range for reference
    """
    Y, X, Z = volume.shape

    # Calculate central range
    z_center = Z // 2
    z_start = max(0, z_center - n_bscans // 2)
    z_end = min(Z, z_start + n_bscans)

    # Adjust if we're at the boundaries
    if z_end - z_start < n_bscans:
        if z_start == 0:
            z_end = min(Z, n_bscans)
        else:
            z_start = max(0, Z - n_bscans)

    # Extract and average
    central_bscans = volume[:, :, z_start:z_end]  # (Y, X, n)
    averaged_bscan = np.mean(central_bscans, axis=2)  # (Y, X)

    print(f"  Extracted B-scans [{z_start}:{z_end}] ({z_end - z_start} scans)")
    print(f"  Averaged shape: {averaged_bscan.shape}")

    return averaged_bscan, (z_start, z_end)


def shift_bscan_2d(bscan, dx, dy):
    """
    Apply 2D shift to a single B-scan using OpenCV.

    Args:
        bscan: 2D B-scan (Y, X)
        dx: X-axis shift in pixels (lateral, positive = right)
        dy: Y-axis shift in pixels (depth, positive = down)

    Returns:
        shifted_bscan: Shifted B-scan with same shape (Y, X)
    """
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return bscan.copy()

    H, W = bscan.shape

    # Create affine transformation matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    # Apply shift
    shifted = cv2.warpAffine(
        bscan.astype(np.float32),
        M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return shifted


def rotate_bscan_2d(bscan, angle_degrees):
    """
    Apply rotation to a single B-scan using scipy.ndimage.

    Args:
        bscan: 2D B-scan (Y, X)
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)

    Returns:
        rotated_bscan: Rotated B-scan with same shape (Y, X)
    """
    from scipy import ndimage

    if abs(angle_degrees) < 0.01:
        return bscan.copy()

    # Rotate around center
    rotated = ndimage.rotate(
        bscan,
        angle=angle_degrees,
        axes=(0, 1),  # Rotate in YX plane
        reshape=False,  # Keep original shape
        order=1,  # Bilinear interpolation
        mode='constant',
        cval=0
    )

    return rotated


def save_averaged_bscan(bscan, output_path, normalize=True):
    """
    Save averaged B-scan as PNG image.

    Args:
        bscan: 2D B-scan (Y, X)
        output_path: Path to save PNG file
        normalize: Whether to normalize to 0-255 range (default: True)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        # Normalize to 0-255
        bscan_min = bscan.min()
        bscan_max = bscan.max()
        if bscan_max > bscan_min:
            bscan_norm = ((bscan - bscan_min) / (bscan_max - bscan_min) * 255).astype(np.uint8)
        else:
            bscan_norm = np.zeros_like(bscan, dtype=np.uint8)
    else:
        bscan_norm = bscan.astype(np.uint8)

    cv2.imwrite(str(output_path), bscan_norm)
    print(f"  Saved: {output_path}")


def save_averaged_bscan_npy(bscan, output_path):
    """
    Save averaged B-scan as numpy array for further processing.

    Args:
        bscan: 2D B-scan (Y, X)
        output_path: Path to save .npy file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, bscan)
    print(f"  Saved: {output_path}")


def visualize_alignment_step(bscan_ref, bscan_before, bscan_after,
                             step_name, output_path,
                             transform_description="",
                             show_difference=True):
    """
    Visualize before/after comparison for an alignment step.

    Args:
        bscan_ref: Reference B-scan (Y, X) - unchanging
        bscan_before: B-scan before transformation (Y, X)
        bscan_after: B-scan after transformation (Y, X)
        step_name: Name of the alignment step (e.g., "X-Shift", "Y-Alignment")
        output_path: Path to save visualization
        transform_description: Description of applied transformation
        show_difference: Whether to show difference images
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if show_difference:
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(20, 7))
        gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Normalize for display
    def normalize_for_display(img):
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img

    ref_norm = normalize_for_display(bscan_ref)
    before_norm = normalize_for_display(bscan_before)
    after_norm = normalize_for_display(bscan_after)

    # Row 1: B-scans
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ref_norm, cmap='gray', aspect='auto')
    ax1.set_title('Reference (V1)', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(before_norm, cmap='gray', aspect='auto')
    ax2.set_title(f'Before {step_name}', fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(after_norm, cmap='gray', aspect='auto')
    ax3.set_title(f'After {step_name}\n{transform_description}',
                  fontsize=14, fontweight='bold')
    ax3.axis('off')

    if show_difference:
        # Row 2: Difference images
        diff_before = np.abs(ref_norm - before_norm)
        diff_after = np.abs(ref_norm - after_norm)

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(diff_before, cmap='hot', aspect='auto', vmin=0, vmax=1)
        ax5.set_title('Difference Before', fontsize=12)
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(diff_after, cmap='hot', aspect='auto', vmin=0, vmax=1)
        ax6.set_title('Difference After', fontsize=12)
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

        # Calculate improvement
        error_before = np.mean(diff_before)
        error_after = np.mean(diff_after)
        improvement = ((error_before - error_after) / error_before) * 100

        fig.text(0.5, 0.02,
                f'Mean Absolute Difference: Before={error_before:.4f}, After={error_after:.4f} '
                f'(Improvement: {improvement:+.1f}%)',
                ha='center', fontsize=12, fontweight='bold')

    plt.suptitle(f'{step_name} - Alignment Visualization',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Visualization saved: {output_path}")


def visualize_surface_comparison(bscan_ref, bscan_aligned,
                                  surface_ref, surface_aligned,
                                  step_name, output_path,
                                  transform_description=""):
    """
    Visualize B-scans with detected surface contours overlaid.

    Args:
        bscan_ref: Reference B-scan (Y, X)
        bscan_aligned: Aligned B-scan (Y, X)
        surface_ref: Reference surface Y-positions (X,)
        surface_aligned: Aligned surface Y-positions (X,)
        step_name: Name of the alignment step
        output_path: Path to save visualization
        transform_description: Description of applied transformation
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Normalize for display
    def normalize_for_display(img):
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img

    ref_norm = normalize_for_display(bscan_ref)
    aligned_norm = normalize_for_display(bscan_aligned)

    X = bscan_ref.shape[1]
    x_coords = np.arange(X)

    # Plot 1: Reference with surface
    axes[0].imshow(ref_norm, cmap='gray', aspect='auto')
    axes[0].plot(x_coords, surface_ref, 'r-', linewidth=2, label='Surface')
    axes[0].set_title('Reference (V1) + Surface', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].axis('off')

    # Plot 2: Aligned with surface
    axes[1].imshow(aligned_norm, cmap='gray', aspect='auto')
    axes[1].plot(x_coords, surface_aligned, 'g-', linewidth=2, label='Surface')
    axes[1].set_title(f'After {step_name} + Surface\n{transform_description}',
                      fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].axis('off')

    # Plot 3: Surface difference
    valid_mask = ~np.isnan(surface_ref) & ~np.isnan(surface_aligned)
    if np.any(valid_mask):
        surface_diff = surface_ref - surface_aligned
        axes[2].plot(x_coords[valid_mask], surface_diff[valid_mask], 'b-', linewidth=1)
        axes[2].axhline(y=0, color='r', linestyle='--', linewidth=1)
        axes[2].set_title('Surface Difference (Ref - Aligned)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('X Position (pixels)', fontsize=12)
        axes[2].set_ylabel('Y Difference (pixels)', fontsize=12)
        axes[2].grid(True, alpha=0.3)

        # Add statistics
        mean_diff = np.mean(surface_diff[valid_mask])
        std_diff = np.std(surface_diff[valid_mask])
        axes[2].text(0.02, 0.98,
                    f'Mean: {mean_diff:.2f} px\nStd: {std_diff:.2f} px',
                    transform=axes[2].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
    else:
        axes[2].text(0.5, 0.5, 'No valid surface data',
                    ha='center', va='center', fontsize=14)
        axes[2].axis('off')

    plt.suptitle(f'{step_name} - Surface Alignment Visualization',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Surface visualization saved: {output_path}")


def crop_to_overlap_region(bscan_ref, bscan_mov, x_offset_total):
    """
    Crop two B-scans to their overlap region after X-shift.

    Args:
        bscan_ref: Reference B-scan (Y, X)
        bscan_mov: Moving B-scan (Y, X) - already shifted
        x_offset_total: Total X offset applied to moving B-scan

    Returns:
        bscan_ref_cropped: Cropped reference B-scan
        bscan_mov_cropped: Cropped moving B-scan
        crop_info: Dictionary with crop parameters
    """
    Y, X_ref = bscan_ref.shape
    Y, X_mov = bscan_mov.shape

    if x_offset_total >= 0:
        # Moving B-scan shifted right
        # Overlap: ref[0:X-offset] overlaps with mov[offset:X]
        # But mov is already shifted, so we crop from the beginning
        x_start_ref = 0
        x_end_ref = X_ref - x_offset_total
        x_start_mov = 0
        x_end_mov = X_mov - x_offset_total
    else:
        # Moving B-scan shifted left
        x_start_ref = abs(x_offset_total)
        x_end_ref = X_ref
        x_start_mov = abs(x_offset_total)
        x_end_mov = X_mov

    # Ensure positive widths
    x_end_ref = max(x_start_ref + 1, x_end_ref)
    x_end_mov = max(x_start_mov + 1, x_end_mov)

    bscan_ref_cropped = bscan_ref[:, x_start_ref:x_end_ref]
    bscan_mov_cropped = bscan_mov[:, x_start_mov:x_end_mov]

    # Make sure they have the same width
    min_width = min(bscan_ref_cropped.shape[1], bscan_mov_cropped.shape[1])
    bscan_ref_cropped = bscan_ref_cropped[:, :min_width]
    bscan_mov_cropped = bscan_mov_cropped[:, :min_width]

    crop_info = {
        'ref_crop': (x_start_ref, x_end_ref),
        'mov_crop': (x_start_mov, x_end_mov),
        'overlap_width': min_width,
        'x_offset_total': x_offset_total
    }

    return bscan_ref_cropped, bscan_mov_cropped, crop_info


def create_panorama_from_aligned_bscans(bscan_v1, bscan_v2, bscan_v3,
                                        x_offset_v2, x_offset_v3,
                                        y_shift_v2, y_shift_v3):
    """
    Create a panoramic B-scan by stitching 3 aligned B-scans together.

    Args:
        bscan_v1: Aligned B-scan 1 (reference) (Y, X)
        bscan_v2: Aligned B-scan 2 (Y, X)
        bscan_v3: Aligned B-scan 3 (Y, X)
        x_offset_v2: Total X offset for V2 relative to V1
        x_offset_v3: Total X offset for V3 relative to V2
        y_shift_v2: Y shift for V2
        y_shift_v3: Y shift for V3

    Returns:
        panorama: Stitched panoramic B-scan
        positions: Dictionary with position info for each B-scan
    """
    # Calculate absolute positions
    # V1 is at position 0
    # V2 is offset from V1 by x_offset_v2
    # V3 is offset from V2 by x_offset_v3

    # Convert to absolute X positions
    x_pos_v1 = 0
    x_pos_v2 = x_offset_v2
    x_pos_v3 = x_offset_v2 + x_offset_v3

    # Find the bounding box for the panorama
    x_min = min(x_pos_v1, x_pos_v2, x_pos_v3)
    x_max = max(x_pos_v1 + bscan_v1.shape[1],
                x_pos_v2 + bscan_v2.shape[1],
                x_pos_v3 + bscan_v3.shape[1])

    # Adjust positions to be non-negative
    x_pos_v1 -= x_min
    x_pos_v2 -= x_min
    x_pos_v3 -= x_min

    # Calculate Y positions (all start at 0, but may have shifts)
    y_pos_v1 = 0
    y_pos_v2 = int(y_shift_v2)
    y_pos_v3 = int(y_shift_v3)

    # Find Y bounding box
    y_min = min(y_pos_v1, y_pos_v2, y_pos_v3)
    y_max = max(y_pos_v1 + bscan_v1.shape[0],
                y_pos_v2 + bscan_v2.shape[0],
                y_pos_v3 + bscan_v3.shape[0])

    # Adjust Y positions
    y_pos_v1 -= y_min
    y_pos_v2 -= y_min
    y_pos_v3 -= y_min

    # Create panorama canvas
    panorama_width = int(x_max - x_min)
    panorama_height = int(y_max - y_min)
    panorama = np.zeros((panorama_height, panorama_width), dtype=np.float32)
    weight_map = np.zeros((panorama_height, panorama_width), dtype=np.float32)

    # Helper function to place B-scan with alpha blending in overlap regions
    def place_bscan(panorama, weight_map, bscan, x_pos, y_pos):
        y_start = y_pos
        y_end = y_pos + bscan.shape[0]
        x_start = x_pos
        x_end = x_pos + bscan.shape[1]

        # Add to panorama with weighting
        panorama[y_start:y_end, x_start:x_end] += bscan
        weight_map[y_start:y_end, x_start:x_end] += 1.0

    # Place all three B-scans
    place_bscan(panorama, weight_map, bscan_v1, x_pos_v1, y_pos_v1)
    place_bscan(panorama, weight_map, bscan_v2, x_pos_v2, y_pos_v2)
    place_bscan(panorama, weight_map, bscan_v3, x_pos_v3, y_pos_v3)

    # Average overlapping regions
    mask = weight_map > 0
    panorama[mask] /= weight_map[mask]

    positions = {
        'v1': {'x': x_pos_v1, 'y': y_pos_v1},
        'v2': {'x': x_pos_v2, 'y': y_pos_v2},
        'v3': {'x': x_pos_v3, 'y': y_pos_v3},
        'width': panorama_width,
        'height': panorama_height
    }

    return panorama, positions


def visualize_three_averaged_bscans(bscan_v1, bscan_v2, bscan_v3, output_path, title=""):
    """
    Visualize all three averaged B-scans side by side.

    Args:
        bscan_v1: Volume 1 averaged B-scan (Y, X)
        bscan_v2: Volume 2 averaged B-scan (Y, X)
        bscan_v3: Volume 3 averaged B-scan (Y, X)
        output_path: Path to save visualization
        title: Optional title for the figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Normalize for display
    def normalize_for_display(img):
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img

    v1_norm = normalize_for_display(bscan_v1)
    v2_norm = normalize_for_display(bscan_v2)
    v3_norm = normalize_for_display(bscan_v3)

    axes[0].imshow(v1_norm, cmap='gray', aspect='auto')
    axes[0].set_title('Volume 1 (Reference)', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(v2_norm, cmap='gray', aspect='auto')
    axes[1].set_title('Volume 2', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(v3_norm, cmap='gray', aspect='auto')
    axes[2].set_title('Volume 3', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    if title:
        plt.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Three B-scans visualization saved: {output_path}")
