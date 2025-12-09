#!/usr/bin/env python3
"""
3D Visualization Module for OCT Volume Alignment

This module provides 3D visualization functions for OCT volumes,
including multi-angle projections and merged volume visualization.

Based on notebook 06_visualize_results.ipynb
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for parallel rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import shift as nd_shift
from multiprocessing import Pool, cpu_count
from PIL import Image
import io


def create_expanded_merged_volume(volume_0, volume_1, transform_3d):
    """
    Create expanded volume that fits both volumes without data loss.

    Handles volumes of different sizes (e.g., after zero-cropping).

    Args:
        volume_0: Reference volume (h0, w0, d0)
        volume_1: Volume to merge (h1, w1, d1) - may differ from volume_0 after cropping
        transform_3d: {'dy': Y_offset, 'dx': X_offset, 'dz': Z_offset}

    Returns:
        expanded_volume: Merged volume with NO data loss
        metadata: Information about the merge
    """
    h0, w0, d0 = volume_0.shape
    h1, w1, d1 = volume_1.shape
    dy, dx, dz = int(transform_3d['dy']), int(transform_3d['dx']), int(transform_3d['dz'])

    print(f"Volume 0 shape: {volume_0.shape}")
    print(f"Volume 1 shape: {volume_1.shape}")
    print(f"Translation: dy={dy}, dx={dx}, dz={dz}")

    # Calculate expanded dimensions to fit both volumes
    # Must account for potentially different sizes and offsets
    if dy >= 0:
        new_h = max(h0, h1 + dy)
    else:
        new_h = max(h0 - dy, h1)

    if dx >= 0:
        new_w = max(w0, w1 + dx)
    else:
        new_w = max(w0 - dx, w1)

    if dz >= 0:
        new_d = max(d0, d1 + dz)
    else:
        new_d = max(d0 - dz, d1)

    print(f"Expanded volume shape: ({new_h}, {new_w}, {new_d})")
    print(f"  Height: {new_h} pixels")
    print(f"  Width: {new_w} pixels")
    print(f"  Depth: {new_d} B-scans")

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
    print(f"  Volume 0: y={y0_start}:{y0_start+h0}, x={x0_start}:{x0_start+w0}, z={z0_start}:{z0_start+d0}")
    print(f"  Volume 1: y={y1_start}:{y1_start+h1}, x={x1_start}:{x1_start+w1}, z={z1_start}:{z1_start+d1}")

    # Place volume 0
    expanded[y0_start:y0_start+h0, x0_start:x0_start+w0, z0_start:z0_start+d0] = volume_0

    # Calculate overlap region
    y_overlap_start = max(y0_start, y1_start)
    y_overlap_end = min(y0_start+h0, y1_start+h1)
    x_overlap_start = max(x0_start, x1_start)
    x_overlap_end = min(x0_start+w0, x1_start+w1)
    z_overlap_start = max(z0_start, z1_start)
    z_overlap_end = min(z0_start+d0, z1_start+d1)

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
    mask_v1[y0_start:y0_start+h0, x0_start:x0_start+w0, z0_start:z0_start+d0] = False

    v1_placed = np.zeros_like(expanded)
    v1_placed[y1_start:y1_start+h1, x1_start:x1_start+w1, z1_start:z1_start+d1] = volume_1

    if has_overlap:
        v1_mask = (v1_placed > 0) & (expanded == 0)
        expanded[v1_mask] = v1_placed[v1_mask]
    else:
        expanded[y1_start:y1_start+h1, x1_start:x1_start+w1, z1_start:z1_start+d1] = volume_1

    total_voxels = (expanded > 0).sum()

    # Create source label volume (0=volume_0, 1=volume_1, 2=overlap)
    source_labels = np.zeros((new_h, new_w, new_d), dtype=np.uint8)

    # Label volume 0 regions
    source_labels[y0_start:y0_start+h0, x0_start:x0_start+w0, z0_start:z0_start+d0] = 0

    # Label volume 1 regions
    v1_mask_full = np.zeros((new_h, new_w, new_d), dtype=bool)
    v1_mask_full[y1_start:y1_start+h1, x1_start:x1_start+w1, z1_start:z1_start+d1] = True

    # Mark volume 1 non-overlap regions
    if has_overlap:
        overlap_mask = np.zeros((new_h, new_w, new_d), dtype=bool)
        overlap_mask[y_overlap_start:y_overlap_end,
                    x_overlap_start:x_overlap_end,
                    z_overlap_start:z_overlap_end] = True
        source_labels[v1_mask_full & ~overlap_mask] = 1
        source_labels[overlap_mask] = 2  # Mark overlap regions
    else:
        source_labels[v1_mask_full] = 1

    metadata = {
        'volume_0_shape': (h0, w0, d0),
        'volume_1_shape': (h1, w1, d1),
        'expanded_shape': (new_h, new_w, new_d),
        'translation': (dy, dx, dz),
        'has_overlap': has_overlap,
        'overlap_voxels': overlap_voxels,
        'total_voxels': total_voxels,
        'data_loss': 0,
        'source_labels': source_labels  # Add source labels to metadata
    }

    return expanded, metadata


def _render_single_angle_view(args):
    """
    Worker function to render a single 3D angle view.

    Runs in a separate process for parallel rendering.

    Args:
        args: Tuple of (y, x, z, colors, scatter_kwargs, title, elev, azim)

    Returns:
        PIL Image of the rendered view
    """
    y, x, z, colors, scatter_kwargs, view_title, elev, azim, labels_sub = args

    # Create figure for this single view
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(y, x, z, **scatter_kwargs)
    ax.set_xlabel('Height (Y)')
    ax.set_ylabel('Width (X)')
    ax.set_zlabel('Depth (Z)')
    ax.set_title(view_title, fontsize=12, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)

    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Convert to PIL Image
    buf.seek(0)
    img = Image.open(buf)

    return img


def visualize_3d_multiangle(volume, title, output_path, subsample=4, percentile=70, z_crop_front=0, source_labels=None):
    """
    Create 3D volume projections from multiple angles (PARALLELIZED).

    OPTIMIZED: Renders 4 angle views in PARALLEL for 2x speedup.

    Shows 4 views: X-axis (side), Y-axis (front), Z-axis (top), 45° angle

    Args:
        volume: 3D volume to visualize
        title: Title for the visualization
        output_path: Where to save the image
        subsample: Subsampling factor for rendering
        percentile: Percentile threshold for voxel filtering
        z_crop_front: Remove this many B-scans from the front (default: 0)
        source_labels: Optional label volume (0=vol0, 1=vol1, 2=overlap) for color coding
    """
    print(f"\nCreating 3D multi-angle visualization (PARALLEL): {title}")

    # Subsample volume for visualization
    vol_sub = volume[::subsample, ::subsample, ::subsample]
    print(f"  Subsampled to: {vol_sub.shape}")

    # Subsample source labels if provided
    labels_sub = None
    if source_labels is not None:
        labels_sub = source_labels[::subsample, ::subsample, ::subsample]

    # Crop back B-scans if requested (in subsampled space)
    # Note: "front" in parameter name is back from user's perspective
    z_crop_sub = z_crop_front // subsample
    if z_crop_sub > 0:
        vol_sub = vol_sub[:, :, :-z_crop_sub]  # Crop from END (back)
        if labels_sub is not None:
            labels_sub = labels_sub[:, :, :-z_crop_sub]
        print(f"  Cropped back {z_crop_front} B-scans ({z_crop_sub} after subsampling)")
        print(f"  New shape: {vol_sub.shape}")

    # Threshold to show only high-intensity voxels
    threshold = np.percentile(vol_sub[vol_sub > 0], percentile)
    print(f"  Intensity threshold: {threshold:.1f}")

    # Get coordinates of voxels above threshold
    z, y, x = np.where(vol_sub > threshold)

    # Prepare colors based on source labels or intensity
    if labels_sub is not None:
        # Color code based on volume source
        # Volume 0: Cyan, Volume 1: Magenta, Overlap: Yellow
        labels_at_voxels = labels_sub[z, y, x]
        colors = np.zeros((len(x), 3))  # RGB colors
        colors[labels_at_voxels == 0] = [0, 1, 1]      # Cyan for volume 0
        colors[labels_at_voxels == 1] = [1, 0, 1]      # Magenta for volume 1
        colors[labels_at_voxels == 2] = [1, 1, 0]      # Yellow for overlap
        print(f"  Color coding: Cyan=Volume0, Magenta=Volume1, Yellow=Overlap")
    else:
        # Use intensity-based coloring (original behavior)
        colors = vol_sub[z, y, x]
        colors_norm = (colors - colors.min()) / (colors.max() - colors.min())
        colors = colors_norm  # Will use with cmap='hot'

    print(f"  Rendering {len(x):,} voxels in PARALLEL (4 views)...")

    # Determine scatter plot parameters based on color type
    if labels_sub is not None:
        # RGB colors - don't use cmap
        scatter_kwargs = {'c': colors, 's': 0.5, 'alpha': 0.6, 'marker': '.'}
    else:
        # Intensity-based - use cmap
        scatter_kwargs = {'c': colors, 'cmap': 'hot', 's': 0.5, 'alpha': 0.6, 'marker': '.'}

    title_suffix = f' [Front {z_crop_front} B-scans removed]' if z_crop_front > 0 else ''

    # Define 4 angle views to render in parallel
    view_configs = [
        (f'X-Axis View (Side/Sagittal){title_suffix}', 0, 0),
        (f'Y-Axis View (Front/Coronal){title_suffix}', 0, 90),
        (f'Z-Axis View (Top/Axial){title_suffix}', 90, -90),
        (f'45° Angle View (Isometric){title_suffix}', 30, 45)
    ]

    # Prepare arguments for parallel rendering
    render_args = [
        (y, x, z, colors, scatter_kwargs, view_title, elev, azim, labels_sub)
        for view_title, elev, azim in view_configs
    ]

    # Render all 4 views in parallel
    num_workers = min(4, cpu_count())
    with Pool(processes=num_workers) as pool:
        rendered_images = pool.map(_render_single_angle_view, render_args)

    print(f"  ✓ Rendered 4 views in parallel")

    # Combine 4 images into 2x2 grid
    # Get dimensions from first image
    img_width, img_height = rendered_images[0].size

    # Create combined image (2x2 grid)
    combined = Image.new('RGB', (img_width * 2, img_height * 2))

    # Paste images into grid
    combined.paste(rendered_images[0], (0, 0))           # Top-left
    combined.paste(rendered_images[1], (img_width, 0))   # Top-right
    combined.paste(rendered_images[2], (0, img_height))  # Bottom-left
    combined.paste(rendered_images[3], (img_width, img_height))  # Bottom-right

    # Add overall title using matplotlib
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111)
    ax.imshow(combined)
    ax.axis('off')
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def _render_single_comparison_view(args):
    """
    Worker function to render a single volume view for comparison.

    Args:
        args: Tuple of (y, x, z, colors_norm, title, elev, azim)

    Returns:
        PIL Image of the rendered view
    """
    y, x, z, colors_norm, view_title, elev, azim = args

    # Create figure for this single view
    fig = plt.figure(figsize=(6.67, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(y, x, z, c=colors_norm, cmap='hot', s=0.5, alpha=0.6, marker='.')
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    ax.set_title(view_title, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)

    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Convert to PIL Image
    buf.seek(0)
    img = Image.open(buf)

    return img


def visualize_3d_comparison(volume_0, volume_1_aligned, merged_volume, transform, output_path, subsample=4, percentile=70, z_crop_front=0, z_crop_back=0):
    """
    Create side-by-side 3D volume comparison (PARALLELIZED).

    OPTIMIZED: Renders 12 views in PARALLEL for 3x speedup.

    Shows: Volume 0, Volume 1 (aligned), Merged volume
    From 4 angles: X-axis, Y-axis, Z-axis, 45°

    Args:
        z_crop_front: Number of B-scans to remove from front (default: 0)
        z_crop_back: Number of B-scans to remove from back (default: 0)
    """
    print("\nCreating side-by-side 3D comparison (PARALLEL)...")
    if z_crop_front > 0 or z_crop_back > 0:
        print(f"  Cropping: {z_crop_front} B-scans from front, {z_crop_back} from back")

    # Crop B-scans if requested
    if z_crop_back > 0:
        volume_0 = volume_0[:, :, :-z_crop_back]
        volume_1_aligned = volume_1_aligned[:, :, :-z_crop_back]
        merged_volume = merged_volume[:, :, :-z_crop_back]
    if z_crop_front > 0:
        volume_0 = volume_0[:, :, z_crop_front:]
        volume_1_aligned = volume_1_aligned[:, :, z_crop_front:]
        merged_volume = merged_volume[:, :, z_crop_front:]

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
    print(f"  Rendering 12 views in PARALLEL (3 volumes × 4 angles)...")

    # Define 12 views: 4 angles × 3 volumes
    # Order: (volume_data, volume_name, angle_name, elev, azim)
    view_data = [
        # Row 1: X-axis view (side)
        (y0, x0, z0, colors0_norm, 'Volume 0 (Reference)\nX-axis view', 0, 0),
        (y1, x1, z1, colors1_norm, f'Volume 1 (Aligned, Y{transform["dy"]:+.0f}px)\nX-axis view', 0, 0),
        (ym, xm, zm, colors_merged_norm, 'Merged Volume\nX-axis view', 0, 0),
        # Row 2: Y-axis view (front)
        (y0, x0, z0, colors0_norm, 'Y-axis view', 0, 90),
        (y1, x1, z1, colors1_norm, 'Y-axis view', 0, 90),
        (ym, xm, zm, colors_merged_norm, 'Y-axis view', 0, 90),
        # Row 3: Z-axis view (top)
        (y0, x0, z0, colors0_norm, 'Z-axis view', 90, -90),
        (y1, x1, z1, colors1_norm, 'Z-axis view', 90, -90),
        (ym, xm, zm, colors_merged_norm, 'Z-axis view', 90, -90),
        # Row 4: 45° angle view
        (y0, x0, z0, colors0_norm, '45° angle view', 30, 45),
        (y1, x1, z1, colors1_norm, '45° angle view', 30, 45),
        (ym, xm, zm, colors_merged_norm, '45° angle view', 30, 45),
    ]

    # Render all 12 views in parallel
    num_workers = min(12, cpu_count())
    with Pool(processes=num_workers) as pool:
        rendered_images = pool.map(_render_single_comparison_view, view_data)

    print(f"  ✓ Rendered 12 views in parallel")

    # Combine 12 images into 4×3 grid
    # Get dimensions from first image
    img_width, img_height = rendered_images[0].size

    # Create combined image (4 rows × 3 columns)
    combined = Image.new('RGB', (img_width * 3, img_height * 4))

    # Paste images into grid (row by row)
    for row in range(4):
        for col in range(3):
            idx = row * 3 + col
            x_pos = col * img_width
            y_pos = row * img_height
            combined.paste(rendered_images[idx], (x_pos, y_pos))

    # Add overall title using matplotlib
    fig = plt.figure(figsize=(20, 24))
    ax = fig.add_subplot(111)
    ax.imshow(combined)
    ax.axis('off')
    plt.suptitle('Side-by-Side 3D Volume Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
