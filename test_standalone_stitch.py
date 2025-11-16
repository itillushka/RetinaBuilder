#!/usr/bin/env python3
"""
Standalone OCT Volume Stitching Test Script

Applies specific transformations to 2 volumes and merges them:
- X-shift: -412 pixels
- Z-shift: 5 B-scans
- Y-shift: 6 pixels
- Z-rotation: 5 degrees

No dependencies on existing pipeline code.
"""

import numpy as np
from scipy import ndimage
from PIL import Image
import glob
import os
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_bmp_volume(folder_path, sidebar_width=250, crop_top=100, crop_bottom=50):
    """
    Load BMP files from a folder and create a 3D volume.

    Args:
        folder_path: Path to folder containing BMP files
        sidebar_width: Width to crop from left side
        crop_top: Pixels to crop from top
        crop_bottom: Pixels to crop from bottom

    Returns:
        3D numpy array (height, width, depth)
    """
    print(f"\nLoading volume from: {folder_path}")

    # Find all BMP files
    bmp_files = glob.glob(os.path.join(folder_path, "*.bmp"))

    if not bmp_files:
        raise ValueError(f"No BMP files found in {folder_path}")

    # Sort files numerically
    def extract_number(filename):
        numbers = re.findall(r'\d+', os.path.basename(filename))
        return int(numbers[-1]) if numbers else 0

    bmp_files.sort(key=extract_number)

    print(f"  Found {len(bmp_files)} BMP files")

    # Load first image to get dimensions
    first_img = Image.open(bmp_files[0])
    first_array = np.array(first_img)

    # Convert to grayscale if needed
    if len(first_array.shape) == 3:
        first_array = np.mean(first_array, axis=2)

    # Apply cropping
    if first_array.shape[1] > sidebar_width:
        first_array = first_array[:, sidebar_width:]
    if first_array.shape[0] > crop_top:
        first_array = first_array[crop_top:, :]
    if first_array.shape[0] > crop_bottom:
        first_array = first_array[:-crop_bottom, :]

    height, width = first_array.shape
    num_slices = len(bmp_files)

    # Initialize volume array
    volume = np.zeros((height, width, num_slices), dtype=np.float32)

    # Load all images
    for i, file_path in enumerate(bmp_files):
        if i % 50 == 0:
            print(f"  Loading B-scan {i+1}/{num_slices}")

        img = Image.open(file_path)
        img_array = np.array(img)

        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_array = np.mean(img_array, axis=2)

        # Apply same cropping
        if img_array.shape[1] > sidebar_width:
            img_array = img_array[:, sidebar_width:]
        if img_array.shape[0] > crop_top:
            img_array = img_array[crop_top:, :]
        if img_array.shape[0] > crop_bottom:
            img_array = img_array[:-crop_bottom, :]

        volume[:, :, i] = img_array

    print(f"  ✓ Loaded volume: {volume.shape}")
    return volume


def apply_transformations_with_expansion(volume, dx, dy, dz, rotation_degrees):
    """
    Apply transformations to a volume with proper canvas expansion to avoid data loss.

    Args:
        volume: Input volume (h, w, d)
        dx: X-axis shift in pixels (negative = shift left)
        dy: Y-axis shift in pixels (positive = shift down)
        dz: Z-axis shift in B-scans (positive = shift forward)
        rotation_degrees: Z-rotation in degrees

    Returns:
        Transformed volume with expanded canvas
    """
    print(f"\nApplying transformations:")
    print(f"  Input shape: {volume.shape}")
    print(f"  dx={dx}, dy={dy}, dz={dz}, rotation={rotation_degrees}°")

    h, w, d = volume.shape

    # Calculate needed expansion for shifts
    # Expand canvas BEFORE applying transformations to avoid clipping

    expand_y_neg = abs(min(0, dy))  # Expansion needed at top
    expand_y_pos = abs(max(0, dy))  # Expansion needed at bottom
    expand_x_neg = abs(min(0, dx))  # Expansion needed at left
    expand_x_pos = abs(max(0, dx))  # Expansion needed at right
    expand_z_neg = abs(min(0, dz))  # Expansion needed at back
    expand_z_pos = abs(max(0, dz))  # Expansion needed at front

    # Add extra padding for rotation (conservative estimate: diagonal)
    rotation_padding = int(np.ceil(np.sqrt(h**2 + w**2) * abs(np.sin(np.radians(rotation_degrees)))))

    expand_y_neg += rotation_padding
    expand_y_pos += rotation_padding
    expand_x_neg += rotation_padding
    expand_x_pos += rotation_padding

    # Create expanded canvas
    new_h = h + expand_y_neg + expand_y_pos
    new_w = w + expand_x_neg + expand_x_pos
    new_d = d + expand_z_neg + expand_z_pos

    print(f"  Expanded canvas: ({new_h}, {new_w}, {new_d})")

    # Place original volume in center of expanded canvas
    expanded = np.zeros((new_h, new_w, new_d), dtype=np.float32)
    expanded[expand_y_neg:expand_y_neg+h,
             expand_x_neg:expand_x_neg+w,
             expand_z_neg:expand_z_neg+d] = volume

    # Now apply transformations on the expanded canvas

    # Step 1: XZ shift
    print(f"  [1] Applying XZ shift...")
    transformed = ndimage.shift(
        expanded,
        shift=(0, dx, dz),
        order=1,
        mode='constant',
        cval=0
    )

    # Step 2: Y shift
    print(f"  [2] Applying Y shift...")
    transformed = ndimage.shift(
        transformed,
        shift=(dy, 0, 0),
        order=1,
        mode='constant',
        cval=0
    )

    # Step 3: Z-rotation (around center of expanded canvas)
    print(f"  [3] Applying Z-rotation...")
    transformed = ndimage.rotate(
        transformed,
        angle=rotation_degrees,
        axes=(0, 1),  # Y-X plane
        reshape=True,  # Allow expansion to preserve all data
        order=1,
        mode='constant',
        cval=0
    )

    print(f"  ✓ Final transformed shape: {transformed.shape}")

    return transformed


def merge_volumes(volume_0, volume_1_transformed, dx, dy, dz):
    """
    Merge two volumes by placing them on a common canvas.

    Args:
        volume_0: Reference volume (original size)
        volume_1_transformed: Transformed volume (potentially larger due to expansion)
        dx, dy, dz: Original transformation offsets (for calculating placement)

    Returns:
        merged_volume, source_labels
    """
    print(f"\nMerging volumes:")
    print(f"  Volume 0 shape: {volume_0.shape}")
    print(f"  Volume 1 transformed shape: {volume_1_transformed.shape}")

    h0, w0, d0 = volume_0.shape
    h1, w1, d1 = volume_1_transformed.shape

    # Calculate canvas size to fit both volumes
    # Volume 0 will be placed at origin
    # Volume 1 is already positioned (transformations applied)
    # We need a canvas that fits both

    canvas_h = max(h0, h1)
    canvas_w = max(w0, w1)
    canvas_d = max(d0, d1)

    print(f"  Canvas size: ({canvas_h}, {canvas_w}, {canvas_d})")

    # Create merged canvas
    merged = np.zeros((canvas_h, canvas_w, canvas_d), dtype=np.float32)
    source_labels = np.zeros((canvas_h, canvas_w, canvas_d), dtype=np.uint8)

    # Place volume 0 at origin
    merged[0:h0, 0:w0, 0:d0] = volume_0
    source_labels[0:h0, 0:w0, 0:d0] = 1  # Label as volume 0

    # Place volume 1 (centered to align with volume 0's center)
    y1_start = (canvas_h - h1) // 2
    x1_start = (canvas_w - w1) // 2
    z1_start = (canvas_d - d1) // 2

    print(f"  Volume 0 placed at: [0:0+{h0}, 0:0+{w0}, 0:0+{d0}]")
    print(f"  Volume 1 placed at: [{y1_start}:{y1_start+h1}, {x1_start}:{x1_start+w1}, {z1_start}:{z1_start+d1}]")

    # Blend where volumes overlap
    v1_region = merged[y1_start:y1_start+h1, x1_start:x1_start+w1, z1_start:z1_start+d1]
    v1_mask = volume_1_transformed > 0

    # Where both have data, average; otherwise take whichever has data
    overlap_mask = (v1_region > 0) & v1_mask
    merged[y1_start:y1_start+h1, x1_start:x1_start+w1, z1_start:z1_start+d1] = np.where(
        overlap_mask,
        (v1_region + volume_1_transformed) / 2,  # Average in overlap
        np.where(v1_mask, volume_1_transformed, v1_region)  # Otherwise take non-zero
    )

    # Update labels
    labels_region = source_labels[y1_start:y1_start+h1, x1_start:x1_start+w1, z1_start:z1_start+d1]
    source_labels[y1_start:y1_start+h1, x1_start:x1_start+w1, z1_start:z1_start+d1] = np.where(
        overlap_mask,
        3,  # Overlap
        np.where(v1_mask, 2, labels_region)  # Volume 1 or keep existing
    )

    print(f"  ✓ Merged volume created: {merged.shape}")

    return merged, source_labels


def visualize_3d(merged_volume, source_labels, output_path, subsample=8, percentile=75):
    """
    Create 3D visualization with color coding.

    Args:
        merged_volume: Merged 3D volume
        source_labels: Source labels (1=vol0, 2=vol1, 3=overlap)
        output_path: Where to save visualization
        subsample: Subsampling factor
        percentile: Intensity threshold percentile
    """
    print(f"\nGenerating 3D visualization...")

    # Subsample
    vol_sub = merged_volume[::subsample, ::subsample, ::subsample]
    labels_sub = source_labels[::subsample, ::subsample, ::subsample]

    print(f"  Subsampled to: {vol_sub.shape}")

    # Threshold
    threshold = np.percentile(vol_sub[vol_sub > 0], percentile)
    print(f"  Threshold: {threshold:.1f}")

    # Get voxel coordinates
    z, y, x = np.where(vol_sub > threshold)

    # Color by source
    labels_at_voxels = labels_sub[z, y, x]
    colors = np.zeros((len(x), 3))
    colors[labels_at_voxels == 1] = [0, 1, 1]  # Cyan = Volume 0
    colors[labels_at_voxels == 2] = [1, 0, 1]  # Magenta = Volume 1
    colors[labels_at_voxels == 3] = [1, 1, 0]  # Yellow = Overlap

    print(f"  Rendering {len(x):,} voxels...")

    # Create figure with 4 views
    fig = plt.figure(figsize=(20, 16))

    # View 1: X-axis (side)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(y, x, z, c=colors, s=0.5, alpha=0.6, marker='.')
    ax1.set_xlabel('Y'); ax1.set_ylabel('X'); ax1.set_zlabel('Z')
    ax1.set_title('Side View (X-axis)', fontweight='bold')
    ax1.view_init(elev=0, azim=0)

    # View 2: Y-axis (front)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.scatter(y, x, z, c=colors, s=0.5, alpha=0.6, marker='.')
    ax2.set_xlabel('Y'); ax2.set_ylabel('X'); ax2.set_zlabel('Z')
    ax2.set_title('Front View (Y-axis)', fontweight='bold')
    ax2.view_init(elev=0, azim=90)

    # View 3: Z-axis (top)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.scatter(y, x, z, c=colors, s=0.5, alpha=0.6, marker='.')
    ax3.set_xlabel('Y'); ax3.set_ylabel('X'); ax3.set_zlabel('Z')
    ax3.set_title('Top View (Z-axis)', fontweight='bold')
    ax3.view_init(elev=90, azim=-90)

    # View 4: 45° angle
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.scatter(y, x, z, c=colors, s=0.5, alpha=0.6, marker='.')
    ax4.set_xlabel('Y'); ax4.set_ylabel('X'); ax4.set_zlabel('Z')
    ax4.set_title('45° View', fontweight='bold')
    ax4.view_init(elev=30, azim=45)

    plt.suptitle('Standalone Stitching Test (Cyan=Vol0, Magenta=Vol1, Yellow=Overlap)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def main():
    """Main test script."""
    print("="*70)
    print("STANDALONE OCT VOLUME STITCHING TEST")
    print("="*70)

    # Paths - MODIFY THESE TO YOUR DATA LOCATIONS
    oct_data_dir = r"C:\Users\illia\Desktop\diploma\RetinaBuilder\OCT_DATA"

    # Find F001 volumes
    import pathlib
    oct_path = pathlib.Path(oct_data_dir)

    bmp_dirs = []
    for bmp_file in oct_path.rglob('*.bmp'):
        vol_dir = bmp_file.parent
        if vol_dir not in bmp_dirs:
            bmp_dirs.append(vol_dir)

    f001_vols = sorted([v for v in bmp_dirs if 'F001_IP' in str(v)])

    if len(f001_vols) < 2:
        print("ERROR: Need at least 2 F001_IP volumes")
        return

    print(f"\nVolume 0: {f001_vols[0].name}")
    print(f"Volume 1: {f001_vols[1].name}")

    # Load volumes
    volume_0 = load_bmp_volume(str(f001_vols[0]))
    volume_1 = load_bmp_volume(str(f001_vols[1]))

    # Apply transformations to volume 1
    dx = -412  # X-shift
    dy = 6     # Y-shift
    dz = 5     # Z-shift
    rotation = 5  # degrees

    volume_1_transformed = apply_transformations_with_expansion(
        volume_1, dx, dy, dz, rotation
    )

    # Merge volumes
    merged, source_labels = merge_volumes(volume_0, volume_1_transformed, dx, dy, dz)

    # Visualize
    output_path = "standalone_stitch_test.png"
    visualize_3d(merged, source_labels, output_path)

    # Save merged volume
    np.save("standalone_merged_volume.npy", merged)
    print(f"\n✓ Saved merged volume: standalone_merged_volume.npy")

    print("\n" + "="*70)
    print("✅ TEST COMPLETE!")
    print("="*70)
    print(f"\nCheck the output:")
    print(f"  - {output_path}")
    print(f"  - standalone_merged_volume.npy")


if __name__ == "__main__":
    main()
