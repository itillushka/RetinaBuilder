#!/usr/bin/env python3
"""
Visualize X-Section Panorama from Alignment Transforms (Vertical Volumes)

Loads alignment transforms from JSON file and creates a panorama by:
1. Loading original volumes
2. Extracting central X-sections (vol[:, x, :] for vertical volumes)
3. Denoising and normalizing brightness
4. Applying transforms DIRECTLY (rotation, then shift)
5. Stitching into panorama

Usage:
    python visualize_bscan_panorama.py <alignment_transforms.json>
    python visualize_bscan_panorama.py results_5vol_em005/alignment_transforms.json --n-slices 50
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from helpers import OCTImageProcessor, OCTVolumeLoader


def extract_central_xsection(volume, n_slices=30):
    """
    Extract and average central X-axis cross-sections from a volume.

    For vertical volumes, we use X-axis cross-sections vol[:, x, :]
    instead of B-scans vol[:, :, z].

    Args:
        volume: OCT volume (Y, X, Z)
        n_slices: Number of central X-sections to average

    Returns:
        averaged_xsection: 2D averaged X-section (Y, Z)
    """
    Y, X, Z = volume.shape
    x_center = X // 2
    x_start = max(0, x_center - n_slices // 2)
    x_end = min(X, x_start + n_slices)

    central_xsections = volume[:, x_start:x_end, :]
    averaged_xsection = np.mean(central_xsections, axis=1)

    return averaged_xsection


def denoise_bscan(img):
    """
    Denoise B-scan using the same pipeline as the alignment.

    Pipeline:
      1. Normalize to 0-255
      2. Non-local means denoising (h=30)
      3. Bilateral filtering (sigma=180)
      4. Median filter (kernel=19)
      5. Threshold (85% of Otsu)
      6. CLAHE contrast enhancement

    Args:
        img: Input B-scan (2D array, float)

    Returns:
        Denoised uint8 image
    """
    # Normalize to 0-255
    img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    # Step 1: Non-local means denoising
    denoised = cv2.fastNlMeansDenoising(img_norm, h=30, templateWindowSize=7, searchWindowSize=21)

    # Step 2: Bilateral filtering
    denoised = cv2.bilateralFilter(denoised, d=11, sigmaColor=180, sigmaSpace=180)

    # Step 3: Median filter
    denoised = cv2.medianBlur(denoised, 19)

    # Step 4: Threshold (85% of Otsu)
    thresh_val = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_val = int(thresh_val * 0.85)
    denoised[denoised < thresh_val] = 0

    # Step 5: CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.6, tileGridSize=(8, 8))
    denoised = clahe.apply(denoised)

    return denoised


def normalize_brightness(bscans):
    """
    Normalize brightness across all B-scans to have consistent appearance.

    Uses percentile-based normalization to match all images to same intensity range.
    This preserves relative intensities better than histogram matching.

    Args:
        bscans: Dict of B-scans {'v1': bscan, ...}

    Returns:
        Dict of normalized B-scans (float, 0-1 range)
    """
    # First pass: find global percentiles across all images
    all_nonzero = []
    for vol_name, bscan in bscans.items():
        bscan_f = bscan.astype(np.float32)
        nonzero = bscan_f[bscan_f > 0]
        if len(nonzero) > 0:
            all_nonzero.append(nonzero)

    if len(all_nonzero) == 0:
        # Fallback
        normalized = {}
        for vol_name, bscan in bscans.items():
            normalized[vol_name] = np.zeros_like(bscan, dtype=np.float32)
        return normalized

    # Calculate global percentiles for consistent mapping
    all_values = np.concatenate(all_nonzero)
    global_p2 = np.percentile(all_values, 2)
    global_p98 = np.percentile(all_values, 98)

    normalized = {}
    for vol_name, bscan in bscans.items():
        bscan_norm = bscan.astype(np.float32)

        # Get local percentiles
        nonzero_mask = bscan_norm > 0
        if np.any(nonzero_mask):
            local_nonzero = bscan_norm[nonzero_mask]
            local_p2 = np.percentile(local_nonzero, 2)
            local_p98 = np.percentile(local_nonzero, 98)

            if local_p98 > local_p2:
                # Map local range to global range
                bscan_norm[nonzero_mask] = (local_nonzero - local_p2) / (local_p98 - local_p2) * (global_p98 - global_p2) + global_p2

        # Final normalization to 0-1 using global range
        if global_p98 > global_p2:
            bscan_norm = np.clip((bscan_norm - global_p2) / (global_p98 - global_p2), 0, 1)
        else:
            bscan_norm = np.zeros_like(bscan_norm)

        normalized[vol_name] = bscan_norm

    return normalized


def apply_transform_to_bscan(bscan, dx, dy, rotation):
    """
    Apply transforms to a B-scan: first rotation, then shift.

    Args:
        bscan: 2D B-scan (Y, X)
        dx: X shift (positive = right)
        dy: Y shift (positive = down)
        rotation: Rotation angle in degrees (counter-clockwise)

    Returns:
        transformed_bscan: Transformed B-scan
    """
    result = bscan.copy()

    # Apply rotation first (around center)
    if abs(rotation) > 0.01:
        result = ndimage.rotate(
            result,
            angle=rotation,  # Direct rotation for B-scans
            axes=(0, 1),
            reshape=False,
            order=1,
            mode='constant',
            cval=0
        )

    return result


def create_panorama(xsections, transforms, output_path):
    """
    Create panorama by stitching transformed X-sections.

    VERTICAL VERSION: Uses dz (Z-offset) for horizontal positioning instead of dx.
    X-sections have shape (Y, Z), so we stitch along Z-axis.

    Args:
        xsections: Dict of X-sections {'v1': xsection, ...}
        transforms: Dict of transforms {'v1': {'dz': ..., 'dy': ..., 'rotation': ...}, ...}
        output_path: Path to save panorama

    Returns:
        panorama: Stitched panorama image
    """
    # Calculate positions (V1 is at origin)
    # For vertical volumes: dz is horizontal position (Z-axis), dy is vertical (Y-axis)
    positions = {}
    for vol_name, trans in transforms.items():
        positions[vol_name] = {
            'dz': trans.get('dz', 0),  # Use dz for horizontal (Z-axis) positioning
            'dy': trans['dy']
        }

    # Find bounding box
    # For X-sections (Y, Z): shape[0]=Y (height), shape[1]=Z (width)
    all_z = []
    all_y = []
    for vol_name, xsection in xsections.items():
        pos = positions[vol_name]
        all_z.extend([pos['dz'], pos['dz'] + xsection.shape[1]])  # Z dimension (horizontal)
        all_y.extend([pos['dy'], pos['dy'] + xsection.shape[0]])  # Y dimension (vertical)

    z_min, z_max = int(min(all_z)), int(max(all_z))
    y_min, y_max = int(min(all_y)), int(max(all_y))

    # Adjust positions to be non-negative
    for vol_name in positions:
        positions[vol_name]['dz'] -= z_min
        positions[vol_name]['dy'] -= y_min

    # Create canvas (Y height, Z width)
    panorama_width = z_max - z_min   # Z dimension
    panorama_height = y_max - y_min  # Y dimension
    panorama = np.zeros((panorama_height, panorama_width), dtype=np.float32)
    weight_map = np.zeros((panorama_height, panorama_width), dtype=np.float32)

    print(f"\n  Creating panorama...")
    print(f"  Canvas size: {panorama_height} (Y) x {panorama_width} (Z)")

    # Place X-sections (order: V5, V4, V1, V2, V3 for spatial order)
    # Use wide feathered blending for smooth transitions
    feather_width = 80  # pixels to feather at edges (increased for smoother blend)

    for vol_name in ['v5', 'v4', 'v1', 'v2', 'v3']:
        if vol_name not in xsections:
            continue

        xsection = xsections[vol_name]
        pos = positions[vol_name]
        z_start = int(pos['dz'])
        y_start = int(pos['dy'])
        z_end = z_start + xsection.shape[1]  # Z dimension
        y_end = y_start + xsection.shape[0]  # Y dimension

        # Handle boundary cases
        z_end = min(z_end, panorama_width)
        y_end = min(y_end, panorama_height)

        xs_h = y_end - y_start
        xs_w = z_end - z_start

        # Create feathered weight mask for this X-section using smooth cosine blend
        weight = np.ones((xs_h, xs_w), dtype=np.float32)

        # Feather left edge with smooth cosine transition
        fw_left = min(feather_width, xs_w // 3)
        if z_start > 0 and fw_left > 0:
            # Cosine blend: smoother than linear
            blend = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fw_left)))
            weight[:, :fw_left] *= blend[np.newaxis, :]

        # Feather right edge with smooth cosine transition
        fw_right = min(feather_width, xs_w // 3)
        if z_end < panorama_width and fw_right > 0:
            blend = 0.5 * (1 - np.cos(np.linspace(np.pi, 0, fw_right)))
            weight[:, -fw_right:] *= blend[np.newaxis, :]

        panorama[y_start:y_end, z_start:z_end] += xsection[:xs_h, :xs_w] * weight
        weight_map[y_start:y_end, z_start:z_end] += weight
        print(f"    Placed {vol_name.upper()} at z={z_start}, y={y_start}")

    # Weighted average in overlapping regions
    mask = weight_map > 0
    panorama[mask] /= weight_map[mask]

    # Crop to non-zero bounds
    nonzero_rows = np.any(panorama > 0, axis=1)
    nonzero_cols = np.any(panorama > 0, axis=0)
    if np.any(nonzero_rows) and np.any(nonzero_cols):
        row_indices = np.where(nonzero_rows)[0]
        col_indices = np.where(nonzero_cols)[0]
        panorama = panorama[row_indices[0]:row_indices[-1]+1, col_indices[0]:col_indices[-1]+1]

    print(f"  Final panorama shape: {panorama.shape}")

    # Save as PNG and NPY
    output_path = Path(output_path)
    panorama_norm = ((panorama - panorama.min()) / (panorama.max() - panorama.min()) * 255).astype(np.uint8)

    # Resize to wider aspect ratio (stretch horizontally by 3x)
    h, w = panorama_norm.shape
    new_w = int(w * 3)  # Make 3x wider
    panorama_wide = cv2.resize(panorama_norm, (new_w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(output_path), panorama_wide)
    print(f"  [SAVED] {output_path} (resized to {h}x{new_w})")

    # Save as numpy array for further analysis (e.g., curvature analysis)
    npy_path = str(output_path).replace('.png', '.npy')
    np.save(npy_path, panorama)
    print(f"  [SAVED] {npy_path}")

    # Create labeled visualization with wider aspect ratio
    # Scale to make panorama wider and lower
    fig, ax = plt.subplots(figsize=(24, 5))
    ax.imshow(panorama, cmap='gray', aspect=0.3)  # aspect < 1 makes it wider
    ax.set_title('5-Volume Averaged X-Section Panorama (Vertical)', fontsize=14, fontweight='bold')
    ax.axis('off')
    fig.text(0.5, 0.02, 'Spatial Order: V5 <- V4 <- V1 -> V2 -> V3',
             ha='center', fontsize=10, style='italic')
    plt.tight_layout()

    labeled_path = str(output_path).replace('.png', '_labeled.png')
    plt.savefig(labeled_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {labeled_path}")

    return panorama


def main():
    parser = argparse.ArgumentParser(
        description='Create B-scan panorama from alignment transforms',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('json_path', type=str, help='Path to alignment_transforms.json')
    parser.add_argument('--n-slices', type=int, default=30,
                        help='Number of central X-sections to average (default: 30)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for panorama (default: same dir as JSON)')

    args = parser.parse_args()

    # Load JSON
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)

    print("=" * 70)
    print("B-SCAN PANORAMA VISUALIZATION")
    print("=" * 70)

    with open(json_path, 'r') as f:
        alignment_data = json.load(f)

    print(f"\n  Patient: {alignment_data['patient']}")
    print(f"  Timestamp: {alignment_data['timestamp']}")

    # Setup output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(alignment_data['output_dir'])
        output_path = output_dir / 'averaged_middle_xsections_panorama.png'

    # Load volumes and extract X-sections (for vertical volumes)
    print("\n  Loading volumes and extracting X-sections...")
    processor = OCTImageProcessor(sidebar_width=250, crop_top=100, crop_bottom=50)
    loader = OCTVolumeLoader(processor)

    xsections_raw = {}
    transforms = {}

    for vol_name, vol_data in alignment_data['volumes'].items():
        vol_path = Path(vol_data['path'])
        print(f"    Loading {vol_name.upper()}: {vol_path.name}")

        volume = loader.load_volume_from_directory(str(vol_path))
        xsection = extract_central_xsection(volume, args.n_slices)

        # Denoise X-section (same as alignment pipeline)
        print(f"      Denoising...")
        xsection = denoise_bscan(xsection)

        # Apply rotation transform directly to X-section
        rotation = vol_data.get('rotation', 0.0)
        if abs(rotation) > 0.01:
            xsection = apply_transform_to_bscan(xsection, 0, 0, rotation)
            print(f"      Applied rotation: {rotation:+.3f} deg")

        xsections_raw[vol_name] = xsection
        transforms[vol_name] = {
            'dx': vol_data.get('dx', 0),
            'dy': vol_data.get('dy', 0),
            'dz': vol_data.get('dz', 0),  # VERTICAL: use dz for horizontal positioning
            'rotation': rotation
        }

        # Free volume memory
        del volume

    # Normalize brightness across all X-sections
    print("\n  Normalizing brightness across all X-sections...")
    xsections = normalize_brightness(xsections_raw)

    # Save individual X-sections
    output_dir = Path(alignment_data['output_dir'])
    print("\n  Saving individual X-sections...")
    for vol_name, xsection in xsections.items():
        # Normalize to 0-255
        xs_norm = ((xsection - xsection.min()) / (xsection.max() - xsection.min() + 1e-8) * 255).astype(np.uint8)
        # Resize to wider aspect (3x horizontal stretch)
        h, w = xs_norm.shape
        xs_wide = cv2.resize(xs_norm, (w * 3, h), interpolation=cv2.INTER_LINEAR)
        xs_path = output_dir / f'xsection_{vol_name}.png'
        cv2.imwrite(str(xs_path), xs_wide)
        print(f"    [SAVED] {xs_path}")

    # Print transforms
    print("\n  Transforms (relative to V1):")
    for vol_name, trans in transforms.items():
        if vol_name == 'v1':
            print(f"    {vol_name.upper()}: reference")
        else:
            print(f"    {vol_name.upper()}: dz={trans['dz']:+d}, dy={trans['dy']:+.1f}, rot={trans['rotation']:+.3f}deg")

    # Create panorama
    panorama = create_panorama(xsections, transforms, output_path)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
