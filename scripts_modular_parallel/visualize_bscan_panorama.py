#!/usr/bin/env python3
"""
Visualize B-Scan Panorama from Alignment Transforms

Loads alignment transforms from JSON file and creates a panorama by:
1. Loading original volumes
2. Extracting central B-scans
3. Denoising and normalizing brightness
4. Applying transforms DIRECTLY (rotation, then shift)
5. Stitching into panorama

Usage:
    python visualize_bscan_panorama.py <alignment_transforms.json>
    python visualize_bscan_panorama.py results_5vol_em005/alignment_transforms.json --n-bscans 50
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


def extract_central_bscan(volume, n_bscans=30):
    """
    Extract and average central B-scans from a volume.

    Args:
        volume: OCT volume (Y, X, Z) where Z is the B-scan index
        n_bscans: Number of central B-scans to average

    Returns:
        averaged_bscan: 2D averaged B-scan (Y, X)
    """
    Y, X, Z = volume.shape
    z_center = Z // 2
    z_start = max(0, z_center - n_bscans // 2)
    z_end = min(Z, z_start + n_bscans)

    central_bscans = volume[:, :, z_start:z_end]
    averaged_bscan = np.mean(central_bscans, axis=2)

    return averaged_bscan


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

    Uses percentile-based normalization to handle different exposure levels.

    Args:
        bscans: Dict of B-scans {'v1': bscan, ...}

    Returns:
        Dict of normalized B-scans (float, 0-1 range)
    """
    # Calculate global statistics from all B-scans
    all_values = []
    for bscan in bscans.values():
        # Only consider non-zero pixels (tissue)
        nonzero = bscan[bscan > 0]
        if len(nonzero) > 0:
            all_values.append(nonzero)

    if not all_values:
        return bscans

    all_values = np.concatenate(all_values)

    # Use percentiles for robust normalization
    p_low = np.percentile(all_values, 2)
    p_high = np.percentile(all_values, 98)

    # Normalize each B-scan
    normalized = {}
    for vol_name, bscan in bscans.items():
        bscan_norm = bscan.astype(np.float32)
        bscan_norm = np.clip(bscan_norm, p_low, p_high)
        bscan_norm = (bscan_norm - p_low) / (p_high - p_low + 1e-8)
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


def create_panorama(bscans, transforms, output_path):
    """
    Create panorama by stitching transformed B-scans.

    Args:
        bscans: Dict of B-scans {'v1': bscan, ...}
        transforms: Dict of transforms {'v1': {'dx': ..., 'dy': ..., 'rotation': ...}, ...}
        output_path: Path to save panorama

    Returns:
        panorama: Stitched panorama image
    """
    # Calculate positions (V1 is at origin)
    positions = {}
    for vol_name, trans in transforms.items():
        positions[vol_name] = {
            'dx': trans['dx'],
            'dy': trans['dy']
        }

    # Find bounding box
    all_x = []
    all_y = []
    for vol_name, bscan in bscans.items():
        pos = positions[vol_name]
        all_x.extend([pos['dx'], pos['dx'] + bscan.shape[1]])
        all_y.extend([pos['dy'], pos['dy'] + bscan.shape[0]])

    x_min, x_max = int(min(all_x)), int(max(all_x))
    y_min, y_max = int(min(all_y)), int(max(all_y))

    # Adjust positions to be non-negative
    for vol_name in positions:
        positions[vol_name]['dx'] -= x_min
        positions[vol_name]['dy'] -= y_min

    # Create canvas
    panorama_width = x_max - x_min
    panorama_height = y_max - y_min
    panorama = np.zeros((panorama_height, panorama_width), dtype=np.float32)
    weight_map = np.zeros((panorama_height, panorama_width), dtype=np.float32)

    print(f"\n  Creating panorama...")
    print(f"  Canvas size: {panorama_height} x {panorama_width}")

    # Place B-scans (order: V5, V4, V1, V2, V3 for spatial order)
    for vol_name in ['v5', 'v4', 'v1', 'v2', 'v3']:
        if vol_name not in bscans:
            continue

        bscan = bscans[vol_name]
        pos = positions[vol_name]
        x_start = int(pos['dx'])
        y_start = int(pos['dy'])
        x_end = x_start + bscan.shape[1]
        y_end = y_start + bscan.shape[0]

        # Handle boundary cases
        x_end = min(x_end, panorama_width)
        y_end = min(y_end, panorama_height)

        bscan_h = y_end - y_start
        bscan_w = x_end - x_start

        panorama[y_start:y_end, x_start:x_end] += bscan[:bscan_h, :bscan_w]
        weight_map[y_start:y_end, x_start:x_end] += 1.0
        print(f"    Placed {vol_name.upper()} at x={x_start}, y={y_start}")

    # Average overlapping regions
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
    cv2.imwrite(str(output_path), panorama_norm)
    print(f"  [SAVED] {output_path}")

    # Save as numpy array for further analysis (e.g., curvature analysis)
    npy_path = str(output_path).replace('.png', '.npy')
    np.save(npy_path, panorama)
    print(f"  [SAVED] {npy_path}")

    # Create labeled visualization
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.imshow(panorama, cmap='gray', aspect='auto')
    ax.set_title('5-Volume Averaged B-Scan Panorama', fontsize=14, fontweight='bold')
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
    parser.add_argument('--n-bscans', type=int, default=30,
                        help='Number of central B-scans to average (default: 30)')
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
        output_path = output_dir / 'averaged_middle_bscans_panorama.png'

    # Load volumes and extract B-scans
    print("\n  Loading volumes and extracting B-scans...")
    processor = OCTImageProcessor(sidebar_width=250, crop_top=100, crop_bottom=50)
    loader = OCTVolumeLoader(processor)

    bscans_raw = {}
    transforms = {}

    for vol_name, vol_data in alignment_data['volumes'].items():
        vol_path = Path(vol_data['path'])
        print(f"    Loading {vol_name.upper()}: {vol_path.name}")

        volume = loader.load_volume_from_directory(str(vol_path))
        bscan = extract_central_bscan(volume, args.n_bscans)

        # Denoise B-scan (same as alignment pipeline)
        print(f"      Denoising...")
        bscan = denoise_bscan(bscan)

        # Apply rotation transform directly to B-scan
        rotation = vol_data.get('rotation', 0.0)
        if abs(rotation) > 0.01:
            bscan = apply_transform_to_bscan(bscan, 0, 0, rotation)
            print(f"      Applied rotation: {rotation:+.3f} deg")

        bscans_raw[vol_name] = bscan
        transforms[vol_name] = {
            'dx': vol_data['dx'],
            'dy': vol_data['dy'],
            'rotation': rotation
        }

        # Free volume memory
        del volume

    # Normalize brightness across all B-scans
    print("\n  Normalizing brightness across all B-scans...")
    bscans = normalize_brightness(bscans_raw)

    # Print transforms
    print("\n  Transforms (relative to V1):")
    for vol_name, trans in transforms.items():
        if vol_name == 'v1':
            print(f"    {vol_name.upper()}: reference")
        else:
            print(f"    {vol_name.upper()}: dx={trans['dx']:+d}, dy={trans['dy']:+.1f}, rot={trans['rotation']:+.3f}deg")

    # Create panorama
    panorama = create_panorama(bscans, transforms, output_path)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
