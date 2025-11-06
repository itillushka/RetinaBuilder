#!/usr/bin/env python3
"""
Step 1: XZ Alignment Only - Show What We Have

This script ONLY does XZ alignment and shows:
1. Center B-scan (X-axis slice) - full vs cropped to overlap
2. Center Y-slice - full vs cropped to overlap

NO rotation, NO Y alignment yet.

Usage:
    python step1_xz_alignment_only.py --auto
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from scipy.fft import fft2
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from oct_volumetric_viewer import OCTImageProcessor, OCTVolumeLoader


def phase_correlation_2d(img1, img2, max_shift=50):
    """2D phase correlation for XZ alignment."""
    img1_norm = (img1 - img1.mean()) / (img1.std() + 1e-8)
    img2_norm = (img2 - img2.mean()) / (img2.std() + 1e-8)

    f1 = fft2(img1_norm)
    f2 = fft2(img2_norm)

    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-8)
    correlation = np.fft.ifft2(cross_power).real
    correlation = np.fft.fftshift(correlation)

    center_y, center_x = np.array(correlation.shape) // 2
    y_start = max(0, center_y - max_shift)
    y_end = min(correlation.shape[0], center_y + max_shift + 1)
    x_start = max(0, center_x - max_shift)
    x_end = min(correlation.shape[1], center_x + max_shift + 1)

    search_region = correlation[y_start:y_end, x_start:x_end]
    peak_y, peak_x = np.unravel_index(np.argmax(search_region), search_region.shape)

    offset_y = (y_start + peak_y) - center_y
    offset_x = (x_start + peak_x) - center_x

    peak_value = search_region[peak_y, peak_x]
    confidence = peak_value / (search_region.std() + 1e-8)

    return offset_x, offset_y, confidence


def main():
    print("="*70)
    print("STEP 1: XZ ALIGNMENT ONLY")
    print("="*70)

    # Setup
    data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'
    oct_data_dir = Path(__file__).parent.parent / 'oct_data'

    processor = OCTImageProcessor(sidebar_width=250, crop_top=100, crop_bottom=50)
    loader = OCTVolumeLoader(processor)

    # Load F001 volumes
    print("\n1. Loading volumes...")
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

    print(f"\n  V0 shape: {volume_0.shape} (Y, X, Z)")
    print(f"  V1 shape: {volume_1.shape}")

    # XZ alignment
    print("\n2. XZ alignment (phase correlation)...")
    mip_v0 = np.max(volume_0, axis=0)
    mip_v1 = np.max(volume_1, axis=0)

    offset_x, offset_z, confidence = phase_correlation_2d(mip_v0, mip_v1)

    print(f"  Offset X: {offset_x} pixels")
    print(f"  Offset Z: {offset_z} pixels")
    print(f"  Confidence: {confidence:.2f}")

    # Apply XZ shift
    volume_1_xz_aligned = ndimage.shift(
        volume_1, shift=(0, offset_x, offset_z),
        order=1, mode='constant', cval=0
    )

    # Calculate overlap region
    print("\n3. Calculating overlap region...")
    Y, X, Z = volume_0.shape

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

    print(f"\n  Overlap region bounds:")
    print(f"    V0: X[{x0_start}:{x0_end}], Z[{z0_start}:{z0_end}]")
    print(f"    V1: X[{x1_start}:{x1_end}], Z[{z1_start}:{z1_end}]")
    print(f"    Overlap size: Y={Y}, X={x0_end-x0_start}, Z={z0_end-z0_start}")

    # Extract overlap regions
    overlap_v0 = volume_0[:, x0_start:x0_end, z0_start:z0_end].copy()
    overlap_v1 = volume_1_xz_aligned[:, x1_start:x1_end, z1_start:z1_end].copy()

    print(f"\n  Overlap V0: {overlap_v0.shape}")
    print(f"  Overlap V1: {overlap_v1.shape}")

    # Get center slices
    x_center_full = volume_0.shape[1] // 2
    y_center_full = volume_0.shape[0] // 2
    z_center_full = volume_0.shape[2] // 2

    x_center_overlap = overlap_v0.shape[1] // 2
    y_center_overlap = overlap_v0.shape[0] // 2
    z_center_overlap = overlap_v0.shape[2] // 2

    print(f"\n4. Center slice positions:")
    print(f"  Full volumes: X={x_center_full}, Y={y_center_full}, Z={z_center_full}")
    print(f"  Overlap regions: X={x_center_overlap}, Y={y_center_overlap}, Z={z_center_overlap}")

    # Full volume slices
    full_v0_bscan = volume_0[:, x_center_full, :].copy()
    full_v1_bscan = volume_1_xz_aligned[:, x_center_full, :].copy()

    full_v0_yslice = volume_0[y_center_full, :, :].copy()
    full_v1_yslice = volume_1_xz_aligned[y_center_full, :, :].copy()

    # Overlap region slices
    overlap_v0_bscan = overlap_v0[:, x_center_overlap, :].copy()
    overlap_v1_bscan = overlap_v1[:, x_center_overlap, :].copy()

    overlap_v0_yslice = overlap_v0[y_center_overlap, :, :].copy()
    overlap_v1_yslice = overlap_v1[y_center_overlap, :, :].copy()

    print("\n5. Slice shapes:")
    print(f"  Full B-scan (Y-Z): {full_v0_bscan.shape}")
    print(f"  Overlap B-scan (Y-Z): {overlap_v0_bscan.shape}")
    print(f"  Full Y-slice (X-Z): {full_v0_yslice.shape}")
    print(f"  Overlap Y-slice (X-Z): {overlap_v0_yslice.shape}")

    # Calculate statistics
    print("\n6. Data statistics:")
    print(f"  Full V0 B-scan: mean={full_v0_bscan.mean():.1f}, nonzero={np.count_nonzero(full_v0_bscan)}/{full_v0_bscan.size}")
    print(f"  Full V1 B-scan: mean={full_v1_bscan.mean():.1f}, nonzero={np.count_nonzero(full_v1_bscan)}/{full_v1_bscan.size}")
    print(f"  Overlap V0 B-scan: mean={overlap_v0_bscan.mean():.1f}, nonzero={np.count_nonzero(overlap_v0_bscan)}/{overlap_v0_bscan.size}")
    print(f"  Overlap V1 B-scan: mean={overlap_v1_bscan.mean():.1f}, nonzero={np.count_nonzero(overlap_v1_bscan)}/{overlap_v1_bscan.size}")

    print(f"\n  Full V0 Y-slice: mean={full_v0_yslice.mean():.1f}, nonzero={np.count_nonzero(full_v0_yslice)}/{full_v0_yslice.size}")
    print(f"  Full V1 Y-slice: mean={full_v1_yslice.mean():.1f}, nonzero={np.count_nonzero(full_v1_yslice)}/{full_v1_yslice.size}")
    print(f"  Overlap V0 Y-slice: mean={overlap_v0_yslice.mean():.1f}, nonzero={np.count_nonzero(overlap_v0_yslice)}/{overlap_v0_yslice.size}")
    print(f"  Overlap V1 Y-slice: mean={overlap_v1_yslice.mean():.1f}, nonzero={np.count_nonzero(overlap_v1_yslice)}/{overlap_v1_yslice.size}")

    # Visualization
    print("\n7. Creating visualization...")
    fig = plt.figure(figsize=(24, 14))

    # Top section: B-scans (Y-Z plane)
    # Row 1: Full volumes
    ax1 = plt.subplot(4, 4, 1)
    ax1.imshow(full_v0_bscan, cmap='Reds', aspect='auto', vmin=0, vmax=255)
    ax1.set_title(f'V0 Full B-scan (X={x_center_full})\nShape: {full_v0_bscan.shape}', fontweight='bold')
    ax1.set_ylabel('Y (depth)')
    ax1.set_xlabel('Z (B-scans)')

    ax2 = plt.subplot(4, 4, 2)
    ax2.imshow(full_v1_bscan, cmap='Greens', aspect='auto', vmin=0, vmax=255)
    ax2.set_title(f'V1 Full B-scan (X={x_center_full})\nShape: {full_v1_bscan.shape}', fontweight='bold')
    ax2.set_xlabel('Z (B-scans)')

    ax3 = plt.subplot(4, 4, 3)
    ax3.imshow(full_v0_bscan, cmap='Reds', alpha=0.5, aspect='auto')
    ax3.imshow(full_v1_bscan, cmap='Greens', alpha=0.5, aspect='auto')
    ax3.set_title('Overlay Full', fontweight='bold')
    ax3.set_xlabel('Z (B-scans)')

    # Row 2: Overlap regions
    ax5 = plt.subplot(4, 4, 5)
    ax5.imshow(overlap_v0_bscan, cmap='Reds', aspect='auto', vmin=0, vmax=255)
    ax5.set_title(f'V0 Overlap B-scan (X={x_center_overlap})\nShape: {overlap_v0_bscan.shape}', fontweight='bold')
    ax5.set_ylabel('Y (depth)')
    ax5.set_xlabel('Z (B-scans)')

    ax6 = plt.subplot(4, 4, 6)
    ax6.imshow(overlap_v1_bscan, cmap='Greens', aspect='auto', vmin=0, vmax=255)
    ax6.set_title(f'V1 Overlap B-scan (X={x_center_overlap})\nShape: {overlap_v1_bscan.shape}', fontweight='bold')
    ax6.set_xlabel('Z (B-scans)')

    ax7 = plt.subplot(4, 4, 7)
    ax7.imshow(overlap_v0_bscan, cmap='Reds', alpha=0.5, aspect='auto')
    ax7.imshow(overlap_v1_bscan, cmap='Greens', alpha=0.5, aspect='auto')
    ax7.set_title('Overlay Overlap', fontweight='bold')
    ax7.set_xlabel('Z (B-scans)')

    # Bottom section: Y-slices (X-Z plane)
    # Row 3: Full volumes
    ax9 = plt.subplot(4, 4, 9)
    ax9.imshow(full_v0_yslice, cmap='Reds', aspect='auto', vmin=0, vmax=255)
    ax9.set_title(f'V0 Full Y-slice (Y={y_center_full})\nShape: {full_v0_yslice.shape}', fontweight='bold')
    ax9.set_ylabel('Z (B-scans)')
    ax9.set_xlabel('X (lateral)')

    ax10 = plt.subplot(4, 4, 10)
    ax10.imshow(full_v1_yslice, cmap='Greens', aspect='auto', vmin=0, vmax=255)
    ax10.set_title(f'V1 Full Y-slice (Y={y_center_full})\nShape: {full_v1_yslice.shape}', fontweight='bold')
    ax10.set_xlabel('X (lateral)')

    ax11 = plt.subplot(4, 4, 11)
    ax11.imshow(full_v0_yslice, cmap='Reds', alpha=0.5, aspect='auto')
    ax11.imshow(full_v1_yslice, cmap='Greens', alpha=0.5, aspect='auto')
    ax11.set_title('Overlay Full', fontweight='bold')
    ax11.set_xlabel('X (lateral)')

    # Row 4: Overlap regions
    ax13 = plt.subplot(4, 4, 13)
    ax13.imshow(overlap_v0_yslice, cmap='Reds', aspect='auto', vmin=0, vmax=255)
    ax13.set_title(f'V0 Overlap Y-slice (Y={y_center_overlap})\nShape: {overlap_v0_yslice.shape}', fontweight='bold')
    ax13.set_ylabel('Z (B-scans)')
    ax13.set_xlabel('X (lateral)')

    ax14 = plt.subplot(4, 4, 14)
    ax14.imshow(overlap_v1_yslice, cmap='Greens', aspect='auto', vmin=0, vmax=255)
    ax14.set_title(f'V1 Overlap Y-slice (Y={y_center_overlap})\nShape: {overlap_v1_yslice.shape}', fontweight='bold')
    ax14.set_xlabel('X (lateral)')

    ax15 = plt.subplot(4, 4, 15)
    ax15.imshow(overlap_v0_yslice, cmap='Reds', alpha=0.5, aspect='auto')
    ax15.imshow(overlap_v1_yslice, cmap='Greens', alpha=0.5, aspect='auto')
    ax15.set_title('Overlay Overlap', fontweight='bold')
    ax15.set_xlabel('X (lateral)')

    plt.suptitle(f'Step 1: XZ Alignment Only (ΔX={offset_x}, ΔZ={offset_z})',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()

    output_path = data_dir / 'step1_xz_alignment_slices.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.show()

    print("\n" + "="*70)
    print("✅ Step 1 complete!")
    print("="*70)


if __name__ == '__main__':
    main()
