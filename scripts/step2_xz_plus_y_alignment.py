#!/usr/bin/env python3
"""
Step 2: XZ + Y Alignment with Pinpoint Detection

1. XZ alignment (phase correlation)
2. Find pinpoint (highest surface point) in overlap region
3. Y alignment (vertical) - match pinpoints
4. Show slices at pinpoint location

Usage:
    python step2_xz_plus_y_alignment.py
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


def find_surface_pinpoint(volume):
    """
    Find pinpoint (highest surface point) in volume.

    Returns: (y_pinpoint, x_pinpoint, z_pinpoint) - coordinates of pinpoint
    """
    print("  Finding surface pinpoint...")

    # Calculate en-face projection (max along Y axis)
    enface = np.max(volume, axis=0)  # Shape: (X, Z)

    # Find brightest point in en-face (approximate fovea/pinpoint location)
    x_pin, z_pin = np.unravel_index(np.argmax(enface), enface.shape)

    print(f"    En-face max at: X={x_pin}, Z={z_pin}")

    # Get A-scan at this location
    ascan = volume[:, x_pin, z_pin]

    # Find surface (first significant peak from top)
    threshold = np.percentile(ascan[ascan > 0], 75) if np.any(ascan > 0) else 0

    # Search from top for first pixel above threshold
    y_pin = 0
    for y in range(len(ascan)):
        if ascan[y] > threshold:
            y_pin = y
            break

    print(f"    Surface at Y={y_pin} (intensity={ascan[y_pin]:.1f})")
    print(f"    → Pinpoint: (Y={y_pin}, X={x_pin}, Z={z_pin})")

    return y_pin, x_pin, z_pin


def main():
    print("="*70)
    print("STEP 2: XZ + Y ALIGNMENT WITH PINPOINT")
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

    print(f"  Overlap region bounds:")
    print(f"    V0: X[{x0_start}:{x0_end}], Z[{z0_start}:{z0_end}]")
    print(f"    V1: X[{x1_start}:{x1_end}], Z[{z1_start}:{z1_end}]")
    print(f"    Overlap size: Y={Y}, X={x0_end-x0_start}, Z={z0_end-z0_start}")

    # Extract overlap regions
    overlap_v0 = volume_0[:, x0_start:x0_end, z0_start:z0_end].copy()
    overlap_v1 = volume_1_xz_aligned[:, x1_start:x1_end, z1_start:z1_end].copy()

    print(f"  Overlap V0: {overlap_v0.shape}")
    print(f"  Overlap V1: {overlap_v1.shape}")

    # Find pinpoints in overlap regions
    print("\n4. Finding pinpoints in overlap regions...")
    print("Volume 0:")
    y_pin_v0, x_pin_v0, z_pin_v0 = find_surface_pinpoint(overlap_v0)

    print("\nVolume 1:")
    y_pin_v1, x_pin_v1, z_pin_v1 = find_surface_pinpoint(overlap_v1)

    # Y alignment (vertical) - match pinpoints
    print("\n5. Y alignment (vertical pinpoint matching)...")
    y_shift = y_pin_v0 - y_pin_v1

    print(f"  V0 pinpoint Y: {y_pin_v0}")
    print(f"  V1 pinpoint Y: {y_pin_v1}")
    print(f"  Y shift needed: {y_shift:+.2f} pixels")

    # Apply Y shift to overlap V1
    overlap_v1_y_aligned = ndimage.shift(
        overlap_v1, shift=(y_shift, 0, 0),
        order=1, mode='constant', cval=0
    )

    # Get slices at pinpoint locations
    print("\n6. Extracting slices at pinpoint locations...")

    # BEFORE Y alignment
    bscan_v0_before = overlap_v0[:, x_pin_v0, :].copy()
    bscan_v1_before = overlap_v1[:, x_pin_v1, :].copy()  # Use V1's pinpoint

    yslice_v0_before = overlap_v0[y_pin_v0, :, :].copy()
    yslice_v1_before = overlap_v1[y_pin_v1, :, :].copy()  # Use V1's pinpoint

    # AFTER Y alignment - use V0's pinpoint for both
    bscan_v0_after = overlap_v0[:, x_pin_v0, :].copy()
    bscan_v1_after = overlap_v1_y_aligned[:, x_pin_v0, :].copy()  # Same X as V0

    yslice_v0_after = overlap_v0[y_pin_v0, :, :].copy()
    yslice_v1_after = overlap_v1_y_aligned[y_pin_v0, :, :].copy()  # Same Y as V0

    print(f"  B-scan shapes: {bscan_v0_before.shape}")
    print(f"  Y-slice shapes: {yslice_v0_before.shape}")

    # Statistics
    print("\n7. Data statistics at pinpoint:")
    print(f"  V0 B-scan: mean={bscan_v0_before.mean():.1f}, max={bscan_v0_before.max():.1f}, nonzero={np.count_nonzero(bscan_v0_before)}/{bscan_v0_before.size}")
    print(f"  V1 B-scan (before Y): mean={bscan_v1_before.mean():.1f}, max={bscan_v1_before.max():.1f}, nonzero={np.count_nonzero(bscan_v1_before)}/{bscan_v1_before.size}")
    print(f"  V1 B-scan (after Y): mean={bscan_v1_after.mean():.1f}, max={bscan_v1_after.max():.1f}, nonzero={np.count_nonzero(bscan_v1_after)}/{bscan_v1_after.size}")

    print(f"\n  V0 Y-slice: mean={yslice_v0_before.mean():.1f}, max={yslice_v0_before.max():.1f}, nonzero={np.count_nonzero(yslice_v0_before)}/{yslice_v0_before.size}")
    print(f"  V1 Y-slice (before Y): mean={yslice_v1_before.mean():.1f}, max={yslice_v1_before.max():.1f}, nonzero={np.count_nonzero(yslice_v1_before)}/{yslice_v1_before.size}")
    print(f"  V1 Y-slice (after Y): mean={yslice_v1_after.mean():.1f}, max={yslice_v1_after.max():.1f}, nonzero={np.count_nonzero(yslice_v1_after)}/{yslice_v1_after.size}")

    # Visualization
    print("\n8. Creating visualization...")
    fig = plt.figure(figsize=(24, 16))

    # Top section: B-scans (Y-Z plane) at pinpoint X
    # Row 1: BEFORE Y alignment
    ax1 = plt.subplot(4, 3, 1)
    ax1.imshow(bscan_v0_before, cmap='Reds', aspect='auto', vmin=0, vmax=255)
    ax1.axhline(y=y_pin_v0, color='yellow', linestyle='--', linewidth=2, label='Pinpoint')
    ax1.set_title(f'V0 B-scan @ Pinpoint X={x_pin_v0}\nPinpoint Y={y_pin_v0}', fontweight='bold')
    ax1.set_ylabel('Y (depth)')
    ax1.set_xlabel('Z (B-scans)')
    ax1.legend(loc='upper right')

    ax2 = plt.subplot(4, 3, 2)
    ax2.imshow(bscan_v1_before, cmap='Greens', aspect='auto', vmin=0, vmax=255)
    ax2.axhline(y=y_pin_v1, color='yellow', linestyle='--', linewidth=2, label='Pinpoint')
    ax2.set_title(f'V1 B-scan @ Pinpoint X={x_pin_v1}\nPinpoint Y={y_pin_v1} (BEFORE Y shift)', fontweight='bold')
    ax2.set_xlabel('Z (B-scans)')
    ax2.legend(loc='upper right')

    ax3 = plt.subplot(4, 3, 3)
    ax3.imshow(bscan_v0_before, cmap='Reds', alpha=0.5, aspect='auto')
    ax3.imshow(bscan_v1_before, cmap='Greens', alpha=0.5, aspect='auto')
    ax3.axhline(y=y_pin_v0, color='red', linestyle='--', linewidth=2, label=f'V0 pin Y={y_pin_v0}')
    ax3.axhline(y=y_pin_v1, color='green', linestyle='--', linewidth=2, label=f'V1 pin Y={y_pin_v1}')
    ax3.set_title(f'Overlay BEFORE Y shift\nΔY = {y_shift:+.0f}px', fontweight='bold')
    ax3.set_xlabel('Z (B-scans)')
    ax3.legend(loc='upper right')

    # Row 2: AFTER Y alignment
    ax4 = plt.subplot(4, 3, 4)
    ax4.imshow(bscan_v0_after, cmap='Reds', aspect='auto', vmin=0, vmax=255)
    ax4.axhline(y=y_pin_v0, color='yellow', linestyle='--', linewidth=2, label='Pinpoint')
    ax4.set_title(f'V0 B-scan @ X={x_pin_v0}\nPinpoint Y={y_pin_v0}', fontweight='bold')
    ax4.set_ylabel('Y (depth)')
    ax4.set_xlabel('Z (B-scans)')
    ax4.legend(loc='upper right')

    ax5 = plt.subplot(4, 3, 5)
    ax5.imshow(bscan_v1_after, cmap='Greens', aspect='auto', vmin=0, vmax=255)
    ax5.axhline(y=y_pin_v0, color='yellow', linestyle='--', linewidth=2, label='Pinpoint')
    ax5.set_title(f'V1 B-scan @ X={x_pin_v0}\nAFTER Y shift {y_shift:+.0f}px', fontweight='bold')
    ax5.set_xlabel('Z (B-scans)')
    ax5.legend(loc='upper right')

    ax6 = plt.subplot(4, 3, 6)
    ax6.imshow(bscan_v0_after, cmap='Reds', alpha=0.5, aspect='auto')
    ax6.imshow(bscan_v1_after, cmap='Greens', alpha=0.5, aspect='auto')
    ax6.axhline(y=y_pin_v0, color='yellow', linestyle='--', linewidth=2, label=f'Pinpoints aligned')
    ax6.set_title('Overlay AFTER Y shift\nPinpoints matched!', fontweight='bold')
    ax6.set_xlabel('Z (B-scans)')
    ax6.legend(loc='upper right')

    # Bottom section: Y-slices (X-Z plane) at pinpoint Y
    # Row 3: BEFORE Y alignment
    ax7 = plt.subplot(4, 3, 7)
    ax7.imshow(yslice_v0_before, cmap='Reds', aspect='auto', vmin=0, vmax=255)
    ax7.axvline(x=x_pin_v0, color='yellow', linestyle='--', linewidth=2, label='Pinpoint')
    ax7.set_title(f'V0 Y-slice @ Pinpoint Y={y_pin_v0}\nPinpoint X={x_pin_v0}', fontweight='bold')
    ax7.set_ylabel('Z (B-scans)')
    ax7.set_xlabel('X (lateral)')
    ax7.legend(loc='upper right')

    ax8 = plt.subplot(4, 3, 8)
    ax8.imshow(yslice_v1_before, cmap='Greens', aspect='auto', vmin=0, vmax=255)
    ax8.axvline(x=x_pin_v1, color='yellow', linestyle='--', linewidth=2, label='Pinpoint')
    ax8.set_title(f'V1 Y-slice @ Y={y_pin_v1} (BEFORE Y shift)\nPinpoint X={x_pin_v1}', fontweight='bold')
    ax8.set_xlabel('X (lateral)')
    ax8.legend(loc='upper right')

    ax9 = plt.subplot(4, 3, 9)
    ax9.imshow(yslice_v0_before, cmap='Reds', alpha=0.5, aspect='auto')
    ax9.imshow(yslice_v1_before, cmap='Greens', alpha=0.5, aspect='auto')
    ax9.set_title('Overlay BEFORE Y shift', fontweight='bold')
    ax9.set_xlabel('X (lateral)')

    # Row 4: AFTER Y alignment
    ax10 = plt.subplot(4, 3, 10)
    ax10.imshow(yslice_v0_after, cmap='Reds', aspect='auto', vmin=0, vmax=255)
    ax10.axvline(x=x_pin_v0, color='yellow', linestyle='--', linewidth=2, label='Pinpoint')
    ax10.set_title(f'V0 Y-slice @ Y={y_pin_v0}\nPinpoint X={x_pin_v0}', fontweight='bold')
    ax10.set_ylabel('Z (B-scans)')
    ax10.set_xlabel('X (lateral)')
    ax10.legend(loc='upper right')

    ax11 = plt.subplot(4, 3, 11)
    ax11.imshow(yslice_v1_after, cmap='Greens', aspect='auto', vmin=0, vmax=255)
    ax11.axvline(x=x_pin_v0, color='yellow', linestyle='--', linewidth=2, label='Pinpoint')
    ax11.set_title(f'V1 Y-slice @ Y={y_pin_v0} (AFTER Y shift)\nSame Y as V0', fontweight='bold')
    ax11.set_xlabel('X (lateral)')
    ax11.legend(loc='upper right')

    ax12 = plt.subplot(4, 3, 12)
    ax12.imshow(yslice_v0_after, cmap='Reds', alpha=0.5, aspect='auto')
    ax12.imshow(yslice_v1_after, cmap='Greens', alpha=0.5, aspect='auto')
    ax12.axvline(x=x_pin_v0, color='yellow', linestyle='--', linewidth=2, label='Pinpoint')
    ax12.set_title('Overlay AFTER Y shift\nSame depth!', fontweight='bold')
    ax12.set_xlabel('X (lateral)')
    ax12.legend(loc='upper right')

    plt.suptitle(f'Step 2: XZ (ΔX={offset_x}, ΔZ={offset_z}) + Y Alignment (ΔY={y_shift:+.0f})',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()

    output_path = data_dir / 'step2_xz_plus_y_alignment.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.show()

    # Save aligned overlap region
    np.save(data_dir / 'step2_overlap_v0.npy', overlap_v0)
    np.save(data_dir / 'step2_overlap_v1_y_aligned.npy', overlap_v1_y_aligned)

    # Save parameters
    params = {
        'xz_offset': (offset_x, offset_z),
        'y_shift': float(y_shift),
        'pinpoint_v0': (int(y_pin_v0), int(x_pin_v0), int(z_pin_v0)),
        'pinpoint_v1': (int(y_pin_v1), int(x_pin_v1), int(z_pin_v1)),
        'overlap_bounds_v0': {'x': (x0_start, x0_end), 'z': (z0_start, z0_end)},
        'overlap_bounds_v1': {'x': (x1_start, x1_end), 'z': (z1_start, z1_end)}
    }
    np.save(data_dir / 'step2_alignment_params.npy', params, allow_pickle=True)
    print("✓ Saved: step2_overlap_v0.npy")
    print("✓ Saved: step2_overlap_v1_y_aligned.npy")
    print("✓ Saved: step2_alignment_params.npy")

    print("\n" + "="*70)
    print("✅ Step 2 complete!")
    print("="*70)
    print(f"  XZ offset: ({offset_x}, {offset_z})")
    print(f"  Y shift: {y_shift:+.2f}")
    print(f"  Pinpoints: V0@(Y={y_pin_v0},X={x_pin_v0},Z={z_pin_v0})")
    print(f"             V1@(Y={y_pin_v1},X={x_pin_v1},Z={z_pin_v1})")
    print("="*70)


if __name__ == '__main__':
    main()
