#!/usr/bin/env python3
"""
Visualize All Rotation Angles - Debug Tool

Takes the central B-scan from overlap region and shows it rotated at all test angles.
NO NCC calculation - pure visual inspection to identify the correct rotation.

Usage:
    python visualize_all_rotations.py
    python visualize_all_rotations.py --angle-range 20 --step 2
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

print("="*70)
print("ROTATION ANGLE VISUALIZATION - DEBUG TOOL")
print("="*70)


def visualize_all_rotations(angle_range=30, step=2):
    """
    Show central B-scan rotated at all test angles.

    Args:
        angle_range: ±angle range (degrees)
        step: Angle step size (degrees)
    """
    # Setup paths
    data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'

    # Load step results
    print("\n1. Loading alignment results...")
    step1_file = data_dir / 'step1_results.npy'
    step2_file = data_dir / 'step2_results.npy'

    if not step1_file.exists() or not step2_file.exists():
        print("❌ Error: Need to run Steps 1 & 2 first!")
        print("   Run: python alignment_pipeline.py --steps 1 2")
        return

    step1_results = np.load(step1_file, allow_pickle=True).item()
    step2_results = np.load(step2_file, allow_pickle=True).item()

    overlap_v0 = step1_results['overlap_v0']
    overlap_v1_y_aligned = step2_results['overlap_v1_y_aligned']

    print(f"  ✓ Overlap V0: {overlap_v0.shape}")
    print(f"  ✓ Overlap V1 (Y-aligned): {overlap_v1_y_aligned.shape}")

    # Extract central B-scan
    z_center = overlap_v0.shape[2] // 2
    bscan_v0 = overlap_v0[:, :, z_center]
    bscan_v1 = overlap_v1_y_aligned[:, :, z_center]

    print(f"\n2. Extracted central B-scan: Z={z_center}/{overlap_v0.shape[2]}")
    print(f"   B-scan shape: {bscan_v0.shape} (Y, X)")

    # Generate rotation angles
    angles_to_test = np.arange(-angle_range, angle_range + step, step)
    print(f"\n3. Testing {len(angles_to_test)} rotation angles:")
    print(f"   Range: {angles_to_test[0]}° to {angles_to_test[-1]}°")
    print(f"   Step: {step}°")

    # Calculate grid layout
    n_angles = len(angles_to_test)
    n_cols = 6  # 6 columns
    n_rows = int(np.ceil(n_angles / n_cols))

    print(f"\n4. Creating visualization grid: {n_rows} rows × {n_cols} cols")

    # Create figure
    fig = plt.figure(figsize=(24, 4*n_rows))

    for idx, angle in enumerate(angles_to_test, 1):
        # Rotate B-scan
        bscan_v1_rotated = ndimage.rotate(
            bscan_v1,
            angle,
            axes=(0, 1),  # Rotate Y and X axes
            reshape=False,
            order=1,
            mode='constant',
            cval=0
        )

        # Create overlay
        ax = plt.subplot(n_rows, n_cols, idx)

        # Show as red/green overlay
        ax.imshow(bscan_v0, cmap='Reds', alpha=0.5, aspect='auto', vmin=0, vmax=255)
        ax.imshow(bscan_v1_rotated, cmap='Greens', alpha=0.5, aspect='auto', vmin=0, vmax=255)

        # Title with angle
        if angle == 0:
            ax.set_title(f'{angle:+.1f}°\n(NO ROTATION)',
                        fontsize=10, fontweight='bold', color='blue')
        else:
            ax.set_title(f'{angle:+.1f}°', fontsize=10, fontweight='bold')

        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.tick_params(labelsize=6)

        # Add grid lines
        ax.axhline(y=bscan_v0.shape[0]//2, color='yellow', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.axvline(x=bscan_v0.shape[1]//2, color='yellow', linestyle='--', alpha=0.3, linewidth=0.5)

    plt.suptitle(f'Central B-scan Rotation Visualization (Z={z_center})\n'
                 f'Red = Volume 0 (reference), Green = Volume 1 (rotated)\n'
                 f'Find the angle where RED and GREEN align best',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = data_dir / 'rotation_angles_all.png'
    print(f"\n5. Saving visualization...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")
    plt.close()

    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nOpen the image to visually identify the best rotation angle:")
    print(f"  {output_path}")
    print("\nLook for the angle where:")
    print("  - Retinal layers align (horizontal bands)")
    print("  - Red and Green overlap creates yellow (good alignment)")
    print("  - Minimal color separation")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize all rotation angles for debugging',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--angle-range', type=int, default=30,
                        help='Angle range to test (default: 30 degrees)')
    parser.add_argument('--step', type=int, default=2,
                        help='Angle step size (default: 2 degrees)')

    args = parser.parse_args()

    visualize_all_rotations(
        angle_range=args.angle_range,
        step=args.step
    )


if __name__ == '__main__':
    main()
