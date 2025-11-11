#!/usr/bin/env python3
"""
Demonstration of 3 Possible Rotation Axes for OCT Volumes

Shows a B-scan rotated in 3 different ways to identify the correct rotation axis.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from oct_volumetric_viewer import OCTImageProcessor, OCTVolumeLoader


def demonstrate_rotation_axes():
    """Show a B-scan rotated in 3 different ways."""

    print("="*70)
    print("ROTATION AXIS DEMONSTRATION")
    print("="*70)

    # Load data
    data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'
    oct_data_dir = Path(__file__).parent.parent / 'oct_data'

    # Load one volume
    processor = OCTImageProcessor(sidebar_width=250, crop_top=100, crop_bottom=50)
    loader = OCTVolumeLoader(processor)

    print("\nLoading OCT volume...")
    bmp_dirs = []
    for bmp_file in oct_data_dir.rglob('*.bmp'):
        vol_dir = bmp_file.parent
        if vol_dir not in bmp_dirs:
            bmp_dirs.append(vol_dir)

    f001_vols = sorted([v for v in bmp_dirs if 'F001_IP' in str(v)])

    if len(f001_vols) < 1:
        print("Error: No F001 volumes found!")
        return

    print(f"  Loading from: {f001_vols[0].name}")
    volume = loader.load_volume_from_directory(str(f001_vols[0]))

    print(f"  Volume shape: {volume.shape} (Y, X, Z)")
    print(f"    Y = depth (vitreous to choroid)")
    print(f"    X = lateral width")
    print(f"    Z = B-scan index")

    # Extract one B-scan from the middle
    z_middle = volume.shape[2] // 2
    bscan_original = volume[:, :, z_middle].copy()

    print(f"\n  Extracted B-scan at Z={z_middle}")
    print(f"  B-scan shape: {bscan_original.shape} (Y, X)")

    # Apply test rotation angle
    test_angle = 10.0
    print(f"\n  Applying {test_angle}° rotation in 3 different ways...")

    # Create a 3D mini-volume with just this B-scan for rotation
    # We need 3D for scipy.ndimage.rotate
    mini_volume = volume[:, :, z_middle-2:z_middle+3].copy()  # 5 B-scans centered

    print(f"  Mini volume shape: {mini_volume.shape}")

    # Rotation 1: Current implementation (axes=(1,2) - X and Z)
    # Rotates in en-face plane (around Y-axis / depth axis)
    print("\n  1. Rotating axes=(1, 2): X and Z axes")
    print("     → Rotates in en-face plane (around depth axis)")
    rotated_1 = ndimage.rotate(mini_volume, test_angle, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0)
    bscan_rot1 = rotated_1[:, :, 2]  # Get middle B-scan

    # Rotation 2: Alternative (axes=(0,2) - Y and Z)
    # Rotates B-scan plane (around X-axis / lateral axis) - TILTS B-SCANS
    print("  2. Rotating axes=(0, 2): Y and Z axes")
    print("     → Tilts B-scans (rotation around lateral axis)")
    rotated_2 = ndimage.rotate(mini_volume, test_angle, axes=(0, 2), reshape=False, order=1, mode='constant', cval=0)
    bscan_rot2 = rotated_2[:, :, 2]

    # Rotation 3: Alternative (axes=(0,1) - Y and X)
    # Rotates within B-scan (around Z-axis / B-scan direction axis) - ROTATES LAYERS
    print("  3. Rotating axes=(0, 1): Y and X axes")
    print("     → Rotates layers within B-scan (around B-scan direction axis)")
    rotated_3 = ndimage.rotate(mini_volume, test_angle, axes=(0, 1), reshape=False, order=1, mode='constant', cval=0)
    bscan_rot3 = rotated_3[:, :, 2]

    # Create visualization
    print("\n  Creating comparison visualization...")

    fig = plt.figure(figsize=(24, 16))

    # Original
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(bscan_original, cmap='gray', aspect='auto')
    ax1.set_title('ORIGINAL B-scan\n(No rotation)', fontsize=14, fontweight='bold', color='blue')
    ax1.set_xlabel('X (lateral)', fontsize=12)
    ax1.set_ylabel('Y (depth)', fontsize=12)
    ax1.axhline(y=bscan_original.shape[0]//2, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax1.axvline(x=bscan_original.shape[1]//2, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Rotation 1: axes=(1,2)
    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(bscan_rot1, cmap='gray', aspect='auto')
    ax2.set_title(f'OPTION 1: axes=(1,2) → Rotates X & Z\n'
                  f'(En-face plane rotation, around Y-axis)\n'
                  f'Rotated {test_angle}°',
                  fontsize=14, fontweight='bold', color='green')
    ax2.set_xlabel('X (lateral)', fontsize=12)
    ax2.set_ylabel('Y (depth)', fontsize=12)
    ax2.axhline(y=bscan_rot1.shape[0]//2, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax2.axvline(x=bscan_rot1.shape[1]//2, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Rotation 2: axes=(0,2)
    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(bscan_rot2, cmap='gray', aspect='auto')
    ax3.set_title(f'OPTION 2: axes=(0,2) → Rotates Y & Z\n'
                  f'(B-scan tilt, around X-axis)\n'
                  f'Rotated {test_angle}°',
                  fontsize=14, fontweight='bold', color='orange')
    ax3.set_xlabel('X (lateral)', fontsize=12)
    ax3.set_ylabel('Y (depth)', fontsize=12)
    ax3.axhline(y=bscan_rot2.shape[0]//2, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax3.axvline(x=bscan_rot2.shape[1]//2, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Rotation 3: axes=(0,1)
    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(bscan_rot3, cmap='gray', aspect='auto')
    ax4.set_title(f'OPTION 3: axes=(0,1) → Rotates Y & X\n'
                  f'(Layer rotation, around Z-axis)\n'
                  f'Rotated {test_angle}°',
                  fontsize=14, fontweight='bold', color='purple')
    ax4.set_xlabel('X (lateral)', fontsize=12)
    ax4.set_ylabel('Y (depth)', fontsize=12)
    ax4.axhline(y=bscan_rot3.shape[0]//2, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax4.axvline(x=bscan_rot3.shape[1]//2, color='red', linestyle='--', alpha=0.3, linewidth=1)

    plt.suptitle(f'OCT B-scan Rotation Demonstration: Which Rotation Do You Need?\n'
                 f'(All rotated by {test_angle}° to show the difference)',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_path = data_dir / 'rotation_axis_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved: {output_path}")
    plt.close()

    # Print explanation
    print("\n" + "="*70)
    print("EXPLANATION OF EACH ROTATION")
    print("="*70)

    print("\n📊 OPTION 1: axes=(1,2) - Rotates X & Z")
    print("  Current implementation in rotation_alignment.py")
    print("  What it does:")
    print("    - Rotates in the en-face (horizontal) plane")
    print("    - Like turning your head left/right while looking down")
    print("    - Affects vessel pattern orientation in MIP view")
    print("    - Does NOT tilt the B-scan image itself")
    print("  When to use:")
    print("    - Patient head was rotated during acquisition")
    print("    - Vessel patterns misaligned in horizontal plane")

    print("\n📊 OPTION 2: axes=(0,2) - Rotates Y & Z")
    print("  Alternative: B-scan tilt correction")
    print("  What it does:")
    print("    - Tilts the entire volume forward/backward")
    print("    - Like nodding your head up/down")
    print("    - Makes B-scans slanted")
    print("    - Affects retinal surface angle across B-scans")
    print("  When to use:")
    print("    - Volumes have different tilt angles")
    print("    - One volume appears 'tilted' relative to other")
    print("    - Surface alignment varies systematically across Z")

    print("\n📊 OPTION 3: axes=(0,1) - Rotates Y & X")
    print("  Alternative: Layer rotation within B-scan")
    print("  What it does:")
    print("    - Rotates retinal layers within each B-scan")
    print("    - Like rolling your head side-to-side")
    print("    - Tilts the layers left/right in B-scan view")
    print("    - Most visible in cross-sectional view")
    print("  When to use:")
    print("    - Retinal layers appear tilted in B-scans")
    print("    - One volume's layers slanted vs reference")
    print("    - Less common for OCT registration")

    print("\n" + "="*70)
    print("WHICH ONE IS CORRECT?")
    print("="*70)
    print("\nLook at the output image: rotation_axis_demo.png")
    print("Compare the 4 panels and identify which rotation matches")
    print("the misalignment you see in your actual data.")
    print("\nThen tell me: 1, 2, or 3?")
    print("="*70)


if __name__ == '__main__':
    demonstrate_rotation_axes()
