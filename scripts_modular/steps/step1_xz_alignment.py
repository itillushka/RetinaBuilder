"""
Step 1: XZ Plane Alignment

Uses vessel-enhanced MIP phase correlation to align volumes in the XZ plane.
"""

import numpy as np
from scipy import ndimage
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers.mip_generation import create_vessel_enhanced_mip, register_mip_phase_correlation


def step1_xz_alignment(volume_0, volume_1, data_dir):
    """
    Step 1: XZ plane alignment using Vessel-Enhanced phase correlation.

    Loads pre-computed vessel MIPs if available, otherwise creates them.

    Args:
        volume_0: Reference volume (Y, X, Z)
        volume_1: Volume to align (Y, X, Z)
        data_dir: Directory for saving results

    Returns:
        results: Dictionary containing:
            - volume_1_xz_aligned: Aligned volume
            - overlap_v0, overlap_v1: Overlap regions
            - offset_x, offset_z: Translation offsets
            - confidence: Registration confidence
            - overlap_bounds: Overlap region coordinates
            - vessel_mip_v0, vessel_mip_v1: Vessel-enhanced MIPs
    """
    print("\n" + "="*70)
    print("STEP 1: XZ ALIGNMENT (VESSEL-ENHANCED)")
    print("="*70)

    # Try to load pre-computed MIPs, fall back to creating them
    mip_v0_path = data_dir / 'enface_mip_vessels_volume0.npy'
    mip_v1_path = data_dir / 'enface_mip_vessels_volume1.npy'

    if mip_v0_path.exists() and mip_v1_path.exists():
        print("  Loading pre-computed Vessel-Enhanced MIPs...")
        mip_v0 = np.load(mip_v0_path)
        mip_v1 = np.load(mip_v1_path)
    else:
        print("  Pre-computed MIPs not found. Creating Vessel-Enhanced MIPs (Frangi filter)...")
        print("  This may take 2-3 minutes per volume...")
        mip_v0 = create_vessel_enhanced_mip(volume_0)
        mip_v1 = create_vessel_enhanced_mip(volume_1)

        # Save for future use
        print("  Saving MIPs for future use...")
        np.save(mip_v0_path, mip_v0)
        np.save(mip_v1_path, mip_v1)
        print(f"  [OK] Saved: {mip_v0_path.name}")
        print(f"  [OK] Saved: {mip_v1_path.name}")

    print(f"  Vessel MIP V0: shape={mip_v0.shape}, mean={mip_v0.mean():.1f}, std={mip_v0.std():.1f}")
    print(f"  Vessel MIP V1: shape={mip_v1.shape}, mean={mip_v1.mean():.1f}, std={mip_v1.std():.1f}")

    print("  Running phase correlation on vessel maps...")
    (offset_x, offset_z), confidence, correlation_map = register_mip_phase_correlation(mip_v0, mip_v1)

    print(f"\n  Results:")
    print(f"    Offset X: {offset_x} pixels")
    print(f"    Offset Z: {offset_z} pixels")
    print(f"    Confidence: {confidence:.2f}")

    # Apply shift
    print("  Applying XZ shift...")
    volume_1_xz_aligned = ndimage.shift(
        volume_1, shift=(0, offset_x, offset_z),
        order=1, mode='constant', cval=0
    )

    # Calculate overlap bounds
    # IMPORTANT: After shifting, both volumes are in the SAME coordinate system!
    # We extract the SAME region from both, avoiding zero-padding from the shift.
    Y, X, Z = volume_0.shape

    # Determine valid region (where both volumes have non-padded data)
    if offset_x >= 0:
        # Volume_1 shifted RIGHT -> zeros on left side of volume_1_xz_aligned
        x_start, x_end = offset_x, X
    else:
        # Volume_1 shifted LEFT -> zeros on right side of volume_1_xz_aligned
        x_start, x_end = 0, X + offset_x

    if offset_z >= 0:
        # Volume_1 shifted FORWARD -> zeros on front of volume_1_xz_aligned
        z_start, z_end = offset_z, Z
    else:
        # Volume_1 shifted BACKWARD -> zeros on back of volume_1_xz_aligned
        z_start, z_end = 0, Z + offset_z

    # Extract SAME region from both volumes (both are now in same coordinate system!)
    overlap_bounds = {
        'x': (x_start, x_end),
        'z': (z_start, z_end),
        'size': (Y, x_end - x_start, z_end - z_start)
    }

    print(f"\n  Overlap region (same for both volumes):")
    print(f"    X[{x_start}:{x_end}], Z[{z_start}:{z_end}]")
    print(f"    Size: {overlap_bounds['size']}")

    # Extract overlap regions - SAME indices for both!
    overlap_v0 = volume_0[:, x_start:x_end, z_start:z_end].copy()
    overlap_v1 = volume_1_xz_aligned[:, x_start:x_end, z_start:z_end].copy()

    # Save
    results = {
        'volume_1_xz_aligned': volume_1_xz_aligned,
        'overlap_v0': overlap_v0,
        'overlap_v1': overlap_v1,
        'offset_x': offset_x,
        'offset_z': offset_z,
        'confidence': confidence,
        'overlap_bounds': overlap_bounds,
        'vessel_mip_v0': mip_v0,
        'vessel_mip_v1': mip_v1
    }

    print("\n[OK] Step 1 complete!")
    print("="*70)

    return results
