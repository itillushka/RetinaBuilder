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
from helpers.mip_generation import (
    create_vessel_enhanced_mip,
    register_mip_phase_correlation,
    register_mip_phase_correlation_legacy,
    register_mip_elastix_cli,
    register_mip_ants,
    register_mip_feature_based,
    compare_registration_methods
)
from helpers.rotation_alignment_parallel import shift_volume_xz_parallel


def perform_xz_alignment(ref_volume, mov_volume, max_offset_x=None, max_offset_z=None, method='multiscale', output_dir=None, denoise=False, vessels_only=False, vessel_threshold=0.1):
    """
    XZ-alignment wrapper for multi-volume stitcher compatibility.

    Simplified version that works directly without requiring data_dir.
    MIPs are computed on-the-fly without caching.

    Args:
        ref_volume: Reference volume (Y, X, Z)
        mov_volume: Volume to align (Y, X, Z)
        max_offset_x: Maximum allowed X-axis offset (±pixels). None = no limit.
        max_offset_z: Maximum allowed Z-axis offset (±pixels). None = no limit.
        method: Registration method:
            - 'multiscale': Multi-scale phase correlation (default, fast)
            - 'ants': ANTs translation (robust for large displacements)
            - 'sift': SIFT feature matching (accurate, medical images)
            - 'orb': ORB feature matching (fastest)
            - 'akaze': AKAZE feature matching (good compromise)
            - 'compare': Run all methods and generate comparison chart
            - 'elastix': Elastix CLI (currently failing)
            - 'phase_corr': Legacy single-scale phase correlation
        output_dir: Output directory for saving logs and comparison charts
        denoise: Apply bilateral denoising after Frangi filter (default: False)
        vessels_only: Use only vessel structures, suppress background (default: False)
        vessel_threshold: Threshold for vessels_only mode (0-1, default: 0.1)

    Returns:
        dict containing:
            - 'volume_1_xz_aligned': XZ-aligned moving volume
            - 'offset_x', 'offset_z': Translation offsets
            - 'confidence': Registration confidence
            - 'method': Actual method used (may differ if fallback occurred)
    """
    # Create vessel-enhanced MIPs on-the-fly (with optional denoising/thresholding)
    mip_v0 = create_vessel_enhanced_mip(ref_volume, verbose=False, denoise=denoise,
                                        vessels_only=vessels_only, vessel_threshold=vessel_threshold)
    mip_v1 = create_vessel_enhanced_mip(mov_volume, verbose=False, denoise=denoise,
                                        vessels_only=vessels_only, vessel_threshold=vessel_threshold)

    # Run registration with selected method
    if method == 'compare':
        # Comparison mode: run all methods and generate chart
        print("  Running COMPARISON MODE - testing all methods...")
        if output_dir:
            from pathlib import Path
            comparison_path = Path(output_dir) / 'registration_method_comparison.png'
        else:
            comparison_path = 'registration_method_comparison.png'

        results = compare_registration_methods(mip_v0, mip_v1, output_path=str(comparison_path))

        # Use the best method (highest confidence among successful methods)
        successful = {k: v for k, v in results.items() if v['success']}
        if not successful:
            raise RuntimeError("All registration methods failed in comparison mode!")

        best_method = max(successful, key=lambda k: successful[k]['confidence'])
        print(f"\n[COMPARISON] Best method: {best_method.upper()} "
              f"(confidence: {successful[best_method]['confidence']:.3f})")

        offset_x = successful[best_method]['offset_x']
        offset_z = successful[best_method]['offset_z']
        confidence = successful[best_method]['confidence']
        method = best_method  # Update method to actual method used

    elif method == 'ants':
        try:
            print("  Using ANTs Translation Registration (robust, handles large displacements)...")
            (offset_x, offset_z), confidence, correlation_map = register_mip_ants(
                mip_v0, mip_v1,
                max_offset_x=max_offset_x,
                max_offset_z=max_offset_z
            )
        except (ImportError, RuntimeError, Exception) as e:
            print(f"  [WARNING] ANTs failed ({e}), falling back to SIFT...")
            method = 'sift'

    if method in ['sift', 'orb', 'akaze']:
        try:
            print(f"  Using {method.upper()} Feature-Based Registration...")
            (offset_x, offset_z), confidence, correlation_map = register_mip_feature_based(
                mip_v0, mip_v1,
                max_offset_x=max_offset_x,
                max_offset_z=max_offset_z,
                method=method,
                min_matches=5  # Lower threshold for vessel-enhanced MIPs
            )
        except (RuntimeError, Exception) as e:
            print(f"  [WARNING] {method.upper()} failed ({e}), falling back to multiscale...")
            method = 'multiscale'

    if method == 'elastix':
        try:
            print("  Using Elastix Command-Line Tool (production-grade, most reliable)...")
            (offset_x, offset_z), confidence, correlation_map = register_mip_elastix_cli(
                mip_v0, mip_v1,
                max_offset_x=max_offset_x,
                max_offset_z=max_offset_z,
                output_dir=output_dir
            )
        except (FileNotFoundError, RuntimeError, Exception) as e:
            print(f"  [WARNING] Elastix-CLI failed ({e}), falling back to multiscale...")
            method = 'multiscale'

    if method == 'phase_corr':
        print("  Using Legacy Phase Correlation (single-scale FFT)...")
        (offset_x, offset_z), confidence, correlation_map = register_mip_phase_correlation_legacy(
            mip_v0, mip_v1,
            max_offset_x=max_offset_x,
            max_offset_z=max_offset_z
        )
    elif method == 'multiscale':
        print("  Using Multi-Scale Phase Correlation (modern, fast, robust)...")
        (offset_x, offset_z), confidence, correlation_map = register_mip_phase_correlation(
            mip_v0, mip_v1,
            max_offset_x=max_offset_x,
            max_offset_z=max_offset_z
        )

    # Apply shift using PARALLEL method
    volume_1_xz_aligned = shift_volume_xz_parallel(
        mov_volume, offset_x, offset_z, n_jobs=-1
    )

    return {
        'volume_1_xz_aligned': volume_1_xz_aligned,
        'offset_x': offset_x,
        'offset_z': offset_z,
        'confidence': confidence,
        'method': method
    }


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
        mip_v0 = create_vessel_enhanced_mip(volume_0, verbose=True, denoise=False)
        mip_v1 = create_vessel_enhanced_mip(volume_1, verbose=True, denoise=False)

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

    # Apply shift using PARALLEL method
    print("  Applying XZ shift (PARALLEL)...")
    volume_1_xz_aligned = shift_volume_xz_parallel(
        volume_1, offset_x, offset_z, n_jobs=-1
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
