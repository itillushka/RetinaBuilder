#!/usr/bin/env python3
"""
Central B-scans Alignment for Retinal Curvature Estimation

Extracts central N B-scans from two OCT volumes and applies
full alignment pipeline to estimate retinal curvature without
artifacts from peripheral scans.

Fast, focused approach for validating alignment methods and
estimating retinal curvature from high-quality central data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from scipy.signal import medfilt
import argparse
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

# Import existing modules
from oct_volumetric_viewer import OCTVolumeLoader, OCTImageProcessor
from rotation_alignment import (
    calculate_ncc_3d,
    find_optimal_rotation_z,
    apply_rotation_z,
    find_optimal_rotation_x,
    apply_rotation_x
)
from surface_visualization import load_or_detect_surface


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_central_bscans(volume, n_bscans=20):
    """
    Extract central N B-scans from volume.

    Args:
        volume: (Y, X, Z) array
        n_bscans: Number of central B-scans to extract

    Returns:
        Central subset: (Y, X, n_bscans)
    """
    Z = volume.shape[2]
    center_z = Z // 2
    start_z = center_z - n_bscans // 2
    end_z = start_z + n_bscans

    print(f"  Extracting B-scans {start_z} to {end_z-1} (center at {center_z})")

    return volume[:, :, start_z:end_z].copy()


# ============================================================================
# ALIGNMENT FUNCTIONS
# ============================================================================

def calculate_global_y_shift(vol0, vol1):
    """
    Calculate global Y-shift using center of mass.

    Args:
        vol0: Reference volume (Y, X, Z)
        vol1: Volume to shift (Y, X, Z)

    Returns:
        Y-shift in pixels (positive = vol1 needs to move up)
    """
    y_profile_0 = vol0.sum(axis=(1, 2))
    y_profile_1 = vol1.sum(axis=(1, 2))

    y_coords = np.arange(len(y_profile_0))
    center_0 = np.average(y_coords, weights=y_profile_0 + 1e-8)
    center_1 = np.average(y_coords, weights=y_profile_1 + 1e-8)

    y_shift = center_0 - center_1

    print(f"    Center of mass: V0={center_0:.2f}, V1={center_1:.2f}")
    print(f"    Y-shift: {y_shift:+.2f} px")

    return y_shift


def apply_per_bscan_y_alignment(vol0, vol1, outlier_threshold=50, smooth_window=3, verbose=True):
    """
    Apply per-B-scan Y-alignment using surface matching.

    Args:
        vol0: Reference volume (Y, X, Z)
        vol1: Volume to align (Y, X, Z)
        outlier_threshold: Reject offsets > this value (pixels)
        smooth_window: Median filter window for smoothing offsets
        verbose: Print progress

    Returns:
        aligned_volume: (Y, X, Z) aligned volume
        offsets: (Z,) array of Y-shifts applied to each B-scan
        confidences: (Z,) array of confidence scores [0-1]
    """
    if verbose:
        print("\n  Detecting surfaces for per-B-scan alignment...")

    # Detect surfaces
    surface0 = load_or_detect_surface(vol0, method='peak')  # (X, Z)
    surface1 = load_or_detect_surface(vol1, method='peak')

    Z = vol0.shape[2]
    offsets_raw = np.zeros(Z)
    confidences = np.zeros(Z)

    if verbose:
        print(f"  Calculating offsets for {Z} B-scans...")

    for z in range(Z):
        # Get surface profiles for this B-scan
        profile0 = surface0[:, z]  # (X,) surface Y-positions
        profile1 = surface1[:, z]

        # Calculate height differences
        diff = profile0 - profile1

        # Filter: remove NaN and outliers
        valid = ~np.isnan(diff) & (np.abs(diff) < outlier_threshold)

        if valid.sum() > 10:  # Need at least 10 valid points
            offsets_raw[z] = np.median(diff[valid])
            confidences[z] = valid.sum() / len(diff)
        else:
            offsets_raw[z] = 0
            confidences[z] = 0

    # Smooth offsets to avoid jumps between adjacent B-scans
    if smooth_window > 1:
        offsets_smooth = medfilt(offsets_raw, kernel_size=smooth_window)
        if verbose:
            print(f"  Smoothed offsets with median filter (window={smooth_window})")
    else:
        offsets_smooth = offsets_raw

    if verbose:
        print(f"  Offset range: {offsets_smooth.min():.1f} to {offsets_smooth.max():.1f} px")
        print(f"  Mean offset: {offsets_smooth.mean():.1f} px")
        print(f"  Average confidence: {confidences.mean():.1%}")

    # Apply per-B-scan shifts
    aligned = np.zeros_like(vol1)

    for z in range(Z):
        aligned[:, :, z] = ndimage.shift(
            vol1[:, :, z],
            shift=(offsets_smooth[z], 0),
            order=1,
            mode='constant',
            cval=0
        )

    return aligned, offsets_smooth, confidences


def align_central_volumes(vol0_central, vol1_central, verbose=True):
    """
    Apply full alignment pipeline to central B-scans.

    Steps:
    1. Global Y-shift (center of mass)
    2. Z-axis rotation (B-scan layer alignment)
    3. X-axis rotation (Y-Z plane alignment)
    4. Per-B-scan Y-alignment (surface matching)

    Args:
        vol0_central: Reference volume (Y, X, Z_central)
        vol1_central: Volume to align (Y, X, Z_central)
        verbose: Print progress

    Returns:
        vol1_aligned: Fully aligned volume
        alignment_params: Dict with all alignment parameters
    """

    if verbose:
        print("\n" + "="*70)
        print("ALIGNMENT PIPELINE")
        print("="*70)

    # Step 1: Global Y-shift
    if verbose:
        print("\nStep 1: Global Y-shift (center of mass)")

    y_shift_global = calculate_global_y_shift(vol0_central, vol1_central)
    vol1_y_aligned = ndimage.shift(
        vol1_central,
        shift=(y_shift_global, 0, 0),
        order=1,
        mode='constant',
        cval=0
    )

    ncc_after_y = calculate_ncc_3d(vol0_central, vol1_y_aligned)
    if verbose:
        print(f"    NCC after Y-shift: {ncc_after_y:.4f}")

    # Step 2: Z-axis rotation (B-scan alignment)
    if verbose:
        print("\nStep 2: Z-axis rotation (B-scan layer alignment)")

    rotation_z, metrics_z = find_optimal_rotation_z(
        vol0_central,
        vol1_y_aligned,
        coarse_range=10,
        coarse_step=1,
        fine_range=3,
        fine_step=0.5,
        verbose=False
    )

    vol1_z_rotated = apply_rotation_z(vol1_y_aligned, rotation_z, axes=(0, 1))

    ncc_after_z = calculate_ncc_3d(vol0_central, vol1_z_rotated)
    if verbose:
        print(f"    Rotation angle: {rotation_z:+.2f}°")
        print(f"    Correlation: {metrics_z['optimal_correlation']:.4f}")
        print(f"    NCC after Z-rotation: {ncc_after_z:.4f}")

    # Step 3: X-axis rotation (Y-Z plane alignment)
    if verbose:
        print("\nStep 3: X-axis rotation (Y-Z plane alignment)")

    rotation_x, metrics_x = find_optimal_rotation_x(
        vol0_central,
        vol1_z_rotated,
        coarse_range=10,
        coarse_step=1,
        fine_range=3,
        fine_step=0.5,
        verbose=False
    )

    vol1_x_rotated = apply_rotation_x(vol1_z_rotated, rotation_x, axes=(0, 2))

    ncc_after_x = calculate_ncc_3d(vol0_central, vol1_x_rotated)
    if verbose:
        print(f"    Rotation angle: {rotation_x:+.2f}°")
        print(f"    Correlation: {metrics_x['optimal_correlation']:.4f}")
        print(f"    NCC after X-rotation: {ncc_after_x:.4f}")

    # Step 4: Per-B-scan Y-alignment
    if verbose:
        print("\nStep 4: Per-B-scan Y-alignment (surface matching)")

    vol1_final, per_bscan_offsets, confidences = apply_per_bscan_y_alignment(
        vol0_central,
        vol1_x_rotated,
        outlier_threshold=50,
        smooth_window=3,
        verbose=verbose
    )

    ncc_final = calculate_ncc_3d(vol0_central, vol1_final)
    if verbose:
        print(f"    NCC after per-B-scan: {ncc_final:.4f}")

    # Summary
    if verbose:
        print("\n" + "="*70)
        print("ALIGNMENT SUMMARY")
        print("="*70)
        print(f"  Global Y-shift: {y_shift_global:+.2f} px")
        print(f"  Z-axis rotation: {rotation_z:+.2f}°")
        print(f"  X-axis rotation: {rotation_x:+.2f}°")
        print(f"  Per-B-scan offsets: {per_bscan_offsets.min():.1f} to {per_bscan_offsets.max():.1f} px")
        print(f"  Final NCC: {ncc_final:.4f}")
        print(f"  Total improvement: {ncc_after_y:.4f} → {ncc_final:.4f} ({(ncc_final-ncc_after_y)*100:+.2f}%)")

    alignment_params = {
        'y_shift_global': float(y_shift_global),
        'rotation_z': float(rotation_z),
        'rotation_x': float(rotation_x),
        'per_bscan_offsets': per_bscan_offsets,
        'confidences': confidences,
        'ncc_after_y': float(ncc_after_y),
        'ncc_after_z': float(ncc_after_z),
        'ncc_after_x': float(ncc_after_x),
        'ncc_final': float(ncc_final)
    }

    return vol1_final, alignment_params


# ============================================================================
# CURVATURE ESTIMATION
# ============================================================================

def estimate_retinal_curvature(vol0, vol1_aligned, verbose=True):
    """
    Estimate retinal curvature from aligned central B-scans.

    Args:
        vol0: Reference volume (Y, X, Z)
        vol1_aligned: Aligned volume (Y, X, Z)
        verbose: Print progress

    Returns:
        curvature_profile: (Z,) array of curvature values
        surface_merged: (X, Z) merged surface positions
    """
    if verbose:
        print("\n" + "="*70)
        print("CURVATURE ESTIMATION")
        print("="*70)

    # Detect surfaces
    if verbose:
        print("\n  Detecting surfaces...")

    surface0 = load_or_detect_surface(vol0, method='peak')  # (X, Z)
    surface1 = load_or_detect_surface(vol1_aligned, method='peak')

    # Merge surfaces (average)
    surface_merged = (surface0 + surface1) / 2

    if verbose:
        print(f"  Surface shape: {surface_merged.shape}")
        print(f"  Surface range: Y={surface_merged.min():.1f} to {surface_merged.max():.1f}")

    # Calculate curvature for each B-scan (along X-axis within each B-scan)
    Z = surface_merged.shape[1]
    curvatures = []

    if verbose:
        print(f"\n  Calculating curvature for {Z} B-scans...")

    for z in range(Z):
        profile = surface_merged[:, z]  # (X,) surface heights

        # Fit 2nd order polynomial: y = ax² + bx + c
        x = np.arange(len(profile))

        # Remove NaN values
        valid = ~np.isnan(profile)
        if valid.sum() < 10:
            curvatures.append(0)
            continue

        coeffs = np.polyfit(x[valid], profile[valid], deg=2)
        curvature = coeffs[0]  # Coefficient of x² term (related to curvature)
        curvatures.append(curvature)

    curvature_profile = np.array(curvatures)

    if verbose:
        print(f"\n  Curvature statistics:")
        print(f"    Range: {curvature_profile.min():.6f} to {curvature_profile.max():.6f}")
        print(f"    Mean: {curvature_profile.mean():.6f}")
        print(f"    Std: {curvature_profile.std():.6f}")

    return curvature_profile, surface_merged


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(vol0, vol1_original, vol1_aligned, params,
                      curvature, surface, output_dir):
    """
    Create comprehensive visualization of alignment and curvature results.

    Generates:
    1. Y-axis view comparison (before/after alignment)
    2. Curvature profile
    3. Per-B-scan offset curve
    4. Surface overlay
    """
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # Figure 1: Y-axis view comparison
    print("\n  Creating Y-axis view comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Get central sagittal slice (middle X position)
    x_center = vol0.shape[1] // 2

    # Vol0 reference
    im0 = axes[0].imshow(vol0[:, x_center, :], aspect='auto', cmap='gray', origin='upper')
    axes[0].set_title('Volume 0 (Reference)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('B-scan Position (Z)')
    axes[0].set_ylabel('Depth (Y)')
    plt.colorbar(im0, ax=axes[0])

    # Vol1 before alignment
    im1 = axes[1].imshow(vol1_original[:, x_center, :], aspect='auto', cmap='gray', origin='upper')
    axes[1].set_title('Volume 1 (Before Alignment)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('B-scan Position (Z)')
    axes[1].set_ylabel('Depth (Y)')
    plt.colorbar(im1, ax=axes[1])

    # Vol1 after alignment
    im2 = axes[2].imshow(vol1_aligned[:, x_center, :], aspect='auto', cmap='gray', origin='upper')
    axes[2].set_title('Volume 1 (After Alignment)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('B-scan Position (Z)')
    axes[2].set_ylabel('Depth (Y)')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    output_path = output_dir / 'yaxis_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path.name}")

    # Figure 2: Curvature profile
    print("  Creating curvature profile...")
    plt.figure(figsize=(12, 6))
    z_positions = np.arange(len(curvature))
    plt.plot(z_positions, curvature, 'o-', linewidth=2, markersize=6, color='darkblue')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero curvature')
    plt.xlabel('B-scan Position (Z)', fontsize=12)
    plt.ylabel('Curvature (polynomial coefficient)', fontsize=12)
    plt.title('Retinal Surface Curvature Profile (Central B-scans)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = output_dir / 'curvature_profile.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path.name}")

    # Figure 3: Per-B-scan offsets
    print("  Creating per-B-scan offsets plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(z_positions, params['per_bscan_offsets'], 'o-', linewidth=2, markersize=6, color='darkgreen')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('B-scan Position (Z)', fontsize=12)
    plt.ylabel('Y-offset (pixels)', fontsize=12)
    plt.title('Per-B-scan Y-offsets Applied', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    output_path = output_dir / 'per_bscan_offsets.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path.name}")

    # Figure 4: Surface overlay
    print("  Creating surface overlay...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Surface as 2D image
    im = axes[0].imshow(surface, aspect='auto', cmap='viridis', origin='upper')
    axes[0].set_title('Merged Retinal Surface (2D)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('B-scan Position (Z)')
    axes[0].set_ylabel('Lateral Position (X)')
    plt.colorbar(im, ax=axes[0], label='Depth (Y pixels)')

    # Surface as 3D plot
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(122, projection='3d')
    # surface shape: (X, Z) = (1536, 20)
    Z_grid, X_grid = np.meshgrid(np.arange(surface.shape[1]), np.arange(surface.shape[0]))
    ax.plot_surface(X_grid, Z_grid, surface, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Lateral (X)')
    ax.set_ylabel('B-scan (Z)')
    ax.set_zlabel('Depth (Y)')
    ax.set_title('Merged Retinal Surface (3D)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'surface_overlay.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path.name}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Align central B-scans for retinal curvature estimation'
    )
    parser.add_argument(
        '--n-bscans',
        type=int,
        default=20,
        help='Number of central B-scans to extract (default: 20)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='notebooks/data/central_alignment',
        help='Output directory (default: notebooks/data/central_alignment)'
    )

    args = parser.parse_args()

    # Configuration
    N_BSCANS = args.n_bscans
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / 'notebooks' / 'data'
    oct_data_dir = project_root / 'oct_data'
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CENTRAL B-SCANS ALIGNMENT FOR CURVATURE ESTIMATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Number of central B-scans: {N_BSCANS}")
    print(f"  Output directory: {output_dir}")

    # Load volumes
    print("\n" + "="*70)
    print("LOADING OCT VOLUMES")
    print("="*70)

    processor = OCTImageProcessor(sidebar_width=250, crop_top=100, crop_bottom=50)
    loader = OCTVolumeLoader(processor)

    # Find F001_IP volumes
    print("\nSearching for volumes...")
    bmp_dirs = []
    for bmp_file in oct_data_dir.rglob('*.bmp'):
        vol_dir = bmp_file.parent
        if vol_dir not in bmp_dirs:
            bmp_dirs.append(vol_dir)

    print(f"  Found {len(bmp_dirs)} volume directories total")

    f001_vols = sorted([v for v in bmp_dirs if 'F001_IP' in str(v)])

    print(f"  Found {len(f001_vols)} F001_IP volumes")

    if len(f001_vols) < 2:
        print(f"\n❌ Error: Found only {len(f001_vols)} F001_IP volume(s). Need at least 2.")
        return

    print(f"\nFound {len(f001_vols)} F001_IP volumes:")
    for i, vol_path in enumerate(f001_vols):
        print(f"  {i}: {vol_path.name}")

    print(f"\nLoading volumes 0 and 1...")
    volume_0 = loader.load_volume_from_directory(str(f001_vols[0]))
    volume_1 = loader.load_volume_from_directory(str(f001_vols[1]))

    print(f"  Volume 0: {volume_0.shape} (Y, X, Z)")
    print(f"  Volume 1: {volume_1.shape} (Y, X, Z)")

    # Extract central B-scans
    print("\n" + "="*70)
    print(f"EXTRACTING CENTRAL {N_BSCANS} B-SCANS")
    print("="*70)

    vol0_central = extract_central_bscans(volume_0, N_BSCANS)
    vol1_central = extract_central_bscans(volume_1, N_BSCANS)
    vol1_central_orig = vol1_central.copy()  # Keep original for comparison

    print(f"\n  Central volume shape: {vol0_central.shape}")

    # Align
    vol1_aligned, alignment_params = align_central_volumes(vol0_central, vol1_central, verbose=True)

    # Estimate curvature
    curvature, surface_merged = estimate_retinal_curvature(vol0_central, vol1_aligned, verbose=True)

    # Visualize
    visualize_results(
        vol0_central,
        vol1_central_orig,
        vol1_aligned,
        alignment_params,
        curvature,
        surface_merged,
        output_dir
    )

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    output_file = output_dir / 'central_alignment_results.npz'
    np.savez(
        output_file,
        vol0=vol0_central,
        vol1_original=vol1_central_orig,
        vol1_aligned=vol1_aligned,
        alignment_params_dict=alignment_params,
        curvature=curvature,
        surface_merged=surface_merged,
        n_bscans=N_BSCANS
    )
    print(f"\n  ✓ Saved: {output_file.name}")

    # Final summary
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles generated:")
    print(f"  - yaxis_comparison.png (before/after Y-axis view)")
    print(f"  - curvature_profile.png (curvature vs Z-position)")
    print(f"  - per_bscan_offsets.png (per-B-scan Y-offsets)")
    print(f"  - surface_overlay.png (merged retinal surface)")
    print(f"  - central_alignment_results.npz (all data)")
    print(f"\nAlignment summary:")
    print(f"  Global Y-shift: {alignment_params['y_shift_global']:+.2f} px")
    print(f"  Z-axis rotation: {alignment_params['rotation_z']:+.2f}°")
    print(f"  X-axis rotation: {alignment_params['rotation_x']:+.2f}°")
    print(f"  Per-B-scan offsets: {alignment_params['per_bscan_offsets'].min():.1f} to {alignment_params['per_bscan_offsets'].max():.1f} px")
    print(f"  Final NCC: {alignment_params['ncc_final']:.4f}")
    print(f"\nCurvature summary:")
    print(f"  Range: {curvature.min():.6f} to {curvature.max():.6f}")
    print(f"  Mean: {curvature.mean():.6f}")
    print(f"  Std: {curvature.std():.6f}")


if __name__ == '__main__':
    main()
