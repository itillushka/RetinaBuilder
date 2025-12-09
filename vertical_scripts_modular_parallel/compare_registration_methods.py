"""
Compare Registration Methods for XZ Alignment

Standalone script to test and compare all available registration methods:
- Multi-scale Phase Correlation (FFT-based, modern, robust)
- Legacy Phase Correlation (single-scale, original method)
- ANTs Translation (medical imaging gold standard)
- SIFT Feature Matching (scale-invariant, accurate)
- ORB Feature Matching (fast, rotation-invariant)
- AKAZE Feature Matching (good compromise)

Usage:
    # Compare volumes 0 and 1 for patient EM005
    python compare_registration_methods.py --patient EM005 --volumes 0 1

    # Use VESSELS-ONLY mode (RECOMMENDED - ignores background tissue)
    python compare_registration_methods.py --patient EM005 --volumes 0 1 --vessels-only

    # Adjust vessel threshold (lower = more vessels, higher = only strongest)
    python compare_registration_methods.py --patient EM005 --volumes 0 1 --vessels-only --vessel-threshold 0.15

    # Compare with ground truth
    python compare_registration_methods.py --patient EM005 --volumes 1 2 --vessels-only --ground-truth 0 100

    # Use custom OCT data directory
    python compare_registration_methods.py --patient EM005 --data-dir /path/to/oct_data

    # Load from .npy files directly
    python compare_registration_methods.py --volume1 vol1.npy --volume2 vol2.npy --vessels-only
"""

import numpy as np
import sys
from pathlib import Path
import argparse
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from helpers.mip_generation import (
    create_vessel_enhanced_mip,
    compare_registration_methods
)
from helpers import OCTImageProcessor, OCTVolumeLoader


def visualize_mips_before_alignment(mip1, mip2, output_path, vessels_only=False):
    """
    Visualize the two MIPs being aligned side-by-side.

    Args:
        mip1: Fixed MIP (reference)
        mip2: Moving MIP (to be aligned)
        output_path: Path to save the visualization
        vessels_only: Whether vessels-only mode was used
    """
    import matplotlib.pyplot as plt

    mode_text = " (VESSELS-ONLY MODE)" if vessels_only else ""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Vessel-Enhanced MIPs Before Alignment{mode_text}', fontsize=14, fontweight='bold')

    # Fixed MIP (reference)
    ax = axes[0]
    im1 = ax.imshow(mip1, cmap='gray', aspect='auto')
    ax.set_title(f'Fixed MIP (Reference)\nShape: {mip1.shape}\nRange: [{mip1.min():.0f}, {mip1.max():.0f}]',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Z-axis (pixels)', fontsize=10)
    ax.set_ylabel('X-axis (pixels)', fontsize=10)
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    ax.grid(False)

    # Moving MIP (to be aligned)
    ax = axes[1]
    im2 = ax.imshow(mip2, cmap='gray', aspect='auto')
    ax.set_title(f'Moving MIP (To Align)\nShape: {mip2.shape}\nRange: [{mip2.min():.0f}, {mip2.max():.0f}]',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Z-axis (pixels)', fontsize=10)
    ax.set_ylabel('X-axis (pixels)', fontsize=10)
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ MIP visualization saved: {output_path}")


def load_volumes_from_patient(patient_code, volume_indices, oct_data_dir=None):
    """Load volumes from patient BMP directories."""
    # Set up OCT data directory
    parent_dir = Path(__file__).parent.parent

    if oct_data_dir is None:
        # Try oct_data first (lowercase), fall back to OCT_DATA
        oct_data_dir = parent_dir / 'oct_data'
        if not oct_data_dir.exists():
            oct_data_dir = parent_dir / 'OCT_DATA'

    if not oct_data_dir.exists():
        raise FileNotFoundError(
            f"OCT data directory not found. Tried:\n"
            f"  - {parent_dir / 'oct_data'}\n"
            f"  - {parent_dir / 'OCT_DATA'}\n"
            f"Please specify --data-dir or use --volume1/--volume2"
        )

    print(f"OCT data directory: {oct_data_dir}")

    # Initialize loader
    processor = OCTImageProcessor(sidebar_width=250, crop_top=100, crop_bottom=50)
    loader = OCTVolumeLoader(processor)

    # Find volume directories
    bmp_dirs = []
    for bmp_file in oct_data_dir.rglob('*.bmp'):
        vol_dir = bmp_file.parent
        if vol_dir not in bmp_dirs:
            bmp_dirs.append(vol_dir)

    # Filter volumes by patient ID
    patient_vols = sorted([v for v in bmp_dirs if patient_code in str(v)])

    if len(patient_vols) == 0:
        raise ValueError(f"No volumes found for patient {patient_code} in {oct_data_dir}")

    max_idx = max(volume_indices)
    if len(patient_vols) <= max_idx:
        raise ValueError(
            f"Not enough volumes for patient {patient_code}. "
            f"Found {len(patient_vols)}, but need index {max_idx}"
        )

    print(f"Found {len(patient_vols)} volumes for patient {patient_code}")

    # Load requested volumes
    volumes = []
    for idx in volume_indices:
        vol_path = patient_vols[idx]
        print(f"Loading volume {idx}: {vol_path.name}...")
        volume = loader.load_volume_from_directory(str(vol_path))
        print(f"  ✓ Loaded: shape={volume.shape}")
        volumes.append(volume)

    return volumes


def main():
    parser = argparse.ArgumentParser(description='Compare registration methods')

    # Option 1: Load specific volume files
    parser.add_argument('--volume1', type=str, help='Path to first volume (.npy)')
    parser.add_argument('--volume2', type=str, help='Path to second volume (.npy)')

    # Option 2: Load from patient directory
    parser.add_argument('--patient', type=str, help='Patient code (e.g., EM005, F001_IP)')
    parser.add_argument('--volumes', type=int, nargs=2, default=[0, 1],
                        help='Volume indices to compare (default: 0 1)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override OCT data directory (default: ../oct_data)')

    # Optional ground truth
    parser.add_argument('--ground-truth', type=float, nargs=2,
                        help='Ground truth offsets (offset_x offset_z)')

    # Output
    parser.add_argument('--output', type=str, default='registration_comparison.png',
                        help='Output chart filename')

    # Processing options
    parser.add_argument('--denoise', action='store_true',
                        help='Enable bilateral denoising after Frangi filter (disabled by default)')
    parser.add_argument('--vessels-only', action='store_true',
                        help='Use only vessel structures (threshold background, recommended)')
    parser.add_argument('--vessel-threshold', type=float, default=0.1,
                        help='Vessel threshold (0-1, default: 0.1, lower = more vessels)')

    args = parser.parse_args()

    # Load volumes
    if args.volume1 and args.volume2:
        print(f"Loading volumes from files...")
        volume_1 = np.load(args.volume1)
        volume_2 = np.load(args.volume2)
        output_path = Path(args.output)
        print(f"  ✓ Volume 1: {args.volume1}")
        print(f"  ✓ Volume 2: {args.volume2}")
    elif args.patient:
        data_dir_path = Path(args.data_dir) if args.data_dir else None
        volumes = load_volumes_from_patient(args.patient, args.volumes, data_dir_path)
        volume_1, volume_2 = volumes

        # Create output directory
        patient_safe = args.patient.replace('_', '').replace('/', '').replace('\\', '')
        output_dir = Path(__file__).parent / f'results_3vol_{patient_safe.lower()}'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / args.output
    else:
        parser.error("Either --volume1/--volume2 OR --patient must be specified")

    print(f"\n{'='*70}")
    print("REGISTRATION METHOD COMPARISON")
    print(f"{'='*70}")
    print(f"Volume 1 shape: {volume_1.shape}")
    print(f"Volume 2 shape: {volume_2.shape}")
    print(f"Output: {output_path}")

    # Create vessel-enhanced MIPs
    denoise = args.denoise
    vessels_only = args.vessels_only
    vessel_threshold = args.vessel_threshold

    print(f"\n{'='*70}")
    mode_str = "VESSELS-ONLY" if vessels_only else "Standard"
    print(f"STEP 1: Creating Vessel-Enhanced MIPs ({mode_str} mode)")
    print(f"{'='*70}")

    if vessels_only:
        print(f"Frangi filter (5 scales) + vessel thresholding (threshold={vessel_threshold})...")
    elif denoise:
        print("Frangi filter (5 scales) + bilateral denoising...")
    else:
        print("Frangi filter (5 scales)...")

    mip_start = time.time()
    mip_1 = create_vessel_enhanced_mip(volume_1, verbose=True, denoise=denoise,
                                       vessels_only=vessels_only, vessel_threshold=vessel_threshold)
    mip_2 = create_vessel_enhanced_mip(volume_2, verbose=True, denoise=denoise,
                                       vessels_only=vessels_only, vessel_threshold=vessel_threshold)
    mip_time = time.time() - mip_start

    print(f"  ✓ MIP 1: shape={mip_1.shape}, range=[{mip_1.min():.1f}, {mip_1.max():.1f}]")
    print(f"  ✓ MIP 2: shape={mip_2.shape}, range=[{mip_2.min():.1f}, {mip_2.max():.1f}]")
    print(f"  ✓ MIP creation time: {mip_time:.2f}s")

    # Visualize MIPs before alignment
    mip_viz_path = output_path.parent / 'mips_before_alignment.png'
    print(f"\nVisualizing MIPs...")
    visualize_mips_before_alignment(mip_1, mip_2, mip_viz_path, vessels_only=vessels_only)

    # Parse ground truth if provided (as tuple for compare_registration_methods)
    ground_truth = None
    if args.ground_truth:
        ground_truth = (args.ground_truth[0], args.ground_truth[1])
        print(f"\nGround truth: offset_x={ground_truth[0]:.1f}, "
              f"offset_z={ground_truth[1]:.1f}")

    # Run comparison
    print(f"\n{'='*70}")
    print("STEP 2: Testing All Registration Methods (6 methods)")
    print(f"{'='*70}")

    comparison_start = time.time()
    results = compare_registration_methods(
        mip_1, mip_2,
        ground_truth=ground_truth,
        output_path=str(output_path)
    )
    comparison_time = time.time() - comparison_start

    # Display results
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    # Sort by confidence (descending)
    successful = {k: v for k, v in results.items() if v['success']}
    failed = {k: v for k, v in results.items() if not v['success']}

    if successful:
        sorted_methods = sorted(successful.items(),
                                key=lambda x: x[1]['confidence'],
                                reverse=True)

        print(f"\n{'Method':<15} {'X-Offset':<10} {'Z-Offset':<10} {'Confidence':<12} {'Time (s)':<10}")
        print("-" * 70)

        for method_name, result in sorted_methods:
            print(f"{method_name.upper():<15} "
                  f"{result['offset_x']:>9.1f} "
                  f"{result['offset_z']:>9.1f} "
                  f"{result['confidence']:>11.3f} "
                  f"{result['time']:>9.2f}")

        # Best method
        best_method = sorted_methods[0][0]
        best_result = sorted_methods[0][1]
        print(f"\n{'='*70}")
        print(f"BEST METHOD: {best_method.upper()}")
        print(f"{'='*70}")
        print(f"  Offset X: {best_result['offset_x']:.1f} px")
        print(f"  Offset Z: {best_result['offset_z']:.1f} px")
        print(f"  Confidence: {best_result['confidence']:.3f}")
        print(f"  Time: {best_result['time']:.2f}s")

        # Compare to ground truth if available
        if ground_truth:
            error_x = abs(best_result['offset_x'] - ground_truth[0])
            error_z = abs(best_result['offset_z'] - ground_truth[1])
            error_total = np.sqrt(error_x**2 + error_z**2)
            print(f"\n  Ground Truth Error:")
            print(f"    ΔX: {error_x:.2f} px")
            print(f"    ΔZ: {error_z:.2f} px")
            print(f"    Total: {error_total:.2f} px")

    if failed:
        print(f"\n{'='*70}")
        print("FAILED METHODS")
        print(f"{'='*70}")
        for method_name, result in failed.items():
            print(f"  {method_name.upper()}: {result['error']}")

    print(f"\n{'='*70}")
    print(f"Total comparison time: {comparison_time:.2f}s")
    print(f"Comparison chart saved: {output_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
