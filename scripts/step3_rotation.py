#!/usr/bin/env python3
"""
Step 3: Z-Axis Rotation Alignment

Standalone script to add rotation correction to the OCT alignment pipeline.

This script:
  1. Loads results from Steps 1 & 2 (XZ + Y alignment)
  2. Finds optimal Z-axis rotation angle using NCC
  3. Applies rotation to aligned volume
  4. Generates before/after visualizations
  5. Saves rotated volume and parameters

Usage:
    python step3_rotation.py
    python step3_rotation.py --coarse-range 20 --coarse-step 1
    python step3_rotation.py --no-vis  # Skip visualization
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rotation_alignment import (
    find_optimal_rotation_z,
    apply_rotation_z,
    calculate_ncc_3d,
    visualize_rotation_search,
    visualize_rotation_comparison
)


def main():
    parser = argparse.ArgumentParser(
        description='Step 3: Z-Axis Rotation Alignment',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--coarse-range', type=int, default=30,
                        help='Coarse search angle range (default: 30 degrees)')
    parser.add_argument('--coarse-step', type=int, default=2,
                        help='Coarse search step size (default: 2 degrees)')
    parser.add_argument('--fine-range', type=int, default=3,
                        help='Fine search angle range (default: 3 degrees)')
    parser.add_argument('--fine-step', type=float, default=0.5,
                        help='Fine search step size (default: 0.5 degrees)')
    parser.add_argument('--no-vis', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory path (default: ../notebooks/data)')

    args = parser.parse_args()

    # Setup paths
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'

    print("="*70)
    print("STEP 3: Z-AXIS ROTATION ALIGNMENT")
    print("="*70)
    print(f"Data directory: {data_dir}")

    # Check for required input files
    step1_file = data_dir / 'step1_results.npy'
    step2_file = data_dir / 'step2_results.npy'

    if not step1_file.exists():
        print(f"\n❌ Error: {step1_file} not found!")
        print("   Please run Step 1 first: python alignment_pipeline.py --step 1")
        return 1

    if not step2_file.exists():
        print(f"\n❌ Error: {step2_file} not found!")
        print("   Please run Step 2 first: python alignment_pipeline.py --step 2")
        return 1

    # Load previous results
    print("\nLoading previous alignment results...")
    step1_results = np.load(step1_file, allow_pickle=True).item()
    step2_results = np.load(step2_file, allow_pickle=True).item()

    overlap_v0 = step1_results['overlap_v0']
    overlap_v1_y_aligned = step2_results['overlap_v1_y_aligned']

    print(f"  ✓ Loaded overlap volumes:")
    print(f"    V0 (reference): {overlap_v0.shape}")
    print(f"    V1 (Y-aligned): {overlap_v1_y_aligned.shape}")

    # Calculate baseline NCC
    print("\n" + "="*70)
    print("1. BASELINE MEASUREMENT")
    print("="*70)
    ncc_before = calculate_ncc_3d(overlap_v0, overlap_v1_y_aligned)
    print(f"  Baseline NCC (before rotation): {ncc_before:.4f}")

    # Find optimal rotation
    print("\n" + "="*70)
    print("2. ROTATION SEARCH")
    print("="*70)
    print(f"  Coarse search: ±{args.coarse_range}° with {args.coarse_step}° steps")
    print(f"  Fine search: ±{args.fine_range}° with {args.fine_step}° steps")

    rotation_angle, rotation_metrics = find_optimal_rotation_z(
        overlap_v0,
        overlap_v1_y_aligned,
        coarse_range=args.coarse_range,
        coarse_step=args.coarse_step,
        fine_range=args.fine_range,
        fine_step=args.fine_step,
        verbose=True,
        visualize_masks=True,
        mask_vis_path=data_dir / 'step3_mask_verification.png'
    )

    ncc_after = rotation_metrics['optimal_ncc']

    # Apply rotation
    print("\n" + "="*70)
    print("3. APPLYING ROTATION")
    print("="*70)
    print(f"  Rotating by {rotation_angle:.2f}°...")

    overlap_v1_rotated = apply_rotation_z(
        overlap_v1_y_aligned,
        rotation_angle,
        axes=(0, 1)
    )

    # Verify
    ncc_verify = calculate_ncc_3d(overlap_v0, overlap_v1_rotated)
    improvement = (ncc_after - ncc_before) * 100

    print(f"\n  Results:")
    print(f"    Rotation angle: {rotation_angle:.2f}°")
    print(f"    NCC before: {ncc_before:.4f}")
    print(f"    NCC after: {ncc_after:.4f}")
    print(f"    NCC verified: {ncc_verify:.4f}")
    print(f"    Improvement: {improvement:+.2f}%")

    # Save results
    print("\n" + "="*70)
    print("4. SAVING RESULTS")
    print("="*70)

    step3_results = {
        'rotation_angle': float(rotation_angle),
        'ncc_before': float(ncc_before),
        'ncc_after': float(ncc_after),
        'ncc_improvement_percent': float(improvement),
        'overlap_v1_rotated': overlap_v1_rotated,
        'coarse_results': rotation_metrics['coarse_results'],
        'fine_results': rotation_metrics['fine_results']
    }

    # Combine with previous results
    combined_results = {
        **step1_results,
        **step2_results,
        **step3_results
    }

    output_file = data_dir / 'step3_results.npy'
    np.save(output_file, combined_results, allow_pickle=True)
    print(f"  ✓ Saved: {output_file}")

    # Save rotation parameters separately for easy access
    rotation_params = {
        'rotation_angle_z': float(rotation_angle),
        'ncc_before': float(ncc_before),
        'ncc_after': float(ncc_after),
        'improvement_percent': float(improvement),
        'offset_x': step1_results['offset_x'],
        'offset_z': step1_results['offset_z'],
        'y_shift': step2_results['y_shift']
    }

    params_file = data_dir / 'rotation_params.npy'
    np.save(params_file, rotation_params, allow_pickle=True)
    print(f"  ✓ Saved: {params_file}")

    # Generate visualizations
    if not args.no_vis:
        print("\n" + "="*70)
        print("5. GENERATING VISUALIZATIONS")
        print("="*70)

        # Rotation search plot
        print("\n  Creating rotation search plot...")
        visualize_rotation_search(
            rotation_metrics['coarse_results'],
            rotation_metrics['fine_results'],
            output_path=data_dir / 'step3_rotation_search.png'
        )

        # Before/after comparison
        print("\n  Creating before/after comparison...")
        visualize_rotation_comparison(
            overlap_v0,
            overlap_v1_y_aligned,
            overlap_v1_rotated,
            rotation_angle,
            ncc_before,
            ncc_after,
            output_path=data_dir / 'step3_rotation_comparison.png'
        )

        print("\n  ✓ Visualizations complete!")

    # Final summary
    print("\n" + "="*70)
    print("✅ STEP 3 COMPLETE!")
    print("="*70)
    print(f"\nAlignment Summary:")
    print(f"  Step 1 (XZ): ΔX={step1_results['offset_x']}, ΔZ={step1_results['offset_z']}")
    print(f"  Step 2 (Y):  ΔY={step2_results['y_shift']:+.2f}")
    print(f"  Step 3 (Rotation): θ={rotation_angle:+.2f}° (NCC: {ncc_before:.4f}→{ncc_after:.4f})")

    print(f"\nGenerated files:")
    print(f"  - step3_results.npy (combined alignment results)")
    print(f"  - rotation_params.npy (rotation parameters only)")
    if not args.no_vis:
        print(f"  - step3_rotation_search.png (angle search plot)")
        print(f"  - step3_rotation_comparison.png (before/after comparison)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
