#!/usr/bin/env python3
"""
Standalone Rotation Alignment Test Script

Tests and visualizes rotation alignment using both NCC and contour methods.
Helps debug why certain angles are chosen and compare methods.

Usage:
    python test_rotation_alignment.py --patient data/patient_001 --method both --range 15 --step 1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import argparse
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from helpers.rotation_alignment import (
    calculate_ncc_3d,
    preprocess_oct_for_visualization,
    detect_contour_surface,
    create_rotation_mask,
    detect_surface_in_masked_region,
    calculate_contour_alignment_score
)


def load_test_volumes(patient_dir):
    """
    Load test volumes from patient directory.

    Args:
        patient_dir: Path to patient directory containing alignment results

    Returns:
        volume_1: Reference volume
        volume_2: Moving volume (already Y-aligned)
    """
    patient_dir = Path(patient_dir)

    # Try to load from various possible locations
    possible_files = [
        ('volume_1.npy', 'volume_2.npy'),
        ('step1_v2_to_v1_xz_results.npy', None),  # Will use stored volumes
    ]

    # First try: Load original volumes
    v1_path = patient_dir / 'volume_1.npy'
    v2_path = patient_dir / 'volume_2.npy'

    if v1_path.exists() and v2_path.exists():
        print(f"Loading volumes from {patient_dir}")
        volume_1 = np.load(v1_path)
        volume_2 = np.load(v2_path)
        print(f"  Volume 1 shape: {volume_1.shape}")
        print(f"  Volume 2 shape: {volume_2.shape}")
        return volume_1, volume_2

    # Second try: Load from Y-alignment results
    y_results_path = patient_dir / 'step1_v2_to_v1_y_results.npy'
    if y_results_path.exists():
        print(f"Loading from Y-alignment results...")
        y_results = np.load(y_results_path, allow_pickle=True).item()

        # Load original V1 (should be in parent dir or data dir)
        data_dirs = [patient_dir, patient_dir.parent, patient_dir / 'data']
        for data_dir in data_dirs:
            v1_test = data_dir / 'volume_1.npy'
            if v1_test.exists():
                volume_1 = np.load(v1_test)
                break
        else:
            raise FileNotFoundError("Could not find volume_1.npy")

        # Use Y-aligned volume from results
        volume_2 = y_results.get('volume_1_y_aligned')
        if volume_2 is None:
            raise ValueError("Y-alignment results don't contain 'volume_1_y_aligned'")

        print(f"  Volume 1 shape: {volume_1.shape}")
        print(f"  Volume 2 (Y-aligned) shape: {volume_2.shape}")
        return volume_1, volume_2

    raise FileNotFoundError(f"Could not find test volumes in {patient_dir}")


def test_single_angle_ncc(angle, bscan_ref, bscan_mov):
    """
    Test a single rotation angle using NCC metric.

    Args:
        angle: Rotation angle in degrees
        bscan_ref: Reference B-scan
        bscan_mov: Moving B-scan

    Returns:
        dict with angle, ncc_score, and rotated B-scan
    """
    try:
        # Rotate moving B-scan
        bscan_mov_rotated = ndimage.rotate(
            bscan_mov, angle, axes=(0, 1),
            reshape=False, order=1,
            mode='constant', cval=0
        )

        # Calculate NCC
        ncc_score = calculate_ncc_3d(bscan_ref[:, :, np.newaxis],
                                      bscan_mov_rotated[:, :, np.newaxis])

        return {
            'angle': float(angle),
            'ncc_score': float(ncc_score),
            'bscan_rotated': bscan_mov_rotated
        }
    except Exception as e:
        return {
            'angle': float(angle),
            'ncc_score': -np.inf,
            'error': str(e)
        }


def test_single_angle_contour(angle, bscan_ref, bscan_mov):
    """
    Test a single rotation angle using contour variance metric.

    Args:
        angle: Rotation angle in degrees
        bscan_ref: Reference B-scan
        bscan_mov: Moving B-scan

    Returns:
        dict with angle, variance, surfaces, and rotated B-scan
    """
    try:
        # Rotate moving B-scan
        bscan_mov_rotated = ndimage.rotate(
            bscan_mov, angle, axes=(0, 1),
            reshape=False, order=1,
            mode='constant', cval=0
        )

        # Create mask
        mask_2d, mask_columns = create_rotation_mask(bscan_ref, bscan_mov_rotated)

        if mask_columns.sum() < 10:
            return {
                'angle': float(angle),
                'variance': np.inf,
                'score': -np.inf,
                'valid_pixels': 0
            }

        # Preprocess and detect surfaces
        bscan_ref_denoised = preprocess_oct_for_visualization(bscan_ref)
        bscan_mov_denoised = preprocess_oct_for_visualization(bscan_mov_rotated)

        surface_ref = detect_surface_in_masked_region(bscan_ref_denoised, mask_columns)
        surface_mov = detect_surface_in_masked_region(bscan_mov_denoised, mask_columns)

        # Calculate alignment score
        score, metrics = calculate_contour_alignment_score(surface_ref, surface_mov, mask_columns)

        return {
            'angle': float(angle),
            'variance': metrics['variance'],
            'score': float(score),
            'valid_pixels': metrics['valid_pixels'],
            'surface_ref': surface_ref,
            'surface_mov': surface_mov,
            'mask_columns': mask_columns,
            'bscan_rotated': bscan_mov_rotated,
            'bscan_ref_denoised': bscan_ref_denoised,
            'bscan_mov_denoised': bscan_mov_denoised
        }
    except Exception as e:
        return {
            'angle': float(angle),
            'variance': np.inf,
            'score': -np.inf,
            'error': str(e)
        }


def visualize_angle_result(result, bscan_ref, bscan_mov, method, output_path):
    """
    Visualize result for a single angle.

    Creates a comprehensive visualization showing original, rotated, and metrics.
    """
    angle = result['angle']

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Row 1: Original B-scans
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(bscan_ref, cmap='gray', aspect='auto')
    ax1.set_title('Reference B-scan', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y (depth)')
    ax1.set_xlabel('X (lateral)')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(bscan_mov, cmap='gray', aspect='auto')
    ax2.set_title('Moving B-scan (original)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (lateral)')

    ax3 = fig.add_subplot(gs[0, 2])
    if 'bscan_rotated' in result:
        ax3.imshow(result['bscan_rotated'], cmap='gray', aspect='auto')
    ax3.set_title(f'Rotated by {angle:+.1f}°', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (lateral)')

    # Overlay
    ax4 = fig.add_subplot(gs[0, 3])
    if 'bscan_rotated' in result:
        overlay = np.zeros((*bscan_ref.shape, 3), dtype=np.uint8)
        overlay[:, :, 0] = (bscan_ref / bscan_ref.max() * 255).astype(np.uint8)
        overlay[:, :, 1] = (result['bscan_rotated'] / (result['bscan_rotated'].max() + 1e-8) * 255).astype(np.uint8)
        ax4.imshow(overlay, aspect='auto')
    ax4.set_title('Overlay (R=Ref, G=Mov)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('X (lateral)')

    # Row 2: Method-specific visualizations
    if method == 'ncc':
        ax5 = fig.add_subplot(gs[1, :])
        ncc_score = result.get('ncc_score', -np.inf)
        ax5.text(0.5, 0.5, f'NCC Score: {ncc_score:.4f}\nAngle: {angle:+.2f}°',
                ha='center', va='center', fontsize=20, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax5.axis('off')

    elif method == 'contour':
        # Show surfaces
        ax5 = fig.add_subplot(gs[1, 0])
        if 'bscan_ref_denoised' in result:
            ax5.imshow(result['bscan_ref_denoised'], cmap='gray', aspect='auto')
            if 'surface_ref' in result and 'mask_columns' in result:
                mask = result['mask_columns']
                surf = result['surface_ref']
                x_coords = np.where(mask)[0]
                y_coords = surf[mask]
                valid = ~np.isnan(y_coords)
                ax5.plot(x_coords[valid], y_coords[valid], 'r-', linewidth=2, label='Surface')
                ax5.legend()
        ax5.set_title('Ref + Surface', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Y (depth)')

        ax6 = fig.add_subplot(gs[1, 1])
        if 'bscan_mov_denoised' in result:
            ax6.imshow(result['bscan_mov_denoised'], cmap='gray', aspect='auto')
            if 'surface_mov' in result and 'mask_columns' in result:
                mask = result['mask_columns']
                surf = result['surface_mov']
                x_coords = np.where(mask)[0]
                y_coords = surf[mask]
                valid = ~np.isnan(y_coords)
                ax6.plot(x_coords[valid], y_coords[valid], 'g-', linewidth=2, label='Surface')
                ax6.legend()
        ax6.set_title('Mov + Surface', fontsize=12, fontweight='bold')

        # Surface difference
        ax7 = fig.add_subplot(gs[1, 2:])
        if 'surface_ref' in result and 'surface_mov' in result and 'mask_columns' in result:
            surf_ref = result['surface_ref']
            surf_mov = result['surface_mov']
            mask = result['mask_columns']
            diff = surf_ref - surf_mov

            x_coords = np.arange(len(surf_ref))
            ax7.plot(x_coords[mask], diff[mask], 'b-', linewidth=2, label='Surface Difference')
            ax7.axhline(0, color='k', linestyle='--', linewidth=1)
            ax7.fill_between(x_coords[mask], 0, diff[mask], alpha=0.3)
            ax7.set_xlabel('X (lateral)')
            ax7.set_ylabel('Difference (px)')
            ax7.set_title(f'Surface Difference (Variance: {result["variance"]:.2f} px²)',
                         fontsize=12, fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)

    plt.suptitle(f'Rotation Test: {angle:+.2f}° ({method.upper()} method)',
                 fontsize=14, fontweight='bold')

    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def create_summary_plot(results, method, output_path):
    """
    Create summary plot showing metric vs angle.
    """
    angles = [r['angle'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if method == 'ncc':
        scores = [r.get('ncc_score', -np.inf) for r in results]
        valid_idx = [i for i, s in enumerate(scores) if s > -np.inf]

        if valid_idx:
            valid_angles = [angles[i] for i in valid_idx]
            valid_scores = [scores[i] for i in valid_idx]

            axes[0].plot(valid_angles, valid_scores, 'b-', linewidth=2, marker='o', markersize=8)
            best_idx = np.argmax(valid_scores)
            axes[0].scatter([valid_angles[best_idx]], [valid_scores[best_idx]],
                          c='red', s=300, marker='*', edgecolors='black', linewidths=2,
                          zorder=5, label=f'Best: {valid_angles[best_idx]:+.1f}°')
            axes[0].set_xlabel('Rotation Angle (degrees)', fontsize=12)
            axes[0].set_ylabel('NCC Score', fontsize=12)
            axes[0].set_title('NCC Score vs Rotation Angle', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=12)
            axes[0].axvline(0, color='k', linestyle=':', alpha=0.5, label='0°')

        axes[1].axis('off')

    elif method == 'contour':
        variances = [r.get('variance', np.inf) for r in results]
        scores = [r.get('score', -np.inf) for r in results]

        valid_idx = [i for i, v in enumerate(variances) if v < np.inf]

        if valid_idx:
            valid_angles = [angles[i] for i in valid_idx]
            valid_variances = [variances[i] for i in valid_idx]
            valid_scores = [scores[i] for i in valid_idx]

            # Plot 1: Variance (lower is better)
            axes[0].plot(valid_angles, valid_variances, 'b-', linewidth=2, marker='o', markersize=8)
            best_idx = np.argmin(valid_variances)
            axes[0].scatter([valid_angles[best_idx]], [valid_variances[best_idx]],
                          c='red', s=300, marker='*', edgecolors='black', linewidths=2,
                          zorder=5, label=f'Best: {valid_angles[best_idx]:+.1f}°')
            axes[0].set_xlabel('Rotation Angle (degrees)', fontsize=12)
            axes[0].set_ylabel('Surface Variance (px²)', fontsize=12)
            axes[0].set_title('Surface Variance vs Angle (Lower = Better)', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=12)
            axes[0].axvline(0, color='k', linestyle=':', alpha=0.5)

            # Plot 2: Score (higher is better, = -variance)
            axes[1].plot(valid_angles, valid_scores, 'g-', linewidth=2, marker='o', markersize=8)
            best_idx = np.argmax(valid_scores)
            axes[1].scatter([valid_angles[best_idx]], [valid_scores[best_idx]],
                          c='red', s=300, marker='*', edgecolors='black', linewidths=2,
                          zorder=5, label=f'Best: {valid_angles[best_idx]:+.1f}°')
            axes[1].set_xlabel('Rotation Angle (degrees)', fontsize=12)
            axes[1].set_ylabel('Alignment Score', fontsize=12)
            axes[1].set_title('Score vs Angle (Higher = Better)', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=12)
            axes[1].axvline(0, color='k', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved summary: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description='Test rotation alignment')
    parser.add_argument('--patient', type=str, required=True, help='Patient directory')
    parser.add_argument('--method', type=str, default='both', choices=['ncc', 'contour', 'both'],
                       help='Method to test')
    parser.add_argument('--range', type=float, default=15, help='Angle range (degrees)')
    parser.add_argument('--step', type=float, default=1, help='Angle step (degrees)')
    parser.add_argument('--visualize-all', action='store_true', help='Visualize every angle')
    parser.add_argument('--output', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.patient) / 'rotation_test_results'
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*70)
    print("ROTATION ALIGNMENT TEST")
    print("="*70)

    # Load volumes
    print("\n1. Loading test volumes...")
    volume_1, volume_2 = load_test_volumes(args.patient)

    # Extract central B-scan
    Z = volume_1.shape[2]
    z_mid = Z // 2
    bscan_ref = volume_1[:, :, z_mid]
    bscan_mov = volume_2[:, :, z_mid]
    print(f"  Using central B-scan (z={z_mid}/{Z})")
    print(f"  B-scan shape: {bscan_ref.shape}")

    # Generate test angles
    angles = np.arange(-args.range, args.range + args.step, args.step)
    print(f"\n2. Testing {len(angles)} angles from {-args.range}° to +{args.range}° (step: {args.step}°)")

    # Test methods
    methods_to_test = ['ncc', 'contour'] if args.method == 'both' else [args.method]

    for method in methods_to_test:
        print(f"\n  Testing {method.upper()} method...")

        results = []
        for i, angle in enumerate(angles):
            if method == 'ncc':
                result = test_single_angle_ncc(angle, bscan_ref, bscan_mov)
            else:
                result = test_single_angle_contour(angle, bscan_ref, bscan_mov)

            results.append(result)

            # Print progress
            if (i + 1) % 5 == 0 or (i + 1) == len(angles):
                print(f"    Tested {i+1}/{len(angles)} angles...")

            # Visualize if requested
            if args.visualize_all or angle in [-15, -10, -5, 0, 5, 10, 15]:
                vis_path = output_dir / f'{method}_angle_{angle:+06.1f}.png'
                visualize_angle_result(result, bscan_ref, bscan_mov, method, vis_path)

        # Find best angle
        if method == 'ncc':
            valid_results = [r for r in results if r['ncc_score'] > -np.inf]
            if valid_results:
                best = max(valid_results, key=lambda x: x['ncc_score'])
                print(f"\n  ✓ Best angle (NCC): {best['angle']:+.2f}° (score: {best['ncc_score']:.4f})")
        else:
            valid_results = [r for r in results if r['variance'] < np.inf]
            if valid_results:
                best = min(valid_results, key=lambda x: x['variance'])
                print(f"\n  ✓ Best angle (Contour): {best['angle']:+.2f}° (variance: {best['variance']:.2f} px²)")

        # Create summary plot
        summary_path = output_dir / f'{method}_summary.png'
        create_summary_plot(results, method, summary_path)

        # Save results to JSON
        json_path = output_dir / f'{method}_results.json'
        # Remove non-serializable numpy arrays
        results_clean = []
        for r in results:
            r_clean = {k: v for k, v in r.items()
                      if not isinstance(v, np.ndarray)}
            results_clean.append(r_clean)

        with open(json_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        print(f"  ✓ Saved results: {json_path.name}")

    print(f"\n✓ All results saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
