#!/usr/bin/env python3
"""
Test Rotation Metrics - Comprehensive Metric Evaluation

Tests all 8 alignment metrics across rotation angles to identify which metric
correctly detects the optimal alignment angle where retinal layers are parallel.

Visual inspection identified -10° to -8° as optimal (parallel layers).
This script tests which metric correctly peaks in that range.

Usage:
    python test_rotation_metrics.py
    python test_rotation_metrics.py --angle-start -14 --angle-end -6 --step 0.5
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

# Import our metrics library
from metrics_library import RotationMetrics

print("="*70)
print("ROTATION METRICS COMPREHENSIVE TEST")
print("="*70)


def load_alignment_data(data_dir):
    """
    Load Step 1 & 2 alignment results.

    Returns:
        tuple: (overlap_v0, overlap_v1_y_aligned)
    """
    print("\n1. Loading alignment results...")

    step1_file = data_dir / 'step1_results.npy'
    step2_file = data_dir / 'step2_results.npy'

    if not step1_file.exists() or not step2_file.exists():
        print("❌ Error: Need to run Steps 1 & 2 first!")
        print("   Run: python alignment_pipeline.py --steps 1 2")
        sys.exit(1)

    step1_results = np.load(step1_file, allow_pickle=True).item()
    step2_results = np.load(step2_file, allow_pickle=True).item()

    overlap_v0 = step1_results['overlap_v0']
    overlap_v1_y_aligned = step2_results['overlap_v1_y_aligned']

    print(f"  ✓ Overlap V0: {overlap_v0.shape}")
    print(f"  ✓ Overlap V1 (Y-aligned): {overlap_v1_y_aligned.shape}")

    return overlap_v0, overlap_v1_y_aligned


def extract_central_bscan(overlap_v0, overlap_v1_y_aligned):
    """
    Extract central B-scan from both volumes.

    Returns:
        tuple: (bscan_v0, bscan_v1, z_center)
    """
    z_center = overlap_v0.shape[2] // 2
    bscan_v0 = overlap_v0[:, :, z_center]
    bscan_v1 = overlap_v1_y_aligned[:, :, z_center]

    print(f"\n2. Extracted central B-scan:")
    print(f"   Z-index: {z_center}/{overlap_v0.shape[2]}")
    print(f"   B-scan shape: {bscan_v0.shape} (Y, X)")

    return bscan_v0, bscan_v1, z_center


def create_mask(bscan_v0, bscan_v1, threshold=10):
    """
    Create binary mask for valid overlap region.

    Args:
        bscan_v0: Reference B-scan
        bscan_v1: Moving B-scan
        threshold: Intensity threshold for valid regions

    Returns:
        Binary mask (True = valid region)
    """
    mask_v0 = bscan_v0 > threshold
    mask_v1 = bscan_v1 > threshold
    combined_mask = mask_v0 & mask_v1

    print(f"\n3. Created overlap mask:")
    print(f"   Valid pixels: {combined_mask.sum()} / {combined_mask.size}")
    print(f"   Coverage: {100 * combined_mask.sum() / combined_mask.size:.1f}%")

    return combined_mask


def test_all_metrics_at_angles(bscan_v0, bscan_v1, mask, angle_start, angle_end, step):
    """
    Test all metrics across rotation angle range.

    Args:
        bscan_v0: Reference B-scan
        bscan_v1: Moving B-scan
        mask: Binary mask for valid region
        angle_start: Start angle (degrees)
        angle_end: End angle (degrees)
        step: Angle step size (degrees)

    Returns:
        DataFrame with all metric scores at each angle
    """
    angles = np.arange(angle_start, angle_end + step, step)

    print(f"\n4. Testing {len(angles)} rotation angles:")
    print(f"   Range: {angles[0]:.1f}° to {angles[-1]:.1f}°")
    print(f"   Step: {step:.1f}°")
    print(f"   Expected optimal: -10° to -8° (parallel layers)")

    results = []

    print("\n   Progress:")
    for idx, angle in enumerate(angles):
        # Rotate B-scan
        bscan_v1_rotated = ndimage.rotate(
            bscan_v1,
            angle,
            axes=(0, 1),  # Rotate Y and X axes (layer tilt)
            reshape=False,
            order=1,
            mode='constant',
            cval=0
        )

        # Calculate all metrics
        metrics = RotationMetrics.evaluate_all(
            bscan_v0,
            bscan_v1_rotated,
            mask=mask
        )

        # Add angle to results
        metrics['angle'] = angle
        results.append(metrics)

        # Progress indicator
        if (idx + 1) % 5 == 0 or idx == len(angles) - 1:
            print(f"   [{idx + 1}/{len(angles)}] Tested {angle:+.1f}°")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Reorder columns (angle first)
    cols = ['angle'] + [col for col in df.columns if col != 'angle']
    df = df[cols]

    return df


def identify_best_metrics(df, expected_min=-10, expected_max=-8):
    """
    Identify which metrics correctly peak in the expected range.

    Args:
        df: Results DataFrame
        expected_min: Lower bound of expected optimal range
        expected_max: Upper bound of expected optimal range

    Returns:
        Dictionary with analysis results
    """
    print(f"\n5. Analyzing metric performance:")
    print(f"   Expected optimal range: {expected_min}° to {expected_max}°")

    metric_names = ['ncc', 'ssim', 'lgc', 'eoa', 'hfp', 'nmi', 'gmc', 'mse',
                    'rta', 'rcp', 'lsc', 'hld']
    analysis = {}

    for metric in metric_names:
        # For MSE, lower is better (find minimum)
        if metric == 'mse':
            best_idx = df[metric].idxmin()
            best_angle = df.loc[best_idx, 'angle']
            best_value = df.loc[best_idx, metric]
            operation = "MIN"
        else:
            # For all other metrics, higher is better (find maximum)
            best_idx = df[metric].idxmax()
            best_angle = df.loc[best_idx, 'angle']
            best_value = df.loc[best_idx, metric]
            operation = "MAX"

        # Check if best angle is in expected range
        in_expected_range = expected_min <= best_angle <= expected_max

        analysis[metric] = {
            'best_angle': best_angle,
            'best_value': best_value,
            'operation': operation,
            'correct': in_expected_range
        }

        # Print result
        status = "✅ CORRECT" if in_expected_range else "❌ WRONG"
        print(f"   {metric.upper():4s} {operation}: {best_angle:+5.1f}° (value={best_value:+.4f}) {status}")

    return analysis


def visualize_all_metrics(df, analysis, expected_min, expected_max, output_path):
    """
    Create comprehensive 12-panel visualization of all metrics.

    Args:
        df: Results DataFrame
        analysis: Analysis results from identify_best_metrics
        expected_min: Lower bound of expected optimal range
        expected_max: Upper bound of expected optimal range
        output_path: Path to save visualization
    """
    print(f"\n6. Creating visualization...")

    metric_names = ['ncc', 'ssim', 'lgc', 'eoa', 'hfp', 'nmi', 'gmc', 'mse',
                    'rta', 'rcp', 'lsc', 'hld']
    metric_labels = {
        'ncc': 'NCC\n(Normalized Cross-Correlation)',
        'ssim': 'SSIM\n(Structural Similarity)',
        'lgc': 'LGC\n(Layer Gradient Correlation)',
        'eoa': 'EOA\n(Edge Orientation Alignment)',
        'hfp': 'HFP\n(Horizontal Frequency Peak)',
        'nmi': 'NMI\n(Normalized Mutual Information)',
        'gmc': 'GMC\n(Gradient Magnitude Correlation)',
        'mse': 'MSE\n(Mean Squared Error)',
        'rta': 'RTA\n(Radon Transform Alignment)',
        'rcp': 'RCP\n(Row-Wise Correlation Profile)',
        'lsc': 'LSC\n(Layer Spacing Consistency)',
        'hld': 'HLD\n(Horizontal Line Detection)'
    }

    fig, axes = plt.subplots(6, 2, figsize=(16, 30))
    axes = axes.ravel()

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]

        # Plot metric vs angle
        ax.plot(df['angle'], df[metric], 'b-', linewidth=2, label=metric.upper())

        # Highlight expected optimal range
        ax.axvspan(expected_min, expected_max, alpha=0.2, color='green',
                   label=f'Expected Optimal\n({expected_min}° to {expected_max}°)')

        # Mark best angle for this metric
        best_angle = analysis[metric]['best_angle']
        best_value = analysis[metric]['best_value']
        is_correct = analysis[metric]['correct']

        marker_color = 'green' if is_correct else 'red'
        marker_label = 'CORRECT' if is_correct else 'WRONG'

        ax.axvline(best_angle, color=marker_color, linestyle='--', linewidth=2,
                   label=f'Peak: {best_angle:+.1f}° ({marker_label})')
        ax.plot(best_angle, best_value, 'o', color=marker_color, markersize=10)

        # Labels and title
        ax.set_xlabel('Rotation Angle (degrees)', fontsize=10)
        ax.set_ylabel('Metric Score', fontsize=10)
        ax.set_title(metric_labels[metric], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

        # Note for MSE (lower is better)
        if metric == 'mse':
            ax.text(0.5, 0.95, 'Note: LOWER is better',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=9, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.suptitle('Rotation Metrics Comprehensive Evaluation (12 Metrics)\n' +
                 '8 Original + 4 NEW Parallel-Layer Metrics | Expected Optimal: -10° to -8°',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved visualization: {output_path}")
    plt.close()


def save_results_csv(df, output_path):
    """
    Save all results to CSV file.

    Args:
        df: Results DataFrame
        output_path: Path to save CSV
    """
    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"   ✓ Saved CSV results: {output_path}")


def print_summary(analysis):
    """
    Print final summary of which metrics are correct.

    Args:
        analysis: Analysis results from identify_best_metrics
    """
    print("\n" + "="*70)
    print("SUMMARY: METRIC PERFORMANCE")
    print("="*70)

    correct_metrics = [m for m, a in analysis.items() if a['correct']]
    wrong_metrics = [m for m, a in analysis.items() if not a['correct']]

    print(f"\n✅ CORRECT METRICS ({len(correct_metrics)}):")
    print("   (Peak in expected range -10° to -8°)")
    if correct_metrics:
        for metric in correct_metrics:
            angle = analysis[metric]['best_angle']
            value = analysis[metric]['best_value']
            print(f"   - {metric.upper():4s}: {angle:+5.1f}° (score={value:+.4f})")
    else:
        print("   None! All metrics failed to identify correct range.")

    print(f"\n❌ WRONG METRICS ({len(wrong_metrics)}):")
    print("   (Peak outside expected range)")
    if wrong_metrics:
        for metric in wrong_metrics:
            angle = analysis[metric]['best_angle']
            value = analysis[metric]['best_value']
            print(f"   - {metric.upper():4s}: {angle:+5.1f}° (score={value:+.4f})")

    print("\n" + "="*70)

    if correct_metrics:
        print(f"RECOMMENDATION: Use {', '.join([m.upper() for m in correct_metrics])} for rotation alignment")
    else:
        print("WARNING: No metric correctly identified parallel layer alignment!")
        print("May need to develop new metric or adjust search strategy.")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Test all rotation metrics across angle range',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--angle-start', type=float, default=-14,
                        help='Start angle (default: -14 degrees)')
    parser.add_argument('--angle-end', type=float, default=-6,
                        help='End angle (default: -6 degrees)')
    parser.add_argument('--step', type=float, default=0.5,
                        help='Angle step size (default: 0.5 degrees)')
    parser.add_argument('--expected-min', type=float, default=-10,
                        help='Expected optimal range minimum (default: -10)')
    parser.add_argument('--expected-max', type=float, default=-8,
                        help='Expected optimal range maximum (default: -8)')

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'

    # Load data
    overlap_v0, overlap_v1_y_aligned = load_alignment_data(data_dir)

    # Extract central B-scan
    bscan_v0, bscan_v1, z_center = extract_central_bscan(overlap_v0, overlap_v1_y_aligned)

    # Create mask
    mask = create_mask(bscan_v0, bscan_v1)

    # Test all metrics
    df = test_all_metrics_at_angles(
        bscan_v0, bscan_v1, mask,
        args.angle_start, args.angle_end, args.step
    )

    # Analyze results
    analysis = identify_best_metrics(df, args.expected_min, args.expected_max)

    # Save results
    csv_path = data_dir / 'metric_test_results.csv'
    save_results_csv(df, csv_path)

    # Visualize
    viz_path = data_dir / 'metric_comparison.png'
    visualize_all_metrics(df, analysis, args.expected_min, args.expected_max, viz_path)

    # Print summary
    print_summary(analysis)

    print(f"\n📊 Results saved:")
    print(f"   CSV:  {csv_path}")
    print(f"   Plot: {viz_path}")


if __name__ == '__main__':
    main()
