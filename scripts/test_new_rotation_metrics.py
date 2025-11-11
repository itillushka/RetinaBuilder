#!/usr/bin/env python3
"""
Test New Rotation Metrics - Evaluate Latest Metrics

Tests the 3 new metrics (LCS, DGH, VPC) alongside previous metrics
to determine which correctly identifies parallel retinal layers.

Uses the debug B-scans saved from the actual pipeline data.

Usage:
    python test_new_rotation_metrics.py
    python test_new_rotation_metrics.py --full-range
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
print("NEW ROTATION METRICS EVALUATION")
print("="*70)


def test_metrics_on_angles(bscan_v0, bscan_v1, mask, angles):
    """
    Test all 15 metrics across given angles.

    Returns:
        DataFrame with metric scores at each angle
    """
    results = []

    print(f"\nTesting {len(angles)} rotation angles...")
    print(f"Range: {angles[0]:.1f}° to {angles[-1]:.1f}°")

    for idx, angle in enumerate(angles):
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

        # Calculate all metrics
        print(f"  Testing angle {angle:+.1f}°...", end='\r')

        metrics = RotationMetrics.evaluate_all(
            bscan_v0,
            bscan_v1_rotated,
            mask=mask
        )

        # Add angle to results
        metrics['angle'] = angle
        results.append(metrics)

    print(f"  Completed testing {len(angles)} angles.    ")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Reorder columns (angle first)
    cols = ['angle'] + [col for col in df.columns if col != 'angle']
    df = df[cols]

    return df


def find_best_angles(df):
    """
    Find the best angle for each metric.

    Returns:
        Dictionary with metric: best_angle mappings
    """
    best_angles = {}

    # List all metrics (excluding angle column)
    metrics = [col for col in df.columns if col != 'angle']

    for metric in metrics:
        if metric == 'mse':
            # MSE: lower is better
            best_idx = df[metric].idxmin()
        else:
            # All others: higher is better
            best_idx = df[metric].idxmax()

        best_angles[metric] = df.loc[best_idx, 'angle']

    return best_angles


def visualize_new_metrics_comparison(df, output_path):
    """
    Create focused visualization of the 3 new metrics plus RCP.

    Args:
        df: Results DataFrame
        output_path: Path to save visualization
    """
    print("\nCreating new metrics visualization...")

    # Focus on key metrics
    key_metrics = ['rcp', 'lcs', 'dgh', 'vpc']
    metric_labels = {
        'rcp': 'RCP (Row-Wise Correlation Profile)\nFailed metric - peaks at +22°',
        'lcs': 'LCS (Layer Contour Smoothness)\nDetects smooth horizontal layers',
        'dgh': 'DGH (Directional Gradient Histogram)\nMeasures horizontalness',
        'vpc': 'VPC (Vertical Profile Correlation)\nHandles curved layers'
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    for idx, metric in enumerate(key_metrics):
        ax = axes[idx]

        # Plot metric vs angle
        ax.plot(df['angle'], df[metric], 'b-', linewidth=2)

        # Highlight expected optimal range
        ax.axvspan(-10, -8, alpha=0.2, color='green',
                   label='Visual optimal (-10° to -8°)')

        # Mark best angle
        if metric == 'mse':
            best_idx = df[metric].idxmin()
        else:
            best_idx = df[metric].idxmax()
        best_angle = df.loc[best_idx, 'angle']
        best_value = df.loc[best_idx, metric]

        # Color based on whether it's in expected range
        marker_color = 'green' if -10 <= best_angle <= -8 else 'red'
        ax.axvline(best_angle, color=marker_color, linestyle='--', linewidth=2,
                   label=f'Peak: {best_angle:+.1f}°')
        ax.plot(best_angle, best_value, 'o', color=marker_color, markersize=10)

        # Also mark +22° (where RCP peaks)
        if metric != 'rcp':
            ax.axvline(22, color='orange', linestyle=':', alpha=0.5,
                       label='RCP peak (+22°)')

        # Labels and title
        ax.set_xlabel('Rotation Angle (degrees)', fontsize=10)
        ax.set_ylabel('Metric Score', fontsize=10)
        ax.set_title(metric_labels[metric], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    plt.suptitle('New Metrics for True Parallel Layer Detection\n' +
                 'Goal: Find metrics that peak at -10° to -8° (visually optimal)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization: {output_path}")
    plt.close()


def create_comparison_table(df, best_angles):
    """
    Create a comparison table of all metrics.

    Returns:
        DataFrame with metric performance summary
    """
    summary_data = []

    for metric, best_angle in best_angles.items():
        # Check if in expected range
        in_expected = -10 <= best_angle <= -8

        # Get score at best angle
        best_idx = df[df['angle'] == best_angle].index[0]
        best_score = df.loc[best_idx, metric]

        # Get score at -9° (middle of expected range)
        expected_idx = df[(df['angle'] >= -9.5) & (df['angle'] <= -8.5)].index
        if len(expected_idx) > 0:
            expected_score = df.loc[expected_idx[0], metric]
        else:
            expected_score = np.nan

        summary_data.append({
            'Metric': metric.upper(),
            'Best Angle': f'{best_angle:+.1f}°',
            'Best Score': f'{best_score:.4f}',
            'Score at -9°': f'{expected_score:.4f}' if not np.isnan(expected_score) else 'N/A',
            'Correct?': '✅' if in_expected else '❌'
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description='Test new rotation metrics on debug B-scans',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--full-range', action='store_true',
                        help='Test full -30° to +30° range (default: -15° to +25°)')

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'

    # Load debug B-scans
    print("\n1. Loading debug B-scans...")
    bscan_v0 = np.load(data_dir / 'debug_bscan_v0.npy')
    bscan_v1 = np.load(data_dir / 'debug_bscan_v1.npy')
    mask = np.load(data_dir / 'debug_mask.npy')

    print(f"  ✓ B-scan V0: {bscan_v0.shape}")
    print(f"  ✓ B-scan V1: {bscan_v1.shape}")
    print(f"  ✓ Mask: {mask.sum()} valid pixels ({100*mask.sum()/mask.size:.1f}%)")

    # Define angles to test
    if args.full_range:
        angles = np.arange(-30, 31, 1.0)
    else:
        # Focus on critical range with finer resolution
        angles = np.concatenate([
            np.arange(-15, -5, 0.5),   # Fine resolution around expected optimal
            np.arange(-5, 5, 1.0),      # Medium resolution around zero
            np.arange(5, 26, 1.0)       # Coarser for positive range
        ])

    # Test all metrics
    print(f"\n2. Testing all 15 metrics...")
    df = test_metrics_on_angles(bscan_v0, bscan_v1, mask, angles)

    # Find best angles
    print(f"\n3. Analyzing results...")
    best_angles = find_best_angles(df)

    # Create comparison table
    summary_df = create_comparison_table(df, best_angles)

    # Print summary
    print("\n" + "="*70)
    print("METRIC COMPARISON SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))

    # Identify correct metrics
    correct_metrics = []
    for metric, angle in best_angles.items():
        if -10 <= angle <= -8:
            correct_metrics.append(metric.upper())

    print("\n" + "="*70)
    if correct_metrics:
        print(f"✅ METRICS THAT CORRECTLY IDENTIFY -10° to -8°:")
        for metric in correct_metrics:
            print(f"   - {metric}")
    else:
        print("❌ NO METRICS correctly identified the expected range!")

    # Check our new metrics specifically
    new_metrics = ['lcs', 'dgh', 'vpc']
    print(f"\n📊 NEW METRICS PERFORMANCE:")
    for metric in new_metrics:
        angle = best_angles[metric]
        status = "✅ CORRECT" if -10 <= angle <= -8 else f"❌ Wrong ({angle:+.1f}°)"
        print(f"   {metric.upper()}: {status}")

    print("="*70)

    # Save results
    csv_path = data_dir / 'new_metric_test_results.csv'
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n✓ Saved CSV results: {csv_path}")

    # Visualize
    viz_path = data_dir / 'new_metric_comparison.png'
    visualize_new_metrics_comparison(df, viz_path)

    # Additional visualization: all 15 metrics
    if len(df.columns) > 5:
        print("\nCreating comprehensive visualization...")
        create_all_metrics_plot(df, data_dir / 'all_15_metrics_comparison.png')


def create_all_metrics_plot(df, output_path):
    """Create a plot showing all 15 metrics."""
    # Exclude angle column
    metrics = [col for col in df.columns if col != 'angle']

    # Create subplots
    n_metrics = len(metrics)
    n_cols = 4
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.plot(df['angle'], df[metric], 'b-', linewidth=1)
        ax.axvspan(-10, -8, alpha=0.2, color='green')

        # Find peak
        if metric == 'mse':
            best_idx = df[metric].idxmin()
        else:
            best_idx = df[metric].idxmax()
        best_angle = df.loc[best_idx, 'angle']

        # Mark peak
        marker_color = 'green' if -10 <= best_angle <= -8 else 'red'
        ax.axvline(best_angle, color=marker_color, linestyle='--', alpha=0.5)

        ax.set_title(f'{metric.upper()} (peak: {best_angle:+.1f}°)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Angle', fontsize=8)
        ax.set_ylabel('Score', fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('All 15 Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved all metrics plot: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()