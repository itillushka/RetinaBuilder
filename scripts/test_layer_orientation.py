#!/usr/bin/env python3
"""
Test Layer Orientation Detection

Tests the Hough and gradient-based layer orientation detection methods
on debug B-scans at various rotation angles to determine which rotation
creates truly horizontal layers.

Usage:
    python test_layer_orientation.py
    python test_layer_orientation.py --method gradient
    python test_layer_orientation.py --visualize-all
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

# Import our layer orientation detector
from layer_orientation_detection import LayerOrientationDetector, visualize_detection

print("=" * 70)
print("LAYER ORIENTATION DETECTION TEST")
print("=" * 70)


def test_orientation_at_angles(bscan_v0, bscan_v1, mask, angles, method='hough'):
    """
    Test layer orientation detection at various rotation angles.

    The correct rotation should bring the measured angle close to 0°.

    Args:
        bscan_v0: Reference B-scan
        bscan_v1: Moving B-scan (to be rotated)
        mask: Valid region mask
        angles: List of rotation angles to test
        method: 'hough' or 'gradient'

    Returns:
        DataFrame with results
    """
    results = []

    print(f"\nTesting layer orientation detection ({method}) at {len(angles)} angles...")
    print(f"Goal: Find rotation that brings layer angle closest to 0° (horizontal)")

    for angle in angles:
        print(f"  Testing rotation: {angle:+.1f}°...", end='\r')

        # Rotate B-scan V1
        bscan_v1_rotated = ndimage.rotate(
            bscan_v1,
            angle,
            axes=(0, 1),  # Y and X
            reshape=False,
            order=1,
            mode='constant',
            cval=0
        )

        # Rotate mask
        mask_rotated = ndimage.rotate(
            mask.astype(float),
            angle,
            axes=(0, 1),
            reshape=False,
            order=0,
            mode='constant',
            cval=0
        ) > 0.5

        # Detect layer orientation in rotated B-scan
        try:
            layer_angle, confidence, vis_data = LayerOrientationDetector.detect_orientation(
                bscan_v1_rotated,
                mask=mask_rotated,
                method=method,
                visualize=False
            )

            # Also test on reference B-scan for comparison
            layer_angle_v0, confidence_v0, _ = LayerOrientationDetector.detect_orientation(
                bscan_v0,
                mask=mask,
                method=method,
                visualize=False
            )

        except Exception as e:
            print(f"\n  Error at angle {angle}: {e}")
            layer_angle = np.nan
            confidence = 0.0
            layer_angle_v0 = np.nan
            confidence_v0 = 0.0

        results.append({
            'rotation_angle': angle,
            'detected_layer_angle_v1': layer_angle,
            'confidence_v1': confidence,
            'detected_layer_angle_v0': layer_angle_v0,
            'confidence_v0': confidence_v0,
            'abs_layer_angle': abs(layer_angle),  # How far from horizontal
            'angle_difference': abs(layer_angle - layer_angle_v0)  # Difference from V0
        })

    print(f"  Completed testing {len(angles)} angles.    ")

    return pd.DataFrame(results)


def find_best_rotation(df):
    """
    Find rotation that creates most horizontal layers.

    The best rotation is when detected_layer_angle_v1 is closest to 0°.

    Args:
        df: Results DataFrame

    Returns:
        Dict with best rotation info
    """
    # Find rotation where layers are most horizontal
    best_idx = df['abs_layer_angle'].idxmin()

    best_rotation = df.loc[best_idx, 'rotation_angle']
    best_layer_angle = df.loc[best_idx, 'detected_layer_angle_v1']
    best_confidence = df.loc[best_idx, 'confidence_v1']

    return {
        'best_rotation_angle': best_rotation,
        'layer_angle_at_best': best_layer_angle,
        'confidence': best_confidence,
        'abs_angle': abs(best_layer_angle)
    }


def visualize_results(df, method, save_path):
    """
    Visualize how detected layer angle changes with rotation.

    Args:
        df: Results DataFrame
        method: Detection method used
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Detected layer angle vs rotation angle
    ax = axes[0, 0]
    ax.plot(df['rotation_angle'], df['detected_layer_angle_v1'], 'b-', linewidth=2, label='V1 (rotated)')
    ax.axhline(0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Horizontal (0°)')
    ax.axvspan(-10, -8, alpha=0.2, color='orange', label='Visual optimal (-10° to -8°)')

    # Mark best rotation
    best_idx = df['abs_layer_angle'].idxmin()
    best_rot = df.loc[best_idx, 'rotation_angle']
    best_angle = df.loc[best_idx, 'detected_layer_angle_v1']
    ax.axvline(best_rot, color='red', linestyle='--', linewidth=2, label=f'Best: {best_rot:+.1f}°')
    ax.plot(best_rot, best_angle, 'ro', markersize=12)

    ax.set_xlabel('Applied Rotation Angle (degrees)', fontsize=11)
    ax.set_ylabel('Detected Layer Angle (degrees)', fontsize=11)
    ax.set_title(f'Layer Angle Detection ({method.upper()})\n' +
                 f'Goal: Line crosses 0° at optimal rotation', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Plot 2: Absolute layer angle (distance from horizontal)
    ax = axes[0, 1]
    ax.plot(df['rotation_angle'], df['abs_layer_angle'], 'r-', linewidth=2)
    ax.axvspan(-10, -8, alpha=0.2, color='orange', label='Visual optimal')
    ax.axvline(best_rot, color='red', linestyle='--', linewidth=2, label=f'Best: {best_rot:+.1f}°')
    ax.plot(best_rot, df.loc[best_idx, 'abs_layer_angle'], 'ro', markersize=12)

    ax.set_xlabel('Applied Rotation Angle (degrees)', fontsize=11)
    ax.set_ylabel('Absolute Layer Angle (degrees)', fontsize=11)
    ax.set_title('Distance from Horizontal\n' +
                 'Lower = more horizontal layers', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Plot 3: Confidence scores
    ax = axes[1, 0]
    ax.plot(df['rotation_angle'], df['confidence_v1'], 'g-', linewidth=2, label='V1 confidence')
    ax.axvspan(-10, -8, alpha=0.2, color='orange')
    ax.axvline(best_rot, color='red', linestyle='--', linewidth=2)

    ax.set_xlabel('Applied Rotation Angle (degrees)', fontsize=11)
    ax.set_ylabel('Detection Confidence', fontsize=11)
    ax.set_title('Detection Confidence Scores', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')

    # Create summary text
    summary_lines = [
        f"LAYER ORIENTATION DETECTION SUMMARY",
        f"Method: {method.upper()}",
        f"",
        f"Best Rotation: {best_rot:+.1f}°",
        f"Layer Angle at Best: {best_angle:+.2f}°",
        f"Confidence: {df.loc[best_idx, 'confidence_v1']:.3f}",
        f"",
        f"Key angles:",
    ]

    # Add key rotation angles
    for rot_angle in [-10, -9, -8, 0, best_rot, 20, 22]:
        if rot_angle in df['rotation_angle'].values:
            idx = df[df['rotation_angle'] == rot_angle].index[0]
            layer_ang = df.loc[idx, 'detected_layer_angle_v1']
            conf = df.loc[idx, 'confidence_v1']
            marker = " ← BEST" if rot_angle == best_rot else ""
            marker = " ← Visual" if rot_angle in [-10, -9, -8] and rot_angle != best_rot else marker
            summary_lines.append(f"  {rot_angle:+5.1f}° → layer angle: {layer_ang:+6.2f}° (conf: {conf:.3f}){marker}")

    summary_text = '\n'.join(summary_lines)
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Layer Orientation Detection Test - {method.upper()} Method\n' +
                 f'Finding rotation that creates horizontal layers',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved results visualization: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Test layer orientation detection on debug B-scans',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--method', choices=['hough', 'gradient', 'both'],
                       default='both',
                       help='Detection method to test (default: both)')
    parser.add_argument('--visualize-all', action='store_true',
                       help='Create detailed visualizations at key angles')

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
    # Focus on critical range with key angles
    angles = np.concatenate([
        np.arange(-15, -5, 0.5),   # Fine resolution around expected optimal
        np.arange(-5, 5, 1.0),      # Medium resolution around zero
        np.arange(5, 26, 1.0)       # Coarser for positive range
    ])

    # Also ensure specific key angles are included
    key_angles = [-10, -9, -8, 0, 20, 22, 24]
    for angle in key_angles:
        if angle not in angles:
            angles = np.append(angles, angle)
    angles = np.sort(angles)

    # Test methods
    methods_to_test = []
    if args.method == 'both':
        methods_to_test = ['hough', 'gradient']
    else:
        methods_to_test = [args.method]

    all_results = {}

    for method in methods_to_test:
        print(f"\n{'='*70}")
        print(f"TESTING {method.upper()} METHOD")
        print(f"{'='*70}")

        # Test orientation detection
        print(f"\n2. Testing {method} orientation detection...")
        df = test_orientation_at_angles(bscan_v0, bscan_v1, mask, angles, method=method)

        # Find best rotation
        print(f"\n3. Analyzing results...")
        best_info = find_best_rotation(df)

        # Print summary
        print("\n" + "="*70)
        print(f"{method.upper()} METHOD - RESULTS")
        print("="*70)
        print(f"Best rotation angle:  {best_info['best_rotation_angle']:+.1f}°")
        print(f"Layer angle at best:  {best_info['layer_angle_at_best']:+.2f}°")
        print(f"Confidence:           {best_info['confidence']:.3f}")
        print(f"Abs angle:            {best_info['abs_angle']:.2f}°")
        print()

        # Compare with visual expectation
        visual_range = (-10, -8)
        in_visual_range = visual_range[0] <= best_info['best_rotation_angle'] <= visual_range[1]
        if in_visual_range:
            print(f"✅ MATCHES visual optimal range ({visual_range[0]}° to {visual_range[1]}°)")
        else:
            print(f"⚠️  DIFFERS from visual optimal range ({visual_range[0]}° to {visual_range[1]}°)")
            print(f"   Difference: {abs(best_info['best_rotation_angle'] - (-9)):.1f}° from -9°")

        # Check at specific angles
        print("\nLayer angles at key rotations:")
        for rot_angle in [-10, -9, -8, 0, 20, 22]:
            if rot_angle in df['rotation_angle'].values:
                idx = df[df['rotation_angle'] == rot_angle].index[0]
                layer_ang = df.loc[idx, 'detected_layer_angle_v1']
                conf = df.loc[idx, 'confidence_v1']
                marker = " ← BEST" if rot_angle == best_info['best_rotation_angle'] else ""
                print(f"  Rotation {rot_angle:+5.1f}° → Layer angle: {layer_ang:+6.2f}° (conf: {conf:.3f}){marker}")

        print("="*70)

        # Save results
        csv_path = data_dir / f'layer_orientation_{method}_results.csv'
        df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"\n✓ Saved CSV: {csv_path}")

        # Visualize
        viz_path = data_dir / f'layer_orientation_{method}_test.png'
        visualize_results(df, method, viz_path)

        # Store results
        all_results[method] = {
            'df': df,
            'best_info': best_info
        }

        # Detailed visualizations at key angles
        if args.visualize_all:
            print(f"\n4. Creating detailed visualizations at key angles...")
            key_viz_angles = [-10, -9, -8, 0, 20, 22, best_info['best_rotation_angle']]
            key_viz_angles = sorted(set([a for a in key_viz_angles if a in df['rotation_angle'].values]))

            for viz_angle in key_viz_angles:
                print(f"  Creating visualization at {viz_angle:+.1f}°...")

                # Rotate B-scan
                bscan_rotated = ndimage.rotate(bscan_v1, viz_angle, axes=(0, 1),
                                               reshape=False, order=1, mode='constant', cval=0)
                mask_rotated = ndimage.rotate(mask.astype(float), viz_angle, axes=(0, 1),
                                              reshape=False, order=0, mode='constant', cval=0) > 0.5

                # Detect with visualization
                angle, conf, vis_data = LayerOrientationDetector.detect_orientation(
                    bscan_rotated, mask=mask_rotated, method=method, visualize=True
                )

                # Save visualization
                viz_save_path = data_dir / f'layer_{method}_rot{viz_angle:+.0f}.png'
                visualize_detection(
                    bscan_rotated, angle, conf, vis_data, method=method,
                    save_path=viz_save_path,
                    title=f'Rotation: {viz_angle:+.1f}° | Detected Layer Angle: {angle:+.2f}°'
                )

            print(f"  ✓ Created {len(key_viz_angles)} detailed visualizations")

    # Compare methods if both were tested
    if len(methods_to_test) > 1:
        print(f"\n{'='*70}")
        print("METHOD COMPARISON")
        print("="*70)
        for method in methods_to_test:
            best = all_results[method]['best_info']
            print(f"{method.upper():10s}: {best['best_rotation_angle']:+6.1f}° " +
                  f"(layer angle: {best['layer_angle_at_best']:+6.2f}°, " +
                  f"conf: {best['confidence']:.3f})")
        print("="*70)

    print("\nTest complete!")


if __name__ == '__main__':
    main()
