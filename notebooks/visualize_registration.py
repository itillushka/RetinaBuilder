#!/usr/bin/env python3
"""
Visualize OCT Registration Results

Quick 2D visualization of the complete 3D registration pipeline results.
Shows aligned vessel skeletons, surfaces, and quality metrics.

Usage:
    python visualize_registration.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_registration_data(data_dir: Path):
    """Load all registration outputs."""
    print("Loading registration data...")

    data = {
        # Phase 1: Surfaces
        'surface_v0': np.load(data_dir / 'surface_peaks_volume0.npy'),
        'surface_v1': np.load(data_dir / 'surface_peaks_volume1.npy'),
        'surface_v1_aligned': np.load(data_dir / 'surface_v1_fully_aligned.npy'),

        # Phase 2: Vessels
        'skeleton_v0': np.load(data_dir / 'vessel_skeleton_volume0.npy'),
        'skeleton_v1': np.load(data_dir / 'vessel_skeleton_volume1.npy'),
        'skeleton_v1_aligned': np.load(data_dir / 'skeleton_v1_aligned.npy'),

        'bifurcations_v0': np.load(data_dir / 'bifurcation_coords_volume0.npy'),
        'bifurcations_v1': np.load(data_dir / 'bifurcation_coords_volume1.npy'),
        'bifurcations_v1_aligned': np.load(data_dir / 'bifurcations_v1_aligned.npy'),

        # Phase 3 & 4: Registration
        'xy_params': np.load(data_dir / 'xy_registration_params.npy', allow_pickle=True).item(),
        'registration_3d': np.load(data_dir / 'registration_3d_params.npy', allow_pickle=True).item(),
    }

    print("✓ All data loaded successfully")
    return data


def plot_registration_overview(data):
    """Create comprehensive registration visualization."""

    fig = plt.figure(figsize=(20, 12))

    # ========== Row 1: Vessel Skeletons ==========
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(data['skeleton_v0'], cmap='gray')
    ax1.set_title('Volume 0: Vessel Skeleton\n(Reference)', fontsize=11, fontweight='bold')
    ax1.axis('off')

    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(data['skeleton_v1'], cmap='gray')
    ax2.set_title('Volume 1: Vessel Skeleton\n(Before Alignment)', fontsize=11, fontweight='bold')
    ax2.axis('off')

    ax3 = plt.subplot(3, 4, 3)
    ax3.imshow(data['skeleton_v1_aligned'], cmap='gray')
    ax3.set_title('Volume 1: Vessel Skeleton\n(After XY Alignment)', fontsize=11, fontweight='bold')
    ax3.axis('off')

    ax4 = plt.subplot(3, 4, 4)
    ax4.imshow(data['skeleton_v0'], cmap='Reds', alpha=0.5)
    ax4.imshow(data['skeleton_v1_aligned'], cmap='Greens', alpha=0.5)
    ax4.set_title('Overlay: Aligned Vessels\n(Red=V0, Green=V1)', fontsize=11, fontweight='bold')
    ax4.axis('off')

    # ========== Row 2: Surface Maps ==========
    ax5 = plt.subplot(3, 4, 5)
    im5 = ax5.imshow(data['surface_v0'].T, aspect='auto', cmap='viridis', vmin=300, vmax=500)
    ax5.set_title('Volume 0: Surface Height\n(Reference)', fontsize=11, fontweight='bold')
    ax5.set_xlabel('X (lateral)')
    ax5.set_ylabel('Z (B-scan)')
    plt.colorbar(im5, ax=ax5, label='Y depth (px)')

    ax6 = plt.subplot(3, 4, 6)
    im6 = ax6.imshow(data['surface_v1'].T, aspect='auto', cmap='viridis', vmin=300, vmax=500)
    ax6.set_title('Volume 1: Surface Height\n(Before Alignment)', fontsize=11, fontweight='bold')
    ax6.set_xlabel('X (lateral)')
    ax6.set_ylabel('Z (B-scan)')
    plt.colorbar(im6, ax=ax6, label='Y depth (px)')

    ax7 = plt.subplot(3, 4, 7)
    im7 = ax7.imshow(data['surface_v1_aligned'].T, aspect='auto', cmap='viridis', vmin=300, vmax=500)
    ax7.set_title('Volume 1: Surface Height\n(After 3D Alignment)', fontsize=11, fontweight='bold')
    ax7.set_xlabel('X (lateral)')
    ax7.set_ylabel('Z (B-scan)')
    plt.colorbar(im7, ax=ax7, label='Y depth (px)')

    ax8 = plt.subplot(3, 4, 8)
    diff = np.abs(data['surface_v0'] - data['surface_v1_aligned'])
    im8 = ax8.imshow(diff.T, aspect='auto', cmap='hot', vmax=50)
    ax8.set_title(f'Surface Difference\nMean: {np.nanmean(diff):.2f} px', fontsize=11, fontweight='bold')
    ax8.set_xlabel('X (lateral)')
    ax8.set_ylabel('Z (B-scan)')
    plt.colorbar(im8, ax=ax8, label='|ΔY| (px)')

    # ========== Row 3: Cross-sections and metrics ==========
    ax9 = plt.subplot(3, 4, 9)
    mid_z = data['surface_v0'].shape[1] // 2
    ax9.plot(data['surface_v0'][:, mid_z], label='Vol 0', linewidth=2, color='blue')
    ax9.plot(data['surface_v1'][:, mid_z], label='Vol 1 (before)',
             linewidth=2, linestyle='--', alpha=0.5, color='orange')
    ax9.plot(data['surface_v1_aligned'][:, mid_z], label='Vol 1 (aligned)',
             linewidth=2, color='green')
    ax9.set_xlabel('X position')
    ax9.set_ylabel('Y position (depth, pixels)')
    ax9.set_title(f'Surface Cross-section (Z={mid_z})', fontsize=11, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.invert_yaxis()

    ax10 = plt.subplot(3, 4, 10)
    # Bifurcation overlay
    ax10.imshow(data['skeleton_v0'], cmap='gray', alpha=0.3)
    ax10.scatter(data['bifurcations_v0'][:, 1], data['bifurcations_v0'][:, 0],
                c='red', s=3, alpha=0.6, label=f'Vol 0 ({len(data["bifurcations_v0"])})')
    ax10.scatter(data['bifurcations_v1_aligned'][:, 1], data['bifurcations_v1_aligned'][:, 0],
                c='green', s=3, alpha=0.6, label=f'Vol 1 ({len(data["bifurcations_v1_aligned"])})')
    ax10.set_title('Bifurcation Landmarks\n(Aligned)', fontsize=11, fontweight='bold')
    ax10.legend()
    ax10.axis('off')

    # ========== Metrics Panel ==========
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')

    reg3d = data['registration_3d']
    xy_params = data['xy_params']

    metrics_text = f"""
    ═══════════════════════════════════
    3D REGISTRATION PARAMETERS
    ═══════════════════════════════════

    📍 XY Registration (Phase 3):
    ─────────────────────────────────
    Translation (dy, dx):
      ({reg3d['translation_xy'][0]:.2f}, {reg3d['translation_xy'][1]:.2f}) pixels

    Inlier ratio: {reg3d['xy_inlier_ratio']:.1%}
    Improvement: {reg3d['xy_improvement']*100:.1f}%

    📏 Depth Alignment (Phase 4):
    ─────────────────────────────────
    Method: {reg3d['depth_method_used']}
    Depth offset: {reg3d['depth_offset_final']:.2f} pixels
    Confidence: {reg3d['depth_confidence_final']:.1%}

    Surface difference:
      Before: {reg3d['surface_diff_before']:.2f} px
      After:  {reg3d['surface_diff_after']:.2f} px

    🎯 FINAL 3D TRANSFORM:
    ─────────────────────────────────
    X (lateral):  {reg3d['transform_3d']['dx']:.2f} px
    Y (depth):    {reg3d['transform_3d']['dy']:.2f} px
    Z (B-scan):   {reg3d['transform_3d']['dz']:.2f} px

    ═══════════════════════════════════
    """

    ax11.text(0.1, 0.95, metrics_text, transform=ax11.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # ========== Checkerboard ==========
    ax12 = plt.subplot(3, 4, 12)
    # Create checkerboard for vessel skeletons
    checker_size = 100
    h, w = data['skeleton_v0'].shape
    checkerboard = np.zeros((h, w), dtype=bool)
    for i in range(0, h, checker_size):
        for j in range(0, w, checker_size):
            if ((i // checker_size) + (j // checker_size)) % 2 == 0:
                checkerboard[i:i+checker_size, j:j+checker_size] = True

    checker_img = data['skeleton_v0'].copy()
    checker_img[~checkerboard] = data['skeleton_v1_aligned'][~checkerboard]
    ax12.imshow(checker_img, cmap='gray')
    ax12.set_title('Checkerboard Pattern\n(Alignment Verification)', fontsize=11, fontweight='bold')
    ax12.axis('off')

    plt.suptitle('OCT Volume Registration Results - Complete Pipeline',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    return fig


def main():
    """Main visualization function."""
    print("=" * 70)
    print("OCT REGISTRATION VISUALIZATION")
    print("=" * 70)
    print()

    # Set working directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'

    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        print("   Make sure you've run all 4 notebooks first!")
        sys.exit(1)

    # Load data
    data = load_registration_data(data_dir)

    # Create visualization
    print("\nGenerating visualization...")
    fig = plot_registration_overview(data)

    # Save figure
    output_path = script_dir / 'registration_results.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")

    # Display
    print("\nDisplaying interactive window...")
    print("(Close the window to exit)")
    plt.show()

    print("\n" + "=" * 70)
    print("✓ Visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
