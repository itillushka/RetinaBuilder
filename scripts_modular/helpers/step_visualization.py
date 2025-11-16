"""
Step-by-Step Visualization Helper

Generates YZ plane (sagittal/side) views after each alignment step
to track the progression of alignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage


def visualize_yz_comparison(volume_0, volume_1, step_name, output_dir, overlap_bounds=None):
    """
    Generate YZ plane (sagittal/side view) comparison after an alignment step.

    Shows side-by-side and overlay views to visualize alignment quality.

    Args:
        volume_0: Reference volume
        volume_1: Moving volume (after current step)
        step_name: Name of the step (e.g., "Step1_XZ_Alignment")
        output_dir: Directory to save visualization
        overlap_bounds: Optional overlap region bounds for cropping
    """
    print(f"\n  Generating YZ view for {step_name}...")

    # Create MIP along X-axis (sagittal/side view)
    # Result shows Y (height) × Z (depth)
    mip_0 = np.max(volume_0, axis=1)  # Shape: (Y, Z)
    mip_1 = np.max(volume_1, axis=1)  # Shape: (Y, Z)

    # Crop to overlap region if provided (for clearer comparison)
    if overlap_bounds is not None:
        z_start, z_end = overlap_bounds.get('z', (0, mip_0.shape[1]))
        mip_0 = mip_0[:, z_start:z_end]
        mip_1 = mip_1[:, z_start:z_end]

    # Normalize for visualization
    mip_0_norm = (mip_0 - mip_0.min()) / (mip_0.max() - mip_0.min() + 1e-8)
    mip_1_norm = (mip_1 - mip_1.min()) / (mip_1.max() - mip_1.min() + 1e-8)

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Volume 0 (Reference)
    axes[0].imshow(mip_0_norm, cmap='gray', aspect='auto', origin='lower')
    axes[0].set_title('Volume 0 (Reference)\nYZ Plane (Side View)', fontweight='bold')
    axes[0].set_xlabel('Depth (Z) - B-scans →')
    axes[0].set_ylabel('Height (Y) ↑')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Volume 1 (Moving)
    axes[1].imshow(mip_1_norm, cmap='gray', aspect='auto', origin='lower')
    axes[1].set_title('Volume 1 (Moving)\nYZ Plane (Side View)', fontweight='bold')
    axes[1].set_xlabel('Depth (Z) - B-scans →')
    axes[1].set_ylabel('Height (Y) ↑')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Overlay (Red=Vol0, Green=Vol1, Yellow=Overlap)
    overlay = np.zeros((*mip_0_norm.shape, 3))
    overlay[..., 0] = mip_0_norm  # Red channel = Volume 0
    overlay[..., 1] = mip_1_norm  # Green channel = Volume 1
    # Where both are bright, we get yellow (red + green)

    axes[2].imshow(overlay, aspect='auto', origin='lower')
    axes[2].set_title('Overlay (Red=Vol0, Green=Vol1, Yellow=Aligned)\nYZ Plane (Side View)', fontweight='bold')
    axes[2].set_xlabel('Depth (Z) - B-scans →')
    axes[2].set_ylabel('Height (Y) ↑')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'{step_name} - Sagittal View Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / f'{step_name}_YZ_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path.name}")


def visualize_step2_bscan_alignment(step2_results, output_dir):
    """
    Generate comprehensive B-scan alignment visualization for Step 2.

    Shows:
    - Central B-scans before/after alignment
    - Overlay showing alignment quality
    - NCC curve with peak marked
    - Detected surfaces from contour method

    Args:
        step2_results: Results dictionary from step2_y_alignment
        output_dir: Directory to save visualization
    """
    print("\n  Generating Step 2 B-scan alignment visualization...")

    # Extract data from results
    bscan_v0 = step2_results['bscan_v0_central']
    bscan_v1_before = step2_results['bscan_v1_central']
    y_shift = step2_results['y_shift']
    ncc_scores = step2_results['ncc_scores']
    offsets_tested = step2_results['offsets_tested']
    contour_offset = step2_results['contour_y_offset']
    surface_v0 = step2_results['surface_v0']
    surface_v1 = step2_results['surface_v1']

    # Apply Y-shift to B-scan for visualization
    bscan_v1_after = ndimage.shift(
        bscan_v1_before, shift=(y_shift, 0),
        order=1, mode='constant', cval=0
    )

    # Normalize B-scans for visualization
    def normalize(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    b0_norm = normalize(bscan_v0)
    b1_before_norm = normalize(bscan_v1_before)
    b1_after_norm = normalize(bscan_v1_after)

    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 14))

    # ROW 1: B-scans before alignment
    ax1 = plt.subplot(3, 3, 1)
    ax1.imshow(b0_norm, cmap='gray', aspect='auto', origin='upper')
    ax1.set_title('Volume 0 (Reference)\nCentral B-scan', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Depth (Y) ↓')
    ax1.set_xlabel('Lateral (X) →')

    ax2 = plt.subplot(3, 3, 2)
    ax2.imshow(b1_before_norm, cmap='gray', aspect='auto', origin='upper')
    ax2.set_title('Volume 1 (Before Alignment)\nCentral B-scan', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Depth (Y) ↓')
    ax2.set_xlabel('Lateral (X) →')

    # Overlay before alignment
    ax3 = plt.subplot(3, 3, 3)
    overlay_before = np.zeros((*b0_norm.shape, 3))
    overlay_before[..., 0] = b0_norm  # Red = Vol0
    overlay_before[..., 1] = b1_before_norm  # Green = Vol1
    ax3.imshow(overlay_before, aspect='auto', origin='upper')
    ax3.set_title('Overlay BEFORE\n(Red=V0, Green=V1, Yellow=Aligned)', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Depth (Y) ↓')
    ax3.set_xlabel('Lateral (X) →')

    # ROW 2: B-scans after alignment
    ax4 = plt.subplot(3, 3, 4)
    ax4.imshow(b0_norm, cmap='gray', aspect='auto', origin='upper')
    ax4.set_title('Volume 0 (Reference)\nCentral B-scan', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Depth (Y) ↓')
    ax4.set_xlabel('Lateral (X) →')

    ax5 = plt.subplot(3, 3, 5)
    ax5.imshow(b1_after_norm, cmap='gray', aspect='auto', origin='upper')
    ax5.set_title(f'Volume 1 (After Y-shift={y_shift:+.1f}px)\nCentral B-scan', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Depth (Y) ↓')
    ax5.set_xlabel('Lateral (X) →')

    # Overlay after alignment
    ax6 = plt.subplot(3, 3, 6)
    overlay_after = np.zeros((*b0_norm.shape, 3))
    overlay_after[..., 0] = b0_norm  # Red = Vol0
    overlay_after[..., 1] = b1_after_norm  # Green = Vol1
    ax6.imshow(overlay_after, aspect='auto', origin='upper')
    ax6.set_title('Overlay AFTER\n(More Yellow = Better Alignment)', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Depth (Y) ↓')
    ax6.set_xlabel('Lateral (X) →')

    # ROW 3: NCC curve and surface detection
    # NCC curve
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(offsets_tested, ncc_scores, 'b-', linewidth=2, label='NCC Score')
    ax7.axvline(x=y_shift, color='red', linestyle='--', linewidth=2, label=f'Selected: {y_shift:+.1f}px')
    ax7.axvline(x=contour_offset, color='green', linestyle=':', linewidth=2, label=f'Contour: {contour_offset:+.1f}px')
    ax7.scatter([y_shift], [ncc_scores[offsets_tested == y_shift]], color='red', s=100, zorder=5)
    ax7.set_xlabel('Y-offset (pixels)', fontsize=11)
    ax7.set_ylabel('NCC Score', fontsize=11)
    ax7.set_title('NCC Search Curve\n(Peak = Best Alignment)', fontweight='bold', fontsize=11)
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=9)

    # Surface detection visualization - Volume 0
    ax8 = plt.subplot(3, 3, 8)
    ax8.imshow(b0_norm, cmap='gray', aspect='auto', origin='upper', alpha=0.7)
    x_coords = np.arange(len(surface_v0))
    ax8.plot(x_coords, surface_v0, 'r-', linewidth=2, label='Detected Surface')
    ax8.set_title(f'Volume 0 Surface\nMean Y: {surface_v0.mean():.1f}px', fontweight='bold', fontsize=11)
    ax8.set_ylabel('Depth (Y) ↓')
    ax8.set_xlabel('Lateral (X) →')
    ax8.legend(fontsize=9)

    # Surface detection visualization - Volume 1 (before shift)
    ax9 = plt.subplot(3, 3, 9)
    ax9.imshow(b1_before_norm, cmap='gray', aspect='auto', origin='upper', alpha=0.7)
    ax9.plot(x_coords, surface_v1, 'g-', linewidth=2, label='Detected Surface')
    ax9.set_title(f'Volume 1 Surface (Before)\nMean Y: {surface_v1.mean():.1f}px', fontweight='bold', fontsize=11)
    ax9.set_ylabel('Depth (Y) ↓')
    ax9.set_xlabel('Lateral (X) →')
    ax9.legend(fontsize=9)

    # Overall title
    ncc_offset_for_viz = step2_results.get('ncc_y_offset', 0)
    offset_diff = abs(y_shift - ncc_offset_for_viz)
    agreement = "[OK] High confidence!" if offset_diff < 3 else f"[WARNING] Methods differ ({offset_diff:.1f}px)"
    plt.suptitle(
        f'Step 2: Y-Alignment via Central B-scan Matching (Contour-Based)\n'
        f'Contour={y_shift:+.1f}px (PRIMARY), NCC={ncc_offset_for_viz:+.1f}px (validation) | {agreement}',
        fontsize=16, fontweight='bold'
    )

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / 'Step2_Bscan_Alignment_Details.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path.name}")


def visualize_all_steps(volume_0, step1_results, step2_results, step3_results, output_dir):
    """
    Generate YZ comparisons for all alignment steps.

    Args:
        volume_0: Reference volume
        step1_results: Results from Step 1 (XZ alignment)
        step2_results: Results from Step 2 (Y alignment)
        step3_results: Results from Step 3 (Z-rotation)
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("GENERATING STEP-BY-STEP YZ VISUALIZATIONS")
    print("="*70)

    overlap_bounds = step1_results.get('overlap_bounds', None)

    # After Step 1: XZ Alignment
    if 'volume_1_xz_aligned' in step1_results:
        visualize_yz_comparison(
            volume_0,
            step1_results['volume_1_xz_aligned'],
            'Step1_XZ_Alignment',
            output_dir,
            overlap_bounds
        )

    # After Step 2: Y Alignment
    if 'overlap_v1_y_aligned' in step2_results:
        # YZ comparison (sagittal view)
        # For Step 2, we need to reconstruct full volume_1_y_aligned from overlap regions
        # Note: This is for visualization - actual pipeline uses overlap regions
        print("\n  Note: Step 2 YZ view shows overlap regions (not full volume)")

        # Generate detailed B-scan alignment visualization
        visualize_step2_bscan_alignment(step2_results, output_dir)

    # After Step 3: Z-Rotation (if available)
    if step3_results and 'volume_1_rotated' in step3_results and step3_results['volume_1_rotated'] is not None:
        visualize_yz_comparison(
            volume_0,
            step3_results['volume_1_rotated'],
            'Step3_Z_Rotation',
            output_dir,
            overlap_bounds
        )

    print("\n✓ Step-by-step YZ visualizations complete!")
    print("="*70)
