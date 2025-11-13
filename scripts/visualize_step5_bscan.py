"""
Visualize Step 5 B-spline registration effect on B-scan images.

Shows before/after comparison of actual B-scan images to demonstrate
the alignment improvement from non-rigid registration.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import map_coordinates


def apply_deformation_to_slice(volume_slice, deformation_y, deformation_x):
    """
    Apply 2D deformation field to a single B-scan slice.

    Args:
        volume_slice: 2D array (Y, X) - the B-scan image
        deformation_y: 2D array (Y, X) - Y-displacement at each pixel
        deformation_x: 2D array (Y, X) - X-displacement at each pixel

    Returns:
        Deformed B-scan image
    """
    Y, X = volume_slice.shape

    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(Y), np.arange(X), indexing='ij')

    # Apply deformations (subtract because we're warping the moving image to fixed)
    y_deformed = y_coords - deformation_y
    x_deformed = x_coords - deformation_x

    # Stack coordinates for map_coordinates
    coords = np.array([y_deformed, x_deformed])

    # Apply deformation with cubic interpolation
    deformed_slice = map_coordinates(volume_slice, coords, order=3, mode='constant', cval=0)

    return deformed_slice


def main():
    # Setup paths
    data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'
    checkpoint_path = data_dir / 'step4_checkpoint.npz'
    deformation_path = data_dir / 'step5_deformation_field.npy'
    step3_results_path = data_dir / 'step3_results.npy'

    # Check files exist
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Run the pipeline first: python scripts/alignment_pipeline.py --steps 1 2 3")
        return

    if not deformation_path.exists():
        print(f"Error: Deformation field not found at {deformation_path}")
        print("Step 5 needs to complete successfully first")
        return

    print("="*70)
    print("STEP 5 B-SPLINE REGISTRATION - B-SCAN VISUALIZATION")
    print("="*70)

    # Load checkpoint data
    print("\n1. Loading checkpoint data...")
    checkpoint = np.load(checkpoint_path)
    overlap_v0 = checkpoint['overlap_v0']  # Fixed volume (reference)
    overlap_v1_before = checkpoint['overlap_v1_windowed']  # Moving volume after Step 4
    ncc_step4 = float(checkpoint['ncc_step4'])

    print(f"   ✓ Overlap volumes: {overlap_v0.shape}")
    print(f"   ✓ Step 4 NCC: {ncc_step4:.4f}")

    # Load deformation field
    print("\n2. Loading deformation field...")
    deformation_field = np.load(deformation_path)
    print(f"   ✓ Deformation field: {deformation_field.shape}")

    # Load Step 5 results for final NCC
    step3_results = np.load(step3_results_path, allow_pickle=True).item()
    ncc_step5 = step3_results.get('ncc_after_bspline', ncc_step4)
    print(f"   ✓ Step 5 NCC: {ncc_step5:.4f}")

    # Select B-scan slice to visualize (middle slice)
    Z = overlap_v0.shape[2]
    slice_z = Z // 2
    print(f"\n3. Visualizing B-scan slice {slice_z}/{Z}...")

    # Extract B-scan slices
    bscan_v0 = overlap_v0[:, :, slice_z]  # Fixed (reference)
    bscan_v1_before = overlap_v1_before[:, :, slice_z]  # Moving before Step 5

    # Extract deformation for this slice
    deformation_y = deformation_field[:, :, slice_z, 0]
    deformation_x = deformation_field[:, :, slice_z, 1]

    # Apply deformation to create "after Step 5" image
    print("   Applying B-spline deformation to B-scan...")
    bscan_v1_after = apply_deformation_to_slice(bscan_v1_before, deformation_y, deformation_x)

    # Calculate difference images
    diff_before = np.abs(bscan_v0 - bscan_v1_before)
    diff_after = np.abs(bscan_v0 - bscan_v1_after)

    # Create comprehensive visualization
    print("\n4. Creating visualization...")
    fig = plt.figure(figsize=(20, 12))

    # Row 1: B-scan images
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(bscan_v0, cmap='gray', aspect='auto')
    ax1.set_title(f'Volume 0 (Reference)\nB-scan {slice_z}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (A-scans)', fontsize=11)
    ax1.set_ylabel('Y (depth)', fontsize=11)
    plt.colorbar(im1, ax=ax1, label='Intensity')

    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(bscan_v1_before, cmap='gray', aspect='auto')
    ax2.set_title(f'Volume 1 BEFORE Step 5\n(After Step 4, NCC={ncc_step4:.4f})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (A-scans)', fontsize=11)
    ax2.set_ylabel('Y (depth)', fontsize=11)
    plt.colorbar(im2, ax=ax2, label='Intensity')

    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(bscan_v1_after, cmap='gray', aspect='auto')
    ax3.set_title(f'Volume 1 AFTER Step 5\n(B-spline FFD, NCC={ncc_step5:.4f})', fontsize=14, fontweight='bold', color='green')
    ax3.set_xlabel('X (A-scans)', fontsize=11)
    ax3.set_ylabel('Y (depth)', fontsize=11)
    plt.colorbar(im3, ax=ax3, label='Intensity')

    # Row 2: Difference images
    vmax_diff = max(diff_before.max(), diff_after.max())

    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(diff_before, cmap='hot', aspect='auto', vmin=0, vmax=vmax_diff)
    ax4.set_title('Difference BEFORE Step 5\n|Volume 0 - Volume 1|', fontsize=14, fontweight='bold')
    ax4.set_xlabel('X (A-scans)', fontsize=11)
    ax4.set_ylabel('Y (depth)', fontsize=11)
    plt.colorbar(im4, ax=ax4, label='Absolute Difference')
    mean_diff_before = np.mean(diff_before)
    ax4.text(0.02, 0.98, f'Mean diff: {mean_diff_before:.2f}',
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(diff_after, cmap='hot', aspect='auto', vmin=0, vmax=vmax_diff)
    ax5.set_title('Difference AFTER Step 5\n|Volume 0 - Volume 1|', fontsize=14, fontweight='bold', color='green')
    ax5.set_xlabel('X (A-scans)', fontsize=11)
    ax5.set_ylabel('Y (depth)', fontsize=11)
    plt.colorbar(im5, ax=ax5, label='Absolute Difference')
    mean_diff_after = np.mean(diff_after)
    improvement = (mean_diff_before - mean_diff_after) / mean_diff_before * 100
    ax5.text(0.02, 0.98, f'Mean diff: {mean_diff_after:.2f}\nImprovement: {improvement:.1f}%',
             transform=ax5.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Difference reduction map
    ax6 = plt.subplot(2, 3, 6)
    diff_reduction = diff_before - diff_after
    im6 = ax6.imshow(diff_reduction, cmap='RdYlGn', aspect='auto',
                     vmin=-vmax_diff*0.3, vmax=vmax_diff*0.3)
    ax6.set_title('Error Reduction Map\n(Green = Improved, Red = Worse)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('X (A-scans)', fontsize=11)
    ax6.set_ylabel('Y (depth)', fontsize=11)
    plt.colorbar(im6, ax=ax6, label='Difference Reduction')

    # Add deformation magnitude overlay
    deformation_mag = np.sqrt(deformation_y**2 + deformation_x**2)
    mean_deform = np.mean(deformation_mag)
    max_deform = np.max(deformation_mag)
    ax6.text(0.02, 0.98, f'Mean deformation: {mean_deform:.2f}px\nMax deformation: {max_deform:.2f}px',
             transform=ax6.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Overall title
    fig.suptitle(f'Step 5 B-Spline Registration Effect on B-scan {slice_z}\n'
                 f'NCC Improvement: {ncc_step4:.4f} → {ncc_step5:.4f} (+{(ncc_step5-ncc_step4)*100:.2f}%)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = data_dir / 'step5_bscan_comparison.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path.name}")

    # Also create a zoomed version showing a region of interest
    print("\n5. Creating zoomed region-of-interest view...")
    create_zoomed_view(bscan_v0, bscan_v1_before, bscan_v1_after,
                       deformation_y, deformation_x, slice_z, data_dir)

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - {output_path}")
    print(f"  - {data_dir / 'step5_bscan_zoomed.png'}")
    print(f"\nKey findings:")
    print(f"  - Mean difference BEFORE Step 5: {mean_diff_before:.2f}")
    print(f"  - Mean difference AFTER Step 5:  {mean_diff_after:.2f}")
    print(f"  - Error reduction: {improvement:.1f}%")
    print(f"  - Mean deformation applied: {mean_deform:.2f} pixels")


def create_zoomed_view(bscan_v0, bscan_v1_before, bscan_v1_after,
                       deformation_y, deformation_x, slice_z, data_dir):
    """Create zoomed view of a region with visible layer structure."""

    # Select region with visible layer structure (upper portion)
    Y, X = bscan_v0.shape
    y_start = Y // 4
    y_end = 3 * Y // 4
    x_start = X // 4
    x_end = 3 * X // 4

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, img, title in zip(axes,
                              [bscan_v0, bscan_v1_before, bscan_v1_after],
                              ['Volume 0 (Reference)',
                               f'Volume 1 BEFORE Step 5',
                               'Volume 1 AFTER Step 5']):
        zoomed = img[y_start:y_end, x_start:x_end]
        im = ax.imshow(zoomed, cmap='gray', aspect='auto')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (A-scans)', fontsize=11)
        ax.set_ylabel('Y (depth)', fontsize=11)
        plt.colorbar(im, ax=ax, label='Intensity')

    # Highlight the improved alignment with colored title for "AFTER"
    axes[2].set_title('Volume 1 AFTER Step 5', fontsize=14, fontweight='bold', color='green')

    fig.suptitle(f'Zoomed View: B-scan {slice_z} - Retinal Layer Detail\n'
                 f'Region: Y[{y_start}:{y_end}], X[{x_start}:{x_end}]',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_path = data_dir / 'step5_bscan_zoomed.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path.name}")


if __name__ == "__main__":
    main()
