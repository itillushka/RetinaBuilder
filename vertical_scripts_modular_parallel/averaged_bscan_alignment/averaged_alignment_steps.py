"""
Alignment functions adapted for 2D averaged X-sections.

VERTICAL PIPELINE: This module works with X-sections (Y, Z) instead of B-scans (Y, X).
For vertical volumes, alignment happens along Z-axis instead of X-axis.

This module provides Y-alignment and rotation alignment specifically
for averaged X-sections, reusing the proven contour-based methods from
the full volume pipeline.
"""

import numpy as np
from scipy import ndimage
import sys
from pathlib import Path

# Add parent directory to path to import helpers
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.rotation_alignment import (
    find_optimal_rotation_z_contour,
    preprocess_oct_for_visualization,
    detect_contour_surface,
    calculate_ncc
)
from steps.step2_y_alignment import (
    contour_based_y_offset,
    ncc_search_y_offset
)


def perform_z_fine_tuning_2d(xsection_ref, xsection_mov, search_range=600, verbose=True):
    """
    Fine-tune Z-alignment on two 2D averaged X-sections using profile correlation.

    VERTICAL VERSION: Aligns along Z-axis (axis 1) instead of X-axis.
    X-sections have shape (Y, Z).

    This refines the coarse Z-alignment from phase correlation by analyzing
    vertical tissue profiles in the averaged X-sections.

    Args:
        xsection_ref: Reference X-section (Y, Z)
        xsection_mov: Moving X-section to align (Y, Z)
        search_range: Z-shift search range in pixels (default: 600)
        verbose: Print detailed information (default: True)

    Returns:
        dict containing:
            - 'x_fine_tune': Fine-tuning Z-shift to apply (named x_fine_tune for backward compat)
            - 'ncc_score': NCC score at best offset
            - 'confidence': Alignment confidence
    """
    if verbose:
        print("\n" + "="*70)
        print("Z-AXIS FINE-TUNING (2D Profile Correlation)")
        print("="*70)
        print(f"  Search range: ±{search_range} pixels")

    Y, Z = xsection_ref.shape

    # Preprocess X-sections
    xs0_proc = preprocess_oct_for_visualization(xsection_ref)
    xs1_proc = preprocess_oct_for_visualization(xsection_mov)

    # Create vertical profiles (sum along Y-axis) -> profile along Z
    profile_ref = np.sum(xs0_proc, axis=0)  # (Z,)
    profile_mov = np.sum(xs1_proc, axis=0)  # (Z,)

    # Normalize profiles
    profile_ref = (profile_ref - profile_ref.mean()) / (profile_ref.std() + 1e-8)
    profile_mov = (profile_mov - profile_mov.mean()) / (profile_mov.std() + 1e-8)

    # Search for best Z-offset using cross-correlation
    offsets = np.arange(-search_range, search_range + 1)
    scores = []

    for offset in offsets:
        if offset >= 0:
            # Shift right in Z
            p_ref = profile_ref[offset:]
            p_mov = profile_mov[:len(p_ref)]
        else:
            # Shift left in Z
            p_mov = profile_mov[-offset:]
            p_ref = profile_ref[:len(p_mov)]

        # Calculate correlation
        if len(p_ref) > 0 and len(p_mov) > 0:
            corr = np.corrcoef(p_ref, p_mov)[0, 1]
            scores.append(corr)
        else:
            scores.append(-1.0)

    scores = np.array(scores)
    best_idx = np.argmax(scores)
    best_offset = offsets[best_idx]
    best_score = scores[best_idx]

    # Also calculate NCC on full images at best offset
    if best_offset >= 0:
        xs0_crop = xs0_proc[:, best_offset:]
        xs1_crop = xs1_proc[:, :xs0_crop.shape[1]]
    else:
        xs1_crop = xs1_proc[:, -best_offset:]
        xs0_crop = xs0_proc[:, :xs1_crop.shape[1]]

    ncc_score = calculate_ncc(xs0_crop, xs1_crop)

    if verbose:
        print(f"  Best Z fine-tune offset: {best_offset:+d} px")
        print(f"  Profile correlation: {best_score:.4f}")
        print(f"  Full image NCC: {ncc_score:.4f}")
        print("="*70)

    return {
        'x_fine_tune': int(best_offset),  # Keep name for backward compatibility
        'z_fine_tune': int(best_offset),  # Also add proper name
        'profile_correlation': float(best_score),
        'ncc_score': float(ncc_score),
        'confidence': float(best_score)
    }


# Backward compatibility alias
def perform_x_fine_tuning_2d(bscan_ref, bscan_mov, search_range=600, verbose=True):
    """
    Backward compatibility wrapper - calls perform_z_fine_tuning_2d.

    For vertical volumes, this aligns along Z-axis, not X-axis.
    """
    return perform_z_fine_tuning_2d(bscan_ref, bscan_mov, search_range, verbose)



def perform_y_alignment_2d(bscan_ref, bscan_mov, search_range=50, verbose=True):
    """
    Perform Y-alignment on two 2D averaged B-scans.

    Uses the same contour-based surface detection and NCC search
    as the full volume pipeline, but operates directly on 2D images.

    Args:
        bscan_ref: Reference B-scan (Y, X)
        bscan_mov: Moving B-scan to align (Y, X)
        search_range: Y-shift search range in pixels (default: 50)
        verbose: Print detailed information (default: True)

    Returns:
        dict containing:
            - 'y_shift': Applied Y-shift (inverted from detected offset)
            - 'contour_offset': Raw contour-based offset
            - 'ncc_offset': Raw NCC-based offset
            - 'ncc_score': NCC score at best offset
            - 'surface_ref': Detected surface in reference (X,)
            - 'surface_mov': Detected surface in moving B-scan (X,)
            - 'confidence': Alignment confidence metric
    """
    if verbose:
        print("\n" + "="*70)
        print("Y-ALIGNMENT (2D Averaged B-Scan)")
        print("="*70)

    # Step 1: Contour-based Y-offset detection
    if verbose:
        print("  1. Detecting surfaces using contour method...")

    contour_offset, surface_ref, surface_mov = contour_based_y_offset(
        bscan_ref, bscan_mov
    )

    if verbose:
        print(f"     Contour offset: {contour_offset:+.2f} px")

    # Step 2: NCC-based validation
    if verbose:
        print("  2. Validating with NCC search...")

    ncc_offset, ncc_scores, offsets_tested = ncc_search_y_offset(
        bscan_ref, bscan_mov,
        search_range=search_range
    )

    max_ncc = np.max(ncc_scores)

    if verbose:
        print(f"     NCC offset: {ncc_offset:+.2f} px (NCC={max_ncc:.4f})")

    # Step 3: Choose offset (prefer contour, validate with NCC)
    # Check if contour and NCC agree (within 5 pixels)
    agreement = abs(contour_offset - ncc_offset) < 5

    if agreement:
        final_offset = contour_offset
        confidence = max_ncc
        method = "contour (NCC validated)"
    else:
        # If disagreement, use NCC (more robust for averaged B-scans)
        final_offset = ncc_offset
        confidence = max_ncc
        method = "NCC (contour disagreed)"

    # Use offset directly (NO inversion for averaged B-scans)
    y_shift = final_offset

    if verbose:
        print(f"  3. Final decision:")
        print(f"     Method: {method}")
        print(f"     Detected offset: {final_offset:+.2f} px")
        print(f"     Applied shift: {y_shift:+.2f} px (direct, no inversion)")
        print(f"     Confidence: {confidence:.4f}")
        print("="*70)

    return {
        'y_shift': float(y_shift),
        'contour_offset': float(contour_offset),
        'ncc_offset': float(ncc_offset),
        'ncc_score': float(max_ncc),
        'surface_ref': surface_ref,
        'surface_mov': surface_mov,
        'confidence': float(confidence),
        'agreement': agreement,
        'method': method
    }


def perform_rotation_alignment_2d(bscan_ref, bscan_mov,
                                   coarse_range=15, coarse_step=1,
                                   fine_range=3, fine_step=0.5,
                                   verbose=True):
    """
    Perform rotation alignment on two 2D averaged B-scans.

    Uses contour-based surface variance minimization to find optimal
    rotation angle, same as the full volume pipeline.

    Args:
        bscan_ref: Reference B-scan (Y, X)
        bscan_mov: Moving B-scan to align (Y, X)
        coarse_range: Coarse search range in degrees (±range)
        coarse_step: Coarse search step size in degrees
        fine_range: Fine search range in degrees (±range from coarse optimum)
        fine_step: Fine search step size in degrees
        verbose: Print detailed information (default: True)

    Returns:
        dict containing:
            - 'rotation_angle': Applied rotation angle (inverted from detected)
            - 'detected_angle': Raw detected angle
            - 'variance': Surface variance at optimal angle
            - 'score': Alignment score at optimal angle
            - 'ncc_after': NCC score after rotation
    """
    if verbose:
        print("\n" + "="*70)
        print("ROTATION ALIGNMENT (2D Averaged B-scan)")
        print("="*70)
        print(f"  Method: Contour-based surface variance minimization")
        print(f"  Coarse search: ±{coarse_range}° (step={coarse_step}°)")
        print(f"  Fine search: ±{fine_range}° (step={fine_step}°)")

    # Add dummy Z dimension for compatibility with volume functions
    # Shape: (Y, X) → (Y, X, 1)
    bscan_ref_3d = bscan_ref.reshape(bscan_ref.shape[0], bscan_ref.shape[1], 1)
    bscan_mov_3d = bscan_mov.reshape(bscan_mov.shape[0], bscan_mov.shape[1], 1)

    # Use existing contour-based rotation function
    detected_angle, rotation_metrics = find_optimal_rotation_z_contour(
        bscan_ref_3d,
        bscan_mov_3d,
        coarse_range=coarse_range,
        coarse_step=coarse_step,
        fine_range=fine_range,
        fine_step=fine_step,
        verbose=verbose
    )

    # Extract metrics
    optimal_variance = rotation_metrics.get('optimal_variance', np.inf)
    optimal_score = rotation_metrics.get('optimal_score', -np.inf)

    # Use angle directly (NO inversion for averaged B-scans)
    rotation_angle = detected_angle

    if verbose:
        print(f"\n  Results:")
        print(f"     Detected angle: {detected_angle:+.2f}°")
        print(f"     Applied angle: {rotation_angle:+.2f}° (direct, no inversion)")
        print(f"     Surface variance: {optimal_variance:.2f} px²")
        print(f"     Alignment score: {optimal_score:.2f}")

    # Calculate NCC after rotation (if rotation is significant)
    if abs(detected_angle) > 0.5:
        bscan_mov_rotated = ndimage.rotate(
            bscan_mov,
            angle=rotation_angle,
            axes=(0, 1),
            reshape=False,
            order=1,
            mode='constant',
            cval=0
        )
        ncc_after = calculate_ncc(bscan_ref, bscan_mov_rotated)
    else:
        ncc_after = calculate_ncc(bscan_ref, bscan_mov)

    if verbose:
        print(f"     NCC after rotation: {ncc_after:.4f}")
        print("="*70)

    return {
        'rotation_angle': float(rotation_angle),
        'detected_angle': float(detected_angle),
        'variance': float(optimal_variance),
        'score': float(optimal_score),
        'ncc_after': float(ncc_after),
        'coarse_results': rotation_metrics.get('coarse_results', []),
        'fine_results': rotation_metrics.get('fine_results', [])
    }


def visualize_rotation_results_2d(bscan_ref, bscan_mov, rotation_results,
                                   output_path, bscan_name=""):
    """
    Create comprehensive visualization for rotation alignment results.

    Args:
        bscan_ref: Reference B-scan (Y, X)
        bscan_mov: Moving B-scan (Y, X)
        rotation_results: Results dict from perform_rotation_alignment_2d
        output_path: Path to save visualization
        bscan_name: Name identifier for the B-scan
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Normalize for display
    def normalize_for_display(img):
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img

    ref_norm = normalize_for_display(bscan_ref)
    mov_norm = normalize_for_display(bscan_mov)

    # Apply rotation
    rotation_angle = rotation_results['rotation_angle']
    mov_rotated = ndimage.rotate(
        bscan_mov,
        angle=rotation_angle,
        axes=(0, 1),
        reshape=False,
        order=1,
        mode='constant',
        cval=0
    )
    mov_rotated_norm = normalize_for_display(mov_rotated)

    # Row 1: B-scans
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ref_norm, cmap='gray', aspect='auto')
    ax1.set_title('Reference', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mov_norm, cmap='gray', aspect='auto')
    ax2.set_title('Before Rotation', fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(mov_rotated_norm, cmap='gray', aspect='auto')
    ax3.set_title(f'After Rotation ({rotation_angle:+.2f}°)',
                  fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Row 2: Analysis
    # Plot coarse search results
    if rotation_results.get('coarse_results'):
        coarse_results = rotation_results['coarse_results']
        angles = [r['angle'] for r in coarse_results]
        scores = [r['score'] for r in coarse_results]

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(angles, scores, 'b-o', markersize=4, linewidth=2)
        ax4.axvline(x=rotation_results['detected_angle'], color='r',
                   linestyle='--', linewidth=2, label='Optimal')
        ax4.set_xlabel('Rotation Angle (degrees)', fontsize=12)
        ax4.set_ylabel('Alignment Score', fontsize=12)
        ax4.set_title('Coarse Search', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

    # Plot fine search results
    if rotation_results.get('fine_results'):
        fine_results = rotation_results['fine_results']
        angles = [r['angle'] for r in fine_results]
        scores = [r['score'] for r in fine_results]

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(angles, scores, 'g-o', markersize=4, linewidth=2)
        ax5.axvline(x=rotation_results['detected_angle'], color='r',
                   linestyle='--', linewidth=2, label='Optimal')
        ax5.set_xlabel('Rotation Angle (degrees)', fontsize=12)
        ax5.set_ylabel('Alignment Score', fontsize=12)
        ax5.set_title('Fine Search', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()

    # Difference images
    ax6 = fig.add_subplot(gs[1, 2])
    diff_after = np.abs(ref_norm - mov_rotated_norm)
    im = ax6.imshow(diff_after, cmap='hot', aspect='auto', vmin=0, vmax=1)
    ax6.set_title('Difference After Rotation', fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

    # Add summary text
    summary_text = (
        f"Detected Angle: {rotation_results['detected_angle']:+.2f}°\n"
        f"Applied Angle: {rotation_results['rotation_angle']:+.2f}° (inverted)\n"
        f"Surface Variance: {rotation_results['variance']:.2f} px²\n"
        f"Alignment Score: {rotation_results['score']:.2f}\n"
        f"NCC After: {rotation_results['ncc_after']:.4f}"
    )
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    title = f'Rotation Alignment - {bscan_name}' if bscan_name else 'Rotation Alignment'
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Rotation visualization saved: {output_path}")
