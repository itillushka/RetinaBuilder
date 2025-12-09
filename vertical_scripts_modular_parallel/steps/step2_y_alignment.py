"""
Step 2: Y-Axis Alignment (VERTICAL VOLUMES)

Aligns volumes along Y-axis using central X-section matching.
Uses NCC search and contour-based surface detection for accurate alignment.

VERTICAL VERSION: Uses X-axis cross-sections (Y, Z) instead of B-scans (Y, X).
"""

import numpy as np
from scipy import ndimage
from pathlib import Path
import sys
from multiprocessing import cpu_count
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import cv2

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers.rotation_alignment import (
    preprocess_oct_for_visualization,
    detect_contour_surface,
    calculate_ncc
)
from helpers.rotation_alignment_parallel import shift_volume_y_parallel


def visualize_bscan_with_surface(bscan_ref_denoised, bscan_mov_denoised, surface_ref, surface_mov,
                                  y_shift, output_path, prefix=""):
    """
    Create visualization of DENOISED X-sections with detected surfaces for traceability.

    Shows the overlap region X-sections (cropped) with surface detection.

    Args:
        bscan_ref_denoised: Denoised reference X-section (Y, Z) - overlap region
        bscan_mov_denoised: Denoised moving X-section (Y, Z) - overlap region
        surface_ref: Detected surface for reference
        surface_mov: Detected surface for moving
        y_shift: Applied Y-shift value
        output_path: Path to save visualization
        prefix: Prefix for title (e.g., "V2_to_V1")
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Reference DENOISED X-section with surface overlay
    axes[0, 0].imshow(bscan_ref_denoised, cmap='gray', aspect='auto')
    x_coords = np.arange(len(surface_ref))
    axes[0, 0].plot(x_coords, surface_ref, 'g-', linewidth=2, label='Detected surface')
    axes[0, 0].set_title(f'{prefix} Reference X-section DENOISED (overlap region)', fontsize=12)
    axes[0, 0].set_xlabel('Z (pixels)')
    axes[0, 0].set_ylabel('Y (pixels)')
    axes[0, 0].legend(loc='upper right')

    # Top-right: Moving DENOISED X-section with surface overlay
    axes[0, 1].imshow(bscan_mov_denoised, cmap='gray', aspect='auto')
    axes[0, 1].plot(x_coords, surface_mov, 'r-', linewidth=2, label='Detected surface')
    axes[0, 1].set_title(f'{prefix} Moving X-section DENOISED (overlap region)', fontsize=12)
    axes[0, 1].set_xlabel('Z (pixels)')
    axes[0, 1].set_ylabel('Y (pixels)')
    axes[0, 1].legend(loc='upper right')

    # Bottom-left: Both surfaces overlaid (before alignment)
    axes[1, 0].plot(x_coords, surface_ref, 'g-', linewidth=2, label='Reference surface')
    axes[1, 0].plot(x_coords, surface_mov, 'r-', linewidth=2, label='Moving surface')
    axes[1, 0].set_title(f'Surface comparison (BEFORE alignment)', fontsize=12)
    axes[1, 0].set_xlabel('Z (pixels)')
    axes[1, 0].set_ylabel('Y (pixels)')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Both surfaces overlaid (after alignment)
    surface_mov_aligned = surface_mov + y_shift
    axes[1, 1].plot(x_coords, surface_ref, 'g-', linewidth=2, label='Reference surface')
    axes[1, 1].plot(x_coords, surface_mov_aligned, 'r--', linewidth=2, label=f'Moving surface (shifted {y_shift:+.1f}px)')
    axes[1, 1].set_title(f'Surface comparison (AFTER alignment, Y-shift={y_shift:+.1f}px)', fontsize=12)
    axes[1, 1].set_xlabel('Z (pixels)')
    axes[1, 1].set_ylabel('Y (pixels)')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {output_path.name}")


def _test_single_y_offset(offset, b0_norm, b1_norm, Y):
    """
    Worker function to test a single Y-offset.

    Args:
        offset: Y-offset to test
        b0_norm: Normalized reference B-scan
        b1_norm: Normalized moving B-scan
        Y: Height of B-scans

    Returns:
        NCC score for this offset
    """
    try:
        # Determine overlap region after shifting
        if offset >= 0:
            # V1 shifted down
            b1_crop = b1_norm[0:Y-offset, :]
            b0_crop = b0_norm[offset:Y, :]
        else:
            # V1 shifted up
            b1_crop = b1_norm[-offset:Y, :]
            b0_crop = b0_norm[0:Y+offset, :]

        # Calculate NCC
        if b0_crop.shape == b1_crop.shape and b0_crop.size > 0:
            return calculate_ncc(b0_crop, b1_crop)
        else:
            return -1.0  # Invalid
    except Exception:
        return -1.0


def ncc_search_y_offset(bscan_v0, bscan_v1, search_range=50):
    """
    Find optimal Y-offset between two B-scans using NCC search.

    Searches for the Y-shift that maximizes normalized cross-correlation
    between the two B-scans.

    Args:
        bscan_v0: Reference B-scan (Y, X)
        bscan_v1: B-scan to align (Y, X)
        search_range: Search offsets from -search_range to +search_range

    Returns:
        best_offset: Optimal Y-offset (positive = shift v1 down)
        ncc_scores: Array of NCC scores for each offset tested
        offsets: Array of offsets tested
    """
    Y, X = bscan_v0.shape

    # Preprocess B-scans for better matching
    b0_proc = preprocess_oct_for_visualization(bscan_v0)
    b1_proc = preprocess_oct_for_visualization(bscan_v1)

    # Normalize B-scans
    b0_norm = (b0_proc - b0_proc.mean()) / (b0_proc.std() + 1e-8)
    b1_norm = (b1_proc - b1_proc.mean()) / (b1_proc.std() + 1e-8)

    # Search range
    offsets = np.arange(-search_range, search_range + 1)

    # PARALLEL PROCESSING: Test all offsets in parallel
    # Use ThreadPoolExecutor instead of multiprocessing to avoid memory issues
    num_workers = min(cpu_count(), len(offsets))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        test_func = partial(
            _test_single_y_offset,
            b0_norm=b0_norm,
            b1_norm=b1_norm,
            Y=Y
        )
        ncc_scores = np.array(list(executor.map(test_func, offsets)))

    # Find best offset
    best_idx = np.argmax(ncc_scores)
    best_offset = offsets[best_idx]

    return best_offset, ncc_scores, offsets


def contour_based_y_offset(bscan_v0, bscan_v1):
    """
    Calculate Y-offset using contour-based surface detection.

    Detects retinal surface in both B-scans and calculates
    the offset needed to align them.

    Args:
        bscan_v0: Reference B-scan (Y, X) or X-section (Y, Z)
        bscan_v1: B-scan/X-section to align

    Returns:
        y_offset: Offset to align surfaces (positive = shift v1 down)
        surface_v0: Detected surface for v0
        surface_v1: Detected surface for v1
        b0_proc: Denoised reference (for visualization)
        b1_proc: Denoised moving (for visualization)
    """
    # Preprocess for surface detection
    b0_proc = preprocess_oct_for_visualization(bscan_v0)
    b1_proc = preprocess_oct_for_visualization(bscan_v1)

    # Find where tissue actually starts (skip empty black region at top)
    # Look for the first row with significant signal
    Y = b0_proc.shape[0]
    row_means = np.mean(b0_proc, axis=1)
    tissue_threshold = np.max(row_means) * 0.1  # 10% of max row mean
    tissue_rows = np.where(row_means > tissue_threshold)[0]
    if len(tissue_rows) > 0:
        start_y = max(0, tissue_rows[0] - 50)  # Start 50px above first tissue
    else:
        start_y = Y // 3  # Fallback: start from 1/3 down

    # Detect surfaces starting from where tissue begins
    surface_v0 = detect_contour_surface(b0_proc, start_y=start_y)
    surface_v1 = detect_contour_surface(b1_proc, start_y=start_y)

    # Calculate median offset (robust to outliers)
    surface_diff = surface_v0 - surface_v1
    y_offset = np.median(surface_diff)

    return y_offset, surface_v0, surface_v1, b0_proc, b1_proc


def perform_y_alignment(ref_volume, mov_volume, position='right', offset_z=0, output_dir=None, prefix=""):
    """
    Y-alignment using CROPPED overlap region for calculation.

    VERTICAL VERSION: Uses X-axis cross-sections instead of B-scans.
    Crops X-sections to the overlap region for accurate alignment at seams,
    then applies the calculated shift to the full volume.

    Args:
        ref_volume: Reference volume (Y, X, Z)
        mov_volume: Volume to align (Y, X, Z)
        position: 'right' or 'left' - position of moving volume relative to reference
        offset_z: Z offset from step 1 (VERTICAL: Z is the unlimited axis)
        output_dir: Directory to save visualization (if provided)
        prefix: Prefix for saved files (e.g., 'v2_to_v1')

    Returns:
        dict containing:
            - 'volume_1_y_aligned': Y-aligned moving volume
            - 'y_shift': Y-shift value applied (raw, no multiplier)
            - 'contour_y_offset': Offset from contour method
            - 'ncc_y_offset': Offset from NCC method
    """
    Y, X_ref, Z_ref = ref_volume.shape
    _, X_mov, Z_mov = mov_volume.shape
    x_center = X_ref // 2  # VERTICAL: use X center instead of Z center

    # Extract central X-sections (full depth first) - VERTICAL: (Y, Z) instead of (Y, X)
    xsection_ref_full = ref_volume[:, x_center, :].copy()  # (Y, Z)
    xsection_mov_full = mov_volume[:, x_center, :].copy()  # (Y, Z)

    # Calculate overlap depth and CROP X-sections (VERTICAL: crop on Z axis)
    offset_z_int = int(round(offset_z))

    if position == 'right':
        # V2 is to the right of V1 (in Z direction for vertical)
        # offset_z is NEGATIVE (e.g., -462 means V2 shifts left 462px to align)
        overlap_depth = min(Z_ref, Z_mov) - abs(offset_z_int)
        if overlap_depth > 0:
            xsection_ref = xsection_ref_full[:, :overlap_depth]   # V1's FIRST N slices
            xsection_mov = xsection_mov_full[:, :overlap_depth]   # V2's FIRST N slices
            print(f"  [Overlap] position='right', offset_z={offset_z_int}, overlap_depth={overlap_depth}")
            print(f"  [Overlap] Cropped V1 to FIRST {overlap_depth} slices, V2 to FIRST {overlap_depth} slices")
        else:
            xsection_ref = xsection_ref_full
            xsection_mov = xsection_mov_full
            print(f"  [Overlap] No overlap (offset_z={offset_z_int}), using full X-sections")

    elif position == 'left':
        # V2 is to the left of V1 (in Z direction for vertical)
        # offset_z is POSITIVE (e.g., +462 means V2 shifts right 462px to align)
        overlap_depth = min(Z_ref, Z_mov) - abs(offset_z_int)
        if overlap_depth > 0:
            xsection_ref = xsection_ref_full[:, -overlap_depth:]  # V1's LAST N slices
            xsection_mov = xsection_mov_full[:, -overlap_depth:]  # V2's LAST N slices
            print(f"  [Overlap] position='left', offset_z={offset_z_int}, overlap_depth={overlap_depth}")
            print(f"  [Overlap] Cropped V1 to LAST {overlap_depth} slices, V2 to LAST {overlap_depth} slices")
        else:
            xsection_ref = xsection_ref_full
            xsection_mov = xsection_mov_full
            print(f"  [Overlap] No overlap (offset_z={offset_z_int}), using full X-sections")

    else:
        # Fallback: unknown position
        xsection_ref = xsection_ref_full
        xsection_mov = xsection_mov_full
        print(f"  [Overlap] Unknown position='{position}', using full X-sections")

    print(f"  [Overlap] X-section shapes: ref={xsection_ref.shape}, mov={xsection_mov.shape}")

    # Rename for compatibility with rest of function (still called bscan internally)
    bscan_ref = xsection_ref
    bscan_mov = xsection_mov

    # Method 1: Contour-based surface detection (on CROPPED B-scans)
    # Returns denoised B-scans for visualization
    contour_offset, surface_ref, surface_mov, bscan_ref_denoised, bscan_mov_denoised = contour_based_y_offset(bscan_ref, bscan_mov)

    # Method 2: NCC search (on CROPPED B-scans)
    ncc_offset, ncc_scores, offsets_tested = ncc_search_y_offset(bscan_ref, bscan_mov, search_range=50)

    # Always prioritize contour-based alignment (more reliable for surface detection)
    y_shift = contour_offset
    print(f"  Using contour offset ({contour_offset:+.1f}) [NCC was {ncc_offset:+.1f}]")

    # Save visualization if output_dir provided
    # Use OVERLAP region (cropped) for visualization - same as horizontal pipeline
    # This avoids showing black gaps from XZ-shift zero-padding
    if output_dir is not None:
        output_dir = Path(output_dir)
        vis_filename = f"step2_y_alignment_{prefix}.png" if prefix else "step2_y_alignment.png"

        visualize_bscan_with_surface(
            bscan_ref_denoised, bscan_mov_denoised,  # OVERLAP region (from contour detection)
            surface_ref, surface_mov,
            y_shift,
            output_dir / vis_filename,
            prefix=prefix.replace('_', ' ').upper() if prefix else ""
        )

    # Apply Y-shift to full volume using PARALLEL method
    # (NO 2.0x multiplier - that's only for visualization)
    volume_1_y_aligned = shift_volume_y_parallel(
        mov_volume, y_shift, n_jobs=-1
    )

    return {
        'volume_1_y_aligned': volume_1_y_aligned,
        'y_shift': float(y_shift),  # Store the shift that was applied
        'contour_y_offset': float(contour_offset),
        'ncc_y_offset': float(ncc_offset)
    }


def step2_y_alignment(step1_results, data_dir):
    """
    Step 2: Y alignment using central X-section matching.

    VERTICAL VERSION: Uses X-axis cross-sections instead of B-scans.
    Uses contour-based surface detection (primary) with NCC search validation
    to accurately align retinal structures between volumes.

    Args:
        step1_results: Dictionary from step1_xz_alignment containing:
            - overlap_v0: Overlap region from volume 0
            - overlap_v1: Overlap region from volume 1
        data_dir: Directory for saving results

    Returns:
        results: Dictionary containing:
            - overlap_v1_y_aligned: Y-aligned overlap region (cropped)
            - overlap_v0: Updated overlap region (cropped)
            - y_shift: Calculated Y shift amount (from contour method)
            - ncc_scores: NCC scores for all tested offsets
            - offsets_tested: Array of offsets tested in NCC search
            - contour_y_offset: Y-offset from contour method (primary)
            - ncc_y_offset: Y-offset from NCC method (validation)
            - surface_v0, surface_v1: Detected surfaces from central X-sections
            - bscan_v0_central, bscan_v1_central: Central X-sections used
            - y_crop_bounds: Crop region bounds
    """
    print("\n" + "="*70)
    print("STEP 2: Y ALIGNMENT (X-SECTION BASED - VERTICAL)")
    print("="*70)

    overlap_v0 = step1_results['overlap_v0']
    overlap_v1 = step1_results['overlap_v1']

    Y, X, Z = overlap_v0.shape
    x_center = X // 2  # VERTICAL: use X center instead of Z center

    # Extract central X-sections - VERTICAL: (Y, Z) instead of (Y, X)
    print(f"\n1. Extracting central X-sections at X={x_center}...")
    bscan_v0 = overlap_v0[:, x_center, :].copy()  # (Y, Z)
    bscan_v1 = overlap_v1[:, x_center, :].copy()  # (Y, Z)
    print(f"   X-section shape: {bscan_v0.shape}")

    # Method 1: Contour-based surface detection (primary method)
    print(f"\n2. Contour-based surface detection (primary method)...")
    contour_offset, surface_v0, surface_v1 = contour_based_y_offset(bscan_v0, bscan_v1)
    print(f"   [OK] Y-offset (contour): {contour_offset:+.2f} px")
    print(f"   [OK] Mean surface position V0: {surface_v0.mean():.2f}")
    print(f"   [OK] Mean surface position V1: {surface_v1.mean():.2f}")

    # Method 2: NCC search (validation)
    print(f"\n3. NCC Search (validation)...")
    print(f"   Searching Y-offsets: -50 to +50 pixels")
    ncc_offset, ncc_scores, offsets_tested = ncc_search_y_offset(bscan_v0, bscan_v1, search_range=50)
    print(f"   [OK] Best Y-offset (NCC): {ncc_offset:+.2f} px")
    print(f"   [OK] Peak NCC score: {ncc_scores.max():.4f}")

    # Compare methods and prioritize higher displacement
    offset_diff = abs(ncc_offset - contour_offset)
    print(f"\n4. Method Comparison:")
    print(f"   Contour offset: {contour_offset:+.2f} px")
    print(f"   NCC offset:     {ncc_offset:+.2f} px")
    print(f"   Difference:     {offset_diff:.2f} px")

    # Always prioritize contour-based alignment (more reliable for surface detection)
    y_shift = contour_offset
    print(f"   [SELECTED] Contour offset ({contour_offset:+.1f}) [NCC was {ncc_offset:+.1f}]")

    print(f"\n5. Applying Y-shift: {y_shift:+.2f} px (PARALLEL)...")

    # Apply Y shift using PARALLEL method
    overlap_v1_y_aligned = shift_volume_y_parallel(
        overlap_v1, y_shift, n_jobs=-1
    )
    print(f"   [OK] Applied Y-shift to full overlap volume")

    # Crop Y dimension to remove zero-padded regions
    if y_shift >= 0:
        # V1 shifted DOWN -> zeros at top
        y_start, y_end = int(np.ceil(y_shift)), Y
    else:
        # V1 shifted UP -> zeros at bottom
        y_start, y_end = 0, Y + int(np.floor(y_shift))

    print(f"\n6. Cropping Y dimension to remove padding:")
    print(f"   Original Y size: {Y}")
    print(f"   Valid Y range: [{y_start}:{y_end}]")
    print(f"   Cropped Y size: {y_end - y_start}")

    # Crop both volumes to valid Y region (same indices for both!)
    overlap_v0_cropped = overlap_v0[y_start:y_end, :, :].copy()
    overlap_v1_y_aligned_cropped = overlap_v1_y_aligned[y_start:y_end, :, :].copy()

    print(f"   Final overlap shape: {overlap_v0_cropped.shape}")

    results = {
        'overlap_v1_y_aligned': overlap_v1_y_aligned_cropped,
        'overlap_v0': overlap_v0_cropped,
        'y_shift': float(y_shift),
        # Contour method data (primary)
        'contour_y_offset': float(contour_offset),
        'surface_v0': surface_v0,
        'surface_v1': surface_v1,
        # NCC search data (validation)
        'ncc_y_offset': float(ncc_offset),
        'ncc_scores': ncc_scores,
        'offsets_tested': offsets_tested,
        # Central X-sections for visualization (VERTICAL)
        'xsection_v0_central': bscan_v0,
        'xsection_v1_central': bscan_v1,
        'x_center': x_center,
        # Crop info
        'y_crop_bounds': (y_start, y_end)
    }

    print("\n" + "="*70)
    print("✅ STEP 2 COMPLETE!")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Y-shift applied: {y_shift:+.2f} px (from contour method)")
    print(f"  Method agreement: {offset_diff:.2f} px difference")
    print(f"  Final overlap shape: {overlap_v0_cropped.shape}")

    return results
