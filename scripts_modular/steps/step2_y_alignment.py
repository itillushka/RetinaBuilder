"""
Step 2: Y-Axis Alignment

Aligns volumes along Y-axis using central B-scan matching.
Uses NCC search and contour-based surface detection for accurate alignment.
"""

import numpy as np
from scipy import ndimage
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers.rotation_alignment import (
    preprocess_oct_for_visualization,
    detect_contour_surface,
    calculate_ncc
)


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
    ncc_scores = np.zeros(len(offsets))

    for i, offset in enumerate(offsets):
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
            ncc = calculate_ncc(b0_crop, b1_crop)
            ncc_scores[i] = ncc
        else:
            ncc_scores[i] = -1  # Invalid

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
        bscan_v0: Reference B-scan (Y, X)
        bscan_v1: B-scan to align (Y, X)

    Returns:
        y_offset: Offset to align surfaces (positive = shift v1 down)
        surface_v0: Detected surface for v0 (X,)
        surface_v1: Detected surface for v1 (X,)
    """
    # Preprocess for surface detection
    b0_proc = preprocess_oct_for_visualization(bscan_v0)
    b1_proc = preprocess_oct_for_visualization(bscan_v1)

    # Detect surfaces
    surface_v0 = detect_contour_surface(b0_proc)  # (X,)
    surface_v1 = detect_contour_surface(b1_proc)  # (X,)

    # Calculate median offset (robust to outliers)
    surface_diff = surface_v0 - surface_v1
    y_offset = np.median(surface_diff)

    return y_offset, surface_v0, surface_v1


def step2_y_alignment(step1_results, data_dir):
    """
    Step 2: Y alignment using central B-scan matching.

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
            - surface_v0, surface_v1: Detected surfaces from central B-scans
            - bscan_v0_central, bscan_v1_central: Central B-scans used
            - y_crop_bounds: Crop region bounds
    """
    print("\n" + "="*70)
    print("STEP 2: Y ALIGNMENT (B-SCAN BASED)")
    print("="*70)

    overlap_v0 = step1_results['overlap_v0']
    overlap_v1 = step1_results['overlap_v1']

    Y, X, Z = overlap_v0.shape
    z_center = Z // 2

    # Extract central B-scans
    print(f"\n1. Extracting central B-scans at Z={z_center}...")
    bscan_v0 = overlap_v0[:, :, z_center].copy()  # (Y, X)
    bscan_v1 = overlap_v1[:, :, z_center].copy()  # (Y, X)
    print(f"   B-scan shape: {bscan_v0.shape}")

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

    # Compare methods
    offset_diff = abs(ncc_offset - contour_offset)
    print(f"\n4. Method Comparison:")
    print(f"   Contour offset (PRIMARY): {contour_offset:+.2f} px")
    print(f"   NCC offset (validation):  {ncc_offset:+.2f} px")
    print(f"   Difference:               {offset_diff:.2f} px")
    if offset_diff < 3:
        print(f"   [OK] Methods agree (diff < 3 px) - High confidence!")
    else:
        print(f"   [WARNING] Methods differ by {offset_diff:.1f} px - Using contour (more reliable for surface alignment)")

    # Use contour offset as primary (directly aligns retinal surfaces)
    y_shift = contour_offset

    print(f"\n5. Applying Y-shift: {y_shift:+.2f} px...")

    # Apply Y shift
    overlap_v1_y_aligned = ndimage.shift(
        overlap_v1, shift=(y_shift, 0, 0),
        order=1, mode='constant', cval=0
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
        # Central B-scans for visualization
        'bscan_v0_central': bscan_v0,
        'bscan_v1_central': bscan_v1,
        'z_center': z_center,
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
