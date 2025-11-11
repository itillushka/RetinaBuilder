#!/usr/bin/env python3
"""
Rotation Alignment Module for OCT Volume Registration

Implements Z-axis (in-plane XY) rotation correction for parallel layer alignment.
Uses ECC-STYLE CORRELATION with aggressive denoising and thresholding to isolate
retinal layer structures and find optimal rotation angle.

This module provides:
  - Z-axis rotation detection (coarse-to-fine grid search)
  - Aggressive preprocessing (denoising + thresholding) to isolate tissue layers
  - Normalized correlation scoring for alignment quality
  - Rotation visualization functions

Key Insight:
  OCT images have significant speckle noise. By aggressively denoising and thresholding
  to keep only bright retinal layers, we can use simple correlation to find the
  rotation that best aligns the layer structures.

Usage:
    from rotation_alignment import find_optimal_rotation_z, apply_rotation_z

    angle, metrics = find_optimal_rotation_z(overlap_v0, overlap_v1)
    volume_rotated = apply_rotation_z(volume, angle)
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import cv2


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_oct_for_rotation(img, mask=None):
    """
    Preprocess OCT B-scan with VERY aggressive denoising and thresholding.

    This isolates clean retinal layer structures by removing speckle noise
    and zeroing out low-intensity regions.

    Pipeline:
      1. Non-local means denoising (h=12)
      2. Bilateral filtering (sigma=90)
      3. Median filter (kernel=7)
      4. Otsu thresholding (80% threshold - keeps only bright tissue)
      5. CLAHE contrast enhancement
      6. Horizontal kernel filter (emphasizes layer structures)

    Args:
        img: Input B-scan (2D array)
        mask: Optional binary mask to focus on valid regions

    Returns:
        Preprocessed uint8 image with isolated layer structures
    """
    # Normalize to 0-255 first
    img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    # Step 1: Non-local means denoising (very effective for OCT speckle)
    denoised1 = cv2.fastNlMeansDenoising(img_norm, h=12, templateWindowSize=7, searchWindowSize=21)

    # Step 2: Bilateral filtering for edge-preserving smoothing
    denoised2 = cv2.bilateralFilter(denoised1, d=9, sigmaColor=90, sigmaSpace=90)

    # Step 3: Median filter to remove remaining noise
    denoised3 = cv2.medianBlur(denoised2, 7)

    # Step 4: Threshold to keep only tissue layers (zero out noise/background)
    thresh_val = cv2.threshold(denoised3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_val = int(thresh_val * 0.8)  # 80% of Otsu threshold (conservative)

    thresholded = denoised3.copy()
    thresholded[denoised3 < thresh_val] = 0

    # Step 5: Enhance contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(thresholded)

    # Step 6: Emphasize horizontal structures (retinal layers)
    kernel = np.ones((1, 5), np.float32) / 5
    layer_enhanced = cv2.filter2D(enhanced, -1, kernel)

    # Apply mask if provided
    if mask is not None:
        layer_enhanced = layer_enhanced.astype(float)
        layer_enhanced[~mask] = 0
        layer_enhanced = layer_enhanced.astype(np.uint8)

    return layer_enhanced


# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def calculate_ncc(img1, img2, mask=None):
    """
    Normalized Cross-Correlation - standard metric for image registration.

    NCC = sum((img1 - mean1) * (img2 - mean2)) / (std1 * std2 * N)

    Args:
        img1: Reference image
        img2: Image to compare
        mask: Optional binary mask to focus on valid regions

    Returns:
        NCC score: Range [-1, 1], where 1 = perfect match
    """
    if mask is not None:
        img1 = img1[mask]
        img2 = img2[mask]

    # Normalize images (remove mean, scale by std)
    img1_norm = (img1 - img1.mean()) / (img1.std() + 1e-8)
    img2_norm = (img2 - img2.mean()) / (img2.std() + 1e-8)

    # Calculate correlation
    ncc = np.mean(img1_norm * img2_norm)

    return float(ncc)


def calculate_ncc_3d(volume1, volume2, mask=None):
    """
    Calculate NCC for 3D volumes.

    Args:
        volume1: Reference volume (Y, X, Z)
        volume2: Volume to compare (Y, X, Z)
        mask: Optional binary mask (Y, X, Z)

    Returns:
        NCC score for the entire volume
    """
    if mask is not None:
        volume1 = volume1[mask]
        volume2 = volume2[mask]
    else:
        volume1 = volume1.ravel()
        volume2 = volume2.ravel()

    # Remove zero regions (background)
    valid = (volume1 > 0) & (volume2 > 0)
    if valid.sum() < 100:  # Need at least 100 pixels
        return -1.0

    volume1 = volume1[valid]
    volume2 = volume2[valid]

    # Normalize
    v1_norm = (volume1 - volume1.mean()) / (volume1.std() + 1e-8)
    v2_norm = (volume2 - volume2.mean()) / (volume2.std() + 1e-8)

    ncc = np.mean(v1_norm * v2_norm)

    return float(ncc)


def calculate_rcp(img1, img2, mask=None):
    """
    Row-Wise Correlation Profile (RCP) - OPTIMAL metric for parallel layer detection.

    Calculates correlation for each row independently and measures uniformity.
    When retinal layers are parallel, corresponding rows correlate uniformly.

    Score = mean_correlation × (1 - std_correlation)
    High mean + low variance = uniformly high correlation = parallel layers

    Args:
        img1: Reference image (2D numpy array)
        img2: Image to compare (2D numpy array)
        mask: Optional binary mask (2D boolean array)

    Returns:
        RCP score: Range [0, 1], where 1 = perfect parallel layer alignment
    """
    H, W = img1.shape

    # Calculate correlation for each row
    row_correlations = []

    for row_idx in range(H):
        row1 = img1[row_idx, :]
        row2 = img2[row_idx, :]

        # Skip if mask excludes this row
        if mask is not None:
            row_mask = mask[row_idx, :]
            if row_mask.sum() < W * 0.3:  # Skip if < 30% valid
                continue
            row1 = row1[row_mask]
            row2 = row2[row_mask]

        # Skip if row is too uniform
        if row1.std() < 1 or row2.std() < 1:
            continue

        # Calculate correlation
        row1_norm = (row1 - row1.mean()) / (row1.std() + 1e-8)
        row2_norm = (row2 - row2.mean()) / (row2.std() + 1e-8)
        corr = np.mean(row1_norm * row2_norm)

        row_correlations.append(corr)

    if len(row_correlations) < 10:  # Need at least 10 rows
        return 0.0

    row_correlations = np.array(row_correlations)

    # Score: mean correlation with penalty for high variance
    # High mean + low variance = uniformly high correlation = parallel layers
    mean_corr = np.mean(row_correlations)
    std_corr = np.std(row_correlations)

    # Normalize: high mean, low std is best
    rcp = mean_corr * (1 - std_corr)

    return float(np.clip(rcp, 0, 1))


# ============================================================================
# ROTATION FUNCTIONS
# ============================================================================

def apply_rotation_z(volume, angle_degrees, axes=(0, 1), reshape=False):
    """
    Apply rotation around Z-axis (B-scan direction axis).

    This rotates Y and X axes, which tilts retinal layers within B-scans.
    Corrects for roll/tilt in the cross-sectional view.

    Args:
        volume: 3D numpy array (Y, X, Z) where Y=depth, X=lateral, Z=B-scan index
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)
        axes: Tuple of axes to rotate (default: (0, 1) = Y and X)
        reshape: If True, expand array to fit rotated volume

    Returns:
        Rotated volume with same shape (or expanded if reshape=True)
    """
    volume_rotated = ndimage.rotate(
        volume,
        angle=angle_degrees,
        axes=axes,
        reshape=reshape,
        order=1,  # Bilinear interpolation
        mode='constant',
        cval=0
    )

    return volume_rotated


def create_overlap_mask(volume, percentile=10):
    """
    Create a binary mask for valid (non-zero) regions in volume.

    Useful for focusing NCC calculation on tissue regions only.

    Args:
        volume: 3D volume
        percentile: Threshold percentile for valid pixels

    Returns:
        Binary mask (True = valid tissue)
    """
    threshold = np.percentile(volume[volume > 0], percentile) if (volume > 0).any() else 0
    mask = volume > threshold
    return mask


# ============================================================================
# ROTATION SEARCH FUNCTIONS
# ============================================================================

def find_optimal_translation_for_rotated_bscan(bscan_v0, bscan_v1_rotated,
                                                 search_range=10, verbose=False):
    """
    Find optimal X,Y translation for a rotated B-scan.

    Uses phase correlation for fast, sub-pixel accurate shift detection.

    Args:
        bscan_v0: Reference B-scan (Y, X)
        bscan_v1_rotated: Rotated B-scan to align (Y, X)
        search_range: Maximum shift to search (pixels)
        verbose: Print debug info

    Returns:
        (shift_y, shift_x): Optimal shifts in pixels
        ncc: NCC score at optimal shift
    """
    from scipy.signal import correlate

    # Use phase correlation for initial estimate
    # Correlate the two images
    correlation = correlate(bscan_v0, bscan_v1_rotated, mode='same', method='fft')

    # Find peak
    y_peak, x_peak = np.unravel_index(np.argmax(correlation), correlation.shape)

    # Convert to shift
    center_y, center_x = np.array(bscan_v0.shape) // 2
    shift_y = y_peak - center_y
    shift_x = x_peak - center_x

    # Clamp to search range
    shift_y = np.clip(shift_y, -search_range, search_range)
    shift_x = np.clip(shift_x, -search_range, search_range)

    # Apply shift and calculate NCC
    bscan_shifted = ndimage.shift(bscan_v1_rotated, shift=(shift_y, shift_x),
                                   order=1, mode='constant', cval=0)

    # Calculate NCC on overlapping region
    mask = (bscan_v0 > 0) & (bscan_shifted > 0)
    if mask.sum() > 100:
        ncc = calculate_ncc(bscan_v0, bscan_shifted, mask=mask)
    else:
        ncc = -1.0

    if verbose:
        print(f"      Translation: ({shift_y:+.1f}, {shift_x:+.1f}) px, NCC={ncc:.4f}")

    return (shift_y, shift_x), ncc


def find_optimal_rotation_z_coarse(overlap_v0, overlap_v1,
                                     angle_range=10, step=1,
                                     verbose=True):
    """
    Coarse search for optimal rotation angle around Z-axis using ECC-style correlation.

    OPTIMIZATION: Uses only the CENTRAL B-scan for speed.
    Rotates Y and X axes (tilts retinal layers within B-scans).
    Searches rotation angles from -angle_range to +angle_range with given step.
    Uses aggressive preprocessing + correlation to find best alignment.

    Args:
        overlap_v0: Reference volume (Y, X, Z)
        overlap_v1: Volume to rotate (Y, X, Z)
        angle_range: Maximum angle to search (degrees)
        step: Angle step size (degrees)
        verbose: Print progress

    Returns:
        best_angle: Optimal rotation angle (degrees)
        best_score: Correlation score at optimal angle
        results: List of dicts with all tested angles and their scores
    """
    angles_to_test = np.arange(-angle_range, angle_range + step, step)

    # Extract CENTRAL B-scan for fast optimization
    z_center = overlap_v0.shape[2] // 2
    bscan_v0 = overlap_v0[:, :, z_center]  # (Y, X)
    bscan_v1 = overlap_v1[:, :, z_center]  # (Y, X)

    if verbose:
        print(f"\n{'='*70}")
        print(f"COARSE ROTATION SEARCH (ECC-STYLE CORRELATION)")
        print(f"{'='*70}")
        print(f"  Method: Aggressive denoising + correlation scoring")
        print(f"  Goal: Find rotation that maximizes layer structure overlap")
        print(f"  Angle range: ±{angle_range}°")
        print(f"  Step size: {step}°")
        print(f"  Total angles: {len(angles_to_test)}")
        print(f"  Using central B-scan: Z={z_center}/{overlap_v0.shape[2]}")
        print(f"  B-scan shape: {bscan_v0.shape} (Y, X)")

    # Create mask for valid regions
    mask_v0 = bscan_v0 > np.percentile(bscan_v0[bscan_v0 > 0], 10) if (bscan_v0 > 0).any() else bscan_v0 > 0

    # Preprocess reference B-scan ONCE
    if verbose:
        print(f"  Preprocessing reference B-scan...")
    bscan_v0_proc = preprocess_oct_for_rotation(bscan_v0, mask=mask_v0)

    results = []
    best_score = -1.0
    best_angle = 0.0

    if verbose:
        print(f"  Starting rotation search...")

    for i, angle in enumerate(angles_to_test):
        if verbose and i % 5 == 0:
            print(f"  Progress: {i}/{len(angles_to_test)} angles tested... " +
                  f"(current best: {best_angle:+.1f}° with score {best_score:.4f})")

        # Rotate B-scan using scipy (same method as apply function for consistency)
        bscan_v1_rotated = ndimage.rotate(bscan_v1, angle, axes=(0, 1),
                                           reshape=False, order=1,
                                           mode='constant', cval=0)

        # Create mask for rotated B-scan
        mask_v1 = bscan_v1_rotated > np.percentile(bscan_v1_rotated[bscan_v1_rotated > 0], 10) if (bscan_v1_rotated > 0).any() else bscan_v1_rotated > 0
        mask_combined = mask_v0 & mask_v1

        # Preprocess rotated B-scan
        bscan_v1_proc = preprocess_oct_for_rotation(bscan_v1_rotated, mask=mask_combined)

        # Calculate correlation on preprocessed images
        if mask_combined.sum() < 100:
            score = -1.0
        else:
            try:
                i1 = bscan_v0_proc[mask_combined].astype(float)
                i2 = bscan_v1_proc[mask_combined].astype(float)

                # Normalize
                i1_norm = (i1 - i1.mean()) / (i1.std() + 1e-8)
                i2_norm = (i2 - i2.mean()) / (i2.std() + 1e-8)

                score = np.mean(i1_norm * i2_norm)
            except Exception:
                score = -1.0

        results.append({
            'angle': float(angle),
            'correlation': float(score),
            'valid_pixels': int(mask_combined.sum())
        })

        # Track best
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    if verbose:
        print(f"\n  ✓ Coarse search complete!")
        print(f"    Best rotation: {best_angle:+.1f}°")
        print(f"    Best correlation: {best_score:.4f}")

    return best_angle, best_score, results


def find_optimal_rotation_z_fine(overlap_v0, overlap_v1,
                                   coarse_angle, angle_range=3, step=0.5,
                                   verbose=True):
    """
    Fine search for optimal rotation angle around Z-axis using ECC-style correlation.

    OPTIMIZATION: Uses only the CENTRAL B-scan for speed.
    Rotates Y and X axes (tilts retinal layers within B-scans).
    Refines rotation angle with smaller step size around coarse estimate.
    Uses aggressive preprocessing + correlation to find best alignment.

    Args:
        overlap_v0: Reference volume (Y, X, Z)
        overlap_v1: Volume to rotate (Y, X, Z)
        coarse_angle: Starting angle from coarse search (degrees)
        angle_range: Search range around coarse_angle (degrees)
        step: Angle step size (degrees)
        verbose: Print progress

    Returns:
        best_angle: Refined rotation angle (degrees)
        best_score: Correlation score at refined angle
        results: List of dicts with all tested angles and their scores
    """
    angles_to_test = np.arange(coarse_angle - angle_range,
                                coarse_angle + angle_range + step,
                                step)

    # Extract CENTRAL B-scan for fast optimization
    z_center = overlap_v0.shape[2] // 2
    bscan_v0 = overlap_v0[:, :, z_center]  # (Y, X)
    bscan_v1 = overlap_v1[:, :, z_center]  # (Y, X)

    if verbose:
        print(f"\n{'='*70}")
        print(f"FINE ROTATION SEARCH (ECC-STYLE CORRELATION)")
        print(f"{'='*70}")
        print(f"  Method: Aggressive denoising + correlation scoring")
        print(f"  Center angle: {coarse_angle:+.1f}°")
        print(f"  Search range: ±{angle_range}°")
        print(f"  Step size: {step}°")
        print(f"  Total angles: {len(angles_to_test)}")
        print(f"  Using central B-scan: Z={z_center}/{overlap_v0.shape[2]}")

    # Create mask for valid regions
    mask_v0 = bscan_v0 > np.percentile(bscan_v0[bscan_v0 > 0], 10) if (bscan_v0 > 0).any() else bscan_v0 > 0

    # Preprocess reference B-scan ONCE
    if verbose:
        print(f"  Preprocessing reference B-scan...")
    bscan_v0_proc = preprocess_oct_for_rotation(bscan_v0, mask=mask_v0)

    results = []
    best_score = -1.0
    best_angle = coarse_angle

    for angle in angles_to_test:
        # Rotate B-scan using scipy (same method as apply function for consistency)
        bscan_v1_rotated = ndimage.rotate(bscan_v1, angle, axes=(0, 1),
                                           reshape=False, order=1,
                                           mode='constant', cval=0)

        # Create mask for rotated B-scan
        mask_v1 = bscan_v1_rotated > np.percentile(bscan_v1_rotated[bscan_v1_rotated > 0], 10) if (bscan_v1_rotated > 0).any() else bscan_v1_rotated > 0
        mask_combined = mask_v0 & mask_v1

        # Preprocess rotated B-scan
        bscan_v1_proc = preprocess_oct_for_rotation(bscan_v1_rotated, mask=mask_combined)

        # Calculate correlation on preprocessed images
        if mask_combined.sum() < 100:
            score = -1.0
        else:
            try:
                i1 = bscan_v0_proc[mask_combined].astype(float)
                i2 = bscan_v1_proc[mask_combined].astype(float)

                # Normalize
                i1_norm = (i1 - i1.mean()) / (i1.std() + 1e-8)
                i2_norm = (i2 - i2.mean()) / (i2.std() + 1e-8)

                score = np.mean(i1_norm * i2_norm)
            except Exception:
                score = -1.0

        results.append({
            'angle': float(angle),
            'correlation': float(score),
            'valid_pixels': int(mask_combined.sum())
        })

        # Track best
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    if verbose:
        print(f"\n  ✓ Fine search complete!")
        print(f"    Refined rotation: {best_angle:+.2f}°")
        print(f"    Best correlation: {best_score:.4f}")

    return best_angle, best_score, results


def find_optimal_rotation_z(overlap_v0, overlap_v1,
                              coarse_range=10, coarse_step=1,
                              fine_range=3, fine_step=0.5,
                              verbose=True,
                              visualize_masks=False,
                              mask_vis_path=None):
    """
    Find optimal rotation around Z-axis using coarse-to-fine search with ECC-style correlation.

    Rotates Y and X axes (tilts retinal layers within B-scans).
    This corrects for roll/tilt in cross-sectional view.
    Uses aggressive preprocessing + correlation to find best alignment.

    Two-stage approach:
      1. Coarse search: ±10° with 1° steps (21 angles)
      2. Fine search: ±3° with 0.5° steps around coarse optimum (13 angles)

    Args:
        overlap_v0: Reference overlap volume (Y, X, Z)
        overlap_v1: Volume to rotate (Y, X, Z)
        coarse_range: Coarse search angle range (degrees)
        coarse_step: Coarse search step size (degrees)
        fine_range: Fine search angle range (degrees)
        fine_step: Fine search step size (degrees)
        verbose: Print progress
        visualize_masks: If True, create mask visualization before search
        mask_vis_path: Path to save mask visualization

    Returns:
        optimal_angle: Best rotation angle (degrees)
        metrics: Dictionary with correlation scores and search results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"ROTATION AROUND Z-AXIS (ECC-STYLE CORRELATION)")
        print(f"{'='*70}")
        print(f"  Method: Aggressive denoising + correlation scoring")
        print(f"  Goal: Find rotation that maximizes layer structure overlap")
        print(f"  Rotation only (no translation)")
        print(f"  Rotates Y & X axes → Tilts retinal layers within B-scans")
        print(f"  OPTIMIZATION: Using central B-scan for speed (will apply to full volume)")

    # Create masks for visualization/debugging
    if visualize_masks:
        if verbose:
            print(f"\n  Creating masks for visualization...")
        mask_v0 = create_overlap_mask(overlap_v0)
        mask_v1 = create_overlap_mask(overlap_v1)
        mask_combined = mask_v0 & mask_v1

        visualize_masking(
            overlap_v0, overlap_v1,
            mask_v0, mask_v1, mask_combined,
            output_path=mask_vis_path
        )

    # Stage 1: Coarse search
    coarse_angle, coarse_score, coarse_results = find_optimal_rotation_z_coarse(
        overlap_v0, overlap_v1,
        angle_range=coarse_range,
        step=coarse_step,
        verbose=verbose
    )

    # Stage 2: Fine search
    fine_angle, fine_score, fine_results = find_optimal_rotation_z_fine(
        overlap_v0, overlap_v1,
        coarse_angle=coarse_angle,
        angle_range=fine_range,
        step=fine_step,
        verbose=verbose
    )

    metrics = {
        'optimal_angle': fine_angle,
        'optimal_correlation': fine_score,
        'coarse_angle': coarse_angle,
        'coarse_correlation': coarse_score,
        'coarse_results': coarse_results,
        'fine_results': fine_results
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"✓ ROTATION ALIGNMENT COMPLETE")
        print(f"{'='*70}")
        print(f"  Optimal rotation: {fine_angle:+.2f}°")
        print(f"  Correlation: {coarse_score:.4f} → {fine_score:.4f}")
        print(f"  Total angles tested: {len(coarse_results) + len(fine_results)}")
        print(f"  Rotation type: Layer tilt (axes Y & X)")
        print(f"  Method: Aggressive denoising + correlation scoring")
        print(f"  Goal: Maximize overlap of clean layer structures")

    return fine_angle, metrics


# ============================================================================
# X-AXIS ROTATION FUNCTIONS (Step 3.5 - Y-Z Plane Alignment)
# ============================================================================

def apply_rotation_x(volume, angle_degrees, axes=(0, 2), reshape=False):
    """
    Apply rotation around X-axis (lateral axis).

    This rotates Y and Z axes, which tilts retinal layers visible in sagittal/coronal view.
    Corrects for pitch/tilt in the Y-Z plane (visible in "Y-axis view").

    Args:
        volume: 3D numpy array (Y, X, Z) where Y=depth, X=lateral, Z=B-scan index
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)
        axes: Tuple of axes to rotate (default: (0, 2) = Y and Z)
        reshape: If True, expand array to fit rotated volume

    Returns:
        Rotated volume with same shape (or expanded if reshape=True)
    """
    volume_rotated = ndimage.rotate(
        volume,
        angle=angle_degrees,
        axes=axes,
        reshape=reshape,
        order=1,  # Bilinear interpolation
        mode='constant',
        cval=0
    )

    return volume_rotated


def find_optimal_rotation_x_coarse(overlap_v0, overlap_v1,
                                     angle_range=10, step=1,
                                     verbose=True):
    """
    Coarse search for optimal rotation angle around X-axis using ECC-style correlation.

    OPTIMIZATION: Uses only the CENTRAL SAGITTAL slice (Y-Z plane) for speed.
    Rotates Y and Z axes (tilts retinal layers in Y-Z plane / "Y-axis view").
    Searches rotation angles from -angle_range to +angle_range with given step.
    Uses aggressive preprocessing + correlation to find best alignment.

    Args:
        overlap_v0: Reference volume (Y, X, Z)
        overlap_v1: Volume to rotate (Y, X, Z)
        angle_range: Maximum angle to search (degrees)
        step: Angle step size (degrees)
        verbose: Print progress

    Returns:
        best_angle: Optimal rotation angle (degrees)
        best_score: Correlation score at optimal angle
        results: List of dicts with all tested angles and their scores
    """
    angles_to_test = np.arange(-angle_range, angle_range + step, step)

    # Extract CENTRAL SAGITTAL slice (Y-Z plane at center X) for fast optimization
    x_center = overlap_v0.shape[1] // 2
    sagittal_v0 = overlap_v0[:, x_center, :]  # (Y, Z)
    sagittal_v1 = overlap_v1[:, x_center, :]  # (Y, Z)

    if verbose:
        print(f"\n{'='*70}")
        print(f"COARSE X-AXIS ROTATION SEARCH (ECC-STYLE CORRELATION)")
        print(f"{'='*70}")
        print(f"  Method: Aggressive denoising + correlation scoring")
        print(f"  Goal: Find rotation that maximizes layer structure overlap in Y-Z plane")
        print(f"  Angle range: ±{angle_range}°")
        print(f"  Step size: {step}°")
        print(f"  Total angles: {len(angles_to_test)}")
        print(f"  Using central sagittal slice: X={x_center}/{overlap_v0.shape[1]}")
        print(f"  Sagittal slice shape: {sagittal_v0.shape} (Y, Z)")

    # Create mask for valid regions
    mask_v0 = sagittal_v0 > np.percentile(sagittal_v0[sagittal_v0 > 0], 10) if (sagittal_v0 > 0).any() else sagittal_v0 > 0

    # Preprocess reference sagittal slice ONCE
    if verbose:
        print(f"  Preprocessing reference sagittal slice...")
    sagittal_v0_proc = preprocess_oct_for_rotation(sagittal_v0, mask=mask_v0)

    results = []
    best_score = -1.0
    best_angle = 0.0

    if verbose:
        print(f"  Starting rotation search...")

    for i, angle in enumerate(angles_to_test):
        if verbose and i % 5 == 0:
            print(f"  Progress: {i}/{len(angles_to_test)} angles tested... " +
                  f"(current best: {best_angle:+.1f}° with score {best_score:.4f})")

        # Rotate sagittal slice using scipy in Y-Z plane
        sagittal_v1_rotated = ndimage.rotate(sagittal_v1, angle, axes=(0, 1),
                                              reshape=False, order=1,
                                              mode='constant', cval=0)

        # Create mask for rotated sagittal slice
        mask_v1 = sagittal_v1_rotated > np.percentile(sagittal_v1_rotated[sagittal_v1_rotated > 0], 10) if (sagittal_v1_rotated > 0).any() else sagittal_v1_rotated > 0
        mask_combined = mask_v0 & mask_v1

        # Preprocess rotated sagittal slice
        sagittal_v1_proc = preprocess_oct_for_rotation(sagittal_v1_rotated, mask=mask_combined)

        # Calculate correlation on preprocessed images
        if mask_combined.sum() < 100:
            score = -1.0
        else:
            try:
                i1 = sagittal_v0_proc[mask_combined].astype(float)
                i2 = sagittal_v1_proc[mask_combined].astype(float)

                # Normalize
                i1_norm = (i1 - i1.mean()) / (i1.std() + 1e-8)
                i2_norm = (i2 - i2.mean()) / (i2.std() + 1e-8)

                score = np.mean(i1_norm * i2_norm)
            except Exception:
                score = -1.0

        results.append({
            'angle': float(angle),
            'correlation': float(score),
            'valid_pixels': int(mask_combined.sum())
        })

        # Track best
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    if verbose:
        print(f"\n  ✓ Coarse search complete!")
        print(f"    Best rotation: {best_angle:+.1f}°")
        print(f"    Best correlation: {best_score:.4f}")

    return best_angle, best_score, results


def find_optimal_rotation_x_fine(overlap_v0, overlap_v1,
                                   coarse_angle, angle_range=3, step=0.5,
                                   verbose=True):
    """
    Fine search for optimal rotation angle around X-axis using ECC-style correlation.

    OPTIMIZATION: Uses only the CENTRAL SAGITTAL slice (Y-Z plane) for speed.
    Rotates Y and Z axes (tilts retinal layers in Y-Z plane / "Y-axis view").
    Refines the coarse angle with smaller steps in a narrower range.

    Args:
        overlap_v0: Reference volume (Y, X, Z)
        overlap_v1: Volume to rotate (Y, X, Z)
        coarse_angle: Starting angle from coarse search (degrees)
        angle_range: Range around coarse angle to search (degrees)
        step: Angle step size (degrees)
        verbose: Print progress

    Returns:
        best_angle: Optimal rotation angle (degrees)
        best_score: Correlation score at optimal angle
        results: List of dicts with all tested angles and their scores
    """
    # Search around coarse angle
    angles_to_test = np.arange(coarse_angle - angle_range,
                               coarse_angle + angle_range + step,
                               step)

    # Extract CENTRAL SAGITTAL slice for fast optimization
    x_center = overlap_v0.shape[1] // 2
    sagittal_v0 = overlap_v0[:, x_center, :]  # (Y, Z)
    sagittal_v1 = overlap_v1[:, x_center, :]  # (Y, Z)

    if verbose:
        print(f"\n{'='*70}")
        print(f"FINE X-AXIS ROTATION SEARCH")
        print(f"{'='*70}")
        print(f"  Refining around coarse angle: {coarse_angle:+.1f}°")
        print(f"  Fine search range: ±{angle_range}°")
        print(f"  Step size: {step}°")
        print(f"  Total angles: {len(angles_to_test)}")

    # Create mask for valid regions
    mask_v0 = sagittal_v0 > np.percentile(sagittal_v0[sagittal_v0 > 0], 10) if (sagittal_v0 > 0).any() else sagittal_v0 > 0

    # Preprocess reference sagittal slice ONCE
    sagittal_v0_proc = preprocess_oct_for_rotation(sagittal_v0, mask=mask_v0)

    results = []
    best_score = -1.0
    best_angle = coarse_angle

    if verbose:
        print(f"  Starting fine rotation search...")

    for angle in angles_to_test:
        # Rotate sagittal slice using scipy in Y-Z plane
        sagittal_v1_rotated = ndimage.rotate(sagittal_v1, angle, axes=(0, 1),
                                              reshape=False, order=1,
                                              mode='constant', cval=0)

        # Create mask for rotated sagittal slice
        mask_v1 = sagittal_v1_rotated > np.percentile(sagittal_v1_rotated[sagittal_v1_rotated > 0], 10) if (sagittal_v1_rotated > 0).any() else sagittal_v1_rotated > 0
        mask_combined = mask_v0 & mask_v1

        # Preprocess rotated sagittal slice
        sagittal_v1_proc = preprocess_oct_for_rotation(sagittal_v1_rotated, mask=mask_combined)

        # Calculate correlation on preprocessed images
        if mask_combined.sum() < 100:
            score = -1.0
        else:
            try:
                i1 = sagittal_v0_proc[mask_combined].astype(float)
                i2 = sagittal_v1_proc[mask_combined].astype(float)

                # Normalize
                i1_norm = (i1 - i1.mean()) / (i1.std() + 1e-8)
                i2_norm = (i2 - i2.mean()) / (i2.std() + 1e-8)

                score = np.mean(i1_norm * i2_norm)
            except Exception:
                score = -1.0

        results.append({
            'angle': float(angle),
            'correlation': float(score),
            'valid_pixels': int(mask_combined.sum())
        })

        # Track best
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    if verbose:
        print(f"\n  ✓ Fine search complete!")
        print(f"    Best rotation: {best_angle:+.2f}°")
        print(f"    Best correlation: {best_score:.4f}")

    return best_angle, best_score, results


def find_optimal_rotation_x(overlap_v0, overlap_v1,
                              coarse_range=10, coarse_step=1,
                              fine_range=3, fine_step=0.5,
                              verbose=True,
                              visualize_masks=False,
                              mask_vis_path=None):
    """
    Find optimal rotation around X-axis using coarse-to-fine search with ECC-style correlation.

    Rotates Y and Z axes (tilts retinal layers in Y-Z plane / "Y-axis view").
    This corrects for pitch/tilt visible in the coronal/sagittal view.
    Uses aggressive preprocessing + correlation to find best alignment.

    Two-stage approach:
      1. Coarse search: ±10° with 1° steps (21 angles)
      2. Fine search: ±3° with 0.5° steps around coarse optimum (13 angles)

    Args:
        overlap_v0: Reference overlap volume (Y, X, Z)
        overlap_v1: Volume to rotate (Y, X, Z)
        coarse_range: Coarse search angle range (degrees)
        coarse_step: Coarse search step size (degrees)
        fine_range: Fine search angle range (degrees)
        fine_step: Fine search step size (degrees)
        verbose: Print progress
        visualize_masks: If True, create mask visualization before search
        mask_vis_path: Path to save mask visualization

    Returns:
        optimal_angle: Best rotation angle (degrees)
        metrics: Dictionary with correlation scores and search results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"ROTATION AROUND X-AXIS (ECC-STYLE CORRELATION)")
        print(f"{'='*70}")
        print(f"  Method: Aggressive denoising + correlation scoring")
        print(f"  Goal: Find rotation that maximizes layer structure overlap in Y-Z plane")
        print(f"  Rotation only (no translation)")
        print(f"  Rotates Y & Z axes → Tilts retinal layers in sagittal view")
        print(f"  OPTIMIZATION: Using central sagittal slice for speed (will apply to full volume)")

    # Create masks for visualization/debugging if requested
    if visualize_masks:
        if verbose:
            print(f"\n  Creating masks for visualization...")
        mask_v0 = create_overlap_mask(overlap_v0)
        mask_v1 = create_overlap_mask(overlap_v1)
        mask_combined = mask_v0 & mask_v1

        visualize_masking(
            overlap_v0, overlap_v1,
            mask_v0, mask_v1, mask_combined,
            output_path=mask_vis_path
        )

    # Stage 1: Coarse search
    coarse_angle, coarse_score, coarse_results = find_optimal_rotation_x_coarse(
        overlap_v0, overlap_v1,
        angle_range=coarse_range,
        step=coarse_step,
        verbose=verbose
    )

    # Stage 2: Fine search
    fine_angle, fine_score, fine_results = find_optimal_rotation_x_fine(
        overlap_v0, overlap_v1,
        coarse_angle=coarse_angle,
        angle_range=fine_range,
        step=fine_step,
        verbose=verbose
    )

    metrics = {
        'optimal_angle': fine_angle,
        'optimal_correlation': fine_score,
        'coarse_results': coarse_results,
        'fine_results': fine_results
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"✓ X-AXIS ROTATION ALIGNMENT COMPLETE")
        print(f"{'='*70}")
        print(f"  Optimal rotation: {fine_angle:+.2f}°")
        print(f"  Correlation: {coarse_score:.4f} → {fine_score:.4f}")
        print(f"  Total angles tested: {len(coarse_results) + len(fine_results)}")
        print(f"  Rotation type: Y-Z plane tilt (axes Y & Z)")
        print(f"  Method: Aggressive denoising + correlation scoring")
        print(f"  Goal: Maximize overlap of layer structures in sagittal view")

    return fine_angle, metrics


# ============================================================================
# Y-AXIS RE-ALIGNMENT FUNCTIONS (Step 3.1)
# ============================================================================

def calculate_tissue_threshold(img1, img2, percentile=50):
    """
    Calculate tissue threshold for masking based on percentile.

    Uses the median (50th percentile) of non-zero pixels to separate
    tissue from background. This focuses alignment on retinal structures.

    Args:
        img1: First image (2D numpy array)
        img2: Second image (2D numpy array)
        percentile: Percentile to use (default: 50 = median)

    Returns:
        Threshold value (float)
    """
    # Get non-zero pixels
    nz1 = img1[img1 > 0]
    nz2 = img2[img2 > 0]

    if len(nz1) == 0 or len(nz2) == 0:
        return 0.0

    # Calculate percentile threshold for each
    thresh1 = np.percentile(nz1, percentile)
    thresh2 = np.percentile(nz2, percentile)

    # Return average
    return (thresh1 + thresh2) / 2


def find_optimal_y_shift_central_bscan(overlap_v0, overlap_v1_rotated,
                                        search_range=20, step=1,
                                        verbose=True):
    """
    Step 3.1: Find optimal Y-axis shift on central B-scan after rotation.

    After rotation in Step 3, the Y-axis (depth) alignment may be disturbed.
    This function re-aligns along Y-axis using NCC-based search on the
    central B-scan to maximize similarity.

    Args:
        overlap_v0: Reference overlap volume (Y, X, Z)
        overlap_v1_rotated: Rotated volume from Step 3 (Y, X, Z)
        search_range: Search range in pixels (±range)
        step: Step size for search (default: 1 pixel)
        verbose: Print progress

    Returns:
        best_shift: Optimal Y shift correction (pixels)
        best_ncc: NCC score at optimal shift
        results: List of dicts with all tested shifts and their scores
    """
    # Extract central B-scan
    z_center = overlap_v0.shape[2] // 2
    bscan_v0 = overlap_v0[:, :, z_center]  # (Y, X)
    bscan_v1 = overlap_v1_rotated[:, :, z_center]  # (Y, X)

    if verbose:
        print(f"\n{'='*70}")
        print(f"STEP 3.1: Y-AXIS RE-ALIGNMENT (CENTRAL B-SCAN)")
        print(f"{'='*70}")
        print(f"  Correcting vertical shift after rotation")
        print(f"  Using central B-scan: Z={z_center}/{overlap_v0.shape[2]}")
        print(f"  Search range: ±{search_range} pixels")
        print(f"  Step size: {step} pixel(s)")

    # Calculate tissue threshold (50th percentile = median)
    threshold = calculate_tissue_threshold(bscan_v0, bscan_v1, percentile=50)

    if verbose:
        print(f"  Tissue threshold: {threshold:.1f}")

    # Search Y shifts
    y_shifts = np.arange(-search_range, search_range + step, step)
    results = []
    best_shift = 0
    best_ncc = -1.0

    if verbose:
        print(f"  Testing {len(y_shifts)} Y-shifts...")

    for i, y_shift in enumerate(y_shifts):
        # Shift B-scan along Y-axis
        bscan_shifted = ndimage.shift(
            bscan_v1, shift=(y_shift, 0),
            order=1, mode='constant', cval=0
        )

        # Create tissue mask (compare only valid tissue regions)
        mask = (bscan_v0 > threshold) & (bscan_shifted > threshold)

        # Calculate NCC on tissue regions only
        if mask.sum() > 100:  # Need at least 100 pixels
            ncc = calculate_ncc(bscan_v0, bscan_shifted, mask=mask)
        else:
            ncc = -1.0

        results.append({
            'y_shift': float(y_shift),
            'ncc': ncc,
            'valid_pixels': int(mask.sum())
        })

        # Track best
        if ncc > best_ncc:
            best_ncc = ncc
            best_shift = float(y_shift)

        # Progress indicator
        if verbose and (i + 1) % 10 == 0:
            print(f"    Progress: {i + 1}/{len(y_shifts)} shifts tested... (current best: {best_shift:+.1f} px @ NCC={best_ncc:.4f})")

    if verbose:
        print(f"\n  ✓ Y-shift search complete!")
        print(f"    Optimal Y shift: {best_shift:+.1f} px")
        print(f"    NCC at optimal shift: {best_ncc:.4f}")

        if abs(best_shift) < 0.5:
            print(f"    → No significant correction needed")
        else:
            print(f"    → Will apply correction: {best_shift:+.1f} px")

    return best_shift, best_ncc, results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_masking(overlap_v0, overlap_v1, mask_v0, mask_v1, mask_combined, output_path=None):
    """
    Visualize the masking process before rotation search.

    Shows what regions will be compared during NCC calculation.

    Args:
        overlap_v0: Reference overlap volume
        overlap_v1: Volume to rotate (before rotation)
        mask_v0: Mask for volume 0
        mask_v1: Mask for volume 1
        mask_combined: Combined mask (v0 & v1)
        output_path: Path to save figure
    """
    print(f"\n  Creating mask visualization...")

    # Create MIPs for visualization
    mip_v0 = np.mean(overlap_v0, axis=0)
    mip_v1 = np.mean(overlap_v1, axis=0)

    # Create mask MIPs
    mask_v0_mip = np.any(mask_v0, axis=0)
    mask_v1_mip = np.any(mask_v1, axis=0)
    mask_combined_mip = np.any(mask_combined, axis=0)

    # Get central B-scan
    z_center = overlap_v0.shape[2] // 2
    bscan_v0 = overlap_v0[:, :, z_center]
    bscan_v1 = overlap_v1[:, :, z_center]
    mask_v0_bscan = mask_v0[:, :, z_center]
    mask_v1_bscan = mask_v1[:, :, z_center]
    mask_combined_bscan = mask_combined[:, :, z_center]

    fig = plt.figure(figsize=(24, 16))

    # Row 1: Original en-face MIPs
    ax1 = plt.subplot(4, 3, 1)
    ax1.imshow(mip_v0.T, cmap='gray', origin='lower', aspect='auto')
    ax1.set_title('V0 En-face MIP\n(Reference)', fontweight='bold')
    ax1.set_ylabel('Z (B-scans)')
    ax1.set_xlabel('X (lateral)')

    ax2 = plt.subplot(4, 3, 2)
    ax2.imshow(mip_v1.T, cmap='gray', origin='lower', aspect='auto')
    ax2.set_title('V1 En-face MIP\n(Before rotation)', fontweight='bold')
    ax2.set_xlabel('X (lateral)')

    ax3 = plt.subplot(4, 3, 3)
    ax3.imshow(mip_v0.T, cmap='Reds', alpha=0.5, origin='lower', aspect='auto')
    ax3.imshow(mip_v1.T, cmap='Greens', alpha=0.5, origin='lower', aspect='auto')
    ax3.set_title('Overlay\n(Before masking)', fontweight='bold')
    ax3.set_xlabel('X (lateral)')

    # Row 2: Masks (en-face view)
    ax4 = plt.subplot(4, 3, 4)
    ax4.imshow(mask_v0_mip.T, cmap='gray', origin='lower', aspect='auto')
    valid_v0 = mask_v0.sum()
    ax4.set_title(f'V0 Mask\n{valid_v0:,} voxels ({valid_v0/mask_v0.size*100:.1f}%)', fontweight='bold')
    ax4.set_ylabel('Z (B-scans)')
    ax4.set_xlabel('X (lateral)')

    ax5 = plt.subplot(4, 3, 5)
    ax5.imshow(mask_v1_mip.T, cmap='gray', origin='lower', aspect='auto')
    valid_v1 = mask_v1.sum()
    ax5.set_title(f'V1 Mask\n{valid_v1:,} voxels ({valid_v1/mask_v1.size*100:.1f}%)', fontweight='bold')
    ax5.set_xlabel('X (lateral)')

    ax6 = plt.subplot(4, 3, 6)
    ax6.imshow(mask_combined_mip.T, cmap='hot', origin='lower', aspect='auto')
    valid_combined = mask_combined.sum()
    ax6.set_title(f'Combined Mask (V0 & V1)\n{valid_combined:,} voxels ({valid_combined/mask_combined.size*100:.1f}%)',
                  fontweight='bold', color='red')
    ax6.set_xlabel('X (lateral)')

    # Row 3: B-scans (original)
    ax7 = plt.subplot(4, 3, 7)
    ax7.imshow(bscan_v0, cmap='gray', aspect='auto')
    ax7.set_title(f'V0 B-scan @ Z={z_center}', fontweight='bold')
    ax7.set_ylabel('Y (depth)')
    ax7.set_xlabel('X (lateral)')

    ax8 = plt.subplot(4, 3, 8)
    ax8.imshow(bscan_v1, cmap='gray', aspect='auto')
    ax8.set_title(f'V1 B-scan @ Z={z_center}', fontweight='bold')
    ax8.set_xlabel('X (lateral)')

    ax9 = plt.subplot(4, 3, 9)
    ax9.imshow(bscan_v0, cmap='Reds', alpha=0.5, aspect='auto')
    ax9.imshow(bscan_v1, cmap='Greens', alpha=0.5, aspect='auto')
    ax9.set_title('B-scan Overlay', fontweight='bold')
    ax9.set_xlabel('X (lateral)')

    # Row 4: B-scans with masks applied
    ax10 = plt.subplot(4, 3, 10)
    bscan_v0_masked = bscan_v0.copy()
    bscan_v0_masked[~mask_v0_bscan] = 0
    ax10.imshow(bscan_v0_masked, cmap='gray', aspect='auto')
    ax10.set_title('V0 B-scan MASKED', fontweight='bold')
    ax10.set_ylabel('Y (depth)')
    ax10.set_xlabel('X (lateral)')

    ax11 = plt.subplot(4, 3, 11)
    bscan_v1_masked = bscan_v1.copy()
    bscan_v1_masked[~mask_v1_bscan] = 0
    ax11.imshow(bscan_v1_masked, cmap='gray', aspect='auto')
    ax11.set_title('V1 B-scan MASKED', fontweight='bold')
    ax11.set_xlabel('X (lateral)')

    ax12 = plt.subplot(4, 3, 12)
    # Show only the combined mask region
    bscan_v0_final = bscan_v0.copy()
    bscan_v1_final = bscan_v1.copy()
    bscan_v0_final[~mask_combined_bscan] = 0
    bscan_v1_final[~mask_combined_bscan] = 0
    ax12.imshow(bscan_v0_final, cmap='Reds', alpha=0.5, aspect='auto')
    ax12.imshow(bscan_v1_final, cmap='Greens', alpha=0.5, aspect='auto')
    ax12.set_title('FINAL: Only Combined Mask Region\n(This is what NCC compares)',
                   fontweight='bold', color='red')
    ax12.set_xlabel('X (lateral)')

    plt.suptitle('Mask Verification Before Rotation Search\n(White/Hot = included in NCC, Black = excluded)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved mask visualization: {output_path}")

    plt.close()

    # Print statistics
    print(f"\n  Mask Statistics:")
    print(f"    Volume 0 valid voxels: {valid_v0:,} ({valid_v0/mask_v0.size*100:.1f}%)")
    print(f"    Volume 1 valid voxels: {valid_v1:,} ({valid_v1/mask_v1.size*100:.1f}%)")
    print(f"    Combined valid voxels: {valid_combined:,} ({valid_combined/mask_combined.size*100:.1f}%)")
    print(f"    Overlap efficiency: {valid_combined/min(valid_v0, valid_v1)*100:.1f}%")


def visualize_rotation_search(coarse_results, fine_results, output_path=None):
    """
    Visualize rotation angle search results.

    Creates plot showing correlation score vs applied rotation angle
    for both coarse and fine searches.

    Args:
        coarse_results: List of dicts from coarse search (keys: angle, correlation, valid_pixels)
        fine_results: List of dicts from fine search (keys: angle, correlation, valid_pixels)
        output_path: Path to save figure (if None, display only)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Extract data
    coarse_angles = [r['angle'] for r in coarse_results]
    coarse_correlations = [r['correlation'] for r in coarse_results]

    fine_angles = [r['angle'] for r in fine_results]
    fine_correlations = [r['correlation'] for r in fine_results]

    # Find best indices
    best_coarse_idx = np.argmax(coarse_correlations)
    best_fine_idx = np.argmax(fine_correlations)

    # Plot 1: Coarse search - correlation vs rotation
    axes[0].plot(coarse_angles, coarse_correlations, 'b-o', linewidth=2, markersize=6)
    axes[0].plot(coarse_angles[best_coarse_idx], coarse_correlations[best_coarse_idx],
                 'r*', markersize=20, label=f'Best: {coarse_angles[best_coarse_idx]:+.1f}°')
    axes[0].set_xlabel('Applied Rotation (degrees)', fontweight='bold')
    axes[0].set_ylabel('Correlation Score', fontweight='bold')
    axes[0].set_title('Coarse Search: Correlation vs Rotation', fontweight='bold', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Fine search - correlation vs rotation
    axes[1].plot(fine_angles, fine_correlations, 'g-o', linewidth=2, markersize=6)
    axes[1].plot(fine_angles[best_fine_idx], fine_correlations[best_fine_idx],
                 'r*', markersize=20, label=f'Best: {fine_angles[best_fine_idx]:+.2f}°')
    axes[1].set_xlabel('Applied Rotation (degrees)', fontweight='bold')
    axes[1].set_ylabel('Correlation Score', fontweight='bold')
    axes[1].set_title('Fine Search: Correlation vs Rotation', fontweight='bold', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle('Z-Axis Rotation Optimization (ECC-Style Correlation with Aggressive Denoising)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved rotation search plot: {output_path}")

    plt.close()


def visualize_rotation_comparison(overlap_v0, overlap_v1_before, overlap_v1_after,
                                    angle, ncc_before, ncc_after,
                                    output_path=None):
    """
    Visualize before/after rotation alignment.

    Shows en-face MIP and central B-scan comparisons.
    Note: Rotation was optimized using ECC-style correlation with aggressive denoising.
    NCC is shown here for verification/comparison purposes.

    Args:
        overlap_v0: Reference volume
        overlap_v1_before: Volume before rotation
        overlap_v1_after: Volume after rotation (including Y-shift correction)
        angle: Applied rotation angle (degrees)
        ncc_before: NCC before rotation (verification metric)
        ncc_after: NCC after rotation + Y-shift (verification metric)
        output_path: Path to save figure
    """
    fig = plt.figure(figsize=(24, 12))

    # Create MIPs (average projection)
    mip_v0 = np.mean(overlap_v0, axis=0)
    mip_v1_before = np.mean(overlap_v1_before, axis=0)
    mip_v1_after = np.mean(overlap_v1_after, axis=0)

    # Get central B-scan
    z_center = overlap_v0.shape[2] // 2
    bscan_v0 = overlap_v0[:, :, z_center]
    bscan_v1_before = overlap_v1_before[:, :, z_center]
    bscan_v1_after = overlap_v1_after[:, :, z_center]

    # Row 1: En-face MIPs - BEFORE
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(mip_v0.T, cmap='Reds', alpha=0.7, origin='lower', aspect='auto')
    ax1.set_title('V0 MIP (Reference)', fontweight='bold')
    ax1.set_xlabel('X'); ax1.set_ylabel('Z')

    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(mip_v1_before.T, cmap='Greens', alpha=0.7, origin='lower', aspect='auto')
    ax2.set_title('V1 MIP (BEFORE rotation)', fontweight='bold')
    ax2.set_xlabel('X')

    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(mip_v0.T, cmap='Reds', alpha=0.5, origin='lower', aspect='auto')
    ax3.imshow(mip_v1_before.T, cmap='Greens', alpha=0.5, origin='lower', aspect='auto')
    ax3.set_title(f'Overlay BEFORE\nNCC={ncc_before:.4f}', fontweight='bold')
    ax3.set_xlabel('X')

    # B-scan before
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(bscan_v0, cmap='Reds', alpha=0.5, aspect='auto')
    ax4.imshow(bscan_v1_before, cmap='Greens', alpha=0.5, aspect='auto')
    ax4.set_title(f'B-scan BEFORE (Z={z_center})', fontweight='bold')
    ax4.set_xlabel('X'); ax4.set_ylabel('Y')

    # Row 2: En-face MIPs - AFTER
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(mip_v0.T, cmap='Reds', alpha=0.7, origin='lower', aspect='auto')
    ax5.set_title('V0 MIP (Reference)', fontweight='bold')
    ax5.set_xlabel('X'); ax5.set_ylabel('Z')

    ax6 = plt.subplot(2, 4, 6)
    ax6.imshow(mip_v1_after.T, cmap='Greens', alpha=0.7, origin='lower', aspect='auto')
    ax6.set_title(f'V1 MIP (AFTER {angle:.2f}° rotation)', fontweight='bold')
    ax6.set_xlabel('X')

    ax7 = plt.subplot(2, 4, 7)
    ax7.imshow(mip_v0.T, cmap='Reds', alpha=0.5, origin='lower', aspect='auto')
    ax7.imshow(mip_v1_after.T, cmap='Greens', alpha=0.5, origin='lower', aspect='auto')
    ax7.set_title(f'Overlay AFTER\nNCC={ncc_after:.4f} ({(ncc_after-ncc_before)*100:+.2f}%)',
                  fontweight='bold')
    ax7.set_xlabel('X')

    # B-scan after
    ax8 = plt.subplot(2, 4, 8)
    ax8.imshow(bscan_v0, cmap='Reds', alpha=0.5, aspect='auto')
    ax8.imshow(bscan_v1_after, cmap='Greens', alpha=0.5, aspect='auto')
    ax8.set_title(f'B-scan AFTER (Z={z_center})', fontweight='bold')
    ax8.set_xlabel('X'); ax8.set_ylabel('Y')

    plt.suptitle(f'Z-Axis Rotation: {angle:+.2f}° (ECC-optimized) + Y-shift Correction | NCC Verification: {ncc_before:.4f} → {ncc_after:.4f}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved rotation comparison: {output_path}")

    plt.close()


# ============================================================================
# WINDOWED Y-ALIGNMENT FUNCTIONS (Step 4)
# ============================================================================

def calculate_windowed_y_offsets(overlap_v0, overlap_v1,
                                  window_size=20,
                                  outlier_threshold=50,
                                  verbose=True):
    """
    Calculate Y-offsets using windowed sampling with interpolation.

    Samples every window_size-th B-scan, calculates Y-offset using
    surface detection, then interpolates smoothly across all B-scans.

    Args:
        overlap_v0: Reference volume (Y, X, Z)
        overlap_v1: Volume to align (Y, X, Z)
        window_size: Sample every N B-scans (default: 20)
        outlier_threshold: Reject offsets > this value (pixels)
        verbose: Print progress

    Returns:
        y_offsets_interpolated: (Z,) array of Y-shifts for each B-scan
        sampled_offsets: (n_samples,) array of calculated Y-offsets at sampled positions
        sampled_positions: (n_samples,) array of Z-indices where samples were taken
        confidences: (n_samples,) array of confidence scores
    """
    from scipy.interpolate import CubicSpline
    from surface_visualization import load_or_detect_surface

    Z = overlap_v0.shape[2]

    # Calculate sampling positions (centers of windows)
    sampled_positions = np.arange(window_size // 2, Z, window_size)
    n_samples = len(sampled_positions)

    if verbose:
        print(f"\n  Sampling {n_samples} B-scans with window size {window_size}")
        print(f"  Sampled positions: {list(sampled_positions)}")

    # Detect surfaces once for entire volume
    if verbose:
        print(f"  Detecting surfaces...")

    surface_v0 = load_or_detect_surface(overlap_v0, method='peak')  # (X, Z)
    surface_v1 = load_or_detect_surface(overlap_v1, method='peak')

    # Calculate Y-offset at each sampled position
    sampled_offsets = np.zeros(n_samples)
    confidences = np.zeros(n_samples)

    if verbose:
        print(f"  Calculating Y-offsets at sampled positions...")

    for i, z in enumerate(sampled_positions):
        # Get surface profiles for this B-scan
        profile_v0 = surface_v0[:, z]  # (X,) surface Y-positions
        profile_v1 = surface_v1[:, z]

        # Calculate height differences
        diff = profile_v0 - profile_v1

        # Filter: remove NaN and outliers
        valid = ~np.isnan(diff) & (np.abs(diff) < outlier_threshold)

        if valid.sum() > 10:
            sampled_offsets[i] = np.median(diff[valid])
            confidences[i] = valid.sum() / len(diff)
        else:
            sampled_offsets[i] = 0
            confidences[i] = 0

    if verbose:
        print(f"  Sampled offset range: {sampled_offsets.min():.1f} to {sampled_offsets.max():.1f} px")
        print(f"  Average confidence: {confidences.mean():.1%}")

    # Interpolate between sampled positions
    if verbose:
        print(f"  Interpolating offsets across all {Z} B-scans...")

    # Add boundary conditions (extrapolate at edges)
    positions_extended = np.concatenate([[0], sampled_positions, [Z-1]])
    offsets_extended = np.concatenate([[sampled_offsets[0]], sampled_offsets, [sampled_offsets[-1]]])

    # Cubic spline interpolation
    cs = CubicSpline(positions_extended, offsets_extended)
    all_positions = np.arange(Z)
    y_offsets_interpolated = cs(all_positions)

    if verbose:
        print(f"  Interpolated offset range: {y_offsets_interpolated.min():.1f} to {y_offsets_interpolated.max():.1f} px")

    return y_offsets_interpolated, sampled_offsets, sampled_positions, confidences


def apply_windowed_y_alignment(volume, y_offsets, verbose=True):
    """
    Apply interpolated Y-offsets to volume.

    Args:
        volume: (Y, X, Z) volume to align
        y_offsets: (Z,) array of Y-shifts for each B-scan
        verbose: Print progress

    Returns:
        aligned_volume: (Y, X, Z) aligned volume
    """
    if verbose:
        print(f"  Applying Y-offsets to {volume.shape[2]} B-scans...")

    aligned = np.zeros_like(volume)

    for z in range(volume.shape[2]):
        aligned[:, :, z] = ndimage.shift(
            volume[:, :, z],
            shift=(y_offsets[z], 0),
            order=1,
            mode='constant',
            cval=0
        )

    return aligned


def visualize_windowed_offsets(y_offsets, sampled_offsets, sampled_positions,
                                confidences, output_path):
    """
    Visualize windowed Y-offsets with sampling points and interpolation.

    Args:
        y_offsets: (Z,) interpolated Y-offsets for all B-scans
        sampled_offsets: (n_samples,) calculated offsets at sampled positions
        sampled_positions: (n_samples,) Z-indices of sampled B-scans
        confidences: (n_samples,) confidence scores
        output_path: Path to save visualization
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    z_all = np.arange(len(y_offsets))

    # Plot 1: Y-offsets
    ax1.plot(z_all, y_offsets, 'b-', linewidth=1.5, label='Interpolated offsets')
    ax1.scatter(sampled_positions, sampled_offsets,
                c='red', s=100, zorder=5, label='Sampled positions', marker='o')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('B-scan Position (Z)', fontsize=12)
    ax1.set_ylabel('Y-offset (pixels)', fontsize=12)
    ax1.set_title('Windowed Y-offsets (Window size: 20)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: Confidence scores
    ax2.bar(sampled_positions, confidences, width=15, color='green', alpha=0.6)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax2.set_xlabel('B-scan Position (Z)', fontsize=12)
    ax2.set_ylabel('Confidence', fontsize=12)
    ax2.set_title('Surface Detection Confidence at Sampled Positions', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path.name}")


if __name__ == "__main__":
    print("Rotation Alignment Module")
    print("=" * 70)
    print("Functions:")
    print("  - find_optimal_rotation_z(): Coarse-to-fine rotation search (Hough)")
    print("  - apply_rotation_z(): Apply rotation to volume")
    print("  - calculate_ncc_3d(): Calculate NCC metric (verification)")
    print("  - visualize_rotation_search(): Plot angle search results")
    print("  - visualize_rotation_comparison(): Plot before/after comparison")
