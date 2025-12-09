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
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os


# ============================================================================
# PARALLEL PROCESSING HELPER FUNCTIONS
# ============================================================================

def calculate_rotation_edge_margin(height, width, angle_degrees):
    """
    Calculate how many pixels to skip from edges after rotation.

    After rotating by angle θ, corners are zero-padded. To avoid false
    surface detections, skip columns that may have incomplete tissue.

    Args:
        height: Image height (Y dimension)
        width: Image width (X dimension)
        angle_degrees: Rotation angle in degrees

    Returns:
        margin: Number of pixels to skip from each edge (left and right)
    """
    import math
    angle_rad = math.radians(abs(angle_degrees))
    # Margin based on image height and rotation angle
    margin = int(math.ceil(height * math.sin(angle_rad)))
    # Also consider width for safety
    margin = max(margin, int(math.ceil(width * math.sin(angle_rad) * 0.5)))
    # Minimum margin of 5px for any rotation, 0 for no rotation
    if abs(angle_degrees) > 0.1:
        margin = max(margin, 5)
    return margin


def _test_single_rotation_angle(angle, bscan_v0_proc, bscan_v1, mask_v0):
    """
    Worker function to test a single rotation angle.

    This function is designed to be pickled and run in parallel.

    Args:
        angle: Rotation angle to test (degrees)
        bscan_v0_proc: Pre-processed reference B-scan
        bscan_v1: Moving B-scan (will be rotated)
        mask_v0: Mask for reference B-scan

    Returns:
        dict with angle, correlation score, and valid pixels
    """
    try:
        # Rotate B-scan
        bscan_v1_rotated = ndimage.rotate(
            bscan_v1, angle, axes=(0, 1),
            reshape=False, order=1,
            mode='constant', cval=0
        )

        # Create mask for rotated B-scan
        if (bscan_v1_rotated > 0).any():
            threshold = np.percentile(bscan_v1_rotated[bscan_v1_rotated > 0], 10)
            mask_v1 = bscan_v1_rotated > threshold
        else:
            mask_v1 = bscan_v1_rotated > 0

        mask_combined = mask_v0 & mask_v1

        # Exclude edge columns affected by rotation zero-padding
        H, W = bscan_v1_rotated.shape
        edge_margin = calculate_rotation_edge_margin(H, W, angle)
        if edge_margin > 0 and edge_margin < W // 2:
            mask_combined[:, :edge_margin] = False
            mask_combined[:, -edge_margin:] = False

        # Check if enough valid pixels
        if mask_combined.sum() < 100:
            return {
                'angle': float(angle),
                'correlation': -1.0,
                'valid_pixels': 0
            }

        # Preprocess rotated B-scan
        bscan_v1_proc = preprocess_oct_for_rotation(bscan_v1_rotated, mask=mask_combined)

        # Calculate correlation
        i1 = bscan_v0_proc[mask_combined].astype(float)
        i2 = bscan_v1_proc[mask_combined].astype(float)

        # Normalize
        i1_norm = (i1 - i1.mean()) / (i1.std() + 1e-8)
        i2_norm = (i2 - i2.mean()) / (i2.std() + 1e-8)

        score = np.mean(i1_norm * i2_norm)

        return {
            'angle': float(angle),
            'correlation': float(score),
            'valid_pixels': int(mask_combined.sum())
        }

    except Exception as e:
        return {
            'angle': float(angle),
            'correlation': -1.0,
            'valid_pixels': 0
        }


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_oct_for_rotation(img, mask=None):
    """
    Preprocess OCT B-scan with EXTREME aggressive denoising and thresholding.

    This isolates clean retinal layer structures by removing speckle noise
    and zeroing out low-intensity regions.

    EXTREME HARSH DENOISING Pipeline:
      1. Non-local means denoising (h=30) - ULTRA HARSH (+20%)
      2. Bilateral filtering (sigma=180) - ULTRA HARSH (+20%)
      3. Median filter (kernel=19) - ULTRA HARSH (+27%)
      4. Otsu thresholding (85% threshold - keeps only strongest signals)
      5. CLAHE contrast enhancement (clipLimit=3.6) - (+20%)
      6. Horizontal kernel filter (emphasizes layer structures)

    Args:
        img: Input B-scan (2D array)
        mask: Optional binary mask to focus on valid regions

    Returns:
        Preprocessed uint8 image with isolated layer structures
    """
    # Normalize to 0-255 first
    img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    # Step 1: Non-local means denoising (ULTRA HARSH - +20%)
    denoised1 = cv2.fastNlMeansDenoising(img_norm, h=30, templateWindowSize=7, searchWindowSize=21)

    # Step 2: Bilateral filtering for edge-preserving smoothing (ULTRA HARSH - +20%)
    denoised2 = cv2.bilateralFilter(denoised1, d=11, sigmaColor=180, sigmaSpace=180)

    # Step 3: Median filter to remove remaining noise (ULTRA HARSH - +27%)
    denoised3 = cv2.medianBlur(denoised2, 19)

    # Step 4: Threshold to keep only strongest signals (85% of Otsu - EXTREME HARSH)
    thresh_val = cv2.threshold(denoised3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_val = int(thresh_val * 0.85)  # 85% of Otsu threshold = EXTREME HARSH

    thresholded = denoised3.copy()
    thresholded[denoised3 < thresh_val] = 0

    # Step 5: Enhance contrast (CLAHE - +20%)
    clahe = cv2.createCLAHE(clipLimit=3.6, tileGridSize=(8, 8))
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


def preprocess_oct_for_visualization(img):
    """
    Preprocess OCT B-scan for VISUALIZATION ONLY (no horizontal kernel filter).

    Same as preprocess_oct_for_rotation() but WITHOUT the horizontal kernel filter
    that can make images appear dark in matplotlib displays.

    EXTREME HARSH DENOISING (for three_volume_alignment and averaged_bscan_alignment):
    - Stronger NLM denoising (h=30 vs 25) - +20%
    - Stronger bilateral filtering (sigma=180 vs 150) - +20%
    - Larger median filter (19x19 vs 15x15) - +27%
    - EXTREME HARSH threshold (85% of Otsu) - keeps only strongest signals
    - Stronger CLAHE (clipLimit=3.6 vs 3.0) - +20%

    Args:
        img: Input B-scan (2D array)

    Returns:
        Preprocessed uint8 image
    """
    # Normalize to 0-255 first
    img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    # Step 1: Non-local means denoising (EXTRA HARSH - 20% stronger)
    denoised = cv2.fastNlMeansDenoising(img_norm, h=30, templateWindowSize=7, searchWindowSize=21)

    # Step 2: Bilateral filtering (EXTRA HARSH - 20% stronger)
    denoised = cv2.bilateralFilter(denoised, d=11, sigmaColor=180, sigmaSpace=180)

    # Step 3: Median filter (EXTRA HARSH - 27% larger kernel)
    denoised = cv2.medianBlur(denoised, 19)

    # Step 4: Threshold (85% of Otsu - EXTREME HARSH, keeps only strongest signals)
    thresh_val = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_val = int(thresh_val * 0.85)  # 85% of Otsu = EXTREME HARSH
    denoised[denoised < thresh_val] = 0

    # Step 5: CLAHE contrast enhancement (20% stronger)
    clahe = cv2.createCLAHE(clipLimit=3.6, tileGridSize=(8, 8))
    denoised = clahe.apply(denoised)

    # NO Step 6 horizontal kernel - it makes images appear dark for visualization

    return denoised


def detect_contour_surface(bscan_denoised, outlier_threshold=20, edge_margin=0, start_y=0):
    """
    Detect retinal surface using contour method (finds where tissue STARTS).

    This method works perfectly on denoised B-scans by finding the first
    white pixel (top boundary) in each column.

    Includes outlier filtering: points that differ from neighbors by more than
    outlier_threshold pixels are replaced with the average of neighbors.

    Args:
        bscan_denoised: Preprocessed B-scan (2D array, uint8, already denoised)
        outlier_threshold: Max allowed deviation from neighbors (default: 20px)
        edge_margin: Skip this many pixels from each side (default: 0)
        start_y: Start searching from this Y position (default: 0). Use this to skip
                 empty regions at the top of the image where no tissue exists.

    Returns:
        surface: 1D array (X,) with Y positions of detected surface (outliers filtered)
                 Edge margin pixels are filled with the nearest valid value.
    """
    Y, X = bscan_denoised.shape
    surface = np.zeros(X)

    # Simple threshold (70th percentile works well on denoised images)
    threshold = np.percentile(bscan_denoised, 70)
    _, binary = cv2.threshold(bscan_denoised, threshold, 255, cv2.THRESH_BINARY)

    # Determine valid range (skip edge_margin from each side)
    x_start = edge_margin
    x_end = X - edge_margin
    if x_end <= x_start:
        # Edge margin too large, use full range
        x_start = 0
        x_end = X

    # For each column in valid range, find first white pixel (top boundary = tissue start)
    # Start searching from start_y to skip empty regions at the top
    for x in range(x_start, x_end):
        column = binary[start_y:, x]  # Only search from start_y onwards
        white_pixels = np.where(column > 0)[0]
        if len(white_pixels) > 0:
            surface[x] = start_y + white_pixels[0]  # Add start_y offset back
        else:
            # No tissue detected - use previous column or default
            surface[x] = surface[x-1] if x > 0 else Y // 2

    # Fill edge margins with nearest valid value
    if edge_margin > 0 and x_start < x_end:
        # Left margin: fill with first valid value
        surface[:x_start] = surface[x_start]
        # Right margin: fill with last valid value
        surface[x_end:] = surface[x_end - 1]

    # PASS 1: Filter extreme outliers (>80px from global average)
    global_avg = np.mean(surface[x_start:x_end])  # Only use valid range for average
    surface_pass1 = surface.copy()
    for x in range(X):
        if abs(surface[x] - global_avg) > 80:
            surface_pass1[x] = global_avg

    # PASS 2: Filter smaller outliers (>20px from 6 neighbors)
    surface_filtered = surface_pass1.copy()
    for x in range(X):
        left_idx = max(0, x - 3)
        right_idx = min(X, x + 4)
        neighbors = np.concatenate([surface_pass1[left_idx:x], surface_pass1[x+1:right_idx]])
        if len(neighbors) > 0:
            neighbor_avg = np.mean(neighbors)
            if abs(surface_pass1[x] - neighbor_avg) > outlier_threshold:
                surface_filtered[x] = neighbor_avg

    return surface_filtered


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

    # Extract CENTRAL X-section for fast optimization (VERTICAL: use X center)
    x_center = overlap_v0.shape[1] // 2
    bscan_v0 = overlap_v0[:, x_center, :]  # (Y, Z) - VERTICAL
    bscan_v1 = overlap_v1[:, x_center, :]  # (Y, Z) - VERTICAL

    if verbose:
        print(f"\n{'='*70}")
        print(f"COARSE ROTATION SEARCH (ECC-STYLE CORRELATION)")
        print(f"{'='*70}")
        print(f"  Method: Aggressive denoising + correlation scoring")
        print(f"  Goal: Find rotation that maximizes layer structure overlap")
        print(f"  Angle range: ±{angle_range}°")
        print(f"  Step size: {step}°")
        print(f"  Total angles: {len(angles_to_test)}")
        print(f"  Using central X-section: X={x_center}/{overlap_v0.shape[1]}")
        print(f"  X-section shape: {bscan_v0.shape} (Y, Z)")

    # Create mask for valid regions
    mask_v0 = bscan_v0 > np.percentile(bscan_v0[bscan_v0 > 0], 10) if (bscan_v0 > 0).any() else bscan_v0 > 0

    # Preprocess reference B-scan ONCE
    if verbose:
        print(f"  Preprocessing reference B-scan...")
    bscan_v0_proc = preprocess_oct_for_rotation(bscan_v0, mask=mask_v0)

    if verbose:
        print(f"  Starting PARALLEL rotation search (using {min(os.cpu_count(), len(angles_to_test))} workers)...")

    # PARALLEL PROCESSING: Test all angles in parallel
    num_workers = min(os.cpu_count(), len(angles_to_test))

    with Pool(processes=num_workers) as pool:
        # Create partial function with fixed parameters
        test_func = partial(
            _test_single_rotation_angle,
            bscan_v0_proc=bscan_v0_proc,
            bscan_v1=bscan_v1,
            mask_v0=mask_v0
        )

        # Map angles to worker function in parallel
        results = pool.map(test_func, angles_to_test)

    # Find best result
    best_result = max(results, key=lambda x: x['correlation'])
    best_angle = best_result['angle']
    best_score = best_result['correlation']

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

    # Extract CENTRAL X-section for fast optimization (VERTICAL: use X center)
    x_center = overlap_v0.shape[1] // 2
    bscan_v0 = overlap_v0[:, x_center, :]  # (Y, Z) - VERTICAL
    bscan_v1 = overlap_v1[:, x_center, :]  # (Y, Z) - VERTICAL

    if verbose:
        print(f"\n{'='*70}")
        print(f"FINE ROTATION SEARCH (ECC-STYLE CORRELATION)")
        print(f"{'='*70}")
        print(f"  Method: Aggressive denoising + correlation scoring")
        print(f"  Center angle: {coarse_angle:+.1f}°")
        print(f"  Search range: ±{angle_range}°")
        print(f"  Step size: {step}°")
        print(f"  Total angles: {len(angles_to_test)}")
        print(f"  Using central X-section: X={x_center}/{overlap_v0.shape[1]}")

    # Create mask for valid regions
    mask_v0 = bscan_v0 > np.percentile(bscan_v0[bscan_v0 > 0], 10) if (bscan_v0 > 0).any() else bscan_v0 > 0

    # Preprocess reference B-scan ONCE
    if verbose:
        print(f"  Preprocessing reference B-scan...")
    bscan_v0_proc = preprocess_oct_for_rotation(bscan_v0, mask=mask_v0)

    if verbose:
        print(f"  Starting PARALLEL fine search (using {min(os.cpu_count(), len(angles_to_test))} workers)...")

    # PARALLEL PROCESSING: Test all angles in parallel
    num_workers = min(os.cpu_count(), len(angles_to_test))

    with Pool(processes=num_workers) as pool:
        # Create partial function with fixed parameters
        test_func = partial(
            _test_single_rotation_angle,
            bscan_v0_proc=bscan_v0_proc,
            bscan_v1=bscan_v1,
            mask_v0=mask_v0
        )

        # Map angles to worker function in parallel
        results = pool.map(test_func, angles_to_test)

    # Find best result
    best_result = max(results, key=lambda x: x['correlation'])
    best_angle = best_result['angle']
    best_score = best_result['correlation']

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
                                        verbose=True, rotation_angle=0.0):
    """
    Step 3.1: Find optimal Y-axis shift on central B-scan after rotation.

    After rotation in Step 3, the Y-axis (depth) alignment may be disturbed.
    This function re-aligns along Y-axis using CONTOUR-BASED surface detection
    on the central B-scan - much faster and more robust than NCC search.

    METHOD: Detects retinal surface on both B-scans using contour detection,
    then calculates the median difference to find optimal Y-shift.

    Args:
        overlap_v0: Reference overlap volume (Y, X, Z)
        overlap_v1_rotated: Rotated volume from Step 3 (Y, X, Z)
        search_range: Unused (kept for compatibility)
        step: Unused (kept for compatibility)
        verbose: Print progress
        rotation_angle: Rotation angle applied in Step 3 (degrees) - used to calculate edge margin

    Returns:
        best_shift: Optimal Y shift correction (pixels)
        best_ncc: NCC score at optimal shift (calculated for verification)
        results: List with single result dict
    """
    # Extract central X-section (full depth - no cropping) - VERTICAL: use X center
    x_center = overlap_v0.shape[1] // 2
    bscan_v0 = overlap_v0[:, x_center, :]  # (Y, Z) - VERTICAL
    bscan_v1 = overlap_v1_rotated[:, x_center, :]  # (Y, Z) - VERTICAL

    # Calculate edge margin based on rotation angle (to exclude zero-padded corners)
    # Use at least 40px, but more if rotation angle requires it
    H, W = bscan_v1.shape
    rotation_margin = calculate_rotation_edge_margin(H, W, rotation_angle)
    edge_margin = max(40, rotation_margin)  # At least 40px, or more if rotation requires

    if verbose:
        print(f"\n{'='*70}")
        print(f"STEP 3.1: Y-AXIS RE-ALIGNMENT (CONTOUR-BASED)")
        print(f"{'='*70}")
        print(f"  Method: Surface detection (contour) - direct geometric alignment")
        print(f"  Using central X-section: X={x_center}/{overlap_v0.shape[1]}")
        print(f"  Rotation angle from Step 3: {rotation_angle:+.2f}°")
        print(f"  Edge margin: {edge_margin}px (base=40px, rotation={rotation_margin}px)")

    # Apply harsh denoising to both B-scans (same as Steps 3 & 3.5)
    # Use two versions: one for alignment (with horizontal filter), one for visualization (without)
    if verbose:
        print(f"  Applying harsh denoising to B-scans...")

    # For alignment: use full preprocessing with horizontal kernel filter
    bscan_v0_denoised_align = preprocess_oct_for_rotation(bscan_v0)
    bscan_v1_denoised_align = preprocess_oct_for_rotation(bscan_v1)

    # For visualization: use preprocessing WITHOUT horizontal kernel filter (prevents dark images)
    bscan_v0_denoised_vis = preprocess_oct_for_visualization(bscan_v0)
    bscan_v1_denoised_vis = preprocess_oct_for_visualization(bscan_v1)

    # Detect surfaces using contour method (on alignment-version images)
    # Skip 40px from each side to reduce edge artifacts
    if verbose:
        print(f"  Detecting retinal surfaces (skipping {edge_margin}px edges)...")
    surface_v0 = detect_contour_surface(bscan_v0_denoised_align, edge_margin=edge_margin)  # (X,) array
    surface_v1 = detect_contour_surface(bscan_v1_denoised_align, edge_margin=edge_margin)  # (X,) array

    # Calculate surface difference (V0 - V1)
    diff = surface_v0 - surface_v1

    # Filter out outliers (use robust median instead of mean)
    valid = ~np.isnan(diff) & (np.abs(diff) < 100)  # Remove extreme outliers

    if valid.sum() > 10:
        # Median is robust to outliers
        y_shift = np.median(diff[valid])
        std_diff = np.std(diff[valid])
        confidence = valid.sum() / len(diff)
    else:
        y_shift = 0.0
        std_diff = 0.0
        confidence = 0.0

    if verbose:
        print(f"\n  ✓ Surface detection complete!")
        print(f"    Surface difference: {y_shift:+.2f} ± {std_diff:.2f} px")
        print(f"    Valid pixels: {valid.sum()}/{len(diff)} ({confidence:.1%})")

    # Verify with NCC at the detected shift (use alignment-version for NCC calculation)
    if verbose:
        print(f"  Verifying alignment quality with NCC...")

    bscan_shifted = ndimage.shift(
        bscan_v1_denoised_align, shift=(y_shift, 0),
        order=1, mode='constant', cval=0
    )

    threshold = calculate_tissue_threshold(bscan_v0_denoised_align, bscan_v1_denoised_align, percentile=50)
    mask = (bscan_v0_denoised_align > threshold) & (bscan_shifted > threshold)

    if mask.sum() > 100:
        ncc_score = calculate_ncc(bscan_v0_denoised_align, bscan_shifted, mask=mask)
    else:
        ncc_score = -1.0

    if verbose:
        print(f"    NCC at detected shift: {ncc_score:.4f}")

        if abs(y_shift) < 0.5:
            print(f"    → No significant correction needed ({y_shift:+.2f} px < 0.5 px threshold)")
        else:
            print(f"    → Will apply correction: {y_shift:+.2f} px")

    # Return in same format as old function for compatibility, with extra data for visualization
    results = [{
        'y_shift': float(y_shift),
        'ncc': ncc_score,
        'std_diff': float(std_diff),
        'confidence': float(confidence),
        'valid_pixels': int(valid.sum()),
        # Extra data for visualization (use visualization-friendly denoised images)
        'bscan_v0': bscan_v0,
        'bscan_v1': bscan_v1,
        'bscan_v0_denoised': bscan_v0_denoised_vis,  # WITHOUT horizontal kernel
        'bscan_v1_denoised': bscan_v1_denoised_vis,  # WITHOUT horizontal kernel
        'surface_v0': surface_v0,
        'surface_v1': surface_v1,
        'edge_margin': edge_margin  # For visualization
    }]

    return float(y_shift), ncc_score, results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_contour_y_alignment(bscan_v0, bscan_v1,
                                   bscan_v0_denoised, bscan_v1_denoised,
                                   surface_v0, surface_v1, y_shift,
                                   ncc_score, confidence, output_path=None,
                                   edge_margin=0):
    """
    Visualize Step 3.1 contour-based Y-axis alignment.

    Shows:
    - Original and denoised B-scans
    - Detected surface contours
    - Surface difference plot
    - Before/after alignment comparison
    - Edge margin regions (if edge_margin > 0)

    Args:
        bscan_v0: Original reference B-scan (Y, X)
        bscan_v1: Original B-scan to align (Y, X)
        bscan_v0_denoised: Denoised reference B-scan (Y, X)
        bscan_v1_denoised: Denoised B-scan to align (Y, X)
        surface_v0: Detected surface on V0 (X,)
        surface_v1: Detected surface on V1 (X,)
        y_shift: Calculated Y-shift correction (pixels)
        ncc_score: NCC score at optimal shift
        confidence: Detection confidence (0-1)
        output_path: Where to save the figure
        edge_margin: Pixels skipped from each edge (default: 0)
    """
    import matplotlib.pyplot as plt

    Y, X = bscan_v0.shape

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Original B-scans
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(bscan_v0, cmap='gray', aspect='auto')
    ax1.set_title('Volume 0 - Original B-scan (Central)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('X (A-scans)')
    ax1.set_ylabel('Y (depth)')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(bscan_v1, cmap='gray', aspect='auto')
    ax2.set_title('Volume 1 - Original B-scan (Central)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('X (A-scans)')
    ax2.set_ylabel('Y (depth)')

    # Row 1, Col 3: Alignment metrics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    metrics_text = f"""
    STEP 3.1: CONTOUR-BASED Y-ALIGNMENT

    Method: Surface Detection

    Results:
    • Y-shift: {y_shift:+.2f} px
    • NCC score: {ncc_score:.4f}
    • Confidence: {confidence:.1%}

    Interpretation:
    {'→ Significant shift detected' if abs(y_shift) >= 0.5 else '→ No significant shift'}
    {'→ High quality alignment' if ncc_score > 0.75 else '→ Moderate alignment quality'}
    """
    ax3.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Row 2: Denoised B-scans with detected surfaces
    # (denoised images are already uint8 0-255 from preprocess_oct_for_rotation)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(bscan_v0_denoised, cmap='gray', aspect='auto')
    ax4.plot(surface_v0, 'r-', linewidth=4, label='Detected Surface')
    # Show edge margin regions (skipped during detection)
    if edge_margin > 0:
        ax4.axvline(x=edge_margin, color='yellow', linestyle='--', linewidth=2, label=f'Edge margin ({edge_margin}px)')
        ax4.axvline(x=X - edge_margin, color='yellow', linestyle='--', linewidth=2)
        ax4.axvspan(0, edge_margin, alpha=0.2, color='yellow')
        ax4.axvspan(X - edge_margin, X, alpha=0.2, color='yellow')
    ax4.set_title('V0 - Denoised (Single B-scan) + Surface', fontsize=11, fontweight='bold')
    ax4.set_xlabel('X (A-scans)')
    ax4.set_ylabel('Y (depth)')
    ax4.legend(loc='upper right')
    ax4.set_ylim([0, Y // 2])

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(bscan_v1_denoised, cmap='gray', aspect='auto')
    ax5.plot(surface_v1, 'b-', linewidth=4, label='Detected Surface')
    # Show edge margin regions (skipped during detection)
    if edge_margin > 0:
        ax5.axvline(x=edge_margin, color='yellow', linestyle='--', linewidth=2, label=f'Edge margin ({edge_margin}px)')
        ax5.axvline(x=X - edge_margin, color='yellow', linestyle='--', linewidth=2)
        ax5.axvspan(0, edge_margin, alpha=0.2, color='yellow')
        ax5.axvspan(X - edge_margin, X, alpha=0.2, color='yellow')
    ax5.set_title('V1 - Denoised (Single B-scan) + Surface', fontsize=11, fontweight='bold')
    ax5.set_xlabel('X (A-scans)')
    ax5.set_ylabel('Y (depth)')
    ax5.legend(loc='upper right')
    ax5.set_ylim([0, Y // 2])

    # Row 2, Col 3: Surface difference
    ax6 = fig.add_subplot(gs[1, 2])
    diff = surface_v0 - surface_v1
    valid = ~np.isnan(diff) & (np.abs(diff) < 100)
    x_coords = np.arange(X)
    ax6.plot(x_coords[valid], diff[valid], 'k-', linewidth=1, alpha=0.5, label='Raw difference')
    ax6.axhline(y_shift, color='red', linestyle='--', linewidth=2, label=f'Median shift: {y_shift:+.2f} px')
    ax6.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax6.fill_between(x_coords, y_shift - np.std(diff[valid]), y_shift + np.std(diff[valid]),
                      alpha=0.3, color='red', label=f'±1 std: {np.std(diff[valid]):.2f} px')
    # Show edge margin regions on difference plot
    if edge_margin > 0:
        ax6.axvline(x=edge_margin, color='yellow', linestyle='--', linewidth=2)
        ax6.axvline(x=X - edge_margin, color='yellow', linestyle='--', linewidth=2)
        ax6.axvspan(0, edge_margin, alpha=0.2, color='yellow')
        ax6.axvspan(X - edge_margin, X, alpha=0.2, color='yellow')
    ax6.set_title('Surface Difference (V0 - V1)', fontsize=11, fontweight='bold')
    ax6.set_xlabel('X (A-scans)')
    ax6.set_ylabel('Y difference (pixels)')
    ax6.legend(loc='best', fontsize=9)
    ax6.grid(alpha=0.3)

    # Row 3: Surface overlay (before alignment)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.imshow(bscan_v0_denoised, cmap='gray', aspect='auto', alpha=0.7)
    ax7.plot(surface_v0, 'r-', linewidth=4, label='V0 surface')
    ax7.plot(surface_v1, 'b-', linewidth=4, label='V1 surface', alpha=0.8)
    ax7.fill_between(np.arange(X), surface_v0, surface_v1,
                      alpha=0.3, color='yellow', label='Misalignment')
    ax7.set_title('BEFORE: Surface Overlay (Single B-scan)', fontsize=11, fontweight='bold')
    ax7.set_xlabel('X (A-scans)')
    ax7.set_ylabel('Y (depth)')
    ax7.legend(loc='upper right')
    ax7.set_ylim([0, Y // 2])

    # Row 3: Surface overlay (after alignment)
    ax8 = fig.add_subplot(gs[2, 1])
    surface_v1_corrected = surface_v1 + y_shift
    ax8.imshow(bscan_v0_denoised, cmap='gray', aspect='auto', alpha=0.7)
    ax8.plot(surface_v0, 'r-', linewidth=4, label='V0 surface')
    ax8.plot(surface_v1_corrected, 'b-', linewidth=4, label='V1 surface (corrected)', alpha=0.8)
    ax8.fill_between(np.arange(X), surface_v0, surface_v1_corrected,
                      alpha=0.3, color='green', label='Residual misalignment')
    ax8.set_title(f'AFTER: Surface Overlay (Y-shift: {y_shift:+.2f} px)', fontsize=11, fontweight='bold')
    ax8.set_xlabel('X (A-scans)')
    ax8.set_ylabel('Y (depth)')
    ax8.legend(loc='upper right')
    ax8.set_ylim([0, Y // 2])

    # Row 3, Col 3: Residual error after correction
    ax9 = fig.add_subplot(gs[2, 2])
    residual = surface_v0 - surface_v1_corrected
    valid_res = ~np.isnan(residual) & (np.abs(residual) < 50)
    ax9.hist(residual[valid_res], bins=50, color='green', alpha=0.7, edgecolor='black')
    ax9.axvline(np.median(residual[valid_res]), color='red', linestyle='--',
                linewidth=2, label=f'Median: {np.median(residual[valid_res]):.2f} px')
    ax9.axvline(0, color='gray', linestyle='-', linewidth=1)
    ax9.set_title('Residual Error After Correction', fontsize=11, fontweight='bold')
    ax9.set_xlabel('Residual difference (pixels)')
    ax9.set_ylabel('Frequency')
    ax9.legend(loc='upper right')
    ax9.grid(alpha=0.3, axis='y')

    fig.suptitle('Step 3.1: Contour-Based Y-Axis Re-alignment (Central B-scan)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization: {output_path}")
        plt.close()
    else:
        plt.show()


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

    # Get central X-section (VERTICAL: use X center)
    x_center = overlap_v0.shape[1] // 2
    bscan_v0 = overlap_v0[:, x_center, :]  # (Y, Z) - VERTICAL
    bscan_v1 = overlap_v1[:, x_center, :]  # (Y, Z) - VERTICAL
    mask_v0_bscan = mask_v0[:, x_center, :]
    mask_v1_bscan = mask_v1[:, x_center, :]
    mask_combined_bscan = mask_combined[:, x_center, :]

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
    # Convert boolean masks to float for proper visualization (white=included, black=excluded)
    ax4 = plt.subplot(4, 3, 4)
    ax4.imshow(mask_v0_mip.T.astype(float), cmap='gray', origin='lower', aspect='auto', vmin=0, vmax=1)
    valid_v0 = mask_v0.sum()
    ax4.set_title(f'V0 Mask\n{valid_v0:,} voxels ({valid_v0/mask_v0.size*100:.1f}%)', fontweight='bold')
    ax4.set_ylabel('Z (B-scans)')
    ax4.set_xlabel('X (lateral)')

    ax5 = plt.subplot(4, 3, 5)
    ax5.imshow(mask_v1_mip.T.astype(float), cmap='gray', origin='lower', aspect='auto', vmin=0, vmax=1)
    valid_v1 = mask_v1.sum()
    ax5.set_title(f'V1 Mask\n{valid_v1:,} voxels ({valid_v1/mask_v1.size*100:.1f}%)', fontweight='bold')
    ax5.set_xlabel('X (lateral)')

    ax6 = plt.subplot(4, 3, 6)
    ax6.imshow(mask_combined_mip.T.astype(float), cmap='hot', origin='lower', aspect='auto', vmin=0, vmax=1)
    valid_combined = mask_combined.sum()
    ax6.set_title(f'Combined Mask (V0 & V1)\n{valid_combined:,} voxels ({valid_combined/mask_combined.size*100:.1f}%)',
                  fontweight='bold', color='red')
    ax6.set_xlabel('X (lateral)')

    # Row 3: X-sections (VERTICAL)
    ax7 = plt.subplot(4, 3, 7)
    ax7.imshow(bscan_v0, cmap='gray', aspect='auto')
    ax7.set_title(f'V0 X-section @ X={x_center}', fontweight='bold')
    ax7.set_ylabel('Y (depth)')
    ax7.set_xlabel('Z (B-scan index)')

    ax8 = plt.subplot(4, 3, 8)
    ax8.imshow(bscan_v1, cmap='gray', aspect='auto')
    ax8.set_title(f'V1 X-section @ X={x_center}', fontweight='bold')
    ax8.set_xlabel('Z (B-scan index)')

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
    bscan_v0_raw = overlap_v0[:, :, z_center]
    bscan_v1_before_raw = overlap_v1_before[:, :, z_center]
    bscan_v1_after_raw = overlap_v1_after[:, :, z_center]

    # Denoise B-scans for visualization (without horizontal kernel filter)
    bscan_v0 = preprocess_oct_for_visualization(bscan_v0_raw)
    bscan_v1_before = preprocess_oct_for_visualization(bscan_v1_before_raw)
    bscan_v1_after = preprocess_oct_for_visualization(bscan_v1_after_raw)

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

    # B-scan before (denoised)
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(bscan_v0, cmap='Reds', alpha=0.5, aspect='auto')
    ax4.imshow(bscan_v1_before, cmap='Greens', alpha=0.5, aspect='auto')
    ax4.set_title(f'Denoised B-scan BEFORE (Z={z_center})', fontweight='bold')
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

    # B-scan after (denoised)
    ax8 = plt.subplot(2, 4, 8)
    ax8.imshow(bscan_v0, cmap='Reds', alpha=0.5, aspect='auto')
    ax8.imshow(bscan_v1_after, cmap='Greens', alpha=0.5, aspect='auto')
    ax8.set_title(f'Denoised B-scan AFTER (Z={z_center})', fontweight='bold')
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
                                  outlier_threshold=100,
                                  verbose=True):
    """
    Calculate Y-offsets using windowed sampling with interpolation.

    Samples every window_size-th B-scan, calculates Y-offset using
    surface detection, then interpolates smoothly across all B-scans.

    Args:
        overlap_v0: Reference volume (Y, X, Z)
        overlap_v1: Volume to align (Y, X, Z)
        window_size: Sample every N B-scans (default: 20)
        outlier_threshold: Reject offsets > this value (pixels, default: 100)
        verbose: Print progress

    Returns:
        y_offsets_interpolated: (Z,) array of Y-shifts for each B-scan
        sampled_offsets: (n_samples,) array of calculated Y-offsets at sampled positions
        sampled_positions: (n_samples,) array of Z-indices where samples were taken
        confidences: (n_samples,) array of confidence scores

    Note:
        Increased outlier_threshold from 50 to 100 to handle larger residual Y-offsets
        after rotation steps. This prevents valid data from being rejected as outliers.
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

    # Detect surfaces ONLY at sampled positions (performance optimization)
    if verbose:
        print(f"  Detecting surfaces at {n_samples} sampled B-scans (with harsh denoising)...")

    surface_v0 = load_or_detect_surface(overlap_v0, method='peak', sampled_positions=sampled_positions)  # (X, n_samples)
    surface_v1 = load_or_detect_surface(overlap_v1, method='peak', sampled_positions=sampled_positions)

    # Calculate Y-offset at each sampled position
    sampled_offsets = np.zeros(n_samples)
    confidences = np.zeros(n_samples)

    if verbose:
        print(f"  Calculating Y-offsets at sampled positions...")

    for i in range(n_samples):
        # Get surface profiles for this sampled B-scan
        profile_v0 = surface_v0[:, i]  # (X,) surface Y-positions
        profile_v1 = surface_v1[:, i]

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


# ============================================================================
# STEP 5: B-SPLINE FREE-FORM DEFORMATION (NON-RIGID REGISTRATION)
# ============================================================================

def calculate_registration_metrics(fixed, moving, registered, deformation_field=None):
    """
    Calculate quality metrics for registration.

    Parameters:
    -----------
    fixed : np.ndarray
        Reference volume
    moving : np.ndarray
        Original moving volume
    registered : np.ndarray
        Registered moving volume
    deformation_field : np.ndarray, optional
        Deformation field (Y, X, Z, 3) if available

    Returns:
    --------
    metrics : dict
        Registration quality metrics
    """
    from scipy.stats import pearsonr

    # NCC before and after
    ncc_before = pearsonr(fixed.ravel(), moving.ravel())[0]
    ncc_after = pearsonr(fixed.ravel(), registered.ravel())[0]

    metrics = {
        'ncc_before': float(ncc_before),
        'ncc_after': float(ncc_after),
        'ncc_improvement_percent': float((ncc_after - ncc_before) * 100)
    }

    # Deformation statistics if available
    if deformation_field is not None:
        displacements = np.linalg.norm(deformation_field, axis=-1)  # Magnitude at each voxel
        metrics.update({
            'mean_displacement': float(np.mean(displacements)),
            'max_displacement': float(np.max(displacements)),
            'std_displacement': float(np.std(displacements))
        })

    return metrics


def bspline_registration_oct(
    fixed_volume: np.ndarray,
    moving_volume: np.ndarray,
    grid_spacing: tuple = (32, 32, 32),
    max_iterations: int = 500,
    num_resolutions: int = 4,
    bending_weight: float = 0.1,
    verbose: bool = True
):
    """
    Perform B-spline FFD registration on OCT volumes using ITK-Elastix.

    This implements non-rigid registration to handle retinal layer curvature
    that cannot be corrected with rigid transformations.

    Parameters:
    -----------
    fixed_volume : np.ndarray
        Reference volume (Y, X, Z) shape
    moving_volume : np.ndarray
        Volume to align (same shape as fixed)
    grid_spacing : tuple
        Control point spacing in voxels (Y, X, Z)
        - (64, 64, 64): Coarse global curvature (~10s)
        - (32, 32, 32): Recommended for layer alignment (~30s)
        - (16, 16, 16): Fine alignment (~2 min)
    max_iterations : int
        Maximum optimizer iterations per resolution
    num_resolutions : int
        Number of pyramid levels (4 = 8x→4x→2x→1x)
    bending_weight : float
        Regularization weight (higher = smoother)
        - 0.01-0.1: Flexible deformation
        - 0.5-1.0: Conservative, prevents overfitting
    verbose : bool
        Print registration progress

    Returns:
    --------
    registered_volume : np.ndarray
        Aligned moving volume
    deformation_field : np.ndarray
        Dense displacement field (Y, X, Z, 3)
    metrics : dict
        Registration quality metrics
    """
    try:
        import itk
    except ImportError:
        raise ImportError(
            "ITK-Elastix not installed. Install with: pip install itk-elastix"
        )

    if verbose:
        print("="*70)
        print("B-SPLINE FREE-FORM DEFORMATION REGISTRATION")
        print("="*70)
        print(f"Fixed volume shape:  {fixed_volume.shape}")
        print(f"Moving volume shape: {moving_volume.shape}")
        print(f"Grid spacing:        {grid_spacing}")
        print(f"Resolutions:         {num_resolutions}")
        print(f"Bending weight:      {bending_weight}")

    # Validate inputs
    if fixed_volume.shape != moving_volume.shape:
        raise ValueError(f"Volume shape mismatch: fixed={fixed_volume.shape}, moving={moving_volume.shape}")

    if np.isnan(fixed_volume).any() or np.isnan(moving_volume).any():
        raise ValueError("Input volumes contain NaN values")

    if np.all(fixed_volume == 0) or np.all(moving_volume == 0):
        raise ValueError("Input volumes are all zeros")

    # Convert NumPy arrays to ITK images
    if verbose:
        print("\n  Converting NumPy arrays to ITK images...")

    try:
        fixed_image = itk.image_from_array(fixed_volume.astype(np.float32))
        moving_image = itk.image_from_array(moving_volume.astype(np.float32))
    except Exception as e:
        raise RuntimeError(f"Failed to convert NumPy arrays to ITK images: {e}")

    # Create parameter map
    parameter_object = itk.ParameterObject.New()
    bspline_map = parameter_object.GetDefaultParameterMap('bspline')

    # Configure B-spline parameters
    bspline_map['MaximumNumberOfIterations'] = [str(max_iterations)]
    bspline_map['FinalGridSpacingInPhysicalUnits'] = [
        str(float(grid_spacing[0])),
        str(float(grid_spacing[1])),
        str(float(grid_spacing[2]))
    ]

    # Similarity metric: NCC + bending energy penalty
    bspline_map['Metric'] = ['AdvancedNormalizedCorrelation', 'TransformBendingEnergyPenalty']
    bspline_map['Metric0Weight'] = ['1.0']
    bspline_map['Metric1Weight'] = [str(bending_weight)]

    # Multi-resolution pyramid
    bspline_map['NumberOfResolutions'] = [str(num_resolutions)]
    schedule = []
    for i in range(num_resolutions):
        factor = 2 ** (num_resolutions - 1 - i)
        schedule.extend([str(factor)] * 3)  # Y, X, Z
    bspline_map['ImagePyramidSchedule'] = schedule

    # Optimizer settings
    bspline_map['Optimizer'] = ['AdaptiveStochasticGradientDescent']
    bspline_map['AutomaticTransformInitialization'] = ['false']
    bspline_map['AutomaticScalesEstimation'] = ['true']

    # Output settings
    bspline_map['WriteResultImage'] = ['false']
    bspline_map['ResultImageFormat'] = ['npy']

    parameter_object.AddParameterMap(bspline_map)

    # Run registration
    if verbose:
        print("\nRunning Elastix registration...")
        print("  (This may take 30-60 seconds depending on volume size)")

    try:
        result_image, result_transform_parameters = itk.elastix_registration_method(
            fixed_image, moving_image,
            parameter_object=parameter_object,
            log_to_console=False  # Suppress verbose ITK output
        )
    except Exception as e:
        raise RuntimeError(f"Elastix registration failed: {e}")

    if result_image is None:
        raise RuntimeError("Elastix returned None for result_image")

    # Apply transformation using transformix (more reliable on Windows than direct conversion)
    if verbose:
        print("  Applying transformation with transformix...")

    try:
        # Use transformix to apply the transformation - this is more reliable
        result_image_transformix = itk.transformix_filter(
            moving_image,
            result_transform_parameters
        )
    except Exception as e:
        raise RuntimeError(f"Transformix failed to apply transformation: {e}")

    # Convert result back to NumPy using transformix output
    if verbose:
        print("  Converting result to NumPy array...")

    try:
        # Try GetArrayFromImage first (most reliable for transformix output)
        registered_volume = itk.GetArrayFromImage(result_image_transformix)

        # Ensure it's a proper numpy array
        if not isinstance(registered_volume, np.ndarray):
            registered_volume = np.asarray(registered_volume)

        if verbose:
            print(f"    Successfully converted: {registered_volume.shape}, dtype={registered_volume.dtype}")

    except Exception as e:
        if verbose:
            print(f"    GetArrayFromImage failed: {e}")

        # Fallback: try array_from_image
        try:
            registered_volume = itk.array_from_image(result_image_transformix)
            if not isinstance(registered_volume, np.ndarray):
                registered_volume = np.asarray(registered_volume)
            if verbose:
                print(f"    array_from_image succeeded: {registered_volume.shape}")
        except Exception as e2:
            raise RuntimeError(f"Failed to convert transformix result to NumPy: {e2}")

    if registered_volume is None:
        raise RuntimeError("Converted registered volume is None")

    if not isinstance(registered_volume, np.ndarray):
        raise RuntimeError(f"Conversion resulted in wrong type: {type(registered_volume)}")

    if registered_volume.size == 0 or len(registered_volume.shape) == 0:
        raise RuntimeError(f"Converted registered volume is empty or scalar: shape={registered_volume.shape}")

    # Validate shape
    if verbose:
        print(f"  Validating shapes...")
        print(f"    Fixed:      {fixed_volume.shape}")
        print(f"    Moving:     {moving_volume.shape}")
        print(f"    Registered: {registered_volume.shape}")

    if registered_volume.shape != fixed_volume.shape:
        # Try to crop/pad to match
        if verbose:
            print(f"    Warning: Shape mismatch, attempting to match shapes...")

        # Get minimum shape along each axis
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(registered_volume.shape, fixed_volume.shape))

        # Crop both to minimum shape
        registered_volume = registered_volume[:min_shape[0], :min_shape[1], :min_shape[2]]
        fixed_volume_cropped = fixed_volume[:min_shape[0], :min_shape[1], :min_shape[2]]
        moving_volume_cropped = moving_volume[:min_shape[0], :min_shape[1], :min_shape[2]]

        if verbose:
            print(f"    Cropped to: {registered_volume.shape}")
    else:
        fixed_volume_cropped = fixed_volume
        moving_volume_cropped = moving_volume

    # Extract deformation field
    if verbose:
        print("  Computing deformation field...")

    transform_parameter_object = itk.ParameterObject.New()
    transform_parameter_object.AddParameterMap(result_transform_parameters.GetParameterMap(0))

    try:
        deformation_field_image = itk.transformix_deformation_field(
            fixed_image, transform_parameter_object
        )
    except Exception as e:
        raise RuntimeError(f"Failed to compute deformation field: {e}")

    if deformation_field_image is None:
        raise RuntimeError("Transformix returned None for deformation field")

    # Convert deformation field to NumPy
    try:
        # Method 1: GetArrayViewFromImage
        deformation_field = itk.GetArrayViewFromImage(deformation_field_image)
        deformation_field = np.array(deformation_field, copy=True)
        if verbose:
            print(f"    Deformation field converted: {deformation_field.shape}")
    except Exception as e:
        if verbose:
            print(f"    Deformation Method 1 failed: {e}")
        try:
            # Method 2: GetArrayFromImage
            deformation_field = itk.GetArrayFromImage(deformation_field_image)
            if verbose:
                print(f"    Deformation Method 2 succeeded: {deformation_field.shape}")
        except Exception as e2:
            if verbose:
                print(f"    Deformation Method 2 failed: {e2}")
            try:
                # Method 3: array_view_from_image
                deformation_field = np.asarray(itk.array_view_from_image(deformation_field_image))
                if verbose:
                    print(f"    Deformation Method 3 succeeded: {deformation_field.shape}")
            except Exception as e3:
                raise RuntimeError(f"Failed to convert deformation field: {e3}")

    if deformation_field is None:
        raise RuntimeError("Converted deformation field is None")

    if not isinstance(deformation_field, np.ndarray):
        raise RuntimeError(f"Deformation field conversion resulted in wrong type: {type(deformation_field)}")

    if deformation_field.size == 0 or len(deformation_field.shape) == 0:
        raise RuntimeError(f"Converted deformation field is empty or scalar: shape={deformation_field.shape}")

    # Calculate metrics using cropped volumes
    metrics = calculate_registration_metrics(
        fixed_volume_cropped, moving_volume_cropped, registered_volume, deformation_field
    )

    if verbose:
        print("\n" + "="*70)
        print("REGISTRATION COMPLETE")
        print("="*70)
        print(f"NCC before:  {metrics['ncc_before']:.4f}")
        print(f"NCC after:   {metrics['ncc_after']:.4f}")
        print(f"Improvement: {metrics['ncc_improvement_percent']:+.2f}%")
        print(f"Mean deformation: {metrics['mean_displacement']:.2f} voxels")
        print(f"Max deformation:  {metrics['max_displacement']:.2f} voxels")

    return registered_volume, deformation_field, metrics


def visualize_deformation_field(deformation_field, output_path, slice_z=None):
    """
    Visualize B-spline deformation field.

    Creates a 4-panel plot showing:
    1. Y-displacement (depth direction)
    2. X-displacement (A-scan direction)
    3. Displacement magnitude
    4. Quiver plot of deformation vectors

    Parameters:
    -----------
    deformation_field : np.ndarray
        Shape (Y, X, Z, 3) - displacement vectors
    output_path : Path
        Where to save visualization
    slice_z : int or None
        Z-slice to visualize (default: middle slice)
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    Y, X, Z, _ = deformation_field.shape

    if slice_z is None:
        slice_z = Z // 2

    # Extract Y and X displacements at slice
    dy = deformation_field[:, :, slice_z, 0]
    dx = deformation_field[:, :, slice_z, 1]

    # Calculate magnitude
    magnitude = np.sqrt(dy**2 + dx**2)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Y-displacement
    im0 = axes[0, 0].imshow(dy, cmap='RdBu_r', vmin=-20, vmax=20)
    axes[0, 0].set_title(f'Y-displacement (B-scan {slice_z})', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('X (A-scans)', fontsize=11)
    axes[0, 0].set_ylabel('Y (depth)', fontsize=11)
    plt.colorbar(im0, ax=axes[0, 0], label='Pixels')

    # X-displacement
    im1 = axes[0, 1].imshow(dx, cmap='RdBu_r', vmin=-20, vmax=20)
    axes[0, 1].set_title(f'X-displacement (B-scan {slice_z})', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('X (A-scans)', fontsize=11)
    axes[0, 1].set_ylabel('Y (depth)', fontsize=11)
    plt.colorbar(im1, ax=axes[0, 1], label='Pixels')

    # Magnitude
    im2 = axes[1, 0].imshow(magnitude, cmap='hot', vmin=0, vmax=30)
    axes[1, 0].set_title(f'Displacement magnitude (B-scan {slice_z})', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('X (A-scans)', fontsize=11)
    axes[1, 0].set_ylabel('Y (depth)', fontsize=11)
    plt.colorbar(im2, ax=axes[1, 0], label='Pixels')

    # Quiver plot (subsampled for clarity)
    step = 32
    Y_grid, X_grid = np.meshgrid(
        np.arange(0, Y, step),
        np.arange(0, X, step),
        indexing='ij'
    )
    dy_sub = dy[::step, ::step]
    dx_sub = dx[::step, ::step]

    axes[1, 1].quiver(
        X_grid, Y_grid, dx_sub, dy_sub, magnitude[::step, ::step],
        cmap='hot', scale=200, width=0.003
    )
    axes[1, 1].set_title(f'Deformation vectors (B-scan {slice_z})', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('X (A-scans)', fontsize=11)
    axes[1, 1].set_ylabel('Y (depth)', fontsize=11)
    axes[1, 1].invert_yaxis()

    # Add statistics text box
    stats_text = (
        f"Statistics (entire volume):\n"
        f"Mean displacement: {np.mean(np.linalg.norm(deformation_field, axis=-1)):.2f}px\n"
        f"Max displacement:  {np.max(np.linalg.norm(deformation_field, axis=-1)):.2f}px\n"
        f"Std displacement:  {np.std(np.linalg.norm(deformation_field, axis=-1)):.2f}px"
    )
    fig.text(0.99, 0.01, stats_text, ha='right', va='bottom',
             fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path.name}")


# ============================================================================
# CONTOUR-BASED ROTATION ALIGNMENT (NEW)
# ============================================================================

def calculate_contour_alignment_score(surface_v0, surface_v1, mask_columns):
    """
    Calculate alignment quality based on surface contour similarity.

    Uses variance of surface differences as the primary metric - lower variance
    means surfaces are more parallel (better aligned).

    Args:
        surface_v0: Detected surface for reference B-scan (X,)
        surface_v1: Detected surface for moving B-scan (X,)
        mask_columns: Boolean mask indicating valid columns (X,)

    Returns:
        score: Alignment score (higher is better, so we negate variance)
        metrics: Dictionary with detailed metrics
    """
    # Extract valid regions only
    if not mask_columns.any():
        return -np.inf, {'variance': np.inf, 'mad': np.inf, 'valid_pixels': 0}

    surf_v0_valid = surface_v0[mask_columns]
    surf_v1_valid = surface_v1[mask_columns]

    # Calculate surface difference
    diff = surf_v0_valid - surf_v1_valid

    # Primary metric: Variance (lower = more parallel surfaces)
    variance = np.var(diff)

    # Secondary metric: Median Absolute Deviation (robust to outliers)
    median_diff = np.median(diff)
    mad = np.median(np.abs(diff - median_diff))

    # Score: Negative variance (so higher is better)
    score = -variance

    metrics = {
        'variance': float(variance),
        'mad': float(mad),
        'median_diff': float(median_diff),
        'valid_pixels': int(mask_columns.sum()),
        'rms': float(np.sqrt(np.mean(diff**2)))
    }

    return score, metrics


def create_rotation_mask(bscan_v0, bscan_v1_rotated, threshold_percentile=10, min_valid_pixels_per_column=5):
    """
    Create combined mask for valid regions after rotation.

    Excludes:
    - Rotation-induced zero padding
    - Background noise regions
    - Columns with insufficient valid pixels

    Args:
        bscan_v0: Reference B-scan (Y, X)
        bscan_v1_rotated: Rotated moving B-scan (Y, X)
        threshold_percentile: Percentile for tissue threshold (default: 10)
        min_valid_pixels_per_column: Minimum valid pixels required per column

    Returns:
        mask_2d: 2D mask (Y, X) indicating valid regions
        mask_columns: 1D mask (X,) indicating valid columns
    """
    Y, X = bscan_v0.shape

    # Threshold for valid tissue (10th percentile of non-zero pixels)
    if (bscan_v0 > 0).any():
        threshold_v0 = np.percentile(bscan_v0[bscan_v0 > 0], threshold_percentile)
    else:
        threshold_v0 = 0

    if (bscan_v1_rotated > 0).any():
        threshold_v1 = np.percentile(bscan_v1_rotated[bscan_v1_rotated > 0], threshold_percentile)
    else:
        threshold_v1 = 0

    # Create masks for valid tissue regions
    mask_v0 = bscan_v0 > threshold_v0
    mask_v1 = bscan_v1_rotated > threshold_v1

    # Combined mask: both must be valid
    mask_2d = mask_v0 & mask_v1

    # Per-column validity: column must have enough valid pixels
    valid_pixels_per_column = mask_2d.sum(axis=0)
    mask_columns = valid_pixels_per_column >= min_valid_pixels_per_column

    return mask_2d, mask_columns


def detect_surface_in_masked_region(bscan_denoised, mask_columns):
    """
    Detect retinal surface only in valid columns.

    For invalid columns, surface is set to NaN.

    Args:
        bscan_denoised: Preprocessed B-scan (Y, X), uint8
        mask_columns: Boolean mask (X,) indicating valid columns

    Returns:
        surface: Surface array (X,) with NaN for invalid columns
    """
    Y, X = bscan_denoised.shape
    surface = np.full(X, np.nan)

    # Threshold for surface detection
    threshold = np.percentile(bscan_denoised, 70)
    _, binary = cv2.threshold(bscan_denoised, threshold, 255, cv2.THRESH_BINARY)

    # Detect surface only in valid columns
    for x in range(X):
        if not mask_columns[x]:
            continue  # Skip invalid columns

        column = binary[:, x]
        white_pixels = np.where(column > 0)[0]
        if len(white_pixels) > 0:
            surface[x] = white_pixels[0]  # First white pixel from top
        else:
            # No surface detected - try to interpolate from neighbors
            # For now, set to NaN (will be handled by masking)
            surface[x] = np.nan

    return surface


def _test_single_rotation_angle_contour(angle, bscan_v0, bscan_v1):
    """
    Worker function to test a single rotation angle using contour method.

    Args:
        angle: Rotation angle to test (degrees)
        bscan_v0: Reference B-scan (Y, X)
        bscan_v1: Moving B-scan (Y, X)

    Returns:
        Dictionary with angle, score, surfaces, and metrics
    """
    try:
        # Rotate moving B-scan
        bscan_v1_rotated = ndimage.rotate(
            bscan_v1, angle, axes=(0, 1),
            reshape=False, order=1,
            mode='constant', cval=0
        )

        # Create mask for valid regions
        mask_2d, mask_columns = create_rotation_mask(bscan_v0, bscan_v1_rotated)

        # Exclude edge columns affected by rotation zero-padding
        H, W = bscan_v1_rotated.shape
        edge_margin = calculate_rotation_edge_margin(H, W, angle)
        if edge_margin > 0 and edge_margin < W // 2:
            mask_columns[:edge_margin] = False
            mask_columns[-edge_margin:] = False
            mask_2d[:, :edge_margin] = False
            mask_2d[:, -edge_margin:] = False

        # Check if enough valid pixels
        if mask_columns.sum() < 10:  # Need at least 10 valid columns
            return {
                'angle': float(angle),
                'score': -np.inf,
                'variance': np.inf,
                'valid_pixels': 0,
                'surface_v0': None,
                'surface_v1': None,
                'mask_columns': mask_columns
            }

        # Preprocess both B-scans for surface detection
        bscan_v0_denoised = preprocess_oct_for_visualization(bscan_v0)
        bscan_v1_denoised = preprocess_oct_for_visualization(bscan_v1_rotated)

        # Detect surfaces in masked regions
        surface_v0 = detect_surface_in_masked_region(bscan_v0_denoised, mask_columns)
        surface_v1 = detect_surface_in_masked_region(bscan_v1_denoised, mask_columns)

        # Calculate alignment score
        score, metrics = calculate_contour_alignment_score(surface_v0, surface_v1, mask_columns)

        # Also calculate NCC for comparison
        ncc_score = calculate_ncc_3d(bscan_v0[:, :, np.newaxis], bscan_v1_rotated[:, :, np.newaxis])

        return {
            'angle': float(angle),
            'score': float(score),
            'variance': metrics['variance'],
            'mad': metrics['mad'],
            'valid_pixels': metrics['valid_pixels'],
            'ncc_score': float(ncc_score),  # ADDED: NCC for comparison
            'surface_v0': surface_v0,
            'surface_v1': surface_v1,
            'mask_columns': mask_columns,
            'bscan_v0_denoised': bscan_v0_denoised,
            'bscan_v1_denoised': bscan_v1_denoised
        }

    except Exception as e:
        return {
            'angle': float(angle),
            'score': -np.inf,
            'variance': np.inf,
            'valid_pixels': 0,
            'error': str(e)
        }


def find_optimal_rotation_z_contour_coarse(overlap_v0, overlap_v1, angle_range=20, step=1, verbose=True):
    """
    Coarse rotation search using contour-based alignment.

    Tests angles from -angle_range to +angle_range with given step size.
    Uses parallel processing for speed.

    Args:
        overlap_v0: Reference overlap volume (Y, X, Z)
        overlap_v1: Moving overlap volume (Y, X, Z)
        angle_range: Search range in degrees (±range)
        step: Step size in degrees
        verbose: Print progress

    Returns:
        best_angle: Optimal rotation angle
        results: List of result dictionaries for all tested angles
    """
    # Use central B-scan for rotation search (same as NCC method)
    Z = overlap_v0.shape[2]
    z_mid = Z // 2
    bscan_v0 = overlap_v0[:, :, z_mid]
    bscan_v1 = overlap_v1[:, :, z_mid]

    if verbose:
        print(f"\n  Coarse search: Testing angles from {-angle_range}° to +{angle_range}° (step: {step}°)")
        print(f"  Using central B-scan (z={z_mid}/{Z})")

    # Generate angles to test
    angles = np.arange(-angle_range, angle_range + step, step)

    if verbose:
        print(f"  Testing {len(angles)} angles...")

    # Sequential processing (Windows-compatible - parallel had pickling issues)
    results = []
    for angle in angles:
        result = _test_single_rotation_angle_contour(angle, bscan_v0, bscan_v1)
        results.append(result)

    # Find best angle
    valid_results = [r for r in results if r['score'] > -np.inf]
    if not valid_results:
        if verbose:
            print("  ⚠️  No valid rotation angles found!")
        return 0.0, results

    best_result = max(valid_results, key=lambda x: x['score'])
    best_angle = best_result['angle']

    if verbose:
        print(f"  ✓ Coarse best: {best_angle:+.1f}° (score: {best_result['score']:.2f}, variance: {best_result['variance']:.2f})")

    return best_angle, results


def find_optimal_rotation_z_contour_fine(overlap_v0, overlap_v1, coarse_angle, angle_range=3, step=0.5, verbose=True):
    """
    Fine rotation search around coarse optimum using contour-based alignment.

    Args:
        overlap_v0: Reference overlap volume (Y, X, Z)
        overlap_v1: Moving overlap volume (Y, X, Z)
        coarse_angle: Center angle from coarse search
        angle_range: Search range around coarse angle (±range)
        step: Step size in degrees
        verbose: Print progress

    Returns:
        best_angle: Optimal rotation angle
        results: List of result dictionaries for all tested angles
    """
    # Use central B-scan
    Z = overlap_v0.shape[2]
    z_mid = Z // 2
    bscan_v0 = overlap_v0[:, :, z_mid]
    bscan_v1 = overlap_v1[:, :, z_mid]

    if verbose:
        print(f"\n  Fine search: Testing angles around {coarse_angle:+.1f}° (±{angle_range}°, step: {step}°)")

    # Generate angles to test
    angles = np.arange(coarse_angle - angle_range, coarse_angle + angle_range + step, step)

    if verbose:
        print(f"  Testing {len(angles)} angles...")

    # Sequential processing (Windows-compatible - parallel had pickling issues)
    results = []
    for angle in angles:
        result = _test_single_rotation_angle_contour(angle, bscan_v0, bscan_v1)
        results.append(result)

    # Find best angle
    valid_results = [r for r in results if r['score'] > -np.inf]
    if not valid_results:
        if verbose:
            print(f"  ⚠️  No valid rotation angles found in fine search, using coarse: {coarse_angle:+.1f}°")
        return coarse_angle, results

    best_result = max(valid_results, key=lambda x: x['score'])
    best_angle = best_result['angle']

    if verbose:
        print(f"  ✓ Fine best: {best_angle:+.2f}° (score: {best_result['score']:.2f}, variance: {best_result['variance']:.2f})")

    return best_angle, results


def find_optimal_rotation_z_contour(overlap_v0, overlap_v1, coarse_range=20, coarse_step=1,
                                    fine_range=3, fine_step=0.5, verbose=True):
    """
    Find optimal Z-rotation angle using contour-based surface alignment.

    Two-stage coarse-to-fine search:
    1. Coarse: Test angles from -coarse_range to +coarse_range
    2. Fine: Refine around coarse optimum with finer steps

    Args:
        overlap_v0: Reference overlap volume (Y, X, Z)
        overlap_v1: Moving overlap volume (Y, X, Z)
        coarse_range: Coarse search range (±degrees)
        coarse_step: Coarse step size (degrees)
        fine_range: Fine search range around coarse optimum (±degrees)
        fine_step: Fine step size (degrees)
        verbose: Print progress information

    Returns:
        optimal_angle: Best rotation angle in degrees
        metrics: Dictionary with search results and metrics
    """
    if verbose:
        print("\n" + "="*70)
        print("CONTOUR-BASED Z-ROTATION SEARCH")
        print("="*70)
        print(f"  Method: Surface contour variance minimization")
        print(f"  Overlap volume shape: {overlap_v0.shape}")

    # Stage 1: Coarse search
    coarse_angle, coarse_results = find_optimal_rotation_z_contour_coarse(
        overlap_v0, overlap_v1,
        angle_range=coarse_range,
        step=coarse_step,
        verbose=verbose
    )

    # Stage 2: Fine search
    fine_angle, fine_results = find_optimal_rotation_z_contour_fine(
        overlap_v0, overlap_v1,
        coarse_angle=coarse_angle,
        angle_range=fine_range,
        step=fine_step,
        verbose=verbose
    )

    optimal_angle = fine_angle

    # Get best result details
    best_fine = max([r for r in fine_results if r['score'] > -np.inf],
                   key=lambda x: x['score'], default=None)

    if verbose:
        print("\n" + "="*70)
        print(f"  OPTIMAL ROTATION: {optimal_angle:+.2f}°")
        if best_fine:
            print(f"  Surface variance: {best_fine['variance']:.2f} px²")
            print(f"  Valid pixels: {best_fine['valid_pixels']}")
        print("="*70)

    metrics = {
        'optimal_angle': float(optimal_angle),
        'optimal_score': float(best_fine['score']) if best_fine else -np.inf,
        'optimal_variance': float(best_fine['variance']) if best_fine else np.inf,
        'coarse_results': coarse_results,
        'fine_results': fine_results,
        'method': 'contour_variance'
    }

    return optimal_angle, metrics


def visualize_rotation_angle_with_contours(result, bscan_v0, bscan_v1, all_results, output_path):
    """
    Visualize a single rotation angle with contour overlays.

    Creates a 2x3 grid showing:
    - Row 1: Original B-scans, rotated B-scan, overlay
    - Row 2: Denoised + contours, surface difference plot
    - Bottom: Angle info, score, score history graph

    Args:
        result: Result dictionary from _test_single_rotation_angle_contour()
        bscan_v0: Original reference B-scan
        bscan_v1: Original moving B-scan
        all_results: List of all tested angle results (for score history)
        output_path: Path to save visualization
    """
    angle = result['angle']
    score = result['score']
    variance = result['variance']
    valid_pixels = result['valid_pixels']
    surface_v0 = result.get('surface_v0')
    surface_v1 = result.get('surface_v1')
    mask_columns = result.get('mask_columns')
    bscan_v0_denoised = result.get('bscan_v0_denoised')
    bscan_v1_denoised = result.get('bscan_v1_denoised')

    # Rotate the original B-scan for visualization
    bscan_v1_rotated = ndimage.rotate(
        bscan_v1, angle, axes=(0, 1),
        reshape=False, order=1,
        mode='constant', cval=0
    )

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.4], hspace=0.3, wspace=0.3)

    # Row 1: Original B-scans
    # 1.1: Reference B-scan
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(bscan_v0, cmap='gray', aspect='auto')
    ax1.set_title(f'Reference B-scan (V0)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y (depth)', fontsize=10)
    ax1.set_xlabel('X (lateral)', fontsize=10)

    # 1.2: Rotated B-scan
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(bscan_v1_rotated, cmap='gray', aspect='auto')
    ax2.set_title(f'Moving B-scan Rotated ({angle:+.2f}°)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (lateral)', fontsize=10)

    # 1.3: Overlay (Red/Green)
    ax3 = fig.add_subplot(gs[0, 2])
    overlay = np.zeros((*bscan_v0.shape, 3), dtype=np.uint8)
    overlay[:, :, 0] = (bscan_v0 / bscan_v0.max() * 255).astype(np.uint8)  # Red: V0
    overlay[:, :, 1] = (bscan_v1_rotated / (bscan_v1_rotated.max() + 1e-8) * 255).astype(np.uint8)  # Green: V1
    ax3.imshow(overlay, aspect='auto')
    ax3.set_title('Overlay (Red=V0, Green=V1)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (lateral)', fontsize=10)

    # Row 2: Denoised + Contours
    if bscan_v0_denoised is not None and bscan_v1_denoised is not None:
        # 2.1: V0 denoised + contour
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(bscan_v0_denoised, cmap='gray', aspect='auto')
        if surface_v0 is not None and mask_columns is not None:
            # Plot contour only for valid columns
            x_coords = np.where(mask_columns)[0]
            y_coords = surface_v0[mask_columns]
            # Remove NaN values
            valid_mask = ~np.isnan(y_coords)
            ax4.plot(x_coords[valid_mask], y_coords[valid_mask], 'r-', linewidth=2, label='Surface V0')
            ax4.legend(loc='upper right', fontsize=9)
        ax4.set_title('V0 Denoised + Surface', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Y (depth)', fontsize=10)
        ax4.set_xlabel('X (lateral)', fontsize=10)

        # 2.2: V1 denoised + contour
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(bscan_v1_denoised, cmap='gray', aspect='auto')
        if surface_v1 is not None and mask_columns is not None:
            x_coords = np.where(mask_columns)[0]
            y_coords = surface_v1[mask_columns]
            valid_mask = ~np.isnan(y_coords)
            ax5.plot(x_coords[valid_mask], y_coords[valid_mask], 'g-', linewidth=2, label='Surface V1')
            ax5.legend(loc='upper right', fontsize=9)
        ax5.set_title(f'V1 Denoised + Surface ({angle:+.2f}°)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('X (lateral)', fontsize=10)

        # 2.3: Surface difference plot
        ax6 = fig.add_subplot(gs[1, 2])
        if surface_v0 is not None and surface_v1 is not None and mask_columns is not None:
            x_coords = np.arange(len(surface_v0))
            diff = surface_v0 - surface_v1

            # Plot surface difference
            ax6.plot(x_coords[mask_columns], diff[mask_columns], 'b-', linewidth=1.5, label='Surface Difference')
            ax6.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax6.fill_between(x_coords[mask_columns], 0, diff[mask_columns], alpha=0.3)
            ax6.set_xlabel('X (lateral)', fontsize=10)
            ax6.set_ylabel('Difference (px)', fontsize=10)
            ax6.set_title(f'Surface Difference (Variance: {variance:.2f} px²)', fontsize=12, fontweight='bold')
            ax6.legend(loc='upper right', fontsize=9)
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No surface data', ha='center', va='center', fontsize=14)
            ax6.set_title('Surface Difference', fontsize=12, fontweight='bold')

    # Row 3: Score history and info
    ax7 = fig.add_subplot(gs[2, :])

    # Plot score vs angle
    if all_results:
        angles = [r['angle'] for r in all_results if r['score'] > -np.inf]
        scores = [r['score'] for r in all_results if r['score'] > -np.inf]

        if angles:
            ax7.plot(angles, scores, 'b-', linewidth=2, label='Alignment Score')
            ax7.scatter(angles, scores, c='blue', s=30, alpha=0.6, zorder=3)
            # Highlight current angle
            if score > -np.inf:
                ax7.scatter([angle], [score], c='red', s=200, marker='*',
                           edgecolors='black', linewidths=2, zorder=4, label='Current Angle')
            ax7.set_xlabel('Rotation Angle (degrees)', fontsize=11, fontweight='bold')
            ax7.set_ylabel('Score (higher = better)', fontsize=11, fontweight='bold')
            ax7.set_title('Rotation Search Results', fontsize=12, fontweight='bold')
            ax7.grid(True, alpha=0.3)
            ax7.legend(loc='best', fontsize=10)
        else:
            ax7.text(0.5, 0.5, 'No valid results', ha='center', va='center', fontsize=14)

    # Add info text
    ncc_score = result.get('ncc_score', 'N/A')
    ncc_text = f"{ncc_score:.4f}" if isinstance(ncc_score, (int, float)) else ncc_score

    info_text = f"""
Angle: {angle:+.2f}°
Contour Score: {score:.2f}
Variance: {variance:.2f} px²
NCC Score: {ncc_text}
Valid Pixels: {valid_pixels}
    """.strip()

    fig.text(0.02, 0.02, info_text, fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def create_rotation_search_summary(coarse_results, fine_results, optimal_angle, output_path):
    """
    Create summary visualization showing score vs. angle for all tested angles.

    Args:
        coarse_results: List of coarse search results
        fine_results: List of fine search results
        optimal_angle: Optimal rotation angle found
        output_path: Path to save visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Z-Rotation Search Summary (Contour + NCC Comparison)', fontsize=16, fontweight='bold')

    # Extract data
    coarse_angles = [r['angle'] for r in coarse_results if r['score'] > -np.inf]
    coarse_scores = [r['score'] for r in coarse_results if r['score'] > -np.inf]
    coarse_variances = [r['variance'] for r in coarse_results if r['variance'] < np.inf]
    coarse_ncc = [r.get('ncc_score', -np.inf) for r in coarse_results if r['score'] > -np.inf]

    fine_angles = [r['angle'] for r in fine_results if r['score'] > -np.inf]
    fine_scores = [r['score'] for r in fine_results if r['score'] > -np.inf]
    fine_variances = [r['variance'] for r in fine_results if r['variance'] < np.inf]
    fine_ncc = [r.get('ncc_score', -np.inf) for r in fine_results if r['score'] > -np.inf]

    # 1. Coarse search: Variance vs Angle (Lower = Better)
    ax1 = axes[0, 0]
    if coarse_angles and coarse_variances:
        ax1.plot(coarse_angles, coarse_variances, 'b-', linewidth=2, marker='o', markersize=6)
        best_idx = np.argmin(coarse_variances)
        ax1.scatter([coarse_angles[best_idx]], [coarse_variances[best_idx]],
                   c='red', s=200, marker='*', edgecolors='black', linewidths=2, zorder=5, label='Best')
        ax1.set_xlabel('Rotation Angle (degrees)', fontsize=12)
        ax1.set_ylabel('Surface Variance (px²)', fontsize=12)
        ax1.set_title('Coarse: Variance vs Angle (LOWER=BETTER)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.axvline(0, color='k', linestyle=':', alpha=0.5)

    # 2. Coarse search: NCC vs Angle (Higher = Better)
    ax2 = axes[0, 1]
    if coarse_angles and coarse_ncc:
        valid_ncc = [ncc for ncc in coarse_ncc if ncc > -np.inf]
        if valid_ncc:
            ax2.plot(coarse_angles[:len(valid_ncc)], valid_ncc, 'g-', linewidth=2, marker='o', markersize=6)
            best_idx = np.argmax(valid_ncc)
            ax2.scatter([coarse_angles[best_idx]], [valid_ncc[best_idx]],
                       c='red', s=200, marker='*', edgecolors='black', linewidths=2, zorder=5, label='Best')
            ax2.set_xlabel('Rotation Angle (degrees)', fontsize=12)
            ax2.set_ylabel('NCC Score', fontsize=12)
            ax2.set_title('Coarse: NCC vs Angle (HIGHER=BETTER)', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            ax2.axvline(0, color='k', linestyle=':', alpha=0.5)

    # 3. Coarse search: Contour Score vs Angle (Higher = Better, = -variance)
    ax3 = axes[0, 2]
    if coarse_angles and coarse_scores:
        ax3.plot(coarse_angles, coarse_scores, 'm-', linewidth=2, marker='o', markersize=6)
        best_idx = np.argmax(coarse_scores)
        ax3.scatter([coarse_angles[best_idx]], [coarse_scores[best_idx]],
                   c='red', s=200, marker='*', edgecolors='black', linewidths=2, zorder=5, label='Best')
        ax3.set_xlabel('Rotation Angle (degrees)', fontsize=12)
        ax3.set_ylabel('Contour Score', fontsize=12)
        ax3.set_title('Coarse: Contour Score vs Angle (HIGHER=BETTER)', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        ax3.axvline(0, color='k', linestyle=':', alpha=0.5)

    # 4. Fine search: Variance vs Angle
    ax4 = axes[1, 0]
    if fine_angles and fine_variances:
        ax4.plot(fine_angles, fine_variances, 'b-', linewidth=2, marker='o', markersize=6)
        best_idx = np.argmin(fine_variances)
        ax4.scatter([fine_angles[best_idx]], [fine_variances[best_idx]],
                   c='red', s=200, marker='*', edgecolors='black', linewidths=2, zorder=5, label='Optimal')
        ax4.set_xlabel('Rotation Angle (degrees)', fontsize=12)
        ax4.set_ylabel('Surface Variance (px²)', fontsize=12)
        ax4.set_title(f'Fine: Variance vs Angle (Optimal: {optimal_angle:+.2f}°)', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        ax4.axvline(0, color='k', linestyle=':', alpha=0.5)

    # 5. Fine search: NCC vs Angle
    ax5 = axes[1, 1]
    if fine_angles and fine_ncc:
        valid_ncc = [ncc for ncc in fine_ncc if ncc > -np.inf]
        if valid_ncc:
            ax5.plot(fine_angles[:len(valid_ncc)], valid_ncc, 'g-', linewidth=2, marker='o', markersize=6)
            best_idx = np.argmax(valid_ncc)
            ax5.scatter([fine_angles[best_idx]], [valid_ncc[best_idx]],
                       c='red', s=200, marker='*', edgecolors='black', linewidths=2, zorder=5, label='Best NCC')
            ax5.set_xlabel('Rotation Angle (degrees)', fontsize=12)
            ax5.set_ylabel('NCC Score', fontsize=12)
            ax5.set_title(f'Fine: NCC vs Angle (Optimal: {optimal_angle:+.2f}°)', fontsize=13, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            ax5.legend(fontsize=10)
            ax5.axvline(0, color='k', linestyle=':', alpha=0.5)

    # 6. Fine search: Contour Score vs Angle
    ax6 = axes[1, 2]
    if fine_angles and fine_scores:
        ax6.plot(fine_angles, fine_scores, 'm-', linewidth=2, marker='o', markersize=6)
        best_idx = np.argmax(fine_scores)
        ax6.scatter([fine_angles[best_idx]], [fine_scores[best_idx]],
                   c='red', s=200, marker='*', edgecolors='black', linewidths=2, zorder=5, label='Optimal')
        ax6.set_xlabel('Rotation Angle (degrees)', fontsize=12)
        ax6.set_ylabel('Contour Score', fontsize=12)
        ax6.set_title(f'Fine: Contour Score vs Angle (Optimal: {optimal_angle:+.2f}°)', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=10)
        ax6.axvline(0, color='k', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved rotation search summary: {output_path.name}")


if __name__ == "__main__":
    print("Rotation Alignment Module")
    print("=" * 70)
    print("Functions:")
    print("  - find_optimal_rotation_z(): Coarse-to-fine rotation search (NCC)")
    print("  - find_optimal_rotation_z_contour(): Contour-based rotation search (NEW)")
    print("  - apply_rotation_z(): Apply rotation to volume")
    print("  - calculate_ncc_3d(): Calculate NCC metric (verification)")
    print("  - visualize_rotation_search(): Plot angle search results")
    print("  - visualize_rotation_comparison(): Plot before/after comparison")
