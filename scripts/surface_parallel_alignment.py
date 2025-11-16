#!/usr/bin/env python3
"""
Surface Parallel Alignment

Aligns two OCT B-scans by making their detected surface contours parallel.

This method:
1. Denoises both B-scans
2. Detects surface contours
3. Fits lines to both surfaces
4. Calculates angle difference
5. Rotates B-scan 2 to align surfaces parallel to B-scan 1
6. Visualizes the results

Usage from pipeline:
    python alignment_pipeline.py --just-surface
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path


def denoise_bscan_for_surface(bscan):
    """
    Apply harsh denoising to B-scan for robust surface detection.

    Uses the same aggressive preprocessing as rotation alignment.

    Args:
        bscan: 2D B-scan array (Y, X)

    Returns:
        Denoised B-scan (Y, X) as uint8
    """
    # Normalize to 0-255
    img_norm = ((bscan - bscan.min()) / (bscan.max() - bscan.min() + 1e-8) * 255).astype(np.uint8)

    # Step 1: Non-local means denoising (HARSH)
    denoised = cv2.fastNlMeansDenoising(img_norm, h=25, templateWindowSize=7, searchWindowSize=21)

    # Step 2: Bilateral filtering (HARSH)
    denoised = cv2.bilateralFilter(denoised, d=11, sigmaColor=150, sigmaSpace=150)

    # Step 3: Median filter (HARSH)
    denoised = cv2.medianBlur(denoised, 15)

    # Step 4: Threshold (50% of Otsu - preserves tissue layers)
    thresh_val = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_val = int(thresh_val * 0.5)
    denoised[denoised < thresh_val] = 0

    # Step 5: CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    denoised = clahe.apply(denoised)

    return denoised


def detect_surface_contour(bscan_denoised):
    """
    Detect retinal surface using contour method.

    Finds where tissue STARTS by finding the first white pixel in each column.

    Args:
        bscan_denoised: Preprocessed B-scan (2D array, uint8)

    Returns:
        surface: 1D array (X,) with Y positions of detected surface
    """
    Y, X = bscan_denoised.shape
    surface = np.zeros(X)

    # Simple threshold (70th percentile works well on denoised images)
    threshold = np.percentile(bscan_denoised, 70)
    _, binary = cv2.threshold(bscan_denoised, threshold, 255, cv2.THRESH_BINARY)

    # For each column, find first white pixel (top boundary = tissue start)
    for x in range(X):
        column = binary[:, x]
        white_pixels = np.where(column > 0)[0]
        if len(white_pixels) > 0:
            surface[x] = white_pixels[0]  # First from top
        else:
            # No tissue detected - use previous column or default
            surface[x] = surface[x-1] if x > 0 else Y // 4

    return surface


def fit_line_to_surface(surface):
    """
    Fit a linear regression line to surface points.

    Args:
        surface: 1D array (X,) with Y positions

    Returns:
        slope: Slope of fitted line (pixels per X)
        intercept: Y-intercept of fitted line
        angle_degrees: Angle of the line in degrees (relative to horizontal)
    """
    X = len(surface)
    x_coords = np.arange(X)

    # Filter out invalid points (zeros or outliers)
    valid = (surface > 0) & (surface < 1000)
    if valid.sum() < 10:
        return 0.0, 0.0, 0.0

    # Linear regression: y = slope * x + intercept
    slope, intercept = np.polyfit(x_coords[valid], surface[valid], deg=1)

    # Calculate angle in degrees
    angle_degrees = np.degrees(np.arctan(slope))

    return slope, intercept, angle_degrees


def rotate_bscan_to_parallel_surface(bscan, current_angle, target_angle):
    """
    Rotate B-scan to make its surface parallel to target angle.

    Args:
        bscan: 2D B-scan array (Y, X)
        current_angle: Current surface angle (degrees)
        target_angle: Target surface angle (degrees)

    Returns:
        rotated_bscan: Rotated B-scan (Y, X)
        rotation_angle: Angle applied (degrees)
    """
    rotation_angle = target_angle - current_angle

    # Rotate using scipy.ndimage (handles boundary correctly)
    rotated_bscan = ndimage.rotate(bscan, rotation_angle, reshape=False, order=1, mode='constant', cval=0)

    return rotated_bscan, rotation_angle


def align_surfaces_parallel(bscan_v0, bscan_v1, verbose=True):
    """
    Align two B-scans by making their surface contours parallel.

    Main function that:
    1. Denoises both B-scans
    2. Detects surface contours
    3. Fits lines to surfaces
    4. Rotates B-scan 2 to align parallel to B-scan 1

    Args:
        bscan_v0: Reference B-scan (Y, X)
        bscan_v1: B-scan to align (Y, X)
        verbose: Print progress

    Returns:
        Dictionary with:
        - bscan_v0_denoised: Denoised reference B-scan
        - bscan_v1_denoised: Denoised B-scan (before rotation)
        - bscan_v1_aligned: Aligned B-scan (after rotation)
        - surface_v0: Detected surface on V0
        - surface_v1_before: Detected surface on V1 before alignment
        - surface_v1_after: Detected surface on V1 after alignment
        - angle_v0: V0 surface angle (degrees)
        - angle_v1_before: V1 surface angle before alignment (degrees)
        - angle_v1_after: V1 surface angle after alignment (degrees)
        - rotation_applied: Rotation angle applied to V1 (degrees)
    """
    if verbose:
        print("\n" + "="*70)
        print("SURFACE PARALLEL ALIGNMENT")
        print("="*70)
        print("  Method: Align surfaces by making them parallel")
        print("="*70)

    # Step 1: Denoise both B-scans (exact same as visualize_retinal_surface.py)
    if verbose:
        print("\n1. Denoising B-scans...")

    # V0 preprocessing
    img_norm_v0 = ((bscan_v0 - bscan_v0.min()) / (bscan_v0.max() - bscan_v0.min() + 1e-8) * 255).astype(np.uint8)
    bscan_v0_denoised = cv2.fastNlMeansDenoising(img_norm_v0, h=25, templateWindowSize=7, searchWindowSize=21)
    bscan_v0_denoised = cv2.bilateralFilter(bscan_v0_denoised, d=11, sigmaColor=150, sigmaSpace=150)
    bscan_v0_denoised = cv2.medianBlur(bscan_v0_denoised, 15)
    thresh_val = cv2.threshold(bscan_v0_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_val = int(thresh_val * 0.5)
    bscan_v0_denoised[bscan_v0_denoised < thresh_val] = 0
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    bscan_v0_denoised = clahe.apply(bscan_v0_denoised)

    # V1 preprocessing
    img_norm_v1 = ((bscan_v1 - bscan_v1.min()) / (bscan_v1.max() - bscan_v1.min() + 1e-8) * 255).astype(np.uint8)
    bscan_v1_denoised = cv2.fastNlMeansDenoising(img_norm_v1, h=25, templateWindowSize=7, searchWindowSize=21)
    bscan_v1_denoised = cv2.bilateralFilter(bscan_v1_denoised, d=11, sigmaColor=150, sigmaSpace=150)
    bscan_v1_denoised = cv2.medianBlur(bscan_v1_denoised, 15)
    thresh_val = cv2.threshold(bscan_v1_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_val = int(thresh_val * 0.5)
    bscan_v1_denoised[bscan_v1_denoised < thresh_val] = 0
    bscan_v1_denoised = clahe.apply(bscan_v1_denoised)

    # Step 2: Detect surfaces
    if verbose:
        print("2. Detecting surface contours...")
    surface_v0 = detect_surface_contour(bscan_v0_denoised)
    surface_v1_before = detect_surface_contour(bscan_v1_denoised)

    # Step 3: Fit lines to surfaces
    if verbose:
        print("3. Fitting lines to surfaces...")
    slope_v0, intercept_v0, angle_v0 = fit_line_to_surface(surface_v0)
    slope_v1_before, intercept_v1_before, angle_v1_before = fit_line_to_surface(surface_v1_before)

    if verbose:
        print(f"   V0 surface angle: {angle_v0:+.3f}°")
        print(f"   V1 surface angle (before): {angle_v1_before:+.3f}°")
        print(f"   Angle difference: {angle_v1_before - angle_v0:+.3f}°")

    # Step 4: Rotate DENOISED B-scan V1 to align parallel to V0
    if verbose:
        print("4. Rotating V1 to align surfaces parallel...")
    bscan_v1_denoised_rotated, rotation_applied = rotate_bscan_to_parallel_surface(
        bscan_v1_denoised, angle_v1_before, angle_v0
    )

    # Also rotate the raw B-scan for final result
    bscan_v1_aligned, _ = rotate_bscan_to_parallel_surface(
        bscan_v1, angle_v1_before, angle_v0
    )

    # Step 5: Detect surface on rotated B-scan to verify
    surface_v1_after = detect_surface_contour(bscan_v1_denoised_rotated)
    slope_v1_after, intercept_v1_after, angle_v1_after = fit_line_to_surface(surface_v1_after)

    if verbose:
        print(f"   Rotation applied: {rotation_applied:+.3f}°")
        print(f"   V1 surface angle (after): {angle_v1_after:+.3f}°")
        print(f"   Residual angle difference: {angle_v1_after - angle_v0:+.3f}°")

    results = {
        'bscan_v0_denoised': bscan_v0_denoised,
        'bscan_v1_denoised': bscan_v1_denoised,
        'bscan_v1_aligned': bscan_v1_aligned,
        'bscan_v1_aligned_denoised': bscan_v1_denoised_rotated,  # Use rotated denoised version
        'surface_v0': surface_v0,
        'surface_v1_before': surface_v1_before,
        'surface_v1_after': surface_v1_after,
        'angle_v0': float(angle_v0),
        'angle_v1_before': float(angle_v1_before),
        'angle_v1_after': float(angle_v1_after),
        'rotation_applied': float(rotation_applied),
        'slope_v0': float(slope_v0),
        'intercept_v0': float(intercept_v0),
        'slope_v1_before': float(slope_v1_before),
        'intercept_v1_before': float(intercept_v1_before),
        'slope_v1_after': float(slope_v1_after),
        'intercept_v1_after': float(intercept_v1_after)
    }

    return results


def visualize_surface_parallel_alignment(results, output_path=None):
    """
    Visualize surface parallel alignment results.

    Shows:
    - Original B-scans
    - Denoised B-scans with detected surfaces
    - Before/after rotation comparison
    - Surface angle comparison

    Args:
        results: Dictionary from align_surfaces_parallel()
        output_path: Where to save visualization
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    Y, X = results['bscan_v0_denoised'].shape

    # Row 1: Denoised B-scans with detected surfaces
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(results['bscan_v0_denoised'], cmap='gray', aspect='auto', vmin=0, vmax=255)
    ax1.plot(results['surface_v0'], 'r-', linewidth=4, label='Detected Surface')
    # Draw fitted line
    x_coords = np.arange(X)
    fitted_line_v0 = results['slope_v0'] * x_coords + results['intercept_v0']
    ax1.plot(x_coords, fitted_line_v0, 'y--', linewidth=2, alpha=0.7, label=f'Fitted Line (angle: {results["angle_v0"]:+.2f}°)')
    ax1.set_title('V0 - Reference Surface', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (A-scans)')
    ax1.set_ylabel('Y (depth)')
    ax1.legend()
    ax1.set_ylim([0, Y // 2])

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(results['bscan_v1_denoised'], cmap='gray', aspect='auto', vmin=0, vmax=255)
    ax2.plot(results['surface_v1_before'], 'b-', linewidth=4, label='Detected Surface')
    # Draw fitted line
    fitted_line_v1_before = results['slope_v1_before'] * x_coords + results['intercept_v1_before']
    ax2.plot(x_coords, fitted_line_v1_before, 'y--', linewidth=2, alpha=0.7, label=f'Fitted Line (angle: {results["angle_v1_before"]:+.2f}°)')
    ax2.set_title('V1 - Before Alignment', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (A-scans)')
    ax2.set_ylabel('Y (depth)')
    ax2.legend()
    ax2.set_ylim([0, Y // 2])

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(results['bscan_v1_aligned_denoised'], cmap='gray', aspect='auto', vmin=0, vmax=255)
    ax3.plot(results['surface_v1_after'], 'g-', linewidth=4, label='Detected Surface')
    # Draw fitted line
    fitted_line_v1_after = results['slope_v1_after'] * x_coords + results['intercept_v1_after']
    ax3.plot(x_coords, fitted_line_v1_after, 'y--', linewidth=2, alpha=0.7, label=f'Fitted Line (angle: {results["angle_v1_after"]:+.2f}°)')
    ax3.set_title(f'V1 - After Alignment (rotated {results["rotation_applied"]:+.2f}°)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (A-scans)')
    ax3.set_ylabel('Y (depth)')
    ax3.legend()
    ax3.set_ylim([0, Y // 2])

    # Row 2: Surface overlays
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(results['bscan_v0_denoised'], cmap='gray', aspect='auto', alpha=0.7, vmin=0, vmax=255)
    ax4.plot(results['surface_v0'], 'r-', linewidth=4, label='V0 surface')
    ax4.plot(results['surface_v1_before'], 'b-', linewidth=4, label='V1 surface (before)', alpha=0.8)
    ax4.fill_between(x_coords, results['surface_v0'], results['surface_v1_before'],
                      alpha=0.3, color='yellow', label='Misalignment')
    ax4.set_title('BEFORE: Surface Overlay', fontsize=12, fontweight='bold')
    ax4.set_xlabel('X (A-scans)')
    ax4.set_ylabel('Y (depth)')
    ax4.legend()
    ax4.set_ylim([0, Y // 2])

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(results['bscan_v0_denoised'], cmap='gray', aspect='auto', alpha=0.7, vmin=0, vmax=255)
    ax5.plot(results['surface_v0'], 'r-', linewidth=4, label='V0 surface')
    ax5.plot(results['surface_v1_after'], 'g-', linewidth=4, label='V1 surface (after)', alpha=0.8)
    ax5.fill_between(x_coords, results['surface_v0'], results['surface_v1_after'],
                      alpha=0.3, color='green', label='Residual misalignment')
    ax5.set_title('AFTER: Surface Overlay', fontsize=12, fontweight='bold')
    ax5.set_xlabel('X (A-scans)')
    ax5.set_ylabel('Y (depth)')
    ax5.legend()
    ax5.set_ylim([0, Y // 2])

    # Surface difference comparison
    ax6 = fig.add_subplot(gs[1, 2])
    diff_before = results['surface_v0'] - results['surface_v1_before']
    diff_after = results['surface_v0'] - results['surface_v1_after']
    ax6.plot(x_coords, diff_before, 'b-', linewidth=2, alpha=0.7, label=f'Before (std: {np.std(diff_before):.2f}px)')
    ax6.plot(x_coords, diff_after, 'g-', linewidth=2, alpha=0.7, label=f'After (std: {np.std(diff_after):.2f}px)')
    ax6.axhline(0, color='black', linestyle='--', linewidth=1)
    ax6.set_title('Surface Difference (V0 - V1)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('X (A-scans)')
    ax6.set_ylabel('Difference (pixels)')
    ax6.legend()
    ax6.grid(alpha=0.3)

    # Row 3: Angle analysis
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')

    # Create summary text
    summary_text = f"""
    SURFACE PARALLEL ALIGNMENT RESULTS

    Reference Surface (V0):
    • Angle: {results['angle_v0']:+.3f}° (slope: {results['slope_v0']:.4f})

    Volume 1 Surface (BEFORE alignment):
    • Angle: {results['angle_v1_before']:+.3f}° (slope: {results['slope_v1_before']:.4f})
    • Angle difference: {results['angle_v1_before'] - results['angle_v0']:+.3f}°

    Volume 1 Surface (AFTER alignment):
    • Rotation applied: {results['rotation_applied']:+.3f}°
    • Final angle: {results['angle_v1_after']:+.3f}° (slope: {results['slope_v1_after']:.4f})
    • Residual angle difference: {results['angle_v1_after'] - results['angle_v0']:+.3f}°

    Alignment Quality:
    • Surface difference std BEFORE: {np.std(diff_before):.2f} pixels
    • Surface difference std AFTER:  {np.std(diff_after):.2f} pixels
    • Improvement: {(np.std(diff_before) - np.std(diff_after)) / np.std(diff_before) * 100:+.1f}%
    """

    ax7.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    fig.suptitle('Surface Parallel Alignment - Central B-scan Analysis',
                 fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[OK] Saved visualization: {output_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    print("Surface Parallel Alignment Module")
    print("Use from alignment pipeline with: python alignment_pipeline.py --just-surface")
