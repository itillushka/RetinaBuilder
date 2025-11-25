"""
Retina Curvature Analysis

Processes averaged B-scan to estimate retina curvature by fitting a circle.

Steps:
1. Apply harsh denoising and save result
2. Detect surface using contour method (existing)
3. Smooth the detected surface using gradient-based smoothing
4. Fit a circular function to estimate curvature
"""

import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import least_squares
from helpers.rotation_alignment import detect_contour_surface


def harsh_denoise(img):
    """Apply very harsh denoising for surface detection."""
    # Normalize to 0-255
    img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    # Very aggressive NLM denoising
    denoised = cv2.fastNlMeansDenoising(img_norm, h=40, templateWindowSize=7, searchWindowSize=21)

    # Strong bilateral filtering
    denoised = cv2.bilateralFilter(denoised, d=15, sigmaColor=200, sigmaSpace=200)

    # Large median filter
    denoised = cv2.medianBlur(denoised, 21)

    # Harsh threshold (70% of Otsu)
    thresh_val = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    denoised[denoised < int(thresh_val * 0.7)] = 0

    return denoised


def smooth_surface_gradient(surface):
    """Smooth detected surface using gradient-based approach.

    Applies smoothing to reduce noise/jumps in the detected surface line.
    """
    from scipy.ndimage import median_filter, gaussian_filter1d

    # Step 1: Median filter to remove outliers
    surface_smooth = median_filter(surface.astype(np.float64), size=51)

    # Step 2: Gaussian smoothing for gradual transitions
    surface_smooth = gaussian_filter1d(surface_smooth, sigma=30)

    # Step 3: Compute gradient and smooth high-gradient regions more
    gradient = np.abs(np.gradient(surface_smooth))
    high_grad_mask = gradient > np.percentile(gradient, 90)

    # Extra smoothing on high-gradient regions
    if np.any(high_grad_mask):
        surface_extra = gaussian_filter1d(surface_smooth, sigma=60)
        surface_smooth[high_grad_mask] = surface_extra[high_grad_mask]

    return surface_smooth


def fit_circle(x, y):
    """Fit a circle to points (x, y) using least squares.

    Circle equation: (x - cx)^2 + (y - cy)^2 = r^2

    Returns: cx, cy, r (center x, center y, radius)
    """
    def residuals(params, x, y):
        cx, cy, r = params
        return np.sqrt((x - cx)**2 + (y - cy)**2) - r

    # Initial guess: center at mean, radius from spread
    cx0 = np.mean(x)
    cy0 = np.mean(y)
    r0 = np.std(x)

    result = least_squares(residuals, [cx0, cy0, r0], args=(x, y))
    cx, cy, r = result.x

    return cx, cy, abs(r)


def analyze_retina_curvature(data_dir):
    """Main function to analyze retina curvature."""
    data_dir = Path(data_dir)

    # Load averaged B-scan
    avg_bscan = np.load(data_dir / 'averaged_middle_30_bscans.npy')
    print(f"Loaded averaged B-scan: {avg_bscan.shape}")

    # Step 1: Harsh denoising
    print("\n[1] Applying harsh denoising...")
    denoised = harsh_denoise(avg_bscan)
    cv2.imwrite(str(data_dir / 'averaged_bscan_denoised.png'), denoised)
    print(f"  Saved: averaged_bscan_denoised.png")

    # Step 2: Detect surface using contour method (existing)
    print("\n[2] Detecting surface with contour method...")
    surface_raw = detect_contour_surface(denoised)

    # Step 3: Smooth surface using gradient-based smoothing
    print("\n[3] Smoothing surface with gradient method...")
    surface = smooth_surface_gradient(surface_raw)

    # Step 4: Fit circle
    print("\n[4] Fitting circle to surface...")
    x_coords = np.arange(len(surface))

    # Filter out invalid points (where surface wasn't detected)
    valid = (surface > 0) & (surface < avg_bscan.shape[0] - 1)
    x_valid = x_coords[valid]
    y_valid = surface[valid]

    cx, cy, radius = fit_circle(x_valid, y_valid)

    # Calculate curvature (1/r)
    curvature = 1.0 / radius if radius > 0 else 0

    print(f"\n  Circle fit results:")
    print(f"    Center: ({cx:.1f}, {cy:.1f})")
    print(f"    Radius: {radius:.1f} pixels")
    print(f"    Curvature: {curvature:.6f} (1/pixels)")

    # Step 5: Visualize
    print("\n[5] Creating visualization...")

    # Create visualization image
    vis = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    # Draw detected surface (green)
    for x in range(len(surface)):
        if 0 < surface[x] < avg_bscan.shape[0]:
            cv2.circle(vis, (x, int(surface[x])), 1, (0, 255, 0), -1)

    # Draw fitted circle (red)
    cv2.circle(vis, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)

    # Draw center point
    cv2.circle(vis, (int(cx), int(cy)), 5, (255, 0, 0), -1)

    cv2.imwrite(str(data_dir / 'retina_curvature_analysis.png'), vis)
    print(f"  Saved: retina_curvature_analysis.png")

    # Save results
    results = {
        'center_x': cx,
        'center_y': cy,
        'radius_pixels': radius,
        'curvature': curvature,
        'surface': surface
    }
    np.save(data_dir / 'retina_curvature_results.npy', results)
    print(f"  Saved: retina_curvature_results.npy")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze retina curvature from averaged B-scan')
    parser.add_argument('--data_dir', default='results_5vol_em005', help='Directory with averaged B-scan')
    args = parser.parse_args()

    analyze_retina_curvature(args.data_dir)
