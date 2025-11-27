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


def detect_rpe_at_region(bscan, x_start, x_end):
    """
    Detect RPE Y-position in a specific X region.
    Finds all intensity peaks in the averaged column and selects the one
    that is both highest in intensity AND lowest in position (deepest).

    Args:
        bscan: 2D B-scan image
        x_start: Start X coordinate of region
        x_end: End X coordinate of region

    Returns: (x_center, y_rpe) - the point coordinates
    """
    from scipy.signal import find_peaks

    Y, X = bscan.shape
    region = bscan[:, x_start:x_end]

    # Average columns in region for robust detection
    avg_column = np.mean(region, axis=1)

    # Search in lower 60% (skip ILM at top)
    search_start = int(Y * 0.4)
    search_column = avg_column[search_start:]

    # Find all peaks in the column
    peaks, properties = find_peaks(search_column, height=0, prominence=5)

    if len(peaks) == 0:
        # Fallback: use max intensity position
        rpe_y = search_start + np.argmax(search_column)
    else:
        # Get peak intensities
        peak_heights = properties['peak_heights']

        # Find the brightest peak
        max_intensity = np.max(peak_heights)

        # Among peaks with intensity >= 80% of max, pick the lowest (deepest) one
        bright_mask = peak_heights >= 0.8 * max_intensity
        bright_peaks = peaks[bright_mask]

        # Select the lowest (highest Y value = deepest in image)
        rpe_y = search_start + np.max(bright_peaks)

    # X coordinate is center of region
    x_center = (x_start + x_end) // 2

    return x_center, rpe_y


def detect_nine_rpe_points(bscan):
    """
    Detect RPE at 9 regions evenly distributed across the image.

    Args:
        bscan: 2D B-scan image

    Returns: list of 9 points [(x, y), ...] from left to right
    """
    Y, X = bscan.shape

    # Point 1: Left edge (10px from edge, offset 15px higher)
    left_x, left_y = detect_rpe_at_region(bscan, 10, 60)
    p1 = (left_x, left_y - 15)

    # Point 2: 1/8 of image width
    p2_start = X // 8 - 10
    p2 = detect_rpe_at_region(bscan, p2_start, p2_start + 20)

    # Point 3: 1/4 of image width
    p3_start = X // 4 - 10
    p3 = detect_rpe_at_region(bscan, p3_start, p3_start + 20)

    # Point 4: 3/8 of image width
    p4_start = 3 * X // 8 - 10
    p4 = detect_rpe_at_region(bscan, p4_start, p4_start + 20)

    # Point 5: Center (offset 15px lower for proper curvature)
    center_start = X // 2 - 10
    center_x, center_y = detect_rpe_at_region(bscan, center_start, center_start + 20)
    p5 = (center_x, center_y + 15)

    # Point 6: 5/8 of image width
    p6_start = 5 * X // 8 - 10
    p6 = detect_rpe_at_region(bscan, p6_start, p6_start + 20)

    # Point 7: 3/4 of image width
    p7_start = 3 * X // 4 - 10
    p7 = detect_rpe_at_region(bscan, p7_start, p7_start + 20)

    # Point 8: 7/8 of image width
    p8_start = 7 * X // 8 - 10
    p8 = detect_rpe_at_region(bscan, p8_start, p8_start + 20)

    # Point 9: Right edge (10px from edge, offset 15px higher)
    right_x, right_y = detect_rpe_at_region(bscan, X - 60, X - 10)
    p9 = (right_x, right_y - 15)

    return [p1, p2, p3, p4, p5, p6, p7, p8, p9]


def circle_from_three_points(p1, p2, p3):
    """
    Calculate circle passing through 3 points.
    Uses circumcircle formula - exact solution, no least squares needed.

    Args:
        p1, p2, p3: Points as (x, y) tuples

    Returns: (cx, cy, radius) or None if points are collinear
    """
    ax, ay = p1
    bx, by = p2
    cx, cy = p3

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-10:
        return None  # Points are collinear

    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d

    radius = np.sqrt((ax - ux)**2 + (ay - uy)**2)
    return ux, uy, radius


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

    # Harsh threshold (99.9% of Otsu)
    thresh_val = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    denoised[denoised < int(thresh_val * 0.999)] = 0

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


def analyze_retina_curvature(data_dir, input_file=None, pixel_size_mm=0.003906):
    """Main function to analyze retina curvature using 5-point RPE detection.

    Detects RPE at 5 locations and fits circle using least squares.

    Args:
        data_dir: Directory for output files
        input_file: Optional path to input .npy file. If None, uses
                    data_dir/averaged_middle_30_bscans.npy
        pixel_size_mm: Size of one pixel in mm (default: 6mm/1536 = 0.003906)
    """
    data_dir = Path(data_dir)

    # Load averaged B-scan
    if input_file is not None:
        input_path = Path(input_file)
        avg_bscan = np.load(input_path)
    else:
        avg_bscan = np.load(data_dir / 'averaged_middle_30_bscans.npy')
    print(f"Loaded averaged B-scan: {avg_bscan.shape}")

    # Step 1: Harsh denoising
    print("\n[1] Applying harsh denoising...")
    denoised = harsh_denoise(avg_bscan)
    cv2.imwrite(str(data_dir / 'averaged_bscan_denoised.png'), denoised)
    print(f"  Saved: averaged_bscan_denoised.png")

    # Step 2: Detect RPE at 9 points
    print("\n[2] Detecting RPE at 9 regions...")
    rpe_points = detect_nine_rpe_points(denoised)

    labels = ['1-Left', '2-1/8', '3-1/4', '4-3/8', '5-Center', '6-5/8', '7-3/4', '8-7/8', '9-Right']
    for label, pt in zip(labels, rpe_points):
        print(f"  {label}: ({pt[0]}, {pt[1]})")

    # Step 3: Fit circle through 9 points using least squares
    print("\n[3] Fitting circle through 9 points...")
    x_pts = np.array([pt[0] for pt in rpe_points])
    y_pts = np.array([pt[1] for pt in rpe_points])

    cx, cy, radius = fit_circle(x_pts, y_pts)

    # Convert to mm
    radius_mm = radius * pixel_size_mm

    # Calculate curvature (1/r in mm^-1)
    curvature_mm = 1.0 / radius_mm if radius_mm > 0 else 0

    print(f"\n  Circle fit results:")
    print(f"    Center: ({cx:.1f}, {cy:.1f}) pixels")
    print(f"    Radius: {radius:.1f} pixels = {radius_mm:.2f} mm")
    print(f"    Curvature: {curvature_mm:.4f} mm^-1")
    print(f"")
    print(f"  Clinical reference:")
    print(f"    Normal posterior pole radius: ~11-13 mm")
    print(f"    Myopic eyes: larger radius (flatter)")
    print(f"    Hyperopic eyes: smaller radius (steeper)")

    # Step 4: Visualize
    print("\n[4] Creating visualization...")

    # Create visualization image
    vis = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    # Draw the 9 detected RPE points (large green circles)
    for i, pt in enumerate(rpe_points):
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, (0, 255, 0), -1)
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), 2)
        cv2.putText(vis, str(i+1), (int(pt[0]) - 5, int(pt[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw fitted circle (red)
    cv2.circle(vis, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)

    # Draw center point (blue)
    cv2.circle(vis, (int(cx), int(cy)), 5, (255, 0, 0), -1)

    # Add text with results
    cv2.putText(vis, f"Radius: {radius_mm:.2f} mm", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Curvature: {curvature_mm:.4f} mm^-1", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(str(data_dir / 'retina_curvature_analysis.png'), vis)
    print(f"  Saved: retina_curvature_analysis.png")

    # Save results
    results = {
        'center_x': cx,
        'center_y': cy,
        'radius_pixels': radius,
        'radius_mm': radius_mm,
        'curvature_mm': curvature_mm,
        'pixel_size_mm': pixel_size_mm,
        'rpe_points': rpe_points
    }
    np.save(data_dir / 'retina_curvature_results.npy', results)
    print(f"  Saved: retina_curvature_results.npy")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze retina curvature from averaged B-scan')
    parser.add_argument('--data_dir', default='results_5vol_em005', help='Directory for output files')
    parser.add_argument('--input', default=None, help='Input .npy file (default: data_dir/averaged_middle_30_bscans.npy)')
    parser.add_argument('--pixel_size', type=float, default=0.003906,
                        help='Pixel size in mm (default: 6mm/1536 = 0.003906)')
    args = parser.parse_args()

    analyze_retina_curvature(args.data_dir, args.input, args.pixel_size)
