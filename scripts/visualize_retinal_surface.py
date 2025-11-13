"""
Visualize the main retinal surface (ILM - Inner Limiting Membrane) alignment
between merged volumes after rigid alignment (Steps 1-4).

This shows whether the retinal layers are properly aligned without non-rigid deformation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def detect_retinal_surface(bscan, method='top_edge', preprocessed=False):
    """
    Detect the main retinal surface (brightest layer) in a B-scan.

    Args:
        bscan: 2D array (Y, X) - B-scan image
        method: Detection method to use:
            - 'top_edge' (default): First bright pixel from top (simple, robust)
            - 'contour': OpenCV contours + top boundary extraction
            - 'center_mass': Center of mass of bright region per column
            - 'all': Return all 3 methods for comparison
        preprocessed: If True, skip preprocessing (image already denoised)

    Returns:
        surface: 1D array (X,) - Y-coordinate of surface at each X position
                 OR dict with all 3 methods if method='all'
    """
    import cv2

    Y, X = bscan.shape
    surface = np.zeros(X)

    # STEP 1: Preprocessing (skip if already done)
    if preprocessed:
        bscan_processed = bscan
    else:
        img_norm = ((bscan - bscan.min()) / (bscan.max() - bscan.min() + 1e-8) * 255).astype(np.uint8)
        bscan_processed = cv2.GaussianBlur(img_norm, (5, 5), 1.0)

    # Helper: detect where tissue STARTS (top surface)
    def detect_top_edge(img):
        """Find first significant intensity increase = tissue starts"""
        surf = np.zeros(X)

        for x in range(X):
            column = img[:, x].astype(float)

            # Compute gradient (where intensity increases)
            gradient = np.gradient(column)

            # Find strong positive gradients (dark->bright transition)
            # This is where tissue STARTS
            threshold = np.percentile(gradient[gradient > 0], 75) if np.any(gradient > 0) else 0

            candidates = np.where(gradient > threshold)[0]

            if len(candidates) > 0:
                # Take first strong increase = top surface
                surf[x] = candidates[0]
            else:
                # No clear edge - use intensity threshold
                bright = np.where(column > np.percentile(column, 75))[0]
                surf[x] = bright[0] if len(bright) > 0 else (surf[x-1] if x > 0 else Y // 4)

        return surf

    def detect_contour_boundary(img):
        """Use intensity threshold + find top boundary"""
        surf = np.zeros(X)

        # Simple threshold
        threshold = np.percentile(img, 70)
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        # For each column, find first white pixel
        for x in range(X):
            column = binary[:, x]
            white_pixels = np.where(column > 0)[0]
            if len(white_pixels) > 0:
                surf[x] = white_pixels[0]  # First from top
            else:
                surf[x] = surf[x-1] if x > 0 else Y // 4

        return surf

    def detect_center_of_mass(img):
        """Use brightest region center"""
        surf = np.zeros(X)

        threshold = np.percentile(img, 70)

        for x in range(X):
            column = img[:, x]
            bright_pixels = np.where(column > threshold)[0]

            if len(bright_pixels) > 0:
                # Take mean of bright pixels in upper 60%
                search_limit = int(Y * 0.6)
                bright_upper = bright_pixels[bright_pixels < search_limit]
                if len(bright_upper) > 0:
                    surf[x] = np.mean(bright_upper)
                else:
                    surf[x] = np.mean(bright_pixels)
            else:
                surf[x] = surf[x-1] if x > 0 else Y // 4

        return surf

    # Execute detection based on method
    if method == 'top_edge':
        surface = detect_top_edge(bscan_processed)

    elif method == 'contour':
        surface = detect_contour_boundary(bscan_processed)

    elif method == 'center_mass':
        surface = detect_center_of_mass(bscan_processed)

    elif method == 'all':
        # Return all 3 for comparison
        results = {
            'top_edge': detect_top_edge(bscan_processed),
            'contour': detect_contour_boundary(bscan_processed),
            'center_mass': detect_center_of_mass(bscan_processed)
        }

        # Apply smoothing to all
        from scipy.ndimage import median_filter
        for key in results:
            results[key] = median_filter(results[key], size=11)
            window = min(51, len(results[key]) if len(results[key]) % 2 == 1 else len(results[key]) - 1)
            if window >= 5:
                results[key] = savgol_filter(results[key], window_length=window, polyorder=3)

        return results

    else:
        raise ValueError(f"Unknown method: {method}. Use 'top_edge', 'contour', 'center_mass', or 'all'")

    # STEP 2: Smooth the detected surface to remove outliers
    from scipy.ndimage import median_filter

    # Use median filter first to remove spikes
    surface = median_filter(surface, size=11)
    # Then apply Savitzky-Golay filter for smooth curve
    window = min(51, len(surface) if len(surface) % 2 == 1 else len(surface) - 1)
    if window >= 5:
        surface = savgol_filter(surface, window_length=window, polyorder=3)

    return surface


def compute_average_surface(volume, num_samples=20):
    """
    Compute average retinal surface across multiple B-scans.

    Args:
        volume: 3D array (Y, X, Z) - OCT volume
        num_samples: Number of B-scans to sample

    Returns:
        avg_surface: 2D array (X, Z) - Average surface height at each (X, Z) position
    """
    Y, X, Z = volume.shape

    # Sample B-scans evenly across the volume
    z_indices = np.linspace(0, Z-1, num_samples, dtype=int)

    print(f"   Sampling {num_samples} B-scans from Z indices: {z_indices[0]} to {z_indices[-1]}...")

    # Detect surface in each sampled B-scan
    surfaces = []
    for z in z_indices:
        bscan = volume[:, :, z]
        surface = detect_retinal_surface(bscan, method='top_edge')
        surfaces.append(surface)

    surfaces = np.array(surfaces)  # Shape: (num_samples, X)

    # Create full 2D surface by interpolating
    surface_2d = np.zeros((X, Z))

    for x in range(X):
        # Interpolate surface height at this X across all Z
        surface_heights = surfaces[:, x]
        surface_2d[x, :] = np.interp(
            np.arange(Z),
            z_indices,
            surface_heights
        )

    return surface_2d


def main():
    # Setup paths
    data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'
    step1_path = data_dir / 'step1_results.npy'
    step2_path = data_dir / 'step2_results.npy'
    step3_path = data_dir / 'step3_results.npy'

    # Check files exist
    if not step3_path.exists():
        print(f"Error: Step 3 results not found at {step3_path}")
        print("Run the pipeline first: python scripts/alignment_pipeline.py --steps 1 2 3")
        return

    print("="*70)
    print("RETINAL SURFACE ALIGNMENT VISUALIZATION")
    print("="*70)

    # Load results
    print("\n1. Loading alignment results...")
    step1_results = np.load(step1_path, allow_pickle=True).item()
    step2_results = np.load(step2_path, allow_pickle=True).item()
    step3_results = np.load(step3_path, allow_pickle=True).item()

    # Get aligned overlap regions (after Steps 1-4, BEFORE Step 5)
    overlap_v0 = step1_results['overlap_v0']

    # Use Step 4 result if available, otherwise Step 3
    if 'overlap_v1_final' in step3_results:
        overlap_v1 = step3_results['overlap_v1_final']
        step_name = "Step 4 (Windowed Y-alignment)"
        ncc = step3_results.get('ncc_after_windowed', step3_results.get('ncc_after_x', 0))
    else:
        overlap_v1 = step3_results['overlap_v1_rotated_x']
        step_name = "Step 3.5 (X-rotation)"
        ncc = step3_results.get('ncc_after_x', 0)

    print(f"   ✓ Volume 0 (reference): {overlap_v0.shape}")
    print(f"   ✓ Volume 1 (aligned with {step_name}): {overlap_v1.shape}")
    print(f"   ✓ NCC: {ncc:.4f}")

    # SKIP volume-wide surface detection - ONLY do single B-scan
    print("\n2. Skipping volume-wide detection - focusing on single B-scan...")

    Y, X, Z = overlap_v0.shape

    # Figure: Single B-scan detection only
    fig2, axes = plt.subplots(3, 2, figsize=(16, 14))

    # Central B-scan (Z = middle)
    z_mid = Z // 2
    bscan_v0 = overlap_v0[:, :, z_mid]
    bscan_v1 = overlap_v1[:, :, z_mid]

    # Apply HARSH denoising to get clean images
    import cv2

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

    # NOW detect on the DENOISED images - CONTOUR METHOD ONLY
    surface_v0 = detect_retinal_surface(bscan_v0_denoised, method='contour', preprocessed=True)
    surface_v1 = detect_retinal_surface(bscan_v1_denoised, method='contour', preprocessed=True)

    # Row 1: Original vs Denoised
    ax = axes[0, 0]
    ax.imshow(bscan_v0, cmap='gray', aspect='auto')
    ax.set_title(f'Volume 0 - Original B-scan (Z={z_mid})', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (A-scans)')
    ax.set_ylabel('Y (depth)')

    ax = axes[0, 1]
    ax.imshow(bscan_v0_denoised, cmap='gray', aspect='auto')
    ax.set_title(f'Volume 0 - After Harsh Denoising\n(NLM+Bilateral+Median+Otsu50%+CLAHE)', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (A-scans)')
    ax.set_ylabel('Y (depth)')

    # Row 2: Show contour detection ON the denoised images
    ax = axes[1, 0]
    ax.imshow(bscan_v0_denoised, cmap='gray', aspect='auto')
    # Draw BOLD line showing detected surface
    ax.plot(surface_v0, 'r-', linewidth=4, label='Detected Surface (Contour)')
    ax.set_title(f'V0 Denoised + Detected Surface', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (A-scans)')
    ax.set_ylabel('Y (depth)')
    ax.legend()

    ax = axes[1, 1]
    ax.imshow(bscan_v1_denoised, cmap='gray', aspect='auto')
    # Draw BOLD line showing detected surface
    ax.plot(surface_v1, 'r-', linewidth=4, label='Detected Surface (Contour)')
    ax.set_title(f'V1 Denoised + Detected Surface', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (A-scans)')
    ax.set_ylabel('Y (depth)')
    ax.legend()

    # Row 3: V0 vs V1 surface comparison
    ax = axes[2, 0]
    ax.imshow(bscan_v0_denoised, cmap='gray', aspect='auto', alpha=0.7)
    ax.plot(surface_v0, 'r-', linewidth=4, label='Volume 0')
    ax.plot(surface_v1, 'b-', linewidth=4, label='Volume 1', alpha=0.8)
    ax.fill_between(np.arange(X), surface_v0, surface_v1,
                     alpha=0.3, color='yellow', label='Misalignment')
    ax.set_title(f'Surface Overlay\n(Red=V0, Blue=V1, Yellow=Gap)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X (A-scans)')
    ax.set_ylabel('Y (depth)')
    ax.legend()
    ax.set_ylim([0, Y//2])

    # V1 vs V0 difference analysis
    ax = axes[2, 1]
    diff_v1_v0 = surface_v1 - surface_v0

    ax.plot(diff_v1_v0, 'r-', linewidth=2, label=f'V1 - V0 (mean={np.mean(np.abs(diff_v1_v0)):.2f}px)')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(f'Surface Alignment Difference\n(Lower = better alignment)', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (A-scans)')
    ax.set_ylabel('Difference (pixels)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig2.suptitle(f'Single B-scan Surface Detection Test - After {step_name}',
                  fontsize=16, fontweight='bold')

    plt.tight_layout()

    output = data_dir / 'retinal_surface_bscan.png'
    plt.savefig(output, dpi=200, bbox_inches='tight')
    print(f"   ✓ Saved: {output.name}")

    print("\n" + "="*70)
    print("SINGLE B-SCAN DETECTION TEST COMPLETE!")
    print("="*70)
    print(f"\nOutput file: {output}")
    print(f"\nThis shows detection on a SINGLE B-scan (Z={z_mid}) to verify the algorithm works.")
    print(f"Check the image to see if contour detection correctly identifies the tissue surface.")


if __name__ == "__main__":
    main()
