#!/usr/bin/env python3
"""
Debug script for rotation alignment algorithm.
Tests the contour-based rotation alignment on two specific B-scans from EM001.

Loads:
- B-scan 180 from volume 0 (first folder)
- B-scan 180 from volume 1 (second folder)

Provides:
- Extensive logging at every step
- 4 visualizations:
  1. Original B-scans side-by-side
  2. Detected surfaces for multiple angles
  3. Score vs angle plot
  4. Best alignment result with overlay
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

# ============================================================================
# CROPPING FUNCTION (from oct_loader.py)
# ============================================================================

def crop_bscan(bscan, sidebar_width=250, crop_top=100, crop_bottom=50):
    """
    Crop B-scan to remove sidebar, top text, and bottom text.
    This matches the preprocessing done in oct_loader.py when loading volumes.
    """
    height, width = bscan.shape
    # Crop: [top:bottom, left:right]
    cropped = bscan[crop_top:height-crop_bottom, sidebar_width:width]
    return cropped


# ============================================================================
# COPIED FUNCTIONS FROM rotation_alignment.py WITH ADDED LOGGING
# ============================================================================

def preprocess_oct_for_visualization(img, verbose=False):
    """
    Preprocess OCT B-scan for visualization (same as rotation_alignment.py).
    """
    if verbose:
        print(f"    [PREPROCESS] Input shape: {img.shape}, min: {img.min():.2f}, max: {img.max():.2f}")

    # Normalize to 0-255
    img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    # Step 1: Non-local means denoising
    denoised = cv2.fastNlMeansDenoising(img_norm, h=25, templateWindowSize=7, searchWindowSize=21)
    if verbose:
        print(f"    [PREPROCESS] After NLM denoising: min: {denoised.min()}, max: {denoised.max()}")

    # Step 2: Bilateral filtering
    denoised = cv2.bilateralFilter(denoised, d=11, sigmaColor=150, sigmaSpace=150)
    if verbose:
        print(f"    [PREPROCESS] After bilateral: min: {denoised.min()}, max: {denoised.max()}")

    # Step 3: Median filter
    denoised = cv2.medianBlur(denoised, 15)
    if verbose:
        print(f"    [PREPROCESS] After median blur: min: {denoised.min()}, max: {denoised.max()}")

    # Step 4: Threshold
    thresh_val = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_val = int(thresh_val * 0.5)
    denoised[denoised < thresh_val] = 0
    if verbose:
        print(f"    [PREPROCESS] Threshold value: {thresh_val}, non-zero pixels: {(denoised > 0).sum()}")

    # Step 5: CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    denoised = clahe.apply(denoised)
    if verbose:
        print(f"    [PREPROCESS] After CLAHE: min: {denoised.min()}, max: {denoised.max()}")

    return denoised


def create_rotation_mask(bscan_v0, bscan_v1_rotated, threshold_percentile=10,
                        min_valid_pixels_per_column=5, verbose=False):
    """
    Create combined mask for valid regions after rotation.
    """
    Y, X = bscan_v0.shape

    if verbose:
        print(f"    [MASK] Creating rotation mask for images of shape: {bscan_v0.shape}")

    # Threshold for valid tissue
    if (bscan_v0 > 0).any():
        threshold_v0 = np.percentile(bscan_v0[bscan_v0 > 0], threshold_percentile)
    else:
        threshold_v0 = 0

    if (bscan_v1_rotated > 0).any():
        threshold_v1 = np.percentile(bscan_v1_rotated[bscan_v1_rotated > 0], threshold_percentile)
    else:
        threshold_v1 = 0

    if verbose:
        print(f"    [MASK] Thresholds: v0={threshold_v0:.2f}, v1={threshold_v1:.2f}")

    # Create masks
    mask_v0 = bscan_v0 > threshold_v0
    mask_v1 = bscan_v1_rotated > threshold_v1
    mask_2d = mask_v0 & mask_v1

    if verbose:
        print(f"    [MASK] Valid pixels - v0: {mask_v0.sum()}, v1: {mask_v1.sum()}, combined: {mask_2d.sum()}")

    # Per-column validity
    valid_pixels_per_column = mask_2d.sum(axis=0)
    mask_columns = valid_pixels_per_column >= min_valid_pixels_per_column

    if verbose:
        print(f"    [MASK] Valid columns: {mask_columns.sum()}/{X} (threshold: {min_valid_pixels_per_column} pixels/column)")
        print(f"    [MASK] Valid pixels per column - min: {valid_pixels_per_column.min()}, max: {valid_pixels_per_column.max()}, mean: {valid_pixels_per_column.mean():.1f}")

    return mask_2d, mask_columns


def detect_surface_in_masked_region(bscan_denoised, mask_columns, verbose=False):
    """
    Detect retinal surface only in valid columns.
    """
    Y, X = bscan_denoised.shape
    surface = np.full(X, np.nan)

    if verbose:
        print(f"    [SURFACE] Detecting surface in {mask_columns.sum()} valid columns out of {X}")

    # Threshold for surface detection
    threshold = np.percentile(bscan_denoised, 70)
    _, binary = cv2.threshold(bscan_denoised, threshold, 255, cv2.THRESH_BINARY)

    if verbose:
        print(f"    [SURFACE] Binary threshold: {threshold}, white pixels: {(binary > 0).sum()}")

    detected_count = 0
    nan_count = 0

    # Detect surface only in valid columns
    for x in range(X):
        if not mask_columns[x]:
            continue  # Skip invalid columns

        column = binary[:, x]
        white_pixels = np.where(column > 0)[0]
        if len(white_pixels) > 0:
            surface[x] = white_pixels[0]
            detected_count += 1
        else:
            surface[x] = np.nan
            nan_count += 1

    if verbose:
        print(f"    [SURFACE] Detected: {detected_count} columns, NaN: {nan_count} columns")
        valid_surface = surface[~np.isnan(surface)]
        if len(valid_surface) > 0:
            print(f"    [SURFACE] Surface Y-position - min: {valid_surface.min():.1f}, max: {valid_surface.max():.1f}, mean: {valid_surface.mean():.1f}")

    return surface


def calculate_contour_alignment_score(surface_v0, surface_v1, mask_columns, verbose=False):
    """
    Calculate alignment quality based on surface contour similarity.
    WITH OUTLIER REMOVAL (from rotation_alignment.py line 1173).
    """
    # Extract valid regions only
    if not mask_columns.any():
        if verbose:
            print(f"    [SCORE] No valid columns - returning -inf")
        return -np.inf, {'variance': np.inf, 'mad': np.inf, 'valid_pixels': 0}

    surf_v0_valid = surface_v0[mask_columns]
    surf_v1_valid = surface_v1[mask_columns]

    # Calculate surface difference
    diff = surf_v0_valid - surf_v1_valid

    if verbose:
        print(f"    [SCORE] Initial diff points: {len(diff)}")
        non_nan = ~np.isnan(diff)
        if non_nan.sum() > 0:
            print(f"    [SCORE] Non-NaN diff - min: {diff[non_nan].min():.2f}, max: {diff[non_nan].max():.2f}")

    # CRITICAL: OUTLIER REMOVAL - Filter out NaN and extreme outliers (>100px difference)
    # From rotation_alignment.py line 1173
    valid = ~np.isnan(diff) & (np.abs(diff) < 100)
    diff_clean = diff[valid]

    if verbose:
        outliers_removed = len(diff) - len(diff_clean)
        print(f"    [SCORE] After outlier removal: {len(diff_clean)}/{len(diff)} points (removed {outliers_removed} outliers)")
        if len(diff_clean) > 0:
            print(f"    [SCORE] Clean diff - min: {diff_clean.min():.2f}, max: {diff_clean.max():.2f}, mean: {diff_clean.mean():.2f}")

    if len(diff_clean) < 10:
        if verbose:
            print(f"    [SCORE] Too few valid points ({len(diff_clean)}) - returning -inf")
        return -np.inf, {'variance': np.inf, 'mad': np.inf, 'valid_pixels': len(diff_clean)}

    # Primary metric: Variance
    variance = np.var(diff_clean)

    # Secondary metric: MAD
    median_diff = np.median(diff_clean)
    mad = np.median(np.abs(diff_clean - median_diff))

    # Score: Negative variance
    score = -variance

    if verbose:
        print(f"    [SCORE] Variance: {variance:.2f} px², MAD: {mad:.2f} px, Score: {score:.2f}")

    metrics = {
        'variance': float(variance),
        'mad': float(mad),
        'median_diff': float(median_diff),
        'valid_pixels': len(diff_clean),
        'rms': float(np.sqrt(np.mean(diff_clean**2)))
    }

    return score, metrics


def test_single_rotation_angle_with_logging(angle, bscan_v0, bscan_v1, verbose=False):
    """
    Test a single rotation angle with comprehensive logging.
    """
    print(f"\n  {'='*60}")
    print(f"  TESTING ANGLE: {angle:+.1f}°")
    print(f"  {'='*60}")

    try:
        # Rotate moving B-scan
        print(f"  [1] Rotating B-scan by {angle:+.1f}°...")
        bscan_v1_rotated = ndimage.rotate(
            bscan_v1, angle, axes=(0, 1),
            reshape=False, order=1,
            mode='constant', cval=0
        )
        print(f"      Rotated shape: {bscan_v1_rotated.shape}, non-zero: {(bscan_v1_rotated > 0).sum()}")

        # Create mask
        print(f"  [2] Creating rotation mask...")
        mask_2d, mask_columns = create_rotation_mask(bscan_v0, bscan_v1_rotated, verbose=verbose)

        # Check if enough valid pixels
        if mask_columns.sum() < 10:
            print(f"      ⚠️  INSUFFICIENT VALID COLUMNS: {mask_columns.sum()} < 10")
            return {
                'angle': float(angle),
                'score': -np.inf,
                'variance': np.inf,
                'valid_columns': mask_columns.sum(),
                'surface_v0': None,
                'surface_v1': None
            }

        # Preprocess
        print(f"  [3] Preprocessing B-scans...")
        bscan_v0_denoised = preprocess_oct_for_visualization(bscan_v0, verbose=verbose)
        bscan_v1_denoised = preprocess_oct_for_visualization(bscan_v1_rotated, verbose=verbose)

        # Detect surfaces
        print(f"  [4] Detecting surfaces...")
        surface_v0 = detect_surface_in_masked_region(bscan_v0_denoised, mask_columns, verbose=verbose)
        surface_v1 = detect_surface_in_masked_region(bscan_v1_denoised, mask_columns, verbose=verbose)

        # Calculate score
        print(f"  [5] Calculating alignment score...")
        score, metrics = calculate_contour_alignment_score(surface_v0, surface_v1, mask_columns, verbose=verbose)

        print(f"\n  📊 RESULT FOR {angle:+.1f}°:")
        print(f"     Score: {score:.2f}")
        print(f"     Variance: {metrics['variance']:.2f} px²")
        print(f"     Valid pixels: {metrics['valid_pixels']}")

        return {
            'angle': float(angle),
            'score': float(score),
            'variance': metrics['variance'],
            'mad': metrics['mad'],
            'valid_pixels': metrics['valid_pixels'],
            'valid_columns': mask_columns.sum(),
            'surface_v0': surface_v0,
            'surface_v1': surface_v1,
            'mask_columns': mask_columns,
            'bscan_v0_denoised': bscan_v0_denoised,
            'bscan_v1_denoised': bscan_v1_denoised,
            'bscan_v1_rotated': bscan_v1_rotated
        }

    except Exception as e:
        print(f"      ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'angle': float(angle),
            'score': -np.inf,
            'variance': np.inf,
            'valid_columns': 0,
            'error': str(e)
        }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_original_bscans(bscan_v0, bscan_v1, save_path):
    """
    Plot 1: Original B-scans side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(bscan_v0, cmap='gray', aspect='auto')
    axes[0].set_title('B-scan Volume 0 (Reference)\nIndex 180', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X (lateral)')
    axes[0].set_ylabel('Y (depth)')

    axes[1].imshow(bscan_v1, cmap='gray', aspect='auto')
    axes[1].set_title('B-scan Volume 1 (Moving)\nIndex 180', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X (lateral)')
    axes[1].set_ylabel('Y (depth)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_surfaces_at_angles(bscan_v0, bscan_v1, results, angles_to_show, save_path):
    """
    Plot 2: Detected surfaces overlaid for multiple key angles.
    """
    n_angles = len(angles_to_show)
    fig, axes = plt.subplots(2, n_angles, figsize=(4*n_angles, 8))

    for i, angle in enumerate(angles_to_show):
        # Find result for this angle
        result = next((r for r in results if abs(r['angle'] - angle) < 0.01), None)

        if result is None or result.get('surface_v0') is None:
            axes[0, i].text(0.5, 0.5, f'No data for {angle:+.1f}°',
                          ha='center', va='center', transform=axes[0, i].transAxes)
            axes[1, i].text(0.5, 0.5, f'No data for {angle:+.1f}°',
                          ha='center', va='center', transform=axes[1, i].transAxes)
            continue

        # Top row: Reference B-scan with surface
        axes[0, i].imshow(result.get('bscan_v0_denoised', bscan_v0), cmap='gray', aspect='auto')
        if result['surface_v0'] is not None:
            x_coords = np.arange(len(result['surface_v0']))
            axes[0, i].plot(x_coords, result['surface_v0'], 'r-', linewidth=1.5, label='Surface V0')
        axes[0, i].set_title(f'Reference\n{angle:+.1f}°', fontsize=10)
        axes[0, i].set_ylabel('Y (depth)')

        # Bottom row: Rotated B-scan with surface
        axes[1, i].imshow(result.get('bscan_v1_denoised', bscan_v1), cmap='gray', aspect='auto')
        if result['surface_v1'] is not None:
            x_coords = np.arange(len(result['surface_v1']))
            axes[1, i].plot(x_coords, result['surface_v1'], 'b-', linewidth=1.5, label='Surface V1')

        score_text = f"Score: {result['score']:.2f}\nVar: {result['variance']:.2f}\nValid: {result['valid_pixels']}"
        axes[1, i].text(0.02, 0.98, score_text, transform=axes[1, i].transAxes,
                       fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round',
                       facecolor='wheat', alpha=0.8))
        axes[1, i].set_title(f'Rotated {angle:+.1f}°', fontsize=10)
        axes[1, i].set_xlabel('X (lateral)')
        axes[1, i].set_ylabel('Y (depth)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_score_vs_angle(results, save_path):
    """
    Plot 3: Score vs angle graph showing the optimization landscape.
    """
    angles = [r['angle'] for r in results]
    scores = [r['score'] if r['score'] > -np.inf else np.nan for r in results]
    variances = [r['variance'] if r['variance'] < np.inf else np.nan for r in results]
    valid_columns = [r.get('valid_columns', 0) for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Score vs Angle
    axes[0].plot(angles, scores, 'bo-', linewidth=2, markersize=6)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('Rotation Angle (degrees)', fontsize=11)
    axes[0].set_ylabel('Alignment Score\n(higher = better)', fontsize=11)
    axes[0].set_title('Score vs Rotation Angle', fontsize=13, fontweight='bold')

    # Mark best angle
    valid_results = [(a, s) for a, s in zip(angles, scores) if not np.isnan(s)]
    if valid_results:
        best_angle, best_score = max(valid_results, key=lambda x: x[1])
        axes[0].axvline(x=best_angle, color='g', linestyle='--', linewidth=2,
                       label=f'Best: {best_angle:+.1f}° (score={best_score:.2f})')
        axes[0].legend()

    # Plot 2: Variance vs Angle
    axes[1].plot(angles, variances, 'rs-', linewidth=2, markersize=6)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('Rotation Angle (degrees)', fontsize=11)
    axes[1].set_ylabel('Surface Variance\n(lower = better)', fontsize=11)
    axes[1].set_title('Variance vs Rotation Angle', fontsize=13, fontweight='bold')

    if valid_results:
        axes[1].axvline(x=best_angle, color='g', linestyle='--', linewidth=2)

    # Plot 3: Valid Columns vs Angle
    axes[2].plot(angles, valid_columns, 'g^-', linewidth=2, markersize=6)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel('Rotation Angle (degrees)', fontsize=11)
    axes[2].set_ylabel('Valid Columns', fontsize=11)
    axes[2].set_title('Valid Columns vs Rotation Angle', fontsize=13, fontweight='bold')

    if valid_results:
        axes[2].axvline(x=best_angle, color='g', linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_best_alignment(bscan_v0, bscan_v1, best_result, save_path):
    """
    Plot 4: Best alignment result with both surfaces overlaid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Original reference with surface
    axes[0, 0].imshow(bscan_v0, cmap='gray', aspect='auto')
    if best_result.get('surface_v0') is not None:
        x_coords = np.arange(len(best_result['surface_v0']))
        axes[0, 0].plot(x_coords, best_result['surface_v0'], 'r-', linewidth=2, label='Surface V0')
    axes[0, 0].set_title('Reference B-scan (V0)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Y (depth)')
    axes[0, 0].legend()

    # Top-right: Best rotated with surface
    axes[0, 1].imshow(best_result.get('bscan_v1_rotated', bscan_v1), cmap='gray', aspect='auto')
    if best_result.get('surface_v1') is not None:
        x_coords = np.arange(len(best_result['surface_v1']))
        axes[0, 1].plot(x_coords, best_result['surface_v1'], 'b-', linewidth=2, label='Surface V1')
    axes[0, 1].set_title(f'Rotated B-scan (V1) at {best_result["angle"]:+.2f}°',
                        fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Y (depth)')
    axes[0, 1].legend()

    # Bottom-left: Overlay of both surfaces
    axes[1, 0].imshow(bscan_v0, cmap='gray', aspect='auto', alpha=0.7)
    if best_result.get('surface_v0') is not None:
        x_coords = np.arange(len(best_result['surface_v0']))
        axes[1, 0].plot(x_coords, best_result['surface_v0'], 'r-', linewidth=2.5,
                       label='Surface V0', alpha=0.9)
    if best_result.get('surface_v1') is not None:
        x_coords = np.arange(len(best_result['surface_v1']))
        axes[1, 0].plot(x_coords, best_result['surface_v1'], 'b-', linewidth=2.5,
                       label='Surface V1 (aligned)', alpha=0.9)
    axes[1, 0].set_title('Surface Overlay', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('X (lateral)')
    axes[1, 0].set_ylabel('Y (depth)')
    axes[1, 0].legend()

    # Bottom-right: Surface difference plot
    if best_result.get('surface_v0') is not None and best_result.get('surface_v1') is not None:
        surf_diff = best_result['surface_v0'] - best_result['surface_v1']
        x_coords = np.arange(len(surf_diff))

        axes[1, 1].plot(x_coords, surf_diff, 'purple', linewidth=1.5)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].fill_between(x_coords, 0, surf_diff, alpha=0.3, color='purple')
        axes[1, 1].set_title(f'Surface Difference (V0 - V1)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('X (lateral)')
        axes[1, 1].set_ylabel('Difference (pixels)')
        axes[1, 1].grid(True, alpha=0.3)

        # Add stats
        diff_clean = surf_diff[~np.isnan(surf_diff)]
        if len(diff_clean) > 0:
            stats_text = (f"Variance: {best_result['variance']:.2f} px²\n"
                         f"MAD: {best_result['mad']:.2f} px\n"
                         f"Mean diff: {diff_clean.mean():.2f} px\n"
                         f"Valid pixels: {best_result['valid_pixels']}")
            axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.suptitle(f'Best Alignment Result: {best_result["angle"]:+.2f}° (Score: {best_result["score"]:.2f})',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("ROTATION ALIGNMENT DEBUG TEST")
    print("="*70)

    # Define paths - use absolute path
    data_root = Path(r"C:\Users\illia\Desktop\diploma\RetinaBuilder\oct_data\emmetropes\EM001")
    vol0_folder = "F001_IP_20250604_175814_Retina_3D_L_6mm_1536x360_2"
    vol1_folder = "F001_IP_20250604_180102_Retina_3D_L_6mm_1536x360_2"

    bscan_idx = 180

    vol0_path = data_root / vol0_folder / f"F001_IP_20250604_175814_Retina_3D_L_6mm_1536x360_scan{bscan_idx}.bmp"
    vol1_path = data_root / vol1_folder / f"F001_IP_20250604_180102_Retina_3D_L_6mm_1536x360_scan{bscan_idx}.bmp"

    print(f"\nLoading B-scans:")
    print(f"  Volume 0: {vol0_path}")
    print(f"  Volume 1: {vol1_path}")

    # Load B-scans
    if not vol0_path.exists():
        print(f"❌ ERROR: File not found: {vol0_path}")
        return
    if not vol1_path.exists():
        print(f"❌ ERROR: File not found: {vol1_path}")
        return

    bscan_v0_raw = cv2.imread(str(vol0_path), cv2.IMREAD_GRAYSCALE)
    bscan_v1_raw = cv2.imread(str(vol1_path), cv2.IMREAD_GRAYSCALE)

    print(f"\n✓ Loaded B-scans:")
    print(f"  Volume 0 raw shape: {bscan_v0_raw.shape}, dtype: {bscan_v0_raw.dtype}")
    print(f"  Volume 1 raw shape: {bscan_v1_raw.shape}, dtype: {bscan_v1_raw.dtype}")

    # CRITICAL: CROP B-SCANS (same as oct_loader.py does when loading volumes)
    print(f"\n🔪 Cropping B-scans...")
    print(f"  Volume 0: sidebar=250px, top=100px, bottom=50px")
    print(f"  Volume 1: sidebar=250px + 500px overlap offset, top=100px, bottom=50px")

    bscan_v0 = crop_bscan(bscan_v0_raw, sidebar_width=250, crop_top=100, crop_bottom=50)
    # Volume 1: Additional 500px from left to simulate overlap region
    bscan_v1 = crop_bscan(bscan_v1_raw, sidebar_width=250+500, crop_top=100, crop_bottom=50)

    print(f"\n✓ Cropped B-scans:")
    print(f"  Volume 0 shape: {bscan_v0.shape}, range: [{bscan_v0.min()}, {bscan_v0.max()}]")
    print(f"  Volume 1 shape: {bscan_v1.shape}, range: [{bscan_v1.min()}, {bscan_v1.max()}]")

    # Match widths by cropping v0 to match v1's width
    if bscan_v0.shape[1] != bscan_v1.shape[1]:
        width_diff = bscan_v0.shape[1] - bscan_v1.shape[1]
        print(f"\n⚖️  Matching widths (cropping {width_diff}px from right of V0)...")
        bscan_v0 = bscan_v0[:, :bscan_v1.shape[1]]
        print(f"  Volume 0 final shape: {bscan_v0.shape}")
        print(f"  Volume 1 final shape: {bscan_v1.shape}")

    # Create output directory
    output_dir = Path("debug_rotation_output")
    output_dir.mkdir(exist_ok=True)
    print(f"\n✓ Output directory: {output_dir.absolute()}")

    # VISUALIZATION 1: Original B-scans
    print(f"\n{'='*70}")
    print("VISUALIZATION 1: Original B-scans")
    print("="*70)
    plot_original_bscans(bscan_v0, bscan_v1, output_dir / "01_original_bscans.png")

    # TEST ROTATION AT MULTIPLE ANGLES
    print(f"\n{'='*70}")
    print("TESTING ROTATION AT MULTIPLE ANGLES")
    print("="*70)

    # Test angles: -20 to +20 degrees
    test_angles = list(range(-20, 21, 5))  # [-20, -15, -10, -5, 0, 5, 10, 15, 20]

    print(f"\nTesting {len(test_angles)} angles: {test_angles}")

    results = []
    for angle in test_angles:
        result = test_single_rotation_angle_with_logging(angle, bscan_v0, bscan_v1, verbose=True)
        results.append(result)

    # Find best result
    valid_results = [r for r in results if r['score'] > -np.inf]
    if not valid_results:
        print("\n❌ NO VALID RESULTS FOUND!")
        return

    best_result = max(valid_results, key=lambda x: x['score'])

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print("="*70)
    print(f"\n🏆 BEST ANGLE: {best_result['angle']:+.2f}°")
    print(f"   Score: {best_result['score']:.2f}")
    print(f"   Variance: {best_result['variance']:.2f} px²")
    print(f"   MAD: {best_result['mad']:.2f} px")
    print(f"   Valid pixels: {best_result['valid_pixels']}")
    print(f"   Valid columns: {best_result['valid_columns']}")

    # Show all results
    print(f"\nAll tested angles:")
    print(f"  {'Angle':>8}  {'Score':>12}  {'Variance':>12}  {'Valid Cols':>12}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")
    for r in results:
        score_str = f"{r['score']:.2f}" if r['score'] > -np.inf else "-inf"
        var_str = f"{r['variance']:.2f}" if r['variance'] < np.inf else "inf"
        print(f"  {r['angle']:>+8.1f}°  {score_str:>12}  {var_str:>12}  {r.get('valid_columns', 0):>12}")

    # VISUALIZATION 2: Surfaces at key angles
    print(f"\n{'='*70}")
    print("VISUALIZATION 2: Detected Surfaces at Key Angles")
    print("="*70)
    angles_to_show = [-15, -5, 0, 5, 15, best_result['angle']]
    angles_to_show = sorted(list(set(angles_to_show)))  # Remove duplicates
    plot_surfaces_at_angles(bscan_v0, bscan_v1, results, angles_to_show,
                           output_dir / "02_surfaces_at_angles.png")

    # VISUALIZATION 3: Score vs angle
    print(f"\n{'='*70}")
    print("VISUALIZATION 3: Score vs Angle Plot")
    print("="*70)
    plot_score_vs_angle(results, output_dir / "03_score_vs_angle.png")

    # VISUALIZATION 4: Best alignment
    print(f"\n{'='*70}")
    print("VISUALIZATION 4: Best Alignment Result")
    print("="*70)
    plot_best_alignment(bscan_v0, bscan_v1, best_result, output_dir / "04_best_alignment.png")

    print(f"\n{'='*70}")
    print("✅ TEST COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print(f"\nGenerated files:")
    print(f"  1. 01_original_bscans.png")
    print(f"  2. 02_surfaces_at_angles.png")
    print(f"  3. 03_score_vs_angle.png")
    print(f"  4. 04_best_alignment.png")


if __name__ == "__main__":
    main()
