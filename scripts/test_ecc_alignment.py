#!/usr/bin/env python3
"""
Test Rotation-Only Alignment for OCT B-scans with ECC-style Correlation

Uses grid search over rotation angles with normalized correlation scoring.
Includes aggressive denoising and thresholding to isolate retinal layers.
This approach is more robust for uniform retinal structures than feature-based methods.

Usage:
    python test_ecc_alignment.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from scipy import ndimage
from scipy.ndimage import median_filter

print("="*70)
print("ROTATION-ONLY ALIGNMENT TEST (ECC-STYLE CORRELATION)")
print("="*70)

# Setup paths
data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'

# Load from pipeline results (Step 1 and Step 2)
print("\n1. Loading from pipeline results...")
step1_results = np.load(data_dir / 'step1_results.npy', allow_pickle=True).item()
step2_results = np.load(data_dir / 'step2_results.npy', allow_pickle=True).item()

# Extract overlap regions and central B-scan
overlap_v0 = step1_results['overlap_v0']
overlap_v1_y_aligned = step2_results['overlap_v1_y_aligned']

# Get central B-scan
z_center = overlap_v0.shape[2] // 2
bscan_v0 = overlap_v0[:, :, z_center]
bscan_v1 = overlap_v1_y_aligned[:, :, z_center]

# Create mask (non-zero regions in both)
mask = (bscan_v0 > np.percentile(bscan_v0[bscan_v0 > 0], 10)) & \
       (bscan_v1 > np.percentile(bscan_v1[bscan_v1 > 0], 10))

print(f"  ✓ B-scan V0: {bscan_v0.shape}")
print(f"  ✓ B-scan V1: {bscan_v1.shape}")
print(f"  ✓ Mask: {mask.sum()} valid pixels ({100*mask.sum()/mask.size:.1f}%)")
print(f"  ✓ Using central B-scan at Z={z_center}")

def preprocess_oct(img, mask=None):
    """Preprocess OCT B-scan with VERY aggressive denoising and thresholding (20% harsher)."""
    # Normalize to 0-255 first
    img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    # Step 1: Non-local means denoising (20% stronger: h 10→12)
    denoised1 = cv2.fastNlMeansDenoising(img_norm, h=12, templateWindowSize=7, searchWindowSize=21)

    # Step 2: Bilateral filtering (20% stronger: sigma 75→90)
    denoised2 = cv2.bilateralFilter(denoised1, d=9, sigmaColor=90, sigmaSpace=90)

    # Step 3: Median filter (20% stronger: kernel 5→7)
    denoised3 = cv2.medianBlur(denoised2, 7)

    # Step 4: Threshold to keep only tissue layers (20% more conservative: 70%→80%)
    # Use Otsu's method but be more conservative
    thresh_val = cv2.threshold(denoised3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_val = int(thresh_val * 0.8)  # 80% of Otsu threshold (was 70%)

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

def calculate_correlation(img1, img2, mask=None):
    """Calculate normalized correlation coefficient between two images."""
    if mask is not None:
        valid = mask
    else:
        valid = (img1 > 0) & (img2 > 0)

    if valid.sum() < 100:
        return -1.0

    i1 = img1[valid].astype(float)
    i2 = img2[valid].astype(float)

    # Normalize
    i1_norm = (i1 - i1.mean()) / (i1.std() + 1e-8)
    i2_norm = (i2 - i2.mean()) / (i2.std() + 1e-8)

    return np.mean(i1_norm * i2_norm)

def align_oct_bscans(img1, img2, mask=None, verbose=True):
    """Align two OCT B-scans using rotation-only grid search with ECC-style correlation."""
    # Preprocess: enhance layers
    if verbose:
        print("  → Preprocessing images...")
    img1_proc = preprocess_oct(img1, mask=mask)
    img2_proc = preprocess_oct(img2, mask=mask)

    # Grid search for rotation only (no translation)
    if verbose:
        print("  → Coarse rotation search...")

    # Coarse search: -10° to +10° in 1° steps
    coarse_angles = np.arange(-10, 11, 1)
    coarse_scores = []

    for angle in coarse_angles:
        # Rotate using scipy (same method as pipeline for consistency)
        img2_rotated = ndimage.rotate(img2_proc, angle, axes=(0, 1),
                                       reshape=False, order=1,
                                       mode='constant', cval=0)

        # Calculate correlation
        score = calculate_correlation(img1_proc, img2_rotated, mask=mask)
        coarse_scores.append(score)

    best_coarse_idx = np.argmax(coarse_scores)
    best_coarse_angle = coarse_angles[best_coarse_idx]
    best_coarse_score = coarse_scores[best_coarse_idx]

    if verbose:
        print(f"  ✓ Coarse search: best angle = {best_coarse_angle:+.1f}°, correlation = {best_coarse_score:.4f}")

    # Fine search: ±3° around best coarse angle in 0.5° steps
    if verbose:
        print("  → Fine rotation search...")

    fine_angles = np.arange(best_coarse_angle - 3, best_coarse_angle + 3.5, 0.5)
    fine_scores = []

    for angle in fine_angles:
        # Rotate using scipy (same method as pipeline for consistency)
        img2_rotated = ndimage.rotate(img2_proc, angle, axes=(0, 1),
                                       reshape=False, order=1,
                                       mode='constant', cval=0)

        score = calculate_correlation(img1_proc, img2_rotated, mask=mask)
        fine_scores.append(score)

    best_fine_idx = np.argmax(fine_scores)
    best_angle = fine_angles[best_fine_idx]
    best_score = fine_scores[best_fine_idx]

    if verbose:
        print(f"  ✓ Fine search: best angle = {best_angle:+.2f}°, correlation = {best_score:.4f}")

    # Apply best rotation to original image using scipy
    img2_aligned = ndimage.rotate(img2, best_angle, axes=(0, 1),
                                   reshape=False, order=1,
                                   mode='constant', cval=0)

    return img2_aligned, best_angle, best_score

# Run rotation-only alignment
print("\n2. Preprocessing and running rotation-only alignment...")
result = align_oct_bscans(bscan_v0, bscan_v1, mask=mask, verbose=True)

if result[0] is None:
    print("\n❌ ERROR: Alignment failed!")
    print("   Try adjusting preprocessing parameters or mask.")
    sys.exit(1)

bscan_v1_aligned, rotation_angle, correlation = result

print(f"\n{'='*70}")
print(f"ESTIMATED TRANSFORMATION")
print(f"{'='*70}")
print(f"  Rotation:        {rotation_angle:+.2f}°")
print(f"  Translation:     NONE (rotation around center only)")
print(f"  Correlation:     {correlation:.6f}")
print(f"{'='*70}")

# Calculate alignment quality
print("\n3. Calculating alignment quality...")
# Calculate NCC on aligned images
# Rotate mask using scipy to match the rotated image
mask_aligned = ndimage.rotate(mask.astype(np.uint8), rotation_angle, axes=(0, 1),
                               reshape=False, order=0, mode='constant', cval=0).astype(bool)

overlap_mask = mask & mask_aligned
if overlap_mask.sum() > 100:
    # Normalize to 0-255 for fair comparison
    v0_norm = ((bscan_v0 - bscan_v0.min()) / (bscan_v0.max() - bscan_v0.min() + 1e-8) * 255).astype(np.uint8)
    v1_aligned_norm = ((bscan_v1_aligned - bscan_v1_aligned.min()) / (bscan_v1_aligned.max() - bscan_v1_aligned.min() + 1e-8) * 255).astype(np.uint8)

    v0_pixels = v0_norm[overlap_mask].astype(float)
    v1_pixels = v1_aligned_norm[overlap_mask].astype(float)

    v0_norm_vals = (v0_pixels - v0_pixels.mean()) / (v0_pixels.std() + 1e-8)
    v1_norm_vals = (v1_pixels - v1_pixels.mean()) / (v1_pixels.std() + 1e-8)

    ncc_aligned = np.mean(v0_norm_vals * v1_norm_vals)
    print(f"  NCC after alignment: {ncc_aligned:.4f}")
else:
    ncc_aligned = -1.0
    print(f"  Warning: Not enough overlap to calculate NCC")

# Visualize results
print("\n4. Creating visualizations...")
fig = plt.figure(figsize=(20, 12))

# Row 1: Original B-scans
ax1 = plt.subplot(3, 3, 1)
ax1.imshow(bscan_v0, cmap='gray', aspect='auto')
ax1.set_title('V0 (Reference)', fontweight='bold')
ax1.set_ylabel('Y (depth)')
ax1.set_xlabel('X (lateral)')

ax2 = plt.subplot(3, 3, 2)
ax2.imshow(bscan_v1, cmap='gray', aspect='auto')
ax2.set_title('V1 (Before Alignment)', fontweight='bold')
ax2.set_xlabel('X (lateral)')

ax3 = plt.subplot(3, 3, 3)
ax3.imshow(bscan_v0, cmap='Reds', alpha=0.5, aspect='auto')
ax3.imshow(bscan_v1, cmap='Greens', alpha=0.5, aspect='auto')
ax3.set_title('Overlay BEFORE', fontweight='bold')
ax3.set_xlabel('X (lateral)')

# Row 2: Preprocessed images
ax4 = plt.subplot(3, 3, 4)
img1_proc = preprocess_oct(bscan_v0, mask=mask)
ax4.imshow(img1_proc, cmap='gray', aspect='auto')
ax4.set_title('V0 Preprocessed (layer-enhanced)', fontweight='bold')
ax4.set_ylabel('Y (depth)')
ax4.set_xlabel('X (lateral)')

ax5 = plt.subplot(3, 3, 5)
img2_proc = preprocess_oct(bscan_v1, mask=mask)
ax5.imshow(img2_proc, cmap='gray', aspect='auto')
ax5.set_title('V1 Preprocessed (layer-enhanced)', fontweight='bold')
ax5.set_xlabel('X (lateral)')

ax6 = plt.subplot(3, 3, 6)
ax6.imshow(img1_proc, cmap='Reds', alpha=0.5, aspect='auto')
ax6.imshow(img2_proc, cmap='Greens', alpha=0.5, aspect='auto')
ax6.set_title('Preprocessed Overlay', fontweight='bold')
ax6.set_xlabel('X (lateral)')

# Row 3: Aligned result
ax7 = plt.subplot(3, 3, 7)
ax7.imshow(bscan_v0, cmap='gray', aspect='auto')
ax7.set_title('V0 (Reference)', fontweight='bold')
ax7.set_ylabel('Y (depth)')
ax7.set_xlabel('X (lateral)')

ax8 = plt.subplot(3, 3, 8)
ax8.imshow(bscan_v1_aligned, cmap='gray', aspect='auto')
ax8.set_title(f'V1 (AFTER Alignment)\nRotation: {rotation_angle:+.2f}°', fontweight='bold')
ax8.set_xlabel('X (lateral)')

ax9 = plt.subplot(3, 3, 9)
ax9.imshow(bscan_v0, cmap='Reds', alpha=0.5, aspect='auto')
ax9.imshow(bscan_v1_aligned, cmap='Greens', alpha=0.5, aspect='auto')
ax9.set_title(f'Overlay AFTER\nNCC: {ncc_aligned:.4f}, Corr: {correlation:.4f}', fontweight='bold')
ax9.set_xlabel('X (lateral)')

plt.suptitle(f'Rotation-Only Alignment (ECC-style Correlation Scoring)\n' +
             f'Rotation: {rotation_angle:+.2f}° (no translation), ' +
             f'Correlation: {correlation:.6f}',
             fontsize=14, fontweight='bold')

plt.tight_layout()

# Save result
output_path = data_dir / 'ecc_alignment_result.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved visualization: {output_path}")
plt.close()

# Save rotation angle
np.save(data_dir / 'ecc_rotation_angle.npy', rotation_angle)
print(f"  ✓ Saved rotation angle")

print("\n" + "="*70)
print("✅ ROTATION-ONLY ALIGNMENT TEST COMPLETE!")
print("="*70)
print(f"\nEstimated rotation: {rotation_angle:+.2f}°")
print(f"Compare with visual expectation: -10° to -8°")
if -10 <= rotation_angle <= -8:
    print(f"✅ MATCHES visual expectation!")
elif -15 <= rotation_angle <= -5:
    print(f"⚠️  Close to visual expectation (within ±5°)")
else:
    print(f"❌ Differs from visual expectation")

print(f"\nTranslation: NONE (rotation around center only)")
print(f"NCC after alignment: {ncc_aligned:.4f}")
print(f"Correlation score: {correlation:.6f}")
print("="*70)
