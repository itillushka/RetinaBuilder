#!/usr/bin/env python3
"""
Test ORB Feature-Based Alignment for OCT B-scans

Uses OpenCV's ORB (Oriented FAST and Rotated BRIEF) feature detector
to find rotation + translation transformation between two B-scans.

This is a more robust approach than Hough transform because:
- ORB is rotation-invariant by design
- Feature matching handles both rotation and translation jointly
- RANSAC filtering removes outliers
- Directly estimates the transformation matrix

Usage:
    python test_orb_alignment.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

print("="*70)
print("ORB FEATURE-BASED B-SCAN ALIGNMENT TEST")
print("="*70)

# Setup paths
data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'
scripts_dir = Path(__file__).parent

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

# Convert to uint8 for OpenCV (required for ORB)
print("\n2. Converting to uint8, denoising, and enhancing contrast...")
# Normalize to 0-255 range
bscan_v0_norm = ((bscan_v0 - bscan_v0.min()) / (bscan_v0.max() - bscan_v0.min() + 1e-8) * 255).astype(np.uint8)
bscan_v1_norm = ((bscan_v1 - bscan_v1.min()) / (bscan_v1.max() - bscan_v1.min() + 1e-8) * 255).astype(np.uint8)

# Apply VERY STRICT noise removal to prevent ORB from detecting speckle noise
# Step 1: Non-local means denoising (very effective for OCT speckle noise)
print("  → Applying non-local means denoising...")
bscan_v0_denoised1 = cv2.fastNlMeansDenoising(bscan_v0_norm, h=10, templateWindowSize=7, searchWindowSize=21)
bscan_v1_denoised1 = cv2.fastNlMeansDenoising(bscan_v1_norm, h=10, templateWindowSize=7, searchWindowSize=21)

# Step 2: Bilateral filtering for additional edge-preserving smoothing
print("  → Applying bilateral filter...")
bscan_v0_denoised2 = cv2.bilateralFilter(bscan_v0_denoised1, d=9, sigmaColor=75, sigmaSpace=75)
bscan_v1_denoised2 = cv2.bilateralFilter(bscan_v1_denoised1, d=9, sigmaColor=75, sigmaSpace=75)

# Step 3: Median filter to remove any remaining salt-and-pepper noise
print("  → Applying median filter...")
bscan_v0_denoised = cv2.medianBlur(bscan_v0_denoised2, 5)
bscan_v1_denoised = cv2.medianBlur(bscan_v1_denoised2, 5)

# Step 4: Threshold to keep only tissue layers (zero out noise/background)
print("  → Thresholding to keep only tissue layers...")
# Use Otsu's method to find optimal threshold, but be more conservative
thresh_v0 = cv2.threshold(bscan_v0_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
thresh_v1 = cv2.threshold(bscan_v1_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
# Use 70% of Otsu threshold to be more conservative (keep more tissue)
thresh_v0 = int(thresh_v0 * 0.7)
thresh_v1 = int(thresh_v1 * 0.7)
print(f"     Threshold V0: {thresh_v0}, V1: {thresh_v1}")

bscan_v0_thresholded = bscan_v0_denoised.copy()
bscan_v1_thresholded = bscan_v1_denoised.copy()
bscan_v0_thresholded[bscan_v0_denoised < thresh_v0] = 0
bscan_v1_thresholded[bscan_v1_denoised < thresh_v1] = 0

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance features
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
bscan_v0_enhanced = clahe.apply(bscan_v0_thresholded)
bscan_v1_enhanced = clahe.apply(bscan_v1_thresholded)

# Apply mask to focus on retinal tissue
bscan_v0_masked = bscan_v0_enhanced.copy()
bscan_v1_masked = bscan_v1_enhanced.copy()
bscan_v0_masked[~mask] = 0
bscan_v1_masked[~mask] = 0

print(f"  ✓ Converted to uint8")
print(f"  ✓ Applied 4-stage preprocessing: denoise → threshold → CLAHE → mask")
print(f"  ✓ Only tissue layers retained (noise zeroed out)")

# Detect features using ORB
print("\n3. Detecting features with ORB...")
orb = cv2.ORB_create(
    nfeatures=10000,  # Max number of features
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=10
)

kp1, des1 = orb.detectAndCompute(bscan_v0_masked, mask=mask.astype(np.uint8))
kp2, des2 = orb.detectAndCompute(bscan_v1_masked, mask=mask.astype(np.uint8))

print(f"  ✓ Detected {len(kp1)} keypoints in V0")
print(f"  ✓ Detected {len(kp2)} keypoints in V1")

if len(kp1) < 10 or len(kp2) < 10:
    print("\n❌ ERROR: Not enough keypoints detected!")
    print("   Retinal layers may be too uniform for feature detection.")
    sys.exit(1)

# Match features
print("\n4. Matching features...")
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test to filter good matches
good_matches = []
for match_pair in matches:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

print(f"  ✓ Found {len(matches)} raw matches")
print(f"  ✓ Filtered to {len(good_matches)} good matches (Lowe's ratio test)")

if len(good_matches) < 10:
    print("\n❌ ERROR: Not enough good matches!")
    print("   Cannot reliably estimate transformation.")
    sys.exit(1)

# Extract matched keypoints
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Estimate affine transformation (rotation + translation + scale)
print("\n5. Estimating transformation with RANSAC...")
M, inlier_mask = cv2.estimateAffinePartial2D(
    pts2, pts1,
    method=cv2.RANSAC,
    ransacReprojThreshold=3.0,
    maxIters=2000,
    confidence=0.99
)

if M is None:
    print("\n❌ ERROR: Could not estimate transformation!")
    print("   RANSAC failed to find consensus.")
    sys.exit(1)

num_inliers = np.sum(inlier_mask)
print(f"  ✓ Transformation estimated")
print(f"  ✓ Inliers: {num_inliers}/{len(good_matches)} ({100*num_inliers/len(good_matches):.1f}%)")

# Extract rotation angle and translation from transformation matrix
# M = [[cos(θ)  -sin(θ)  tx],
#      [sin(θ)   cos(θ)  ty]]
rotation_angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
translation_x = M[0, 2]
translation_y = M[1, 2]
scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)

print(f"\n{'='*70}")
print(f"ESTIMATED TRANSFORMATION")
print(f"{'='*70}")
print(f"  Rotation:    {rotation_angle:+.2f}°")
print(f"  Translation: ({translation_y:+.1f}, {translation_x:+.1f}) px (Y, X)")
print(f"  Scale:       {scale:.4f}")
print(f"{'='*70}")

# Apply transformation to align B-scan V1
print("\n6. Applying transformation...")
h, w = bscan_v1.shape
bscan_v1_aligned = cv2.warpAffine(
    bscan_v1_norm,
    M,
    (w, h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0
)

print(f"  ✓ Applied transformation to V1")

# Calculate alignment quality
mask_aligned = cv2.warpAffine(
    mask.astype(np.uint8) * 255,
    M,
    (w, h),
    flags=cv2.INTER_NEAREST
) > 127

overlap_mask = mask & mask_aligned
if overlap_mask.sum() > 100:
    # Calculate NCC on aligned images
    v0_pixels = bscan_v0_norm[overlap_mask].astype(float)
    v1_pixels = bscan_v1_aligned[overlap_mask].astype(float)

    v0_norm = (v0_pixels - v0_pixels.mean()) / (v0_pixels.std() + 1e-8)
    v1_norm = (v1_pixels - v1_pixels.mean()) / (v1_pixels.std() + 1e-8)

    ncc_aligned = np.mean(v0_norm * v1_norm)
    print(f"  NCC after alignment: {ncc_aligned:.4f}")
else:
    ncc_aligned = -1.0
    print(f"  Warning: Not enough overlap to calculate NCC")

# Visualize results
print("\n7. Creating visualizations...")
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

# Row 2: Feature matching visualization
ax4 = plt.subplot(3, 3, 4)
# Draw keypoints
img_kp1 = cv2.drawKeypoints(bscan_v0_masked, kp1, None, color=(0, 255, 0), flags=0)
ax4.imshow(img_kp1)
ax4.set_title(f'V0 Keypoints ({len(kp1)})', fontweight='bold')
ax4.set_ylabel('Y (depth)')
ax4.set_xlabel('X (lateral)')

ax5 = plt.subplot(3, 3, 5)
img_kp2 = cv2.drawKeypoints(bscan_v1_masked, kp2, None, color=(0, 255, 0), flags=0)
ax5.imshow(img_kp2)
ax5.set_title(f'V1 Keypoints ({len(kp2)})', fontweight='bold')
ax5.set_xlabel('X (lateral)')

ax6 = plt.subplot(3, 3, 6)
# Draw matches (top 50 for visibility)
img_matches = cv2.drawMatches(
    bscan_v0_masked, kp1,
    bscan_v1_masked, kp2,
    good_matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
ax6.imshow(img_matches)
ax6.set_title(f'Feature Matches (top 50/{len(good_matches)})', fontweight='bold')
ax6.set_xlabel('Concatenated images')

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
ax9.set_title(f'Overlay AFTER\nNCC: {ncc_aligned:.4f}', fontweight='bold')
ax9.set_xlabel('X (lateral)')

plt.suptitle(f'ORB Feature-Based Alignment\n' +
             f'Rotation: {rotation_angle:+.2f}°, Translation: ({translation_y:+.1f}, {translation_x:+.1f}) px, ' +
             f'Inliers: {num_inliers}/{len(good_matches)}',
             fontsize=14, fontweight='bold')

plt.tight_layout()

# Save result
output_path = data_dir / 'orb_alignment_result.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved visualization: {output_path}")
plt.close()

# Save transformation matrix
np.save(data_dir / 'orb_transformation_matrix.npy', M)
print(f"  ✓ Saved transformation matrix")

print("\n" + "="*70)
print("✅ ORB ALIGNMENT TEST COMPLETE!")
print("="*70)
print(f"\nEstimated rotation: {rotation_angle:+.2f}°")
print(f"Compare with visual expectation: -10° to -8°")
if -10 <= rotation_angle <= -8:
    print(f"✅ MATCHES visual expectation!")
elif -15 <= rotation_angle <= -5:
    print(f"⚠️  Close to visual expectation (within ±5°)")
else:
    print(f"❌ Differs from visual expectation")

print(f"\nTranslation: ({translation_y:+.1f}, {translation_x:+.1f}) px")
print(f"NCC after alignment: {ncc_aligned:.4f}")
print(f"Inlier ratio: {100*num_inliers/len(good_matches):.1f}%")
print("="*70)
