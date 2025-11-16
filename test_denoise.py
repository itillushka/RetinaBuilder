#!/usr/bin/env python3
"""Test script to check denoised B-scan pixel values"""

import sys
import os

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_dir)

import numpy as np
import cv2
from pathlib import Path
from oct_volumetric_viewer import OCTVolumeLoader, OCTImageProcessor

# Load one B-scan
data_dir = Path('oct_data/emmetropes/EM001')
f001_dirs = sorted(data_dir.glob('F001*'))
f001_vols = [d for d in f001_dirs if d.is_dir()]
print(f'Found {len(f001_vols)} volumes: {[v.name for v in f001_vols]}')

processor = OCTImageProcessor(sidebar_width=250, crop_top=100, crop_bottom=50)
loader = OCTVolumeLoader(processor)
volume_0 = loader.load_volume_from_directory(str(f001_vols[0]))

bscan = volume_0[:, :, 40]
print(f'Original B-scan stats:')
print(f'  Shape: {bscan.shape}')
print(f'  Min: {bscan.min()}, Max: {bscan.max()}')
print(f'  Mean: {bscan.mean():.2f}')
print(f'  Non-zero pixels: {np.count_nonzero(bscan)} / {bscan.size} ({100*np.count_nonzero(bscan)/bscan.size:.1f}%)')

# Apply denoising
img_norm = ((bscan - bscan.min()) / (bscan.max() - bscan.min() + 1e-8) * 255).astype(np.uint8)
denoised = cv2.fastNlMeansDenoising(img_norm, h=25, templateWindowSize=7, searchWindowSize=21)
denoised = cv2.bilateralFilter(denoised, d=11, sigmaColor=150, sigmaSpace=150)
denoised = cv2.medianBlur(denoised, 15)
thresh_val = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
thresh_val = int(thresh_val * 0.5)
print(f'\nOtsu threshold (50%): {thresh_val}')
denoised[denoised < thresh_val] = 0
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
denoised = clahe.apply(denoised)

print(f'\nDenoised B-scan stats:')
print(f'  Shape: {denoised.shape}')
print(f'  Min: {denoised.min()}, Max: {denoised.max()}')
print(f'  Mean: {denoised.mean():.2f}')
print(f'  Non-zero pixels: {np.count_nonzero(denoised)} / {denoised.size} ({100*np.count_nonzero(denoised)/denoised.size:.1f}%)')
print(f'  Pixels > 50: {np.sum(denoised > 50)} ({100*np.sum(denoised > 50)/denoised.size:.1f}%)')
print(f'  Pixels > 100: {np.sum(denoised > 100)} ({100*np.sum(denoised > 100)/denoised.size:.1f}%)')

# Save denoised B-scan as image to visualize
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(img_norm, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Normalized (before harsh denoising)')
axes[1].imshow(denoised, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('After harsh denoising')
plt.tight_layout()
plt.savefig('notebooks/data/test_denoise.png', dpi=150)
print(f'\nSaved: notebooks/data/test_denoise.png')
