"""
MIP Generation and Registration Utilities

Functions for creating vessel-enhanced maximum intensity projections (MIPs)
and performing phase correlation registration.
"""

import numpy as np
from scipy import signal
from skimage import filters


def create_vessel_enhanced_mip(volume):
    """
    Create Vessel-Enhanced MIP using Frangi filter.

    This enhances tubular structures (blood vessels) for better registration.

    Args:
        volume: 3D OCT volume (Y, X, Z)

    Returns:
        vessel_mip: 2D vessel-enhanced MIP (X, Z)
    """
    # Average projection (better baseline than max)
    enface = np.mean(volume, axis=0)

    # Normalize
    enface_norm = (enface - enface.min()) / (enface.max() - enface.min() + 1e-8)

    # Enhance vessels using Frangi filter
    # Multiple scales to capture vessels of different sizes
    vessels_enhanced = filters.frangi(enface_norm, sigmas=range(1, 10, 2), black_ridges=False)

    # Normalize to 0-255
    vessels_final = ((vessels_enhanced - vessels_enhanced.min()) /
                     (vessels_enhanced.max() - vessels_enhanced.min() + 1e-8) * 255).astype(np.uint8)

    return vessels_final


def register_mip_phase_correlation(mip1, mip2):
    """
    Register two MIP en-face images using phase correlation.

    This is the EXACT algorithm from the working notebook.

    Args:
        mip1: Reference MIP from Volume 0
        mip2: MIP to align from Volume 1

    Returns:
        (offset_x, offset_z): Translation offset (lateral X, B-scan Z)
        confidence: Match quality score
        correlation: Full correlation map
    """
    # Normalize images (remove mean and scale by std)
    mip1_norm = (mip1 - mip1.mean()) / (mip1.std() + 1e-8)
    mip2_norm = (mip2 - mip2.mean()) / (mip2.std() + 1e-8)

    # Compute 2D correlation
    correlation = signal.correlate2d(mip1_norm, mip2_norm, mode='same')

    # Find peak (strongest match position)
    peak_x, peak_z = np.unravel_index(np.argmax(correlation), correlation.shape)
    center_x, center_z = np.array(correlation.shape) // 2

    # Calculate offset from center
    offset_x = peak_x - center_x
    offset_z = peak_z - center_z

    # Confidence = peak strength relative to noise
    confidence = correlation.max() / (correlation.std() + 1e-8)

    return (offset_x, offset_z), confidence, correlation


def find_y_center(volume):
    """
    Find center of mass along Y axis.

    EXACT implementation from notebook Phase 5.

    Args:
        volume: 3D OCT volume (Y, X, Z)

    Returns:
        center_y: Center of mass coordinate (float)
    """
    y_profile = volume.sum(axis=(1, 2))
    y_coords = np.arange(len(y_profile))
    center = np.average(y_coords, weights=y_profile + 1e-8)
    return center
