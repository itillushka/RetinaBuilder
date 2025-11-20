"""
MIP Generation and Registration Utilities

Functions for creating vessel-enhanced maximum intensity projections (MIPs)
and performing phase correlation registration.
"""

import numpy as np
from scipy import signal
from skimage import filters
from joblib import Parallel, delayed
from multiprocessing import cpu_count


def _frangi_single_scale(image, sigma):
    """
    Apply Frangi filter at a single scale.

    Worker function for parallel processing.

    Args:
        image: 2D normalized en-face image
        sigma: Scale parameter for Frangi filter

    Returns:
        Frangi response at this scale
    """
    return filters.frangi(image, sigmas=[sigma], black_ridges=False)


def create_vessel_enhanced_mip(volume, verbose=True):
    """
    Create Vessel-Enhanced MIP using PARALLEL Frangi filter.

    This enhances tubular structures (blood vessels) for better registration.
    Uses joblib to parallelize across multiple sigma scales.

    Args:
        volume: 3D OCT volume (Y, X, Z)
        verbose: Print progress messages

    Returns:
        vessel_mip: 2D vessel-enhanced MIP (X, Z)
    """
    # Average projection (better baseline than max)
    enface = np.mean(volume, axis=0)

    # Normalize
    enface_norm = (enface - enface.min()) / (enface.max() - enface.min() + 1e-8)

    # Enhance vessels using PARALLEL Frangi filter
    # Multiple scales to capture vessels of different sizes
    sigmas = list(range(1, 10, 2))  # [1, 3, 5, 7, 9]

    if verbose:
        print(f"  Applying Frangi filter with {len(sigmas)} scales in PARALLEL (using {cpu_count()} cores)...")

    # PARALLEL PROCESSING: Compute each sigma scale in parallel
    frangi_results = Parallel(n_jobs=cpu_count(), verbose=0)(
        delayed(_frangi_single_scale)(enface_norm, sigma) for sigma in sigmas
    )

    # Combine results (take maximum across scales)
    vessels_enhanced = np.maximum.reduce(frangi_results)

    # Normalize to 0-255
    vessels_final = ((vessels_enhanced - vessels_enhanced.min()) /
                     (vessels_enhanced.max() - vessels_enhanced.min() + 1e-8) * 255).astype(np.uint8)

    if verbose:
        print(f"  ✓ Frangi filter complete!")

    return vessels_final


def register_mip_phase_correlation(mip1, mip2):
    """
    Register two MIP en-face images using FFT-based phase correlation.

    Much faster than spatial correlation (O(n log n) vs O(n²)).
    Uses optimized FFT libraries that can leverage multiple cores.

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

    # FFT-based correlation (MUCH faster than signal.correlate2d)
    # This can leverage optimized FFT libraries (FFTW, MKL) with multi-threading
    fft1 = np.fft.fft2(mip1_norm)
    fft2 = np.fft.fft2(mip2_norm)

    # Cross-power spectrum
    cross_power = fft1 * np.conj(fft2)

    # Inverse FFT to get correlation
    correlation = np.fft.ifft2(cross_power).real

    # Shift zero-frequency component to center (equivalent to mode='same')
    correlation = np.fft.fftshift(correlation)

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
