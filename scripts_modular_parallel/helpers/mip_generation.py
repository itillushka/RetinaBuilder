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


def create_vessel_enhanced_mip(volume, verbose=True, denoise=False, vessels_only=False, vessel_threshold=0.1):
    """
    Create Vessel-Enhanced MIP using PARALLEL Frangi filter.

    This enhances tubular structures (blood vessels) for better registration.
    Uses joblib to parallelize across multiple sigma scales.

    Args:
        volume: 3D OCT volume (Y, X, Z)
        verbose: Print progress messages
        denoise: Apply bilateral denoising after Frangi filter (default: False)
        vessels_only: If True, threshold to keep only strong vessel responses (default: False)
        vessel_threshold: Threshold for vessel-only mode (0-1, default: 0.1)

    Returns:
        vessel_mip: 2D vessel-enhanced MIP (X, Z)
    """
    from skimage.restoration import denoise_bilateral
    from skimage.filters import threshold_otsu

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

    # Optional denoising (preserves edges while reducing noise)
    if denoise:
        if verbose:
            print(f"  Applying bilateral denoising (edge-preserving)...")
        vessels_enhanced = denoise_bilateral(
            vessels_enhanced,
            sigma_color=0.05,    # Color similarity (lower = more smoothing)
            sigma_spatial=2,     # Spatial distance (vessel structure scale)
            channel_axis=None
        )
        if verbose:
            print(f"  ✓ Denoising complete!")

    # VESSELS-ONLY mode: Threshold to keep only strong vessel responses
    if vessels_only:
        if verbose:
            print(f"  Applying vessel-only threshold (threshold={vessel_threshold})...")

        # Normalize to 0-1 for thresholding
        vessels_norm = (vessels_enhanced - vessels_enhanced.min()) / (vessels_enhanced.max() - vessels_enhanced.min() + 1e-8)

        # Apply threshold (keep only strong vessels)
        vessel_mask = vessels_norm > vessel_threshold

        # Mask out non-vessel regions (set to 0)
        vessels_enhanced = vessels_norm * vessel_mask

        # Count vessel pixels
        vessel_pixels = np.sum(vessel_mask)
        total_pixels = vessel_mask.size
        vessel_percentage = (vessel_pixels / total_pixels) * 100

        if verbose:
            print(f"  ✓ Vessel pixels: {vessel_pixels:,} / {total_pixels:,} ({vessel_percentage:.1f}%)")
            print(f"  ✓ Background suppressed (vessels-only mode)")

    # Normalize to 0-255
    vessels_final = ((vessels_enhanced - vessels_enhanced.min()) /
                     (vessels_enhanced.max() - vessels_enhanced.min() + 1e-8) * 255).astype(np.uint8)

    if verbose:
        print(f"  ✓ Frangi filter complete!")

    return vessels_final


def register_mip_phase_correlation_legacy(mip1, mip2, max_offset_x=None, max_offset_z=None):
    """Legacy single-scale phase correlation (kept for reference)."""
    """
    Register two MIP en-face images using FFT-based phase correlation.

    Much faster than spatial correlation (O(n log n) vs O(n²)).
    Uses optimized FFT libraries that can leverage multiple cores.

    Args:
        mip1: Reference MIP from Volume 0
        mip2: MIP to align from Volume 1
        max_offset_x: Maximum allowed X-axis offset (±pixels). None = no limit.
        max_offset_z: Maximum allowed Z-axis offset (±pixels). None = no limit.

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

    # Apply constraints if specified
    if max_offset_x is not None:
        offset_x = np.clip(offset_x, -max_offset_x, max_offset_x)
    if max_offset_z is not None:
        offset_z = np.clip(offset_z, -max_offset_z, max_offset_z)

    # Confidence = peak strength relative to noise
    confidence = correlation.max() / (correlation.std() + 1e-8)

    return (offset_x, offset_z), confidence, correlation


def register_mip_phase_correlation(mip1, mip2, max_offset_x=None, max_offset_z=None):
    """
    Multi-scale phase correlation with peak validation (MODERN METHOD).

    Addresses false peaks in repetitive vessel patterns by:
    1. Gaussian pyramid (coarse-to-fine optimization)
    2. Constrained search windows (not just post-clipping)
    3. Peak sharpness validation
    4. Secondary peak analysis

    Much more robust than single-scale correlation.

    Args:
        mip1: Reference MIP from Volume 0
        mip2: MIP to align from Volume 1
        max_offset_x: Maximum X-axis offset (±pixels). None = image_width/4
        max_offset_z: Maximum Z-axis offset (±pixels). None = image_depth/4

    Returns:
        (offset_x, offset_z): Translation offset
        confidence: Quality metric (higher = better, >3.0 is good)
        correlation: Full correlation map from finest scale
    """
    from scipy.ndimage import zoom

    # Default search limits (1/4 of image size)
    if max_offset_x is None:
        max_offset_x = mip1.shape[0] // 4
    if max_offset_z is None:
        max_offset_z = mip1.shape[1] // 4

    # Build Gaussian pyramid (3 scales: coarse to fine)
    scales = [0.25, 0.5, 1.0]
    cumulative_offset_x = 0.0
    cumulative_offset_z = 0.0

    print(f"  [Multi-scale] Searching with ±{max_offset_x}px X limit, ±{max_offset_z}px Z limit")

    for i, scale in enumerate(scales):
        # Downsample images
        if scale < 1.0:
            mip1_scaled = zoom(mip1, scale, order=1)
            mip2_scaled = zoom(mip2, scale, order=1)
        else:
            mip1_scaled = mip1
            mip2_scaled = mip2

        # Calculate search window at this scale
        if i == 0:  # Coarse scale - full search
            search_x = int(max_offset_x * scale)
            search_z = int(max_offset_z * scale)
        else:  # Finer scales - refine around previous estimate
            search_x = int(15 * scale)  # ±15px refinement window
            search_z = int(15 * scale)

        # Normalize
        mip1_norm = (mip1_scaled - mip1_scaled.mean()) / (mip1_scaled.std() + 1e-8)
        mip2_norm = (mip2_scaled - mip2_scaled.mean()) / (mip2_scaled.std() + 1e-8)

        # FFT correlation
        fft1 = np.fft.fft2(mip1_norm)
        fft2 = np.fft.fft2(mip2_norm)
        cross_power = fft1 * np.conj(fft2)
        correlation = np.fft.ifft2(cross_power).real
        correlation = np.fft.fftshift(correlation)

        # CONSTRAIN SEARCH REGION (not just clip result!)
        center_x, center_z = np.array(correlation.shape) // 2
        x_min = max(0, center_x - search_x)
        x_max = min(correlation.shape[0], center_x + search_x + 1)
        z_min = max(0, center_z - search_z)
        z_max = min(correlation.shape[1], center_z + search_z + 1)

        # Mask out regions outside search window
        correlation_windowed = np.copy(correlation)
        correlation_windowed[:x_min, :] = -np.inf
        correlation_windowed[x_max:, :] = -np.inf
        correlation_windowed[:, :z_min] = -np.inf
        correlation_windowed[:, z_max:] = -np.inf

        # Find peak
        peak_x, peak_z = np.unravel_index(
            np.argmax(correlation_windowed),
            correlation.shape
        )
        offset_x = peak_x - center_x
        offset_z = peak_z - center_z

        # PEAK VALIDATION
        peak_value = correlation[peak_x, peak_z]

        # 1. Peak sharpness (ratio to mean)
        valid_region = correlation_windowed[correlation_windowed > -np.inf]
        sharpness = peak_value / (valid_region.mean() + 1e-8)

        # 2. Secondary peak analysis
        correlation_copy = np.copy(correlation_windowed)
        # Suppress primary peak (5x5 region)
        x_start = max(0, peak_x - 2)
        x_end = min(correlation.shape[0], peak_x + 3)
        z_start = max(0, peak_z - 2)
        z_end = min(correlation.shape[1], peak_z + 3)
        correlation_copy[x_start:x_end, z_start:z_end] = -np.inf

        second_peak_value = np.max(correlation_copy[correlation_copy > -np.inf])
        peak_ratio = peak_value / (second_peak_value + 1e-8)

        # Combined confidence
        confidence = sharpness * (1 - 1/max(peak_ratio, 1.01))

        # Accumulate offset (scale back up to original resolution)
        cumulative_offset_x += offset_x / scale
        cumulative_offset_z += offset_z / scale

        scale_name = ['Coarse', 'Medium', 'Fine'][i]
        print(f"  [{scale_name:6s}] offset=({offset_x:+4.0f}, {offset_z:+4.0f}) "
              f"sharp={sharpness:.2f} ratio={peak_ratio:.2f} conf={confidence:.2f}")

    # Final offsets (from accumulation)
    final_offset_x = int(round(cumulative_offset_x))
    final_offset_z = int(round(cumulative_offset_z))

    # Final confidence check
    if confidence < 2.0:
        print(f"  [WARNING] Low confidence ({confidence:.2f}) - alignment may be unreliable!")

    return (final_offset_x, final_offset_z), confidence, correlation


def register_mip_itk_elastix(mip1, mip2, max_offset_x=None, max_offset_z=None, output_dir=None):
    """
    Modern ITK-Elastix registration for robust XZ alignment.

    State-of-the-art multi-resolution registration framework.
    Much more accurate than phase correlation for repetitive vessel patterns.

    Uses:
    - Multi-resolution pyramid (coarse-to-fine optimization)
    - Advanced normalized correlation metric
    - Adaptive stochastic gradient descent optimizer
    - Automatic parameter estimation for robust convergence
    - Built-in quality metrics and convergence detection

    Args:
        mip1: Reference MIP from Volume 0 (X, Z)
        mip2: MIP to align from Volume 1 (X, Z)
        max_offset_x: Maximum allowed X-axis offset (±pixels). None = no limit.
        max_offset_z: Maximum allowed Z-axis offset (±pixels). None = no limit.
        output_dir: Directory for saving elastix log files. None = temp directory.

    Returns:
        (offset_x, offset_z): Translation offset (lateral X, B-scan Z)
        confidence: Final metric value (higher = better alignment)
        correlation: Not used for Elastix (returns None for API compatibility)
    """
    try:
        import itk
    except ImportError:
        raise ImportError(
            "ITK-Elastix not available. Falling back to phase correlation.\n"
            "To use Elastix: pip install itk-elastix"
        )

    print(f"  [Elastix] Input MIP shapes: fixed={mip1.shape}, moving={mip2.shape}")

    # Validate input images (prevent silent crashes from bad data)
    if np.any(np.isnan(mip1)) or np.any(np.isnan(mip2)):
        raise ValueError("Input MIPs contain NaN values")
    if np.any(np.isinf(mip1)) or np.any(np.isinf(mip2)):
        raise ValueError("Input MIPs contain infinite values")
    if mip1.size == 0 or mip2.size == 0:
        raise ValueError("Input MIPs are empty")

    # Normalize images to 0-1 range for better registration
    mip1_norm = (mip1 - mip1.min()) / (mip1.max() - mip1.min() + 1e-8)
    mip2_norm = (mip2 - mip2.min()) / (mip2.max() - mip2.min() + 1e-8)

    # Convert to float32 (required by ITK)
    mip1_float = (mip1_norm * 255).astype(np.float32)
    mip2_float = (mip2_norm * 255).astype(np.float32)

    print(f"  [Elastix] Normalized ranges: fixed=[{mip1_float.min():.1f}, {mip1_float.max():.1f}], "
          f"moving=[{mip2_float.min():.1f}, {mip2_float.max():.1f}]")

    # WORKAROUND for Windows silent crash: Save to temp files instead of using image_from_array
    # The NumPy bridge can cause silent crashes on Windows
    if output_dir is not None:
        from pathlib import Path
        import tempfile

        temp_dir = Path(output_dir) / 'elastix_temp'
        temp_dir.mkdir(exist_ok=True)

        fixed_path = temp_dir / 'fixed_mip.nii'
        moving_path = temp_dir / 'moving_mip.nii'

        print(f"  [Elastix] Writing temporary files to avoid NumPy bridge crash...")

        # Create ITK images from arrays with explicit ordering
        fixed_image_temp = itk.image_from_array(np.ascontiguousarray(mip1_float))
        moving_image_temp = itk.image_from_array(np.ascontiguousarray(mip2_float))

        # Set spacing and origin
        fixed_image_temp.SetSpacing([1.0, 1.0])
        fixed_image_temp.SetOrigin([0.0, 0.0])
        moving_image_temp.SetSpacing([1.0, 1.0])
        moving_image_temp.SetOrigin([0.0, 0.0])

        # Write to disk
        itk.imwrite(fixed_image_temp, str(fixed_path))
        itk.imwrite(moving_image_temp, str(moving_path))

        # Read back from disk (avoids NumPy bridge issues)
        fixed_image = itk.imread(str(fixed_path), itk.F)
        moving_image = itk.imread(str(moving_path), itk.F)

        print(f"  [Elastix] ✓ Images loaded from disk")
    else:
        # Fallback: Use contiguous arrays to reduce crash risk
        print(f"  [Elastix] Converting NumPy arrays to ITK images...")
        fixed_image = itk.image_from_array(np.ascontiguousarray(mip1_float))
        moving_image = itk.image_from_array(np.ascontiguousarray(mip2_float))

    # Set proper spacing and origin (required by ITK)
    fixed_image.SetSpacing([1.0, 1.0])
    fixed_image.SetOrigin([0.0, 0.0])
    moving_image.SetSpacing([1.0, 1.0])
    moving_image.SetOrigin([0.0, 0.0])

    # Create parameter object
    parameter_object = itk.ParameterObject.New()
    parameter_map = parameter_object.GetDefaultParameterMap('translation')

    # ========================================================================
    # OPTIMIZED PARAMETERS (Based on Elastix Model Zoo and Best Practices)
    # ========================================================================

    # Image dimensions (explicit declaration for 2D)
    parameter_map['FixedImageDimension'] = ['2']
    parameter_map['MovingImageDimension'] = ['2']
    parameter_map['UseDirectionCosines'] = ['false']  # Not needed for 2D

    # Transform initialization
    parameter_map['AutomaticTransformInitialization'] = ['true']
    parameter_map['AutomaticTransformInitializationMethod'] = ['GeometricalCenter']

    # Metric
    parameter_map['Metric'] = ['AdvancedNormalizedCorrelation']
    parameter_map['SubtractMean'] = ['true']  # Narrower cost function valleys
    parameter_map['UseMultiThreadingForMetrics'] = ['false']  # Prevent Windows crashes

    # Optimizer (with automatic parameter estimation)
    parameter_map['Optimizer'] = ['AdaptiveStochasticGradientDescent']
    parameter_map['AutomaticParameterEstimation'] = ['true']  # Auto-calculate SP_a, SP_alpha
    parameter_map['UseAdaptiveStepSizes'] = ['true']  # Robust convergence
    parameter_map['MaximumNumberOfIterations'] = ['256', '256', '128']  # Per resolution

    # Multi-resolution pyramid
    parameter_map['NumberOfResolutions'] = ['3']
    parameter_map['ImagePyramidSchedule'] = ['4', '4', '2', '2', '1', '1']

    # Sampling strategy
    parameter_map['NumberOfSpatialSamples'] = ['2048']
    parameter_map['NewSamplesEveryIteration'] = ['true']

    # Interpolation
    parameter_map['Interpolator'] = ['BSplineInterpolator']
    parameter_map['BSplineInterpolationOrder'] = ['1']
    parameter_map['FinalBSplineInterpolationOrder'] = ['3']

    # Result image format
    parameter_map['ResultImagePixelType'] = ['float']
    parameter_map['ResultImageFormat'] = ['nii']

    # Add parameter map back to parameter object
    parameter_object.SetParameterMap(0, parameter_map)

    # Setup logging
    if output_dir is not None:
        from pathlib import Path
        log_dir = Path(output_dir) / 'elastix_logs'
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / 'elastix.log')
    else:
        log_dir = None
        log_file = None

    # Run registration with comprehensive logging
    print("  [Elastix] Running optimized multi-resolution registration...")
    print("  [Elastix] Expected time: 30-60 seconds")
    print("  [Elastix] Iterations: 256 (coarse) + 256 (medium) + 128 (fine)")
    print("  [Elastix] Starting registration NOW...")

    result_image = None
    result_transform_parameters = None

    try:
        if log_dir is not None:
            print(f"  [Elastix] Output directory: {log_dir}")
            result_image, result_transform_parameters = itk.elastix_registration_method(
                fixed_image,
                moving_image,
                parameter_object=parameter_object,
                log_to_console=True,
                log_to_file=True,
                output_directory=str(log_dir)
            )
            print(f"  [Elastix] Log saved to: {log_file}")
        else:
            print(f"  [Elastix] Running without file logging...")
            result_image, result_transform_parameters = itk.elastix_registration_method(
                fixed_image,
                moving_image,
                parameter_object=parameter_object,
                log_to_console=True
            )
    except SystemExit as e:
        print(f"  [Elastix] FATAL: Registration caused system exit: {e}")
        print(f"  [Elastix] This is likely a silent crash in elastix native code")
        raise RuntimeError("Elastix registration crashed (SystemExit)")
    except Exception as e:
        print(f"  [Elastix] Registration exception: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

    if result_image is None or result_transform_parameters is None:
        raise RuntimeError("Elastix returned None - registration failed silently")

    # Extract translation parameters
    transform_params = result_transform_parameters.GetTransformParameterMap(0)
    translation_params = transform_params['TransformParameters']

    # ITK uses (X, Y) indexing, we want (X, Z) for (lateral, depth)
    offset_x = float(translation_params[0])
    offset_z = float(translation_params[1])

    # Get final metric value as confidence (higher = better)
    final_metric_list = transform_params.get('FinalMetricValue', ['0'])
    final_metric = float(final_metric_list[0]) if final_metric_list else 0.0
    confidence = abs(final_metric)  # Normalized correlation, higher is better

    print(f"  [Elastix] ✓ Registration complete!")
    print(f"  [Elastix] Raw offset: dx={offset_x:.2f}, dz={offset_z:.2f}, metric={confidence:.4f}")

    # Apply constraints if specified
    if max_offset_x is not None:
        offset_x_clamped = np.clip(offset_x, -max_offset_x, max_offset_x)
        if abs(offset_x - offset_x_clamped) > 0.1:
            print(f"  [Elastix] WARNING: X offset {offset_x:.1f} clamped to ±{max_offset_x}")
        offset_x = offset_x_clamped

    if max_offset_z is not None:
        offset_z_clamped = np.clip(offset_z, -max_offset_z, max_offset_z)
        if abs(offset_z - offset_z_clamped) > 0.1:
            print(f"  [Elastix] WARNING: Z offset {offset_z:.1f} clamped to ±{max_offset_z}")
        offset_z = offset_z_clamped

    # Return offsets as integers (pixel-level alignment)
    return (int(round(offset_x)), int(round(offset_z))), confidence, None


def find_elastix_executable():
    """
    Find elastix command-line executable on the system.

    Returns:
        Path to elastix.exe if found, None otherwise
    """
    import shutil
    from pathlib import Path

    # Check common installation locations on Windows
    common_paths = [
        r"C:\Program Files\elastix\elastix.exe",
        r"C:\elastix\bin\elastix.exe",
        r"C:\elastix\elastix.exe",
        Path.home() / "elastix" / "elastix.exe",
    ]

    # Check PATH first
    elastix_path = shutil.which("elastix")
    if elastix_path:
        return elastix_path

    # Check common installation directories
    for path in common_paths:
        if Path(path).exists():
            return str(path)

    return None


def register_mip_elastix_cli(mip1, mip2, max_offset_x=None, max_offset_z=None, output_dir=None):
    """
    Register MIPs using elastix command-line tool via subprocess.

    This bypasses the ITK-Elastix Python binding which has silent crash issues on Windows.
    Uses the rock-solid elastix.exe command-line tool instead.

    Args:
        mip1: Reference MIP from Volume 0 (X, Z)
        mip2: MIP to align from Volume 1 (X, Z)
        max_offset_x: Maximum allowed X-axis offset (±pixels). None = no limit.
        max_offset_z: Maximum allowed Z-axis offset (±pixels). None = no limit.
        output_dir: Directory for saving elastix files. Required for CLI version.

    Returns:
        (offset_x, offset_z): Translation offset (lateral X, B-scan Z)
        confidence: Final metric value (higher = better alignment)
        correlation: Not used (returns None for API compatibility)
    """
    import subprocess
    import tempfile
    from pathlib import Path

    # Find elastix executable
    elastix_exe = find_elastix_executable()
    if elastix_exe is None:
        print("  [Elastix-CLI] ERROR: elastix.exe not found!")
        print("  [Elastix-CLI] Install from: https://elastix.dev/download.php")
        print("  [Elastix-CLI] Falling back to multi-scale phase correlation...")
        raise FileNotFoundError("elastix.exe not found on system")

    print(f"  [Elastix-CLI] Found elastix at: {elastix_exe}")

    # Validate inputs
    if np.any(np.isnan(mip1)) or np.any(np.isnan(mip2)):
        raise ValueError("Input MIPs contain NaN values")
    if np.any(np.isinf(mip1)) or np.any(np.isinf(mip2)):
        raise ValueError("Input MIPs contain infinite values")

    # Setup output directory
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="elastix_"))
    else:
        output_dir = Path(output_dir) / 'elastix_cli'
        output_dir.mkdir(exist_ok=True, parents=True)

    print(f"  [Elastix-CLI] Working directory: {output_dir}")

    # Normalize and convert images
    mip1_norm = (mip1 - mip1.min()) / (mip1.max() - mip1.min() + 1e-8)
    mip2_norm = (mip2 - mip2.min()) / (mip2.max() - mip2.min() + 1e-8)
    mip1_float = (mip1_norm * 255).astype(np.float32)
    mip2_float = (mip2_norm * 255).astype(np.float32)

    # Write images to disk (elastix reads from files)
    try:
        import itk
        fixed_image = itk.image_from_array(np.ascontiguousarray(mip1_float))
        moving_image = itk.image_from_array(np.ascontiguousarray(mip2_float))
        fixed_image.SetSpacing([1.0, 1.0])
        fixed_image.SetOrigin([0.0, 0.0])
        moving_image.SetSpacing([1.0, 1.0])
        moving_image.SetOrigin([0.0, 0.0])

        fixed_path = output_dir / 'fixed.mha'
        moving_path = output_dir / 'moving.mha'
        itk.imwrite(fixed_image, str(fixed_path))
        itk.imwrite(moving_image, str(moving_path))
    except:
        # Fallback: write as raw numpy if ITK fails
        fixed_path = output_dir / 'fixed.npy'
        moving_path = output_dir / 'moving.npy'
        np.save(fixed_path, mip1_float)
        np.save(moving_path, mip2_float)
        print("  [Elastix-CLI] WARNING: Using .npy format (ITK not available)")

    # Generate parameter file
    param_file = output_dir / 'parameters.txt'
    print(f"  [Elastix-CLI] Writing parameter file...")

    param_content = f"""// Elastix Parameter File for 2D Translation Registration
// Generated automatically for OCT volume alignment
// Configured for large displacements (~100px Z-axis)

(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(FixedImageDimension 2)
(MovingImageDimension 2)
(UseDirectionCosines "false")

// Transform (NO automatic initialization - allows finding large displacements)
(Transform "TranslationTransform")
(AutomaticTransformInitialization "false")
(AutomaticScalesEstimation "true")

// Metric
(Metric "AdvancedNormalizedCorrelation")
(SubtractMean "true")
(UseMultiThreadingForMetrics "false")

// Optimizer (configured for LARGE displacements up to 200px)
(Optimizer "AdaptiveStochasticGradientDescent")
(AutomaticParameterEstimation "true")
(UseAdaptiveStepSizes "true")
(MaximumNumberOfIterations 512 512 256)
(MaximumStepLength 200.0)
(MinimumStepLength 0.1)

// Multi-resolution
(Registration "MultiResolutionRegistration")
(NumberOfResolutions 3)
(ImagePyramidSchedule 4 4  2 2  1 1)

// Sampling (very conservative at coarse level for large displacements)
(ImageSampler "Random")
(NumberOfSpatialSamples 512 2048 4096)
(NewSamplesEveryIteration "true")
(MaximumNumberOfSamplingAttempts 5)
(RequiredRatioOfValidSamples 0.10)

// Interpolation
(Interpolator "BSplineInterpolator")
(BSplineInterpolationOrder 1)
(FinalBSplineInterpolationOrder 3)

// Output
(WriteResultImage "false")
(ResultImagePixelType "float")
(ResultImageFormat "mha")
"""

    with open(param_file, 'w') as f:
        f.write(param_content)

    # Run elastix
    print(f"  [Elastix-CLI] Running registration (timeout: 90s)...")
    print(f"  [Elastix-CLI] Configured for large displacements (up to 200px)")
    print(f"  [Elastix-CLI] Iterations: 512 (coarse) + 512 (medium) + 256 (fine)")

    cmd = [
        elastix_exe,
        '-f', str(fixed_path),
        '-m', str(moving_path),
        '-out', str(output_dir),
        '-p', str(param_file)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=90,
            cwd=str(output_dir)
        )

        # Check for errors
        if result.returncode != 0:
            print(f"  [Elastix-CLI] ERROR: elastix returned code {result.returncode}")
            print(f"  [Elastix-CLI] stderr: {result.stderr[:500]}")
            raise RuntimeError(f"Elastix failed with code {result.returncode}")

        print(f"  [Elastix-CLI] ✓ Registration complete!")

    except subprocess.TimeoutExpired:
        print(f"  [Elastix-CLI] ERROR: Registration timeout after 90s")
        raise RuntimeError("Elastix registration timeout")

    # Parse TransformParameters.txt
    transform_file = output_dir / 'TransformParameters.0.txt'
    if not transform_file.exists():
        raise RuntimeError(f"Transform file not found: {transform_file}")

    print(f"  [Elastix-CLI] Parsing results from: {transform_file.name}")

    with open(transform_file, 'r') as f:
        content = f.read()

    # Extract translation parameters
    import re

    # Find TransformParameters line
    match = re.search(r'\(TransformParameters\s+([-\d.]+)\s+([-\d.]+)\)', content)
    if not match:
        raise RuntimeError("Could not parse TransformParameters from output")

    offset_x = float(match.group(1))
    offset_z = float(match.group(2))

    # Extract final metric value
    match_metric = re.search(r'\(FinalMetricValue\s+([-\d.]+)\)', content)
    confidence = float(match_metric.group(1)) if match_metric else 0.0
    confidence = abs(confidence)  # Normalized correlation, higher is better

    print(f"  [Elastix-CLI] Raw offset: dx={offset_x:.2f}, dz={offset_z:.2f}, metric={confidence:.4f}")

    # Apply constraints if specified
    if max_offset_x is not None:
        offset_x_clamped = np.clip(offset_x, -max_offset_x, max_offset_x)
        if abs(offset_x - offset_x_clamped) > 0.1:
            print(f"  [Elastix-CLI] WARNING: X offset {offset_x:.1f} clamped to ±{max_offset_x}")
        offset_x = offset_x_clamped

    if max_offset_z is not None:
        offset_z_clamped = np.clip(offset_z, -max_offset_z, max_offset_z)
        if abs(offset_z - offset_z_clamped) > 0.1:
            print(f"  [Elastix-CLI] WARNING: Z offset {offset_z:.1f} clamped to ±{max_offset_z}")
        offset_z = offset_z_clamped

    return (int(round(offset_x)), int(round(offset_z))), confidence, None


def register_mip_ants(mip1, mip2, max_offset_x=None, max_offset_z=None):
    """
    Register MIPs using ANTs (Advanced Normalization Tools) translation registration.

    ANTs is the gold standard for medical image registration, highly robust for
    large displacements through multi-resolution pyramid optimization.

    Args:
        mip1: Reference MIP from Volume 0 (X, Z)
        mip2: MIP to align from Volume 1 (X, Z)
        max_offset_x: Maximum allowed X-axis offset (±pixels). None = no limit.
        max_offset_z: Maximum allowed Z-axis offset (±pixels). None = no limit.

    Returns:
        (offset_x, offset_z): Translation offset (integers)
        confidence: Normalized cross-correlation (higher = better)
        correlation: None (for API compatibility)
    """
    try:
        import ants
    except ImportError:
        raise ImportError(
            "ANTsPy not available. Install with: pip install antspyx\n"
            "ANTs is separate from itk-elastix and may be more robust."
        )

    print(f"  [ANTs] Input MIP shapes: fixed={mip1.shape}, moving={mip2.shape}")

    # Validate inputs
    if np.any(np.isnan(mip1)) or np.any(np.isnan(mip2)):
        raise ValueError("Input MIPs contain NaN values")
    if np.any(np.isinf(mip1)) or np.any(np.isinf(mip2)):
        raise ValueError("Input MIPs contain infinite values")

    # Normalize to 0-255 range
    mip1_norm = ((mip1 - mip1.min()) / (mip1.max() - mip1.min() + 1e-8) * 255).astype(np.float32)
    mip2_norm = ((mip2 - mip2.min()) / (mip2.max() - mip2.min() + 1e-8) * 255).astype(np.float32)

    print(f"  [ANTs] Normalized ranges: fixed=[{mip1_norm.min():.1f}, {mip1_norm.max():.1f}], "
          f"moving=[{mip2_norm.min():.1f}, {mip2_norm.max():.1f}]")

    # Convert numpy arrays to ANTs images
    fixed_img = ants.from_numpy(mip1_norm)
    moving_img = ants.from_numpy(mip2_norm)

    # Set proper spacing (1.0 = pixel units)
    fixed_img.set_spacing([1.0, 1.0])
    moving_img.set_spacing([1.0, 1.0])

    print("  [ANTs] Running translation-only registration...")
    print("  [ANTs] Multi-resolution pyramid: 4 levels (8x, 4x, 2x, 1x)")
    print("  [ANTs] Optimized for 100px+ displacements")

    # Perform registration
    result = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform='Translation',  # Pure translation only

        # Multi-resolution pyramid (4 levels for large search)
        aff_iterations=(2000, 1000, 500, 250),  # More iterations at coarse levels
        aff_shrink_factors=(8, 4, 2, 1),         # Standard pyramid
        aff_smoothing_sigmas=(4, 2, 1, 0),       # Gaussian smoothing

        # Metric
        aff_metric='meansquares',

        # Sampling
        aff_sampling=64,

        # Optimization
        grad_step=0.2,

        # Output
        verbose=False
    )

    print(f"  [ANTs] ✓ Registration complete!")

    # Extract transformation parameters
    transform_file = result['fwdtransforms'][0]

    # Read transform (ANTsPy API doesn't need dimension parameter)
    try:
        transform = ants.read_transform(transform_file)
    except Exception as e:
        # Fallback: try without dimension parameter or parse file directly
        print(f"  [ANTs] Warning: Could not read transform object: {e}")
        print(f"  [ANTs] Attempting to parse transform file directly...")

        # Parse the .mat file directly to extract translation
        import re
        with open(transform_file, 'r') as f:
            content = f.read()

        # Look for Parameters line in the transform file
        params_match = re.search(r'Parameters:\s+([-\d.e+]+)\s+([-\d.e+]+)', content)
        if params_match:
            offset_x = float(params_match.group(1))
            offset_z = float(params_match.group(2))
        else:
            # Try alternative format
            params_match = re.search(r'FixedParameters:\s+([-\d.e+]+)\s+([-\d.e+]+)', content)
            if params_match:
                offset_x = float(params_match.group(1))
                offset_z = float(params_match.group(2))
            else:
                raise RuntimeError("Could not parse translation parameters from transform file")
    else:
        # Successfully read transform object
        params = transform.parameters
        # Extract translation (ANTs uses [tx, ty] format)
        offset_x = float(params[0])
        offset_z = float(params[1])

    # Compute confidence from normalized cross-correlation
    warped_array = result['warpedmovout'].numpy()
    ncc = np.corrcoef(mip1_norm.flatten(), warped_array.flatten())[0, 1]
    confidence = abs(ncc)

    print(f"  [ANTs] Raw offset: dx={offset_x:.2f}, dz={offset_z:.2f}")
    print(f"  [ANTs] Normalized correlation: {confidence:.4f}")

    # Apply constraints if specified
    if max_offset_x is not None:
        offset_x_clamped = np.clip(offset_x, -max_offset_x, max_offset_x)
        if abs(offset_x - offset_x_clamped) > 0.1:
            print(f"  [ANTs] WARNING: X offset {offset_x:.1f} clamped to ±{max_offset_x}")
        offset_x = offset_x_clamped

    if max_offset_z is not None:
        offset_z_clamped = np.clip(offset_z, -max_offset_z, max_offset_z)
        if abs(offset_z - offset_z_clamped) > 0.1:
            print(f"  [ANTs] WARNING: Z offset {offset_z:.1f} clamped to ±{max_offset_z}")
        offset_z = offset_z_clamped

    return (int(round(offset_x)), int(round(offset_z))), confidence, None


def register_mip_feature_based(mip1, mip2, max_offset_x=None, max_offset_z=None,
                                method='sift', min_matches=5):
    """
    Register MIPs using feature-based matching (SIFT/ORB/AKAZE) with RANSAC.

    Detects keypoints, matches descriptors, uses RANSAC to find rigid transform.
    Robust to large displacements and repetitive patterns.

    Args:
        mip1: Reference MIP from Volume 0 (X, Z)
        mip2: MIP to align from Volume 1 (X, Z)
        max_offset_x: Maximum X-axis offset (±pixels). None = no limit.
        max_offset_z: Maximum Z-axis offset (±pixels). None = no limit.
        method: Feature detector ('sift', 'orb', 'akaze')
        min_matches: Minimum number of good matches required

    Returns:
        (offset_x, offset_z): Translation offset (integers)
        confidence: Match quality score (0-1, inlier ratio)
        correlation: None (for API compatibility)
    """
    import cv2

    print(f"  [Feature-{method.upper()}] Input MIP shapes: fixed={mip1.shape}, moving={mip2.shape}")

    # Convert to 8-bit grayscale (required by feature detectors)
    mip1_8u = ((mip1 - mip1.min()) / (mip1.max() - mip1.min() + 1e-8) * 255).astype(np.uint8)
    mip2_8u = ((mip2 - mip2.min()) / (mip2.max() - mip2.min() + 1e-8) * 255).astype(np.uint8)

    # Create feature detector
    if method.lower() == 'sift':
        detector = cv2.SIFT_create(
            nfeatures=5000,
            contrastThreshold=0.01,  # Lower for vessel patterns (more sensitive)
            edgeThreshold=20,        # Higher to accept more edge-like features
            sigma=1.6
        )
        print(f"  [SIFT] Using SIFT detector (optimized for vessel patterns)")
    elif method.lower() == 'orb':
        detector = cv2.ORB_create(
            nfeatures=5000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31
        )
        print(f"  [ORB] Using ORB detector (fast, free)")
    elif method.lower() == 'akaze':
        detector = cv2.AKAZE_create(
            threshold=0.0001,
            nOctaves=4,
            nOctaveLayers=4,
            diffusivity=cv2.KAZE_DIFF_PM_G2
        )
        print(f"  [AKAZE] Using AKAZE detector (good for medical images)")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sift', 'orb', or 'akaze'")

    # Detect keypoints and compute descriptors
    print(f"  [Feature] Detecting keypoints...")
    kp1, desc1 = detector.detectAndCompute(mip1_8u, None)
    kp2, desc2 = detector.detectAndCompute(mip2_8u, None)

    print(f"  [Feature] Found {len(kp1)} keypoints in reference")
    print(f"  [Feature] Found {len(kp2)} keypoints in moving")

    if len(kp1) < min_matches or len(kp2) < min_matches:
        raise RuntimeError(f"Not enough keypoints detected (need at least {min_matches})")

    # Create matcher
    print(f"  [Feature] Matching descriptors...")
    if method.lower() == 'orb':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # KNN matching (k=2 for ratio test)
    matches = matcher.knnMatch(desc1, desc2, k=2)

    # Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    print(f"  [Feature] Good matches after ratio test: {len(good_matches)}")

    if len(good_matches) < min_matches:
        raise RuntimeError(
            f"Not enough good matches ({len(good_matches)} < {min_matches}). "
            "Try lowering min_matches or using different detector."
        )

    # Extract matched point coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    print(f"  [Feature] Estimating transformation with RANSAC...")

    # RANSAC transformation estimation
    transform_matrix, inlier_mask = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=2000,
        confidence=0.995
    )

    if transform_matrix is None:
        raise RuntimeError("RANSAC failed to find transformation")

    # Count inliers
    inliers = np.sum(inlier_mask)
    inlier_ratio = inliers / len(good_matches)

    print(f"  [Feature] RANSAC inliers: {inliers}/{len(good_matches)} ({inlier_ratio*100:.1f}%)")

    # Extract translation (last column of 2x3 matrix)
    offset_x = float(transform_matrix[0, 2])
    offset_z = float(transform_matrix[1, 2])

    # Extract scale and rotation for diagnostics
    scale = np.sqrt(transform_matrix[0, 0]**2 + transform_matrix[0, 1]**2)
    rotation_deg = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0]) * 180 / np.pi

    print(f"  [Feature] Raw offset: dx={offset_x:.2f}, dz={offset_z:.2f}")
    print(f"  [Feature] Detected scale: {scale:.3f}x, rotation: {rotation_deg:.2f}°")

    if abs(scale - 1.0) > 0.1:
        print(f"  [Feature] WARNING: Significant scaling detected ({scale:.3f}x)")
    if abs(rotation_deg) > 5.0:
        print(f"  [Feature] WARNING: Significant rotation detected ({rotation_deg:.2f}°)")

    # Confidence scoring
    confidence = inlier_ratio
    if abs(scale - 1.0) > 0.05 or abs(rotation_deg) > 2.0:
        confidence *= 0.8

    print(f"  [Feature] Final confidence: {confidence:.3f}")

    # Apply constraints if specified
    if max_offset_x is not None:
        offset_x_clamped = np.clip(offset_x, -max_offset_x, max_offset_x)
        if abs(offset_x - offset_x_clamped) > 0.1:
            print(f"  [Feature] WARNING: X offset {offset_x:.1f} clamped to ±{max_offset_x}")
        offset_x = offset_x_clamped

    if max_offset_z is not None:
        offset_z_clamped = np.clip(offset_z, -max_offset_z, max_offset_z)
        if abs(offset_z - offset_z_clamped) > 0.1:
            print(f"  [Feature] WARNING: Z offset {offset_z:.1f} clamped to ±{max_offset_z}")
        offset_z = offset_z_clamped

    return (int(round(offset_x)), int(round(offset_z))), confidence, None


def compare_registration_methods(mip1, mip2, ground_truth=None, output_path=None):
    """
    Compare all available registration methods for benchmarking.

    Runs multiple registration methods on the same MIP pair and collects
    timing, accuracy, and confidence metrics for comparison.

    Args:
        mip1: Reference MIP from Volume 0 (X, Z)
        mip2: MIP to align from Volume 1 (X, Z)
        ground_truth: Optional (offset_x, offset_z) tuple for accuracy calculation
        output_path: Optional path to save comparison visualization

    Returns:
        results: Dictionary with metrics for each method:
            {method_name: {
                'offset_x': int,
                'offset_z': int,
                'confidence': float,
                'time': float (seconds),
                'success': bool,
                'error': str (if failed)
            }}
    """
    import time

    print("\n" + "=" * 70)
    print("REGISTRATION METHOD COMPARISON")
    print("=" * 70)
    print(f"MIP shapes: fixed={mip1.shape}, moving={mip2.shape}")
    if ground_truth:
        print(f"Ground truth offset: dx={ground_truth[0]}, dz={ground_truth[1]}")
    print("=" * 70)

    # Define methods to compare
    methods = [
        ('multiscale', lambda: register_mip_phase_correlation(mip1, mip2)),
        ('phase_corr', lambda: register_mip_phase_correlation_legacy(mip1, mip2)),
        ('ants', lambda: register_mip_ants(mip1, mip2)),
        ('sift', lambda: register_mip_feature_based(mip1, mip2, method='sift')),
        ('orb', lambda: register_mip_feature_based(mip1, mip2, method='orb')),
        ('akaze', lambda: register_mip_feature_based(mip1, mip2, method='akaze')),
    ]

    results = {}

    for method_name, method_func in methods:
        try:
            print(f"\n{'='*70}")
            print(f"Testing: {method_name.upper()}")
            print('='*70)

            start = time.time()
            (offset_x, offset_z), confidence, _ = method_func()
            elapsed = time.time() - start

            results[method_name] = {
                'offset_x': offset_x,
                'offset_z': offset_z,
                'confidence': confidence,
                'time': elapsed,
                'success': True
            }

            # Calculate accuracy if ground truth provided
            if ground_truth is not None:
                error_x = abs(offset_x - ground_truth[0])
                error_z = abs(offset_z - ground_truth[1])
                total_error = np.sqrt(error_x**2 + error_z**2)
                results[method_name]['error_x'] = error_x
                results[method_name]['error_z'] = error_z
                results[method_name]['total_error'] = total_error

            print(f"\n[{method_name.upper()}] Results:")
            print(f"  Offset: ({offset_x}, {offset_z})")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Time: {elapsed:.2f}s")
            if ground_truth:
                print(f"  Error: {total_error:.2f} pixels (X: {error_x:.2f}, Z: {error_z:.2f})")

        except Exception as e:
            print(f"\n[{method_name.upper()}] FAILED: {e}")
            results[method_name] = {
                'success': False,
                'error': str(e),
                'time': 0,
                'offset_x': 0,
                'offset_z': 0,
                'confidence': 0
            }

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<12} {'Success':<10} {'Z-Offset':<12} {'Confidence':<12} {'Time (s)':<10}")
    print("-" * 70)
    for method_name, result in results.items():
        if result['success']:
            print(f"{method_name.upper():<12} {'✓':<10} {result['offset_z']:<12} {result['confidence']:<12.3f} {result['time']:<10.2f}")
        else:
            print(f"{method_name.upper():<12} {'✗ FAILED':<10} {'-':<12} {'-':<12} {'-':<10}")

    # Visualize if output path provided
    if output_path:
        visualize_comparison_results(results, output_path, ground_truth)

    return results


def visualize_comparison_results(results, output_path='registration_comparison.png', ground_truth=None):
    """
    Create visualization comparing registration methods.

    Generates a 2x2 grid of bar charts showing:
    - Registration speed
    - Confidence scores
    - Z-displacement values
    - Accuracy (if ground truth provided)

    Args:
        results: Dictionary from compare_registration_methods()
        output_path: Path to save the figure
        ground_truth: Optional (offset_x, offset_z) for accuracy plot
    """
    import matplotlib.pyplot as plt

    # Filter successful methods
    methods = [k for k in results.keys() if results[k].get('success', False)]

    if not methods:
        print("  [Visualization] No successful methods to plot!")
        return

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Registration Method Comparison', fontsize=16, fontweight='bold')

    # Colors for bars (6 methods: multiscale, phase_corr, ants, sift, orb, akaze)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    # 1. Time comparison
    ax = axes[0, 0]
    times = [results[m]['time'] for m in methods]
    bars = ax.bar(range(len(methods)), times, color=colors[:len(methods)])
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Registration Speed (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels on bars
    for i, (bar, time) in enumerate(zip(bars, times)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time:.2f}s', ha='center', va='bottom', fontsize=9)

    # 2. Confidence comparison
    ax = axes[0, 1]
    confidences = [results[m]['confidence'] for m in methods]
    bars = ax.bar(range(len(methods)), confidences, color=colors[:len(methods)])
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title('Registration Confidence (Higher is Better)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (>0.7)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (>0.5)')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=8)
    # Add value labels
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{conf:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. Z-displacement comparison
    ax = axes[1, 0]
    offset_z = [results[m]['offset_z'] for m in methods]
    bars = ax.bar(range(len(methods)), offset_z, color=colors[:len(methods)])
    ax.set_ylabel('Z Offset (pixels)', fontsize=12)
    ax.set_title('Detected Z Displacement', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    if ground_truth:
        ax.axhline(y=ground_truth[1], color='red', linestyle='--', linewidth=2,
                   label=f'Ground Truth ({ground_truth[1]}px)')
        ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels
    for i, (bar, z) in enumerate(zip(bars, offset_z)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (2 if z > 0 else -5),
                f'{z}px', ha='center', va='bottom' if z > 0 else 'top', fontsize=9)

    # 4. Accuracy comparison (if ground truth available)
    ax = axes[1, 1]
    if ground_truth and 'total_error' in results[methods[0]]:
        errors = [results[m]['total_error'] for m in methods]
        bars = ax.bar(range(len(methods)), errors, color=colors[:len(methods)])
        ax.set_ylabel('Total Error (pixels)', fontsize=12)
        ax.set_title('Registration Accuracy (Lower is Better)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
        ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='Excellent (<5px)')
        ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Good (<10px)')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=8)
        # Add value labels
        for i, (bar, err) in enumerate(zip(bars, errors)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{err:.1f}px', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Ground Truth\nNot Available\n\n(Provide ground_truth parameter\nto enable accuracy comparison)',
                ha='center', va='center', fontsize=11, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax.set_title('Registration Accuracy', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison chart: {output_path}")
    plt.close()


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
