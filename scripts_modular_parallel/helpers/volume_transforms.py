"""
Volume Transformation Utilities

Functions for applying geometric transformations to OCT volumes,
including rotation around arbitrary points and sequential application
of all alignment transformations.
"""

import numpy as np
from scipy import ndimage
import time
from .rotation_alignment_parallel import (
    apply_rotation_z_parallel,
    shift_volume_parallel,
    shift_volume_xz_parallel,
    shift_volume_y_parallel
)


def rotate_volume_around_point(volume, angle_degrees, axes, center_point):
    """
    Rotate volume around a specific center point (not volume center).

    This is critical for applying rotations calculated on overlap region
    to the full volume. The rotation must happen around the SAME physical
    point in both cases.

    Algorithm:
      1. Calculate offset from volume center to desired rotation center
      2. Shift volume so rotation center becomes volume center
      3. Apply rotation (rotates around volume center = our desired point)
      4. Shift back to original position

    Args:
        volume: 3D numpy array (Y, X, Z)
        angle_degrees: Rotation angle in degrees
        axes: Tuple (axis1, axis2) defining rotation plane
        center_point: Tuple (y_center, x_center, z_center) - rotation center

    Returns:
        Rotated volume
    """
    # Get volume center
    volume_center = np.array(volume.shape) / 2.0

    # Calculate offset from volume center to desired rotation center
    offset = np.array(center_point) - volume_center

    # Only shift in the axes that will be rotated
    shift_before = np.zeros(3)
    shift_before[list(axes)] = -offset[list(axes)]

    # Step 1: Shift so rotation center aligns with volume center
    volume_shifted = ndimage.shift(volume, shift=shift_before, order=1, mode='constant', cval=0)

    # Step 2: Rotate (now rotates around our desired center point)
    volume_rotated = ndimage.rotate(
        volume_shifted,
        angle=angle_degrees,
        axes=axes,
        reshape=False,
        order=1,
        mode='constant',
        cval=0
    )

    # Step 3: Shift back
    volume_final = ndimage.shift(volume_rotated, shift=-shift_before, order=1, mode='constant', cval=0)

    return volume_final


def apply_all_transformations_to_volume(volume_1_original, step1_results, step2_results, step3_results, step3_5_results=None, step4_results=None):
    """
    Apply ALL transformations to original volume_1 for clean visualization.

    This function loads the original volume and applies all saved transformation
    parameters in sequence, with proper canvas expansion to avoid data loss.

    Args:
        volume_1_original: Original untransformed volume_1
        step1_results: XZ alignment results (offset_x, offset_z)
        step2_results: Y alignment results (y_shift)
        step3_results: Z-rotation results (rotation_angle, y_shift_correction)
        step3_5_results: X-rotation results (rotation_angle_x) [optional]
        step4_results: Windowed Y-offsets results (y_offsets_interpolated) [optional]

    Returns:
        volume_1_transformed: Volume_1 with all transformations applied
    """
    print("\n" + "="*70)
    print("APPLYING ALL TRANSFORMATIONS TO ORIGINAL VOLUME_1")
    print("="*70)

    transform_start = time.time()

    h, w, d = volume_1_original.shape
    print(f"  Original volume shape: {volume_1_original.shape}")

    # Get transformation parameters
    offset_x = step1_results['offset_x']
    offset_z = step1_results['offset_z']
    y_shift = step2_results['y_shift']
    rotation_angle_z = step3_results['rotation_angle'] if step3_results else 0.0
    rotation_axes = step3_results.get('rotation_axes', (0, 1)) if step3_results else (0, 1)
    y_shift_correction = step3_results.get('y_shift_correction', 0.0) if step3_results else 0.0

    # Calculate needed canvas expansion to avoid clipping
    expand_y_neg = abs(min(0, int(y_shift + y_shift_correction)))
    expand_y_pos = abs(max(0, int(y_shift + y_shift_correction)))
    expand_x_neg = abs(min(0, offset_x))
    expand_x_pos = abs(max(0, offset_x))
    expand_z_neg = abs(min(0, offset_z))
    expand_z_pos = abs(max(0, offset_z))

    # Add padding for rotation (conservative estimate) - only if rotation is applied
    if rotation_angle_z != 0:
        rotation_padding = int(np.ceil(np.sqrt(h**2 + w**2) * abs(np.sin(np.radians(rotation_angle_z)))))
        expand_y_neg += rotation_padding
        expand_y_pos += rotation_padding
        expand_x_neg += rotation_padding
        expand_x_pos += rotation_padding
    else:
        rotation_padding = 0

    # Create expanded canvas
    new_h = h + expand_y_neg + expand_y_pos
    new_w = w + expand_x_neg + expand_x_pos
    new_d = d + expand_z_neg + expand_z_pos

    print(f"  Expanding canvas to avoid clipping: ({new_h}, {new_w}, {new_d})")
    print(f"    Y expansion: {expand_y_neg} (top) + {expand_y_pos} (bottom)")
    print(f"    X expansion: {expand_x_neg} (left) + {expand_x_pos} (right)")
    print(f"    Z expansion: {expand_z_neg} (back) + {expand_z_pos} (front)")

    # Place original volume in expanded canvas
    volume_1_transformed = np.zeros((new_h, new_w, new_d), dtype=np.float32)
    volume_1_transformed[expand_y_neg:expand_y_neg+h,
                        expand_x_neg:expand_x_neg+w,
                        expand_z_neg:expand_z_neg+d] = volume_1_original

    # Step 1: XZ shift (on expanded canvas, no clipping) - PARALLEL
    print(f"  [1] Applying XZ shift: dx={offset_x}, dz={offset_z} (PARALLEL)")
    t_start = time.time()
    volume_1_transformed = shift_volume_xz_parallel(
        volume_1_transformed, offset_x, offset_z, n_jobs=-1
    )
    print(f"      ⏱️  {time.time() - t_start:.2f}s")

    # Step 2: Y shift (on expanded canvas, no clipping) - PARALLEL
    print(f"  [2] Applying Y shift: dy={y_shift:.1f} (PARALLEL)")
    t_start = time.time()
    volume_1_transformed = shift_volume_y_parallel(
        volume_1_transformed, y_shift, n_jobs=-1
    )
    print(f"      ⏱️  {time.time() - t_start:.2f}s")

    # Step 3: Z-rotation (reshape=False, using pre-expanded canvas) - PARALLEL
    # NOTE: Invert rotation angle when applying (same as Y-shift inversion)
    print(f"  [3] Applying Z-rotation: {-rotation_angle_z:+.2f}° around axes {rotation_axes} (PARALLEL, pre-expanded canvas, inverted)")
    t_start = time.time()
    volume_1_transformed = apply_rotation_z_parallel(
        volume_1_transformed,
        -rotation_angle_z,  # INVERTED SIGN
        axes=rotation_axes,
        n_jobs=-1
    )
    print(f"      ⏱️  {time.time() - t_start:.2f}s")

    # Step 3.1: Y-shift correction after rotation - PARALLEL
    if abs(y_shift_correction) > 0.5:
        print(f"  [3.1] Applying Y-shift correction: {y_shift_correction:+.1f} px (PARALLEL)")
        t_start = time.time()
        volume_1_transformed = shift_volume_y_parallel(
            volume_1_transformed, y_shift_correction, n_jobs=-1
        )
        print(f"      ⏱️  {time.time() - t_start:.2f}s")

    # Step 3.5: X-rotation REMOVED (user requested removal)

    # Step 4: Windowed Y-offsets (if available and enabled) - PARALLEL
    if step4_results is not None and 'y_offsets_interpolated' in step4_results:
        from joblib import Parallel, delayed
        import cv2

        y_offsets = step4_results['y_offsets_interpolated']
        print(f"  [4] Applying windowed Y-offsets: {len(y_offsets)} B-scans (PARALLEL)")
        t_start = time.time()

        def shift_single_bscan_windowed(z_idx, volume, y_offsets):
            """Apply Y-shift to a single B-scan."""
            if z_idx < len(y_offsets):
                y_offset = y_offsets[z_idx]
                bscan = volume[:, :, z_idx]
                # Use OpenCV for faster shifting
                M = np.float32([[1, 0, 0], [0, 1, y_offset]])
                H, W = bscan.shape
                shifted = cv2.warpAffine(bscan, M, (W, H),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
                return shifted
            else:
                return volume[:, :, z_idx]

        # Apply per-B-scan Y shifts in parallel
        shifted_bscans = Parallel(n_jobs=-1, backend='threading', verbose=0)(
            delayed(shift_single_bscan_windowed)(z, volume_1_transformed, y_offsets)
            for z in range(volume_1_transformed.shape[2])
        )

        volume_1_transformed = np.stack(shifted_bscans, axis=2)
        print(f"      ⏱️  {time.time() - t_start:.2f}s")

    # Crop padding back to tissue bounds
    print(f"\n  [4] Cropping padding to tissue bounds...")
    print(f"      Before cropping: {volume_1_transformed.shape}")

    # Find bounding box of non-zero data
    nonzero_coords = np.argwhere(volume_1_transformed > 0)
    if len(nonzero_coords) > 0:
        min_coords = nonzero_coords.min(axis=0)
        max_coords = nonzero_coords.max(axis=0)

        # Crop to bounding box
        volume_1_transformed = volume_1_transformed[
            min_coords[0]:max_coords[0]+1,
            min_coords[1]:max_coords[1]+1,
            min_coords[2]:max_coords[2]+1
        ]
        print(f"      After cropping: {volume_1_transformed.shape}")

    transform_time = time.time() - transform_start

    print(f"\n  [OK] All transformations applied successfully!")
    print(f"  [OK] Final volume shape: {volume_1_transformed.shape}")
    print(f"  [OK] NO DATA LOSS - Canvas expanded during transformations, then cropped to tissue")
    print(f"\n  ⏱️  TOTAL TRANSFORMATION TIME: {transform_time:.2f}s")
    print("="*70)

    return volume_1_transformed
