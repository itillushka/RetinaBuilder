"""
Five-Volume Two-Phase Alignment Pipeline

Aligns five OCT volumes in two phases around V1 (center anchor):
  Phase 1 (Right Chain): V2→V1, V3→V2 → merged_right (V1+V2+V3)
  Phase 2 (Left Chain):  V4→V1, V5→V4 → merged_left (V1+V4+V5)
  Phase 3 (Final Merge): Combine both chains with V1 as static anchor

Volume spatial order: V5 ← V4 ← V1 → V2 → V3

Usage:
    python five_volume_alignment.py --patient F001_IP
    python five_volume_alignment.py --patient EM005 --visual
"""

import numpy as np
import argparse
from pathlib import Path
import time
import json
from datetime import datetime
from multiprocessing import freeze_support

# Import helper modules
from helpers import (
    OCTImageProcessor,
    OCTVolumeLoader
)
from helpers.visualization_3d import (
    create_expanded_merged_volume,
    visualize_3d_multiangle,
    visualize_3d_comparison
)
from helpers.volume_transforms import apply_all_transformations_to_volume

# Import step modules for alignment functions
from steps.step1_xz_alignment import perform_xz_alignment
from steps.step2_y_alignment import perform_y_alignment
from steps.step3_rotation_z import perform_z_rotation_alignment


def merge_two_volumes(volume_ref, volume_mov_aligned, xz_results, y_results):
    """
    Merge two aligned volumes into a single expanded volume.

    Args:
        volume_ref: Reference volume
        volume_mov_aligned: Aligned moving volume (with transformations applied).
                           If None, will use y_results['volume_1_y_aligned']
        xz_results: XZ alignment results (offset_x, offset_z)
        y_results: Y alignment results (y_shift, volume_1_y_aligned)

    Returns:
        merged_volume: Merged volume containing both volumes
        metadata: Merge metadata
    """
    # Create transform dict for merge function
    transform_3d = {
        'dy': float(y_results['y_shift']),
        'dx': float(xz_results['offset_x']),
        'dz': float(xz_results['offset_z'])
    }

    print(f"\n  Merging with transforms: dy={transform_3d['dy']:.1f}, dx={transform_3d['dx']}, dz={transform_3d['dz']}")

    # Use provided aligned volume, or fall back to step results
    if volume_mov_aligned is None and 'volume_1_y_aligned' in y_results:
        volume_mov_aligned = y_results['volume_1_y_aligned']
        print(f"  [INFO] Using aligned volume from y_results (shape: {volume_mov_aligned.shape})")
    else:
        print(f"  [INFO] Using provided aligned volume (shape: {volume_mov_aligned.shape})")

    # Create merged volume
    merged_volume, metadata = create_expanded_merged_volume(
        volume_ref, volume_mov_aligned, transform_3d
    )

    return merged_volume, metadata


def align_volume_pair(ref_volume, mov_volume, position, data_dir, prefix, visualize=False):
    """
    Perform full alignment (XZ, Y, Rotation) between two volumes.

    Args:
        ref_volume: Reference volume
        mov_volume: Moving volume to align
        position: 'right' or 'left' for rotation axis selection
        data_dir: Output directory for results
        prefix: Prefix for saved files (e.g., 'phase1_v2_to_v1')
        visualize: Whether to generate visualizations

    Returns:
        xz_results, y_results, rotation_results, aligned_volume
    """
    # XZ alignment
    print(f"\n  [XZ] XZ Alignment...")
    xz_results = perform_xz_alignment(
        ref_volume, mov_volume,
        max_offset_z=15,
        method='phase_corr',
        output_dir=data_dir
    )
    print(f"    ✓ XZ Offset: dx={xz_results['offset_x']}, dz={xz_results['offset_z']}")
    print(f"    ✓ Confidence: {xz_results['confidence']:.3f}")

    # Y alignment (using CROPPED overlap region for calculation)
    print(f"\n  [Y] Y Alignment (overlap-cropped for calculation)...")
    y_results = perform_y_alignment(
        ref_volume,
        xz_results['volume_1_xz_aligned'],
        position=position,
        offset_x=xz_results['offset_x'],
        output_dir=data_dir,
        prefix=prefix
    )
    print(f"    ✓ Y Shift: {y_results['y_shift']:.2f} px")
    print(f"    ✓ Contour: {y_results['contour_y_offset']:.2f} px, NCC: {y_results['ncc_y_offset']:.2f} px")

    # Z-rotation alignment (using CROPPED overlap region for calculation)
    print(f"\n  [Rotation] Z-Rotation Alignment (position='{position}', overlap-cropped)...")
    rotation_results = perform_z_rotation_alignment(
        ref_volume,
        y_results['volume_1_y_aligned'],
        visualize=visualize,
        position=position,
        output_dir=data_dir,
        vis_interval=5,
        offset_x=xz_results['offset_x']
    )
    print(f"    ✓ Rotation angle: {rotation_results['rotation_angle']:+.3f}°")
    print(f"    ✓ NCC after rotation: {rotation_results['ncc_after']:.4f}")

    # No Step 3.1 Y-correction - Y alignment from Step 2 is sufficient
    rotation_results['y_shift_correction'] = 0.0

    # Apply transformations to original volume
    print(f"\n  [Transform] Applying all transformations...")
    aligned_volume = apply_all_transformations_to_volume(
        mov_volume,
        xz_results,
        y_results,
        step3_results=rotation_results
    )
    print(f"    ✓ Aligned volume shape: {aligned_volume.shape}")

    # Save results
    np.save(data_dir / f'{prefix}_xz_results.npy', xz_results)
    np.save(data_dir / f'{prefix}_y_results.npy', y_results)
    np.save(data_dir / f'{prefix}_rotation_results.npy', rotation_results)
    print(f"  [SAVED] {prefix}_*.npy")

    return xz_results, y_results, rotation_results, aligned_volume


def merge_all_five_volumes(volume_1, volume_2_aligned, volume_3_aligned,
                           volume_4_aligned, volume_5_aligned,
                           transforms_v2, transforms_v3,
                           transforms_v4, transforms_v5):
    """
    Merge all 5 volumes with V1 as static anchor at the center.

    Args:
        volume_1: Original V1 (static anchor, no transforms)
        volume_2_aligned, volume_3_aligned: Right chain aligned volumes
        volume_4_aligned, volume_5_aligned: Left chain aligned volumes
        transforms_v2, transforms_v3: Transform dicts for V2, V3
        transforms_v4, transforms_v5: Transform dicts for V4, V5

    Returns:
        final_merged: Merged volume containing all 5 volumes
        metadata: Merge metadata
    """
    print("\n  Computing canvas size for all 5 volumes...")

    h1, w1, d1 = volume_1.shape

    # Get all transform offsets
    all_offsets = [
        transforms_v2, transforms_v3,
        transforms_v4, transforms_v5
    ]

    # Calculate required canvas expansion
    min_dy = min(0, min(t['dy'] for t in all_offsets))
    max_dy = max(0, max(t['dy'] for t in all_offsets))
    min_dx = min(0, min(t['dx'] for t in all_offsets))
    max_dx = max(0, max(t['dx'] for t in all_offsets))
    min_dz = min(0, min(t['dz'] for t in all_offsets))
    max_dz = max(0, max(t['dz'] for t in all_offsets))

    # Expand canvas to fit all volumes
    expand_y_neg = abs(int(min_dy)) + 50  # Extra padding
    expand_y_pos = abs(int(max_dy)) + 50
    expand_x_neg = abs(int(min_dx)) + 50
    expand_x_pos = abs(int(max_dx)) + 50
    expand_z_neg = abs(int(min_dz)) + 50
    expand_z_pos = abs(int(max_dz)) + 50

    # Also account for aligned volume sizes
    for vol in [volume_2_aligned, volume_3_aligned, volume_4_aligned, volume_5_aligned]:
        h, w, d = vol.shape
        expand_y_pos = max(expand_y_pos, h - h1 + 50)
        expand_x_pos = max(expand_x_pos, w - w1 + 50)
        expand_z_pos = max(expand_z_pos, d - d1 + 50)

    new_h = h1 + expand_y_neg + expand_y_pos
    new_w = w1 + expand_x_neg + expand_x_pos
    new_d = d1 + expand_z_neg + expand_z_pos

    print(f"  Canvas size: ({new_h}, {new_w}, {new_d})")
    print(f"  V1 position: y=[{expand_y_neg}:{expand_y_neg+h1}], "
          f"x=[{expand_x_neg}:{expand_x_neg+w1}], "
          f"z=[{expand_z_neg}:{expand_z_neg+d1}]")

    # Create canvas and place V1 at center (static anchor)
    final_merged = np.zeros((new_h, new_w, new_d), dtype=np.float32)

    # Place V1 (static anchor - no transforms)
    final_merged[expand_y_neg:expand_y_neg+h1,
                 expand_x_neg:expand_x_neg+w1,
                 expand_z_neg:expand_z_neg+d1] = volume_1
    print(f"  ✓ Placed V1 (anchor)")

    # Helper to place aligned volume with offset relative to V1's position
    def place_volume(canvas, vol, transform, vol_name, v1_pos):
        """Place a volume on canvas with transform offset relative to V1."""
        h, w, d = vol.shape
        dy, dx, dz = int(transform['dy']), int(transform['dx']), int(transform['dz'])

        # Position relative to V1's position on canvas
        y_start = v1_pos[0] + dy
        x_start = v1_pos[1] + dx
        z_start = v1_pos[2] + dz

        # Clip to canvas bounds
        y_end = min(y_start + h, canvas.shape[0])
        x_end = min(x_start + w, canvas.shape[1])
        z_end = min(z_start + d, canvas.shape[2])

        y_start = max(0, y_start)
        x_start = max(0, x_start)
        z_start = max(0, z_start)

        vol_y_start = max(0, -dy - v1_pos[0])
        vol_x_start = max(0, -dx - v1_pos[1])
        vol_z_start = max(0, -dz - v1_pos[2])

        vol_y_end = vol_y_start + (y_end - y_start)
        vol_x_end = vol_x_start + (x_end - x_start)
        vol_z_end = vol_z_start + (z_end - z_start)

        # Blend with existing data (average in overlap)
        region = canvas[y_start:y_end, x_start:x_end, z_start:z_end]
        vol_region = vol[vol_y_start:vol_y_end, vol_x_start:vol_x_end, vol_z_start:vol_z_end]

        # Where both have data, average; otherwise take max
        mask_existing = region > 0
        mask_new = vol_region > 0
        overlap = mask_existing & mask_new

        result = region.copy()
        result[mask_new & ~mask_existing] = vol_region[mask_new & ~mask_existing]
        result[overlap] = (region[overlap] + vol_region[overlap]) / 2.0

        canvas[y_start:y_end, x_start:x_end, z_start:z_end] = result
        print(f"  ✓ Placed {vol_name} (offset: dy={dy}, dx={dx}, dz={dz})")

    v1_pos = (expand_y_neg, expand_x_neg, expand_z_neg)

    # Place right chain (V2, V3)
    place_volume(final_merged, volume_2_aligned, transforms_v2, "V2", v1_pos)
    place_volume(final_merged, volume_3_aligned, transforms_v3, "V3", v1_pos)

    # Place left chain (V4, V5)
    place_volume(final_merged, volume_4_aligned, transforms_v4, "V4", v1_pos)
    place_volume(final_merged, volume_5_aligned, transforms_v5, "V5", v1_pos)

    # Crop to non-zero bounds (memory-efficient method)
    print("\n  Cropping to tissue bounds...")
    # Find bounds along each axis without creating huge index arrays
    mask_y = np.any(final_merged > 0, axis=(1, 2))
    mask_x = np.any(final_merged > 0, axis=(0, 2))
    mask_z = np.any(final_merged > 0, axis=(0, 1))

    y_indices = np.where(mask_y)[0]
    x_indices = np.where(mask_x)[0]
    z_indices = np.where(mask_z)[0]

    if len(y_indices) > 0 and len(x_indices) > 0 and len(z_indices) > 0:
        y_min, y_max = y_indices[0], y_indices[-1] + 1
        x_min, x_max = x_indices[0], x_indices[-1] + 1
        z_min, z_max = z_indices[0], z_indices[-1] + 1

        final_merged = final_merged[y_min:y_max, x_min:x_max, z_min:z_max]
    print(f"  ✓ Final shape: {final_merged.shape}")

    metadata = {
        'total_voxels': np.sum(final_merged > 0),
        'shape': final_merged.shape
    }

    return final_merged, metadata


def main():
    """Five-volume two-phase alignment pipeline."""

    parser = argparse.ArgumentParser(
        description='Five-Volume Two-Phase Alignment Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Align 5 volumes for patient F001_IP
  python five_volume_alignment.py --patient F001_IP

  # Align 5 volumes for EM005 with visualization
  python five_volume_alignment.py --patient EM005 --visual
        """
    )

    parser.add_argument('--patient', type=str, default='F001_IP',
                        help='Patient ID to search for (e.g., F001_IP, EM005)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override OCT data directory (default: ../oct_data)')
    parser.add_argument('--visual', action='store_true',
                        help='Generate 3D visualizations')

    args = parser.parse_args()

    # Set up paths
    parent_dir = Path(__file__).parent.parent

    if args.data_dir:
        oct_data_dir = Path(args.data_dir)
    else:
        oct_data_dir = parent_dir / 'oct_data'
        if not oct_data_dir.exists():
            oct_data_dir = parent_dir / 'OCT_DATA'

    # Create patient-specific results directory
    patient_safe = args.patient.replace('_', '').replace('/', '').replace('\\', '')
    data_dir = Path(__file__).parent / f'results_5vol_{patient_safe.lower()}'
    data_dir.mkdir(exist_ok=True)

    # Start total pipeline timer
    pipeline_start = time.time()

    print("=" * 70)
    print("FIVE-VOLUME TWO-PHASE ALIGNMENT PIPELINE")
    print("=" * 70)
    print(f"Patient: {args.patient}")
    print(f"Output directory: {data_dir}")
    print(f"OCT data directory: {oct_data_dir}")
    print("\nVolume Order: V5 <- V4 <- V1 -> V2 -> V3")
    print("\nStrategy:")
    print("  Phase 1: Right Chain - V2→V1, V3→V2 → merged_right")
    print("  Phase 2: Left Chain  - V4→V1, V5→V4 → merged_left")
    print("  Phase 3: Final Merge - Combine with V1 as static anchor")

    # ========================================================================
    # STEP 0: LOAD FIVE VOLUMES
    # ========================================================================
    print("\n" + "=" * 70)
    print("LOADING FIVE OCT VOLUMES")
    print("=" * 70)

    loading_start = time.time()
    processor = OCTImageProcessor(sidebar_width=250, crop_top=100, crop_bottom=50)
    loader = OCTVolumeLoader(processor)

    # Find volume directories
    bmp_dirs = []
    for bmp_file in oct_data_dir.rglob('*.bmp'):
        vol_dir = bmp_file.parent
        if vol_dir not in bmp_dirs:
            bmp_dirs.append(vol_dir)

    # Filter volumes by patient ID
    patient_vols = sorted([v for v in bmp_dirs if args.patient in str(v)])

    if len(patient_vols) < 5:
        raise ValueError(f"Need at least 5 volumes for patient {args.patient}, found {len(patient_vols)}")

    print(f"  Found {len(patient_vols)} volumes for patient {args.patient}")
    print(f"  Volume 1 (center): {patient_vols[0].name}")
    print(f"  Volume 2 (right):  {patient_vols[1].name}")
    print(f"  Volume 3 (far right): {patient_vols[2].name}")
    print(f"  Volume 4 (left):   {patient_vols[3].name}")
    print(f"  Volume 5 (far left): {patient_vols[4].name}")

    volume_1 = loader.load_volume_from_directory(str(patient_vols[0]))
    volume_2 = loader.load_volume_from_directory(str(patient_vols[1]))
    volume_3 = loader.load_volume_from_directory(str(patient_vols[2]))
    volume_4 = loader.load_volume_from_directory(str(patient_vols[3]))
    volume_5 = loader.load_volume_from_directory(str(patient_vols[4]))

    print(f"\n  V1 shape: {volume_1.shape}")
    print(f"  V2 shape: {volume_2.shape}")
    print(f"  V3 shape: {volume_3.shape}")
    print(f"  V4 shape: {volume_4.shape}")
    print(f"  V5 shape: {volume_5.shape}")

    loading_time = time.time() - loading_start
    print(f"\n  Volume loading time: {loading_time:.2f} seconds")

    # ========================================================================
    # PHASE 1: RIGHT CHAIN (V2→V1, V3→V2)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: RIGHT CHAIN ALIGNMENT")
    print("=" * 70)

    phase1_start = time.time()

    # Step 1.1: V2 → V1
    print("\n[1.1] ALIGN V2 → V1")
    print("-" * 50)
    xz_v2, y_v2, rot_v2, volume_2_aligned = align_volume_pair(
        volume_1, volume_2,
        position='right',
        data_dir=data_dir,
        prefix='phase1_v2_to_v1',
        visualize=args.visual
    )

    # Merge V1+V2 (for V3 alignment reference)
    print("\n[1.2] MERGE V1 + V2")
    print("-" * 50)
    merged_v1_v2, merge_meta_12 = merge_two_volumes(
        volume_1, volume_2_aligned, xz_v2, y_v2
    )
    print(f"  ✓ Merged shape: {merged_v1_v2.shape}")
    np.save(data_dir / 'merged_v1_v2.npy', merged_v1_v2)

    # Step 1.3: V3 → V2
    print("\n[1.3] ALIGN V3 → V2")
    print("-" * 50)
    xz_v3, y_v3, rot_v3, volume_3_aligned_v2 = align_volume_pair(
        volume_2, volume_3,
        position='right',
        data_dir=data_dir,
        prefix='phase1_v3_to_v2',
        visualize=args.visual
    )

    # Compute combined V3→V1 transforms
    print("\n[1.4] COMPUTE COMBINED V3→V1 TRANSFORMS")
    print("-" * 50)
    combined_v3_to_v1 = {
        'dx': xz_v3['offset_x'] + xz_v2['offset_x'],
        'dz': xz_v3['offset_z'] + xz_v2['offset_z'],
        'dy': y_v3['y_shift'] + y_v2['y_shift'],
        'rotation': rot_v3['rotation_angle'] + rot_v2['rotation_angle']
    }
    print(f"  V3→V2: dx={xz_v3['offset_x']}, dz={xz_v3['offset_z']}, dy={y_v3['y_shift']:.1f}, rot={rot_v3['rotation_angle']:+.2f}°")
    print(f"  V2→V1: dx={xz_v2['offset_x']}, dz={xz_v2['offset_z']}, dy={y_v2['y_shift']:.1f}, rot={rot_v2['rotation_angle']:+.2f}°")
    print(f"  Combined V3→V1: dx={combined_v3_to_v1['dx']}, dz={combined_v3_to_v1['dz']}, dy={combined_v3_to_v1['dy']:.1f}, rot={combined_v3_to_v1['rotation']:+.2f}°")

    # Apply combined transforms to V3
    combined_xz_v3 = {'offset_x': combined_v3_to_v1['dx'], 'offset_z': combined_v3_to_v1['dz'], 'confidence': xz_v3['confidence']}
    combined_y_v3 = {'y_shift': combined_v3_to_v1['dy'], 'contour_y_offset': combined_v3_to_v1['dy'], 'ncc_y_offset': combined_v3_to_v1['dy']}
    combined_rot_v3 = {'rotation_angle': combined_v3_to_v1['rotation'], 'ncc_after': rot_v3['ncc_after'], 'rotation_axes': rot_v3['rotation_axes']}

    volume_3_aligned = apply_all_transformations_to_volume(
        volume_3, combined_xz_v3, combined_y_v3, step3_results=combined_rot_v3
    )
    print(f"  ✓ V3 aligned to V1: {volume_3_aligned.shape}")

    # Create merged_right (V1+V2+V3)
    print("\n[1.5] CREATE MERGED_RIGHT (V1+V2+V3)")
    print("-" * 50)
    merged_right, merge_meta_right = merge_two_volumes(
        merged_v1_v2, volume_3_aligned, combined_xz_v3, combined_y_v3
    )
    print(f"  ✓ Merged right shape: {merged_right.shape}")
    np.save(data_dir / 'merged_right_v1_v2_v3.npy', merged_right)

    phase1_time = time.time() - phase1_start
    print(f"\n  Phase 1 time: {phase1_time:.2f} seconds")

    # ========================================================================
    # PHASE 2: LEFT CHAIN (V4→V1, V5→V4)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: LEFT CHAIN ALIGNMENT")
    print("=" * 70)

    phase2_start = time.time()

    # Step 2.1: V4 → V1
    print("\n[2.1] ALIGN V4 → V1")
    print("-" * 50)
    xz_v4, y_v4, rot_v4, volume_4_aligned = align_volume_pair(
        volume_1, volume_4,
        position='left',
        data_dir=data_dir,
        prefix='phase2_v4_to_v1',
        visualize=args.visual
    )

    # Merge V1+V4 (for V5 alignment reference)
    print("\n[2.2] MERGE V1 + V4")
    print("-" * 50)
    merged_v1_v4, merge_meta_14 = merge_two_volumes(
        volume_1, volume_4_aligned, xz_v4, y_v4
    )
    print(f"  ✓ Merged shape: {merged_v1_v4.shape}")
    np.save(data_dir / 'merged_v1_v4.npy', merged_v1_v4)

    # Step 2.3: V5 → V4
    print("\n[2.3] ALIGN V5 → V4")
    print("-" * 50)
    xz_v5, y_v5, rot_v5, volume_5_aligned_v4 = align_volume_pair(
        volume_4, volume_5,
        position='left',
        data_dir=data_dir,
        prefix='phase2_v5_to_v4',
        visualize=args.visual
    )

    # Compute combined V5→V1 transforms
    print("\n[2.4] COMPUTE COMBINED V5→V1 TRANSFORMS")
    print("-" * 50)
    combined_v5_to_v1 = {
        'dx': xz_v5['offset_x'] + xz_v4['offset_x'],
        'dz': xz_v5['offset_z'] + xz_v4['offset_z'],
        'dy': y_v5['y_shift'] + y_v4['y_shift'],
        'rotation': rot_v5['rotation_angle'] + rot_v4['rotation_angle']
    }
    print(f"  V5→V4: dx={xz_v5['offset_x']}, dz={xz_v5['offset_z']}, dy={y_v5['y_shift']:.1f}, rot={rot_v5['rotation_angle']:+.2f}°")
    print(f"  V4→V1: dx={xz_v4['offset_x']}, dz={xz_v4['offset_z']}, dy={y_v4['y_shift']:.1f}, rot={rot_v4['rotation_angle']:+.2f}°")
    print(f"  Combined V5→V1: dx={combined_v5_to_v1['dx']}, dz={combined_v5_to_v1['dz']}, dy={combined_v5_to_v1['dy']:.1f}, rot={combined_v5_to_v1['rotation']:+.2f}°")

    # Apply combined transforms to V5
    combined_xz_v5 = {'offset_x': combined_v5_to_v1['dx'], 'offset_z': combined_v5_to_v1['dz'], 'confidence': xz_v5['confidence']}
    combined_y_v5 = {'y_shift': combined_v5_to_v1['dy'], 'contour_y_offset': combined_v5_to_v1['dy'], 'ncc_y_offset': combined_v5_to_v1['dy']}
    combined_rot_v5 = {'rotation_angle': combined_v5_to_v1['rotation'], 'ncc_after': rot_v5['ncc_after'], 'rotation_axes': rot_v5['rotation_axes']}

    volume_5_aligned = apply_all_transformations_to_volume(
        volume_5, combined_xz_v5, combined_y_v5, step3_results=combined_rot_v5
    )
    print(f"  ✓ V5 aligned to V1: {volume_5_aligned.shape}")

    # Create merged_left (V1+V4+V5)
    print("\n[2.5] CREATE MERGED_LEFT (V1+V4+V5)")
    print("-" * 50)
    merged_left, merge_meta_left = merge_two_volumes(
        merged_v1_v4, volume_5_aligned, combined_xz_v5, combined_y_v5
    )
    print(f"  ✓ Merged left shape: {merged_left.shape}")
    np.save(data_dir / 'merged_left_v1_v4_v5.npy', merged_left)

    phase2_time = time.time() - phase2_start
    print(f"\n  Phase 2 time: {phase2_time:.2f} seconds")

    # ========================================================================
    # MEMORY CLEANUP BEFORE FINAL MERGE
    # ========================================================================
    print("\n" + "=" * 70)
    print("MEMORY CLEANUP")
    print("=" * 70)
    import gc

    # Delete original volumes (we have aligned versions saved to disk)
    del volume_2, volume_3, volume_4, volume_5
    # Delete intermediate merged volumes
    del merged_v1_v2, merged_v1_v4, merged_right, merged_left
    # Delete intermediate aligned volumes that aren't needed
    del volume_3_aligned_v2, volume_5_aligned_v4
    # Force garbage collection
    gc.collect()
    print("  ✓ Freed memory from original and intermediate volumes")

    # ========================================================================
    # PHASE 3: FINAL MERGE (V1 AS STATIC ANCHOR)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: FINAL MERGE (V1 AS STATIC ANCHOR)")
    print("=" * 70)

    phase3_start = time.time()

    # Prepare transform dicts for final merge
    transforms_v2 = {'dy': y_v2['y_shift'], 'dx': xz_v2['offset_x'], 'dz': xz_v2['offset_z']}
    transforms_v3 = {'dy': combined_v3_to_v1['dy'], 'dx': combined_v3_to_v1['dx'], 'dz': combined_v3_to_v1['dz']}
    transforms_v4 = {'dy': y_v4['y_shift'], 'dx': xz_v4['offset_x'], 'dz': xz_v4['offset_z']}
    transforms_v5 = {'dy': combined_v5_to_v1['dy'], 'dx': combined_v5_to_v1['dx'], 'dz': combined_v5_to_v1['dz']}

    final_merged, final_metadata = merge_all_five_volumes(
        volume_1,
        volume_2_aligned, volume_3_aligned,
        volume_4_aligned, volume_5_aligned,
        transforms_v2, transforms_v3,
        transforms_v4, transforms_v5
    )

    phase3_time = time.time() - phase3_start
    print(f"\n  Phase 3 time: {phase3_time:.2f} seconds")

    # Save final merged volume
    np.save(data_dir / 'final_merged_5volumes.npy', final_merged)
    print(f"  [SAVED] final_merged_5volumes.npy")

    # ========================================================================
    # SAVE TRANSFORMS TO JSON
    # ========================================================================
    print("\n" + "=" * 70)
    print("SAVING ALIGNMENT TRANSFORMS")
    print("=" * 70)

    # Build JSON structure with all transforms
    # These are DIRECT transforms for B-scans (no inversion needed)
    alignment_data = {
        "patient": args.patient,
        "timestamp": datetime.now().isoformat(),
        "output_dir": str(data_dir),
        "volumes": {
            "v1": {
                "path": str(patient_vols[0]),
                "role": "reference",
                "dx": 0,
                "dy": 0,
                "dz": 0,
                "rotation": 0.0
            },
            "v2": {
                "path": str(patient_vols[1]),
                "dx": int(xz_v2['offset_x']),
                "dy": float(y_v2['y_shift']),
                "dz": int(xz_v2['offset_z']),
                "rotation": float(rot_v2['rotation_angle'])
            },
            "v3": {
                "path": str(patient_vols[2]),
                "dx": int(combined_v3_to_v1['dx']),
                "dy": float(combined_v3_to_v1['dy']),
                "dz": int(combined_v3_to_v1['dz']),
                "rotation": float(combined_v3_to_v1['rotation'])
            },
            "v4": {
                "path": str(patient_vols[3]),
                "dx": int(xz_v4['offset_x']),
                "dy": float(y_v4['y_shift']),
                "dz": int(xz_v4['offset_z']),
                "rotation": float(rot_v4['rotation_angle'])
            },
            "v5": {
                "path": str(patient_vols[4]),
                "dx": int(combined_v5_to_v1['dx']),
                "dy": float(combined_v5_to_v1['dy']),
                "dz": int(combined_v5_to_v1['dz']),
                "rotation": float(combined_v5_to_v1['rotation'])
            }
        }
    }

    json_path = data_dir / 'alignment_transforms.json'
    with open(json_path, 'w') as f:
        json.dump(alignment_data, f, indent=2)
    print(f"  [SAVED] alignment_transforms.json")

    # Print summary of transforms
    print("\n  Transforms (relative to V1):")
    for vol_name, vol_data in alignment_data['volumes'].items():
        if vol_name == 'v1':
            print(f"    {vol_name.upper()}: reference (no transform)")
        else:
            print(f"    {vol_name.upper()}: dx={vol_data['dx']:+d}, dy={vol_data['dy']:+.1f}, "
                  f"dz={vol_data['dz']:+d}, rot={vol_data['rotation']:+.3f}deg")

    # ========================================================================
    # VISUALIZATION (OPTIONAL)
    # ========================================================================
    viz_time = 0
    if args.visual:
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        viz_start = time.time()

        # Final merged volume (multi-angle)
        print("\n  Generating final merged volume visualization...")
        visualize_3d_multiangle(
            final_merged,
            title="Five-Volume Merge (V5+V4+V1+V2+V3)",
            output_path=data_dir / '3d_merged_5volumes.png',
            subsample=8,
            percentile=75
        )
        print(f"  ✓ Saved: 3d_merged_5volumes.png")

        viz_time = time.time() - viz_start
        print(f"\n  Visualization time: {viz_time:.2f} seconds")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    total_time = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("FIVE-VOLUME TWO-PHASE ALIGNMENT COMPLETE!")
    print("=" * 70)
    print("\n  TIMING SUMMARY:")
    print("=" * 70)
    print(f"  Volume Loading:     {loading_time:>8.2f}s")
    print(f"  Phase 1 (Right):    {phase1_time:>8.2f}s")
    print(f"  Phase 2 (Left):     {phase2_time:>8.2f}s")
    print(f"  Phase 3 (Merge):    {phase3_time:>8.2f}s")
    if args.visual:
        print(f"  Visualization:      {viz_time:>8.2f}s")
    print(f"  {'-' * 30}")
    print(f"  TOTAL TIME:         {total_time:>8.2f}s ({total_time/60:.1f} min)")
    print("=" * 70)

    print(f"\n  RESULTS:")
    print(f"  Final merge (all 5):      {final_merged.shape}")
    print(f"  Output directory:         {data_dir}")

    print("\n  SAVED FILES:")
    print(f"  Phase 1:")
    print(f"    - phase1_v2_to_v1_*.npy")
    print(f"    - phase1_v3_to_v2_*.npy")
    print(f"    - merged_v1_v2.npy")
    print(f"    - merged_right_v1_v2_v3.npy")
    print(f"  Phase 2:")
    print(f"    - phase2_v4_to_v1_*.npy")
    print(f"    - phase2_v5_to_v4_*.npy")
    print(f"    - merged_v1_v4.npy")
    print(f"    - merged_left_v1_v4_v5.npy")
    print(f"  Final:")
    print(f"    - final_merged_5volumes.npy")
    print(f"    - alignment_transforms.json")
    if args.visual:
        print(f"  Visualizations:")
        print(f"    - 3d_merged_5volumes.png")

    print("\n  To generate B-scan panorama, run:")
    print(f"    python visualize_bscan_panorama.py {json_path}")


if __name__ == '__main__':
    freeze_support()
    main()
