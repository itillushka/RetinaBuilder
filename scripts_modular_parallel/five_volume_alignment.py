"""
Five-Volume Progressive Alignment Pipeline

Progressively aligns five OCT volumes with center volume as reference:
Layout: V5 - V4 - V1(center) - V2 - V3

Right side chain: V2 → V1, V3 → V2
Left side chain:  V4 → V1, V5 → V4

Final step places all volumes on common canvas with V1 as static reference.

Usage:
    python five_volume_alignment.py --patient F001_IP
    python five_volume_alignment.py --patient EM005 --visual
"""

import numpy as np
import argparse
from pathlib import Path
import time
import gc
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
from helpers.step_visualization import visualize_three_volume_mips

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


def place_volumes_on_canvas(volumes_dict, v1_position):
    """
    Place all aligned volumes on a common canvas with V1 as static reference.

    Args:
        volumes_dict: Dictionary with volume names as keys and (volume, position) tuples as values
                     Example: {'V1': (vol1, (0, 0, 0)), 'V2': (vol2, (dy, dx, dz)), ...}
        v1_position: Position of V1 in the canvas (usually (0, 0, 0))

    Returns:
        final_canvas: Merged volume with all 5 volumes placed
        metadata: Canvas metadata
    """
    print("\n  Calculating canvas size for all 5 volumes...")

    # Find canvas bounds
    min_y, max_y = 0, 0
    min_x, max_x = 0, 0
    min_z, max_z = 0, 0

    for vol_name, (volume, position) in volumes_dict.items():
        dy, dx, dz = position
        Y, X, Z = volume.shape

        min_y = min(min_y, dy)
        max_y = max(max_y, dy + Y)
        min_x = min(min_x, dx)
        max_x = max(max_x, dx + X)
        min_z = min(min_z, dz)
        max_z = max(max_z, dz + Z)

        print(f"    {vol_name}: shape={volume.shape}, position=({dy:.1f}, {dx}, {dz})")

    # Calculate canvas size
    canvas_Y = int(np.ceil(max_y - min_y))
    canvas_X = int(np.ceil(max_x - min_x))
    canvas_Z = int(np.ceil(max_z - min_z))

    print(f"\n  Canvas size: Y={canvas_Y}, X={canvas_X}, Z={canvas_Z}")

    # Create canvas
    canvas = np.zeros((canvas_Y, canvas_X, canvas_Z), dtype=np.float32)
    overlap_count = np.zeros((canvas_Y, canvas_X, canvas_Z), dtype=np.int32)

    # Place each volume on canvas
    print("\n  Placing volumes on canvas...")
    for vol_name, (volume, position) in volumes_dict.items():
        dy, dx, dz = position
        Y, X, Z = volume.shape

        # Calculate placement coordinates (shift by minimum to ensure all positive)
        start_y = int(np.round(dy - min_y))
        start_x = int(np.round(dx - min_x))
        start_z = int(np.round(dz - min_z))

        end_y = start_y + Y
        end_x = start_x + X
        end_z = start_z + Z

        # Place volume on canvas (accumulate for averaging in overlaps)
        canvas[start_y:end_y, start_x:end_x, start_z:end_z] += volume
        overlap_count[start_y:end_y, start_x:end_x, start_z:end_z] += 1

        print(f"    {vol_name}: placed at canvas[{start_y}:{end_y}, {start_x}:{end_x}, {start_z}:{end_z}]")

    # Average overlapping regions
    overlap_mask = overlap_count > 1
    canvas[overlap_mask] /= overlap_count[overlap_mask]

    # Calculate metadata
    total_voxels = np.sum(overlap_count > 0)
    overlap_voxels = np.sum(overlap_mask)

    metadata = {
        'canvas_shape': (canvas_Y, canvas_X, canvas_Z),
        'total_voxels': int(total_voxels),
        'overlap_voxels': int(overlap_voxels),
        'overlap_percent': float(overlap_voxels / total_voxels * 100) if total_voxels > 0 else 0.0
    }

    print(f"\n  ✓ Total voxels: {metadata['total_voxels']:,}")
    print(f"  ✓ Overlap voxels: {metadata['overlap_voxels']:,}")
    print(f"  ✓ Overlap: {metadata['overlap_percent']:.1f}%")

    return canvas, metadata


def main():
    """Five-volume progressive alignment pipeline."""

    parser = argparse.ArgumentParser(
        description='Five-Volume Progressive Alignment Pipeline',
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
        # Try oct_data first (lowercase), fall back to OCT_DATA
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
    print("FIVE-VOLUME PROGRESSIVE ALIGNMENT PIPELINE")
    print("=" * 70)
    print(f"Patient: {args.patient}")
    print(f"Output directory: {data_dir}")
    print(f"OCT data directory: {oct_data_dir}")
    print("\nLayout: V5 - V4 - V1(center) - V2 - V3")
    print("\nStrategy:")
    print("  RIGHT SIDE:")
    print("    1. Align V2 → V1, merge V1+V2")
    print("    2. Align V3 → V2, apply combined transforms (V3→V2+V2→V1)")
    print("    3. Create merged_V1V2V3")
    print("  LEFT SIDE:")
    print("    4. Align V4 → V1, merge V1+V4")
    print("    5. Align V5 → V4, apply combined transforms (V5→V4+V4→V1)")
    print("    6. Create merged_V1V4V5")
    print("  FINAL:")
    print("    7. Place all volumes on common canvas (V1 is static reference)")

    # ========================================================================
    # STEP 0: FIND VOLUME PATHS (Don't load all at once to save memory)
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINDING VOLUME PATHS")
    print("=" * 70)

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
    print(f"  Volume 1 (CENTER): {patient_vols[0].name}")
    print(f"  Volume 2 (right):  {patient_vols[1].name}")
    print(f"  Volume 3 (right):  {patient_vols[2].name}")
    print(f"  Volume 4 (left):   {patient_vols[3].name}")
    print(f"  Volume 5 (left):   {patient_vols[4].name}")
    print("\n  [Memory optimization] Loading volumes in phases to avoid memory issues")

    # ========================================================================
    # PHASE 1: LOAD RIGHT SIDE VOLUMES (V1, V2, V3)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: LOADING RIGHT SIDE VOLUMES (V1, V2, V3)")
    print("=" * 70)

    phase1_loading_start = time.time()

    volume_1 = loader.load_volume_from_directory(str(patient_vols[0]))
    volume_2 = loader.load_volume_from_directory(str(patient_vols[1]))
    volume_3 = loader.load_volume_from_directory(str(patient_vols[2]))

    print(f"  V1 shape: {volume_1.shape} (CENTER)")
    print(f"  V2 shape: {volume_2.shape}")
    print(f"  V3 shape: {volume_3.shape}")

    phase1_loading_time = time.time() - phase1_loading_start
    print(f"  ⏱️  Phase 1 loading time: {phase1_loading_time:.2f} seconds")

    # ========================================================================
    # RIGHT SIDE: ALIGN V2 → V1 → V3
    # ========================================================================
    print("\n" + "=" * 70)
    print("RIGHT SIDE ALIGNMENT: V2 → V1 → V3")
    print("=" * 70)

    # ========================================================================
    # STEP 1: ALIGN VOLUME 2 → VOLUME 1
    # ========================================================================
    print("\n" + "─" * 70)
    print("STEP 1: ALIGN VOLUME 2 → VOLUME 1")
    print("─" * 70)

    step1_start = time.time()

    # XZ alignment (volume 2 to volume 1)
    print("\n[1.1] XZ Alignment (Volume 2 → Volume 1)...")
    xz_results_v2 = perform_xz_alignment(volume_1, volume_2,
                                         max_offset_z=15,  # Limit Z-axis to ±15px
                                         method='phase_corr',  # Classic phase correlation
                                         output_dir=data_dir)
    print(f"  ✓ XZ Offset: dx={xz_results_v2['offset_x']}, dz={xz_results_v2['offset_z']}")
    print(f"  ✓ Confidence: {xz_results_v2['confidence']:.3f}")

    # Y alignment (volume 2 to volume 1)
    print("\n[1.2] Y Alignment (Volume 2 → Volume 1)...")
    y_results_v2 = perform_y_alignment(
        volume_1,
        xz_results_v2['volume_1_xz_aligned']
    )
    print(f"  ✓ Y Shift: {y_results_v2['y_shift']:.2f} px")

    # Z-rotation alignment (volume 2 to volume 1)
    print("\n[1.3] Z-Rotation Alignment (Volume 2 → Volume 1)...")
    step3_results_v2 = perform_z_rotation_alignment(
        volume_1,
        y_results_v2['volume_1_y_aligned'],
        visualize=args.visual,
        position='right',  # V2 is to the right of V1
        output_dir=data_dir,
        vis_interval=5
    )
    print(f"  ✓ Rotation angle: {step3_results_v2['rotation_angle']:+.3f}°")

    step1_time = time.time() - step1_start
    print(f"\n⏱️  Step 1 time: {step1_time:.2f} seconds")

    # Apply transformations to V2
    print("\n  Applying transformations to volume_2...")
    volume_2_aligned = apply_all_transformations_to_volume(
        volume_2,
        xz_results_v2,
        y_results_v2,
        step3_results=step3_results_v2
    )
    print(f"  ✓ Volume 2 transformed: {volume_2_aligned.shape}")

    # Merge V1 + V2
    print("\n  Merging V1 + V2...")
    merged_v1_v2, merge_metadata_12 = merge_two_volumes(
        volume_1,
        volume_2_aligned,
        xz_results_v2,
        y_results_v2
    )
    print(f"  ✓ Merged V1+V2 shape: {merged_v1_v2.shape}")

    # ========================================================================
    # STEP 2: ALIGN VOLUME 3 → VOLUME 2
    # ========================================================================
    print("\n" + "─" * 70)
    print("STEP 2: ALIGN VOLUME 3 → VOLUME 2")
    print("─" * 70)

    step2_start = time.time()

    # XZ alignment (volume 3 to volume 2)
    print("\n[2.1] XZ Alignment (Volume 3 → Volume 2)...")
    xz_results_v3 = perform_xz_alignment(volume_2, volume_3,
                                         max_offset_z=15,
                                         method='phase_corr',
                                         output_dir=data_dir)
    print(f"  ✓ XZ Offset: dx={xz_results_v3['offset_x']}, dz={xz_results_v3['offset_z']}")

    # Y alignment (volume 3 to volume 2)
    print("\n[2.2] Y Alignment (Volume 3 → Volume 2)...")
    y_results_v3 = perform_y_alignment(
        volume_2,
        xz_results_v3['volume_1_xz_aligned']
    )
    print(f"  ✓ Y Shift: {y_results_v3['y_shift']:.2f} px")

    # Z-rotation alignment (volume 3 to volume 2)
    print("\n[2.3] Z-Rotation Alignment (Volume 3 → Volume 2)...")
    step3_results_v3 = perform_z_rotation_alignment(
        volume_2,
        y_results_v3['volume_1_y_aligned'],
        visualize=args.visual,
        position='right',  # V3 is to the right of V2
        output_dir=data_dir,
        vis_interval=5
    )
    print(f"  ✓ Rotation angle: {step3_results_v3['rotation_angle']:+.3f}°")

    step2_time = time.time() - step2_start
    print(f"\n⏱️  Step 2 time: {step2_time:.2f} seconds")

    # Compute combined transforms for V3 (V3→V2 + V2→V1)
    print("\n  Computing combined transforms (V3→V2 + V2→V1)...")
    combined_offset_x_v3 = xz_results_v3['offset_x'] + xz_results_v2['offset_x']
    combined_offset_z_v3 = xz_results_v3['offset_z'] + xz_results_v2['offset_z']
    combined_y_shift_v3 = y_results_v3['y_shift'] + y_results_v2['y_shift']
    combined_rotation_v3 = step3_results_v3['rotation_angle'] + step3_results_v2['rotation_angle']

    print(f"    V3→V2: dx={xz_results_v3['offset_x']}, dz={xz_results_v3['offset_z']}, dy={y_results_v3['y_shift']:.1f}, rot={step3_results_v3['rotation_angle']:+.3f}°")
    print(f"    V2→V1: dx={xz_results_v2['offset_x']}, dz={xz_results_v2['offset_z']}, dy={y_results_v2['y_shift']:.1f}, rot={step3_results_v2['rotation_angle']:+.3f}°")
    print(f"    Combined: dx={combined_offset_x_v3}, dz={combined_offset_z_v3}, dy={combined_y_shift_v3:.1f}, rot={combined_rotation_v3:+.3f}°")

    # Apply combined transforms to V3
    print("\n  Applying combined transforms to volume_3...")
    combined_xz_results_v3 = {
        'offset_x': combined_offset_x_v3,
        'offset_z': combined_offset_z_v3,
        'confidence': xz_results_v3['confidence']
    }
    combined_y_results_v3 = {
        'y_shift': combined_y_shift_v3,
        'contour_y_offset': combined_y_shift_v3,
        'ncc_y_offset': combined_y_shift_v3
    }
    combined_rotation_results_v3 = {
        'rotation_angle': combined_rotation_v3,
        'ncc_after': step3_results_v3['ncc_after'],
        'rotation_axes': step3_results_v3['rotation_axes']
    }

    volume_3_aligned = apply_all_transformations_to_volume(
        volume_3,
        combined_xz_results_v3,
        combined_y_results_v3,
        step3_results=combined_rotation_results_v3
    )
    print(f"  ✓ Volume 3 transformed (combined): {volume_3_aligned.shape}")

    # Merge into final right side volume
    print("\n  Merging V3 into right side (V1+V2+V3)...")
    merged_v1_v2_v3, merge_metadata_123 = merge_two_volumes(
        merged_v1_v2,
        volume_3_aligned,
        combined_xz_results_v3,
        combined_y_results_v3
    )
    print(f"  ✓ Merged V1+V2+V3 shape: {merged_v1_v2_v3.shape}")

    # Save right side results and transformed volumes
    np.save(data_dir / 'merged_right_v1_v2_v3.npy', merged_v1_v2_v3)
    np.save(data_dir / 'volume_2_aligned.npy', volume_2_aligned)
    np.save(data_dir / 'volume_3_aligned.npy', volume_3_aligned)
    np.save(data_dir / 'v2_transforms.npy', {
        'xz': xz_results_v2,
        'y': y_results_v2,
        'rotation': step3_results_v2
    })
    np.save(data_dir / 'v3_transforms.npy', {
        'xz_combined': combined_xz_results_v3,
        'y_combined': combined_y_results_v3,
        'rotation_combined': combined_rotation_results_v3
    })
    print(f"  [SAVED] merged_right_v1_v2_v3.npy")
    print(f"  [SAVED] volume_2_aligned.npy")
    print(f"  [SAVED] volume_3_aligned.npy")
    print(f"  [SAVED] v2_transforms.npy, v3_transforms.npy")

    # Clear right side volumes from memory
    print("\n  [Memory cleanup] Clearing right side volumes from memory...")
    del volume_2, volume_3, volume_2_aligned, volume_3_aligned
    del merged_v1_v2, merged_v1_v2_v3
    gc.collect()
    print("  ✓ Memory cleared")

    # ========================================================================
    # PHASE 2: LOAD LEFT SIDE VOLUMES (V1, V4, V5)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: LOADING LEFT SIDE VOLUMES (V1, V4, V5)")
    print("=" * 70)

    phase2_loading_start = time.time()

    # Reload V1 (or keep if still in memory)
    if 'volume_1' not in locals():
        print("  Reloading V1...")
        volume_1 = loader.load_volume_from_directory(str(patient_vols[0]))
    else:
        print("  V1 still in memory, reusing...")

    volume_4 = loader.load_volume_from_directory(str(patient_vols[3]))
    volume_5 = loader.load_volume_from_directory(str(patient_vols[4]))

    print(f"  V1 shape: {volume_1.shape} (CENTER)")
    print(f"  V4 shape: {volume_4.shape}")
    print(f"  V5 shape: {volume_5.shape}")

    phase2_loading_time = time.time() - phase2_loading_start
    print(f"  ⏱️  Phase 2 loading time: {phase2_loading_time:.2f} seconds")

    # ========================================================================
    # LEFT SIDE: ALIGN V4 → V1 → V5
    # ========================================================================
    print("\n" + "=" * 70)
    print("LEFT SIDE ALIGNMENT: V4 → V1 → V5")
    print("=" * 70)

    # ========================================================================
    # STEP 3: ALIGN VOLUME 4 → VOLUME 1
    # ========================================================================
    print("\n" + "─" * 70)
    print("STEP 3: ALIGN VOLUME 4 → VOLUME 1")
    print("─" * 70)

    step3_start = time.time()

    # XZ alignment (volume 4 to volume 1)
    print("\n[3.1] XZ Alignment (Volume 4 → Volume 1)...")
    xz_results_v4 = perform_xz_alignment(volume_1, volume_4,
                                         max_offset_z=15,
                                         method='phase_corr',
                                         output_dir=data_dir)
    print(f"  ✓ XZ Offset: dx={xz_results_v4['offset_x']}, dz={xz_results_v4['offset_z']}")

    # Y alignment (volume 4 to volume 1)
    print("\n[3.2] Y Alignment (Volume 4 → Volume 1)...")
    y_results_v4 = perform_y_alignment(
        volume_1,
        xz_results_v4['volume_1_xz_aligned']
    )
    print(f"  ✓ Y Shift: {y_results_v4['y_shift']:.2f} px")

    # Z-rotation alignment (volume 4 to volume 1)
    print("\n[3.3] Z-Rotation Alignment (Volume 4 → Volume 1)...")
    step3_results_v4 = perform_z_rotation_alignment(
        volume_1,
        y_results_v4['volume_1_y_aligned'],
        visualize=args.visual,
        position='left',  # V4 is to the left of V1
        output_dir=data_dir,
        vis_interval=5
    )
    print(f"  ✓ Rotation angle: {step3_results_v4['rotation_angle']:+.3f}°")

    step3_time = time.time() - step3_start
    print(f"\n⏱️  Step 3 time: {step3_time:.2f} seconds")

    # Apply transformations to V4
    print("\n  Applying transformations to volume_4...")
    volume_4_aligned = apply_all_transformations_to_volume(
        volume_4,
        xz_results_v4,
        y_results_v4,
        step3_results=step3_results_v4
    )
    print(f"  ✓ Volume 4 transformed: {volume_4_aligned.shape}")

    # Merge V1 + V4
    print("\n  Merging V1 + V4...")
    merged_v1_v4, merge_metadata_14 = merge_two_volumes(
        volume_1,
        volume_4_aligned,
        xz_results_v4,
        y_results_v4
    )
    print(f"  ✓ Merged V1+V4 shape: {merged_v1_v4.shape}")

    # ========================================================================
    # STEP 4: ALIGN VOLUME 5 → VOLUME 4
    # ========================================================================
    print("\n" + "─" * 70)
    print("STEP 4: ALIGN VOLUME 5 → VOLUME 4")
    print("─" * 70)

    step4_start = time.time()

    # XZ alignment (volume 5 to volume 4)
    print("\n[4.1] XZ Alignment (Volume 5 → Volume 4)...")
    xz_results_v5 = perform_xz_alignment(volume_4, volume_5,
                                         max_offset_z=15,
                                         method='phase_corr',
                                         output_dir=data_dir)
    print(f"  ✓ XZ Offset: dx={xz_results_v5['offset_x']}, dz={xz_results_v5['offset_z']}")

    # Y alignment (volume 5 to volume 4)
    print("\n[4.2] Y Alignment (Volume 5 → Volume 4)...")
    y_results_v5 = perform_y_alignment(
        volume_4,
        xz_results_v5['volume_1_xz_aligned']
    )
    print(f"  ✓ Y Shift: {y_results_v5['y_shift']:.2f} px")

    # Z-rotation alignment (volume 5 to volume 4)
    print("\n[4.3] Z-Rotation Alignment (Volume 5 → Volume 4)...")
    step3_results_v5 = perform_z_rotation_alignment(
        volume_4,
        y_results_v5['volume_1_y_aligned'],
        visualize=args.visual,
        position='left',  # V5 is to the left of V4
        output_dir=data_dir,
        vis_interval=5
    )
    print(f"  ✓ Rotation angle: {step3_results_v5['rotation_angle']:+.3f}°")

    step4_time = time.time() - step4_start
    print(f"\n⏱️  Step 4 time: {step4_time:.2f} seconds")

    # Compute combined transforms for V5 (V5→V4 + V4→V1)
    print("\n  Computing combined transforms (V5→V4 + V4→V1)...")
    combined_offset_x_v5 = xz_results_v5['offset_x'] + xz_results_v4['offset_x']
    combined_offset_z_v5 = xz_results_v5['offset_z'] + xz_results_v4['offset_z']
    combined_y_shift_v5 = y_results_v5['y_shift'] + y_results_v4['y_shift']
    combined_rotation_v5 = step3_results_v5['rotation_angle'] + step3_results_v4['rotation_angle']

    print(f"    V5→V4: dx={xz_results_v5['offset_x']}, dz={xz_results_v5['offset_z']}, dy={y_results_v5['y_shift']:.1f}, rot={step3_results_v5['rotation_angle']:+.3f}°")
    print(f"    V4→V1: dx={xz_results_v4['offset_x']}, dz={xz_results_v4['offset_z']}, dy={y_results_v4['y_shift']:.1f}, rot={step3_results_v4['rotation_angle']:+.3f}°")
    print(f"    Combined: dx={combined_offset_x_v5}, dz={combined_offset_z_v5}, dy={combined_y_shift_v5:.1f}, rot={combined_rotation_v5:+.3f}°")

    # Apply combined transforms to V5
    print("\n  Applying combined transforms to volume_5...")
    combined_xz_results_v5 = {
        'offset_x': combined_offset_x_v5,
        'offset_z': combined_offset_z_v5,
        'confidence': xz_results_v5['confidence']
    }
    combined_y_results_v5 = {
        'y_shift': combined_y_shift_v5,
        'contour_y_offset': combined_y_shift_v5,
        'ncc_y_offset': combined_y_shift_v5
    }
    combined_rotation_results_v5 = {
        'rotation_angle': combined_rotation_v5,
        'ncc_after': step3_results_v5['ncc_after'],
        'rotation_axes': step3_results_v5['rotation_axes']
    }

    volume_5_aligned = apply_all_transformations_to_volume(
        volume_5,
        combined_xz_results_v5,
        combined_y_results_v5,
        step3_results=combined_rotation_results_v5
    )
    print(f"  ✓ Volume 5 transformed (combined): {volume_5_aligned.shape}")

    # Merge into final left side volume
    print("\n  Merging V5 into left side (V1+V4+V5)...")
    merged_v1_v4_v5, merge_metadata_145 = merge_two_volumes(
        merged_v1_v4,
        volume_5_aligned,
        combined_xz_results_v5,
        combined_y_results_v5
    )
    print(f"  ✓ Merged V1+V4+V5 shape: {merged_v1_v4_v5.shape}")

    # Save left side results and transformed volumes
    np.save(data_dir / 'merged_left_v1_v4_v5.npy', merged_v1_v4_v5)
    np.save(data_dir / 'volume_4_aligned.npy', volume_4_aligned)
    np.save(data_dir / 'volume_5_aligned.npy', volume_5_aligned)
    np.save(data_dir / 'v4_transforms.npy', {
        'xz': xz_results_v4,
        'y': y_results_v4,
        'rotation': step3_results_v4
    })
    np.save(data_dir / 'v5_transforms.npy', {
        'xz_combined': combined_xz_results_v5,
        'y_combined': combined_y_results_v5,
        'rotation_combined': combined_rotation_results_v5
    })
    print(f"  [SAVED] merged_left_v1_v4_v5.npy")
    print(f"  [SAVED] volume_4_aligned.npy")
    print(f"  [SAVED] volume_5_aligned.npy")
    print(f"  [SAVED] v4_transforms.npy, v5_transforms.npy")

    # Clear left side volumes from memory
    print("\n  [Memory cleanup] Clearing left side volumes from memory...")
    del volume_4, volume_5, volume_4_aligned, volume_5_aligned
    del merged_v1_v4, merged_v1_v4_v5
    gc.collect()
    print("  ✓ Memory cleared")

    # ========================================================================
    # PHASE 3: LOAD ALL TRANSFORMED VOLUMES FOR FINAL CANVAS
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: LOADING TRANSFORMED VOLUMES FOR FINAL CANVAS")
    print("=" * 70)

    phase3_loading_start = time.time()

    # Load all transformed volumes
    print("  Loading transformed volumes...")
    volume_2_aligned = np.load(data_dir / 'volume_2_aligned.npy')
    volume_3_aligned = np.load(data_dir / 'volume_3_aligned.npy')
    volume_4_aligned = np.load(data_dir / 'volume_4_aligned.npy')
    volume_5_aligned = np.load(data_dir / 'volume_5_aligned.npy')

    # Reload transforms for position calculation
    v2_transforms = np.load(data_dir / 'v2_transforms.npy', allow_pickle=True).item()
    v3_transforms = np.load(data_dir / 'v3_transforms.npy', allow_pickle=True).item()
    v4_transforms = np.load(data_dir / 'v4_transforms.npy', allow_pickle=True).item()
    v5_transforms = np.load(data_dir / 'v5_transforms.npy', allow_pickle=True).item()

    print(f"  ✓ V2 aligned: {volume_2_aligned.shape}")
    print(f"  ✓ V3 aligned: {volume_3_aligned.shape}")
    print(f"  ✓ V4 aligned: {volume_4_aligned.shape}")
    print(f"  ✓ V5 aligned: {volume_5_aligned.shape}")

    phase3_loading_time = time.time() - phase3_loading_start
    print(f"  ⏱️  Phase 3 loading time: {phase3_loading_time:.2f} seconds")

    # ========================================================================
    # STEP 5: PLACE ALL VOLUMES ON COMMON CANVAS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: PLACE ALL VOLUMES ON COMMON CANVAS")
    print("=" * 70)
    print("  (V1 is static reference at origin)")

    step5_start = time.time()

    # Extract position information from transforms
    # V2 position (direct V2→V1)
    y_shift_v2 = v2_transforms['y']['y_shift']
    offset_x_v2 = v2_transforms['xz']['offset_x']
    offset_z_v2 = v2_transforms['xz']['offset_z']

    # V3 position (combined V3→V2 + V2→V1)
    y_shift_v3 = v3_transforms['y_combined']['y_shift']
    offset_x_v3 = v3_transforms['xz_combined']['offset_x']
    offset_z_v3 = v3_transforms['xz_combined']['offset_z']

    # V4 position (direct V4→V1)
    y_shift_v4 = v4_transforms['y']['y_shift']
    offset_x_v4 = v4_transforms['xz']['offset_x']
    offset_z_v4 = v4_transforms['xz']['offset_z']

    # V5 position (combined V5→V4 + V4→V1)
    y_shift_v5 = v5_transforms['y_combined']['y_shift']
    offset_x_v5 = v5_transforms['xz_combined']['offset_x']
    offset_z_v5 = v5_transforms['xz_combined']['offset_z']

    # V1 is at origin (0, 0, 0)
    # All other volumes are positioned relative to V1 based on their transforms
    volumes_dict = {
        'V1': (volume_1, (0, 0, 0)),
        'V2': (volume_2_aligned, (y_shift_v2, offset_x_v2, offset_z_v2)),
        'V3': (volume_3_aligned, (y_shift_v3, offset_x_v3, offset_z_v3)),
        'V4': (volume_4_aligned, (y_shift_v4, offset_x_v4, offset_z_v4)),
        'V5': (volume_5_aligned, (y_shift_v5, offset_x_v5, offset_z_v5))
    }

    final_canvas, canvas_metadata = place_volumes_on_canvas(volumes_dict, v1_position=(0, 0, 0))

    print(f"\n  ✓ Final canvas shape: {final_canvas.shape}")

    step5_time = time.time() - step5_start
    print(f"\n⏱️  Step 5 time: {step5_time:.2f} seconds")

    # Save final merged volume
    np.save(data_dir / 'final_merged_volume_5volumes.npy', final_canvas)
    print(f"  [SAVED] final_merged_volume_5volumes.npy")

    # ========================================================================
    # STEP 6: VISUALIZATION (OPTIONAL)
    # ========================================================================
    if args.visual:
        print("\n" + "=" * 70)
        print("STEP 6: GENERATING VISUALIZATIONS")
        print("=" * 70)

        viz_start = time.time()

        # Visualize final merged volume (multi-angle)
        print("\n[Visualization] Generating final merged volume (all 5 volumes)...")
        visualize_3d_multiangle(
            final_canvas,
            title="Five-Volume Progressive Merge (V1+V2+V3+V4+V5)",
            output_path=data_dir / '3d_merged_5volumes.png',
            subsample=8,
            percentile=75
        )
        print(f"  ✓ Saved: 3d_merged_5volumes.png")

        viz_time = time.time() - viz_start
        print(f"\n⏱️  Visualization time: {viz_time:.2f} seconds")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    total_time = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("✅ FIVE-VOLUME PROGRESSIVE ALIGNMENT COMPLETE!")
    print("=" * 70)
    print("\n⏱️  TIMING SUMMARY:")
    print("=" * 70)
    print(f"  Phase 1 loading (V1,V2,V3):  {phase1_loading_time:>8.2f}s")
    print(f"  Step 1 (V2→V1 alignment):    {step1_time:>8.2f}s")
    print(f"  Step 2 (V3→V2 alignment):    {step2_time:>8.2f}s")
    print(f"  Phase 2 loading (V1,V4,V5):  {phase2_loading_time:>8.2f}s")
    print(f"  Step 3 (V4→V1 alignment):    {step3_time:>8.2f}s")
    print(f"  Step 4 (V5→V4 alignment):    {step4_time:>8.2f}s")
    print(f"  Phase 3 loading (aligned):   {phase3_loading_time:>8.2f}s")
    print(f"  Step 5 (Canvas placement):   {step5_time:>8.2f}s")
    if args.visual:
        print(f"  Visualization:               {viz_time:>8.2f}s")
    print(f"  {'─' * 30}")
    print(f"  TOTAL TIME:                  {total_time:>8.2f}s ({total_time/60:.1f} min)")
    print("=" * 70)

    print(f"\n📊 RESULTS:")
    print(f"  Final canvas (all 5):        {final_canvas.shape}")
    print(f"  Output directory:            {data_dir}")
    print("\n📁 SAVED FILES:")
    print(f"  Merged Volumes:")
    print(f"    - merged_right_v1_v2_v3.npy (right side merge)")
    print(f"    - merged_left_v1_v4_v5.npy (left side merge)")
    print(f"    - final_merged_volume_5volumes.npy (all 5 volumes)")
    print(f"  Transformed Volumes:")
    print(f"    - volume_2_aligned.npy, volume_3_aligned.npy")
    print(f"    - volume_4_aligned.npy, volume_5_aligned.npy")
    print(f"  Transform Parameters:")
    print(f"    - v2_transforms.npy, v3_transforms.npy")
    print(f"    - v4_transforms.npy, v5_transforms.npy")
    if args.visual:
        print(f"  Visualizations:")
        print(f"    - 3d_merged_5volumes.png (Multi-angle view)")


if __name__ == '__main__':
    freeze_support()  # Required for Windows multiprocessing
    main()
