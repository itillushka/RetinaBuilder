"""
Three-Volume Progressive Alignment Pipeline

Progressively aligns three OCT volumes:
1. Align volume_2 to volume_1
2. Merge volumes 1 and 2
3. Align volume_3 to the merged volume (1+2)

This approach creates a progressively larger merged volume at each step.

Usage:
    python three_volume_alignment.py --patient F001_IP
    python three_volume_alignment.py --patient EM005 --visual
"""

import numpy as np
import argparse
from pathlib import Path
import time
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


def main():
    """Three-volume progressive alignment pipeline."""

    parser = argparse.ArgumentParser(
        description='Three-Volume Progressive Alignment Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Align 3 volumes for patient F001_IP
  python three_volume_alignment.py --patient F001_IP

  # Align 3 volumes for EM005 with visualization
  python three_volume_alignment.py --patient EM005 --visual
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
    data_dir = Path(__file__).parent / f'results_3vol_{patient_safe.lower()}'
    data_dir.mkdir(exist_ok=True)

    # Start total pipeline timer
    pipeline_start = time.time()

    print("=" * 70)
    print("THREE-VOLUME PROGRESSIVE ALIGNMENT PIPELINE")
    print("=" * 70)
    print(f"Patient: {args.patient}")
    print(f"Output directory: {data_dir}")
    print(f"OCT data directory: {oct_data_dir}")
    print("\nStrategy:")
    print("  1. Align V2 → V1")
    print("  2. Merge V1 + V2")
    print("  3. Align V3 → V2 (uses V2 MIP to avoid size mismatch)")
    print("  4. Merge V3 into final volume using combined transforms (V3→V2 + V2→V1)")

    # ========================================================================
    # STEP 0: LOAD THREE VOLUMES
    # ========================================================================
    print("\n" + "=" * 70)
    print("LOADING THREE OCT VOLUMES")
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

    if len(patient_vols) < 3:
        raise ValueError(f"Need at least 3 volumes for patient {args.patient}, found {len(patient_vols)}")

    print(f"  Found {len(patient_vols)} volumes for patient {args.patient}")
    print(f"  Volume 1: {patient_vols[0].name}")
    print(f"  Volume 2: {patient_vols[1].name}")
    print(f"  Volume 3: {patient_vols[2].name}")

    volume_1 = loader.load_volume_from_directory(str(patient_vols[0]))
    volume_2 = loader.load_volume_from_directory(str(patient_vols[1]))
    volume_3 = loader.load_volume_from_directory(str(patient_vols[2]))

    print(f"\n  V1 shape: {volume_1.shape}")
    print(f"  V2 shape: {volume_2.shape}")
    print(f"  V3 shape: {volume_3.shape}")

    loading_time = time.time() - loading_start
    print(f"\n⏱️  Volume loading time: {loading_time:.2f} seconds")

    # ========================================================================
    # STEP 1: ALIGN VOLUME 2 → VOLUME 1
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: ALIGN VOLUME 2 → VOLUME 1")
    print("=" * 70)

    step1_start = time.time()

    # XZ alignment (volume 2 to volume 1)
    print("\n[1.1] XZ Alignment (Volume 2 → Volume 1)...")
    xz_results_v2 = perform_xz_alignment(volume_1, volume_2, max_offset_x=15)
    print(f"  ✓ XZ Offset: dx={xz_results_v2['offset_x']}, dz={xz_results_v2['offset_z']}")
    print(f"  ✓ Confidence: {xz_results_v2['confidence']:.3f}")

    # Y alignment (volume 2 to volume 1)
    print("\n[1.2] Y Alignment (Volume 2 → Volume 1)...")
    y_results_v2 = perform_y_alignment(
        volume_1,
        xz_results_v2['volume_1_xz_aligned']
    )
    print(f"  ✓ Y Shift: {y_results_v2['y_shift']:.2f} px")
    print(f"  ✓ Contour offset: {y_results_v2['contour_y_offset']:.2f} px")
    print(f"  ✓ NCC offset: {y_results_v2['ncc_y_offset']:.2f} px")

    step1_time = time.time() - step1_start
    print(f"\n⏱️  Step 1 time: {step1_time:.2f} seconds")

    # Save intermediate results
    np.save(data_dir / 'step1_v2_to_v1_xz_results.npy', xz_results_v2)
    np.save(data_dir / 'step1_v2_to_v1_y_results.npy', y_results_v2)
    print(f"  [SAVED] step1_v2_to_v1_xz_results.npy")
    print(f"  [SAVED] step1_v2_to_v1_y_results.npy")

    # ========================================================================
    # STEP 2: MERGE VOLUMES 1 AND 2
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: MERGE VOLUMES 1 AND 2")
    print("=" * 70)

    step2_start = time.time()

    # Apply ALL transformations to ORIGINAL volume_2 (not cropped intermediate)
    print("\n  Applying transformations to volume_2 for merging...")
    volume_2_aligned = apply_all_transformations_to_volume(
        volume_2,
        xz_results_v2,
        y_results_v2,
        step3_results=None  # No rotation for V2→V1
    )
    print(f"  ✓ Volume 2 transformed: {volume_2_aligned.shape}")

    # Merge using the fully transformed volume
    merged_v1_v2, merge_metadata_12 = merge_two_volumes(
        volume_1,
        volume_2_aligned,  # Use transformed volume, not original
        xz_results_v2,
        y_results_v2
    )

    print(f"\n  ✓ Merged volume shape: {merged_v1_v2.shape}")
    print(f"  ✓ Total voxels: {merge_metadata_12['total_voxels']:,}")
    print(f"  ✓ Overlap voxels: {merge_metadata_12['overlap_voxels']:,}")
    print(f"  ✓ Data loss: {merge_metadata_12['data_loss']}%")

    step2_time = time.time() - step2_start
    print(f"\n⏱️  Step 2 time: {step2_time:.2f} seconds")

    # Save intermediate merged volume
    np.save(data_dir / 'merged_v1_v2.npy', merged_v1_v2)
    print(f"  [SAVED] merged_v1_v2.npy")

    # Visualization: V2→V1 alignment (after merge)
    if args.visual:
        print("\n[Visualization] Generating V2→V1 alignment comparison...")
        # Reuse the already transformed volume
        transform_v2_to_v1 = {
            'dy': float(y_results_v2['y_shift']),
            'dx': float(xz_results_v2['offset_x']),
            'dz': float(xz_results_v2['offset_z'])
        }
        visualize_3d_comparison(
            volume_1,
            volume_2_aligned,  # Reuse already transformed volume
            merged_v1_v2,
            transform_v2_to_v1,
            output_path=data_dir / '3d_comparison_v2_to_v1.png',
            subsample=8,
            percentile=75,
            z_crop_front=100,
            z_crop_back=100
        )
        print(f"  ✓ Saved: 3d_comparison_v2_to_v1.png")

    # ========================================================================
    # STEP 3: ALIGN VOLUME 3 → VOLUME 2
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: ALIGN VOLUME 3 → VOLUME 2")
    print("=" * 70)
    print("  (Using V2 as reference to avoid MIP size mismatch)")

    step3_start = time.time()

    # XZ alignment (volume 3 to volume 2)
    print("\n[3.1] XZ Alignment (Volume 3 → Volume 2)...")
    xz_results_v3 = perform_xz_alignment(volume_2, volume_3, max_offset_x=15)
    print(f"  ✓ XZ Offset: dx={xz_results_v3['offset_x']}, dz={xz_results_v3['offset_z']}")
    print(f"  ✓ Confidence: {xz_results_v3['confidence']:.3f}")

    # Y alignment (volume 3 to volume 2)
    print("\n[3.2] Y Alignment (Volume 3 → Volume 2)...")
    y_results_v3 = perform_y_alignment(
        volume_2,
        xz_results_v3['volume_1_xz_aligned']
    )
    print(f"  ✓ Y Shift: {y_results_v3['y_shift']:.2f} px")
    print(f"  ✓ Contour offset: {y_results_v3['contour_y_offset']:.2f} px")
    print(f"  ✓ NCC offset: {y_results_v3['ncc_y_offset']:.2f} px")

    step3_time = time.time() - step3_start
    print(f"\n⏱️  Step 3 time: {step3_time:.2f} seconds")

    # Save intermediate results
    np.save(data_dir / 'step3_v3_to_v2_xz_results.npy', xz_results_v3)
    np.save(data_dir / 'step3_v3_to_v2_y_results.npy', y_results_v3)
    print(f"  [SAVED] step3_v3_to_v2_xz_results.npy")
    print(f"  [SAVED] step3_v3_to_v2_y_results.npy")

    # Apply transformations to ORIGINAL volume_3 for later use
    print("\n  Applying V3→V2 transformations to volume_3...")
    volume_3_aligned_v2 = apply_all_transformations_to_volume(
        volume_3,
        xz_results_v3,
        y_results_v3,
        step3_results=None  # No rotation for V3→V2
    )
    print(f"  ✓ Volume 3 transformed (V3→V2): {volume_3_aligned_v2.shape}")

    # Visualization: V3→V2 alignment
    if args.visual:
        print("\n[Visualization] Generating V3→V2 alignment comparison...")
        # Create temporary merged volume for V2+V3 visualization using transformed volume
        merged_v2_v3_temp, _ = merge_two_volumes(
            volume_2,
            volume_3_aligned_v2,  # Use transformed volume
            xz_results_v3,
            y_results_v3
        )
        transform_v3_to_v2 = {
            'dy': float(y_results_v3['y_shift']),
            'dx': float(xz_results_v3['offset_x']),
            'dz': float(xz_results_v3['offset_z'])
        }
        visualize_3d_comparison(
            volume_2,
            volume_3_aligned_v2,  # Use already transformed volume
            merged_v2_v3_temp,
            transform_v3_to_v2,
            output_path=data_dir / '3d_comparison_v3_to_v2.png',
            subsample=8,
            percentile=75,
            z_crop_front=100,
            z_crop_back=100
        )
        print(f"  ✓ Saved: 3d_comparison_v3_to_v2.png")

    # ========================================================================
    # STEP 4: CREATE FINAL MERGED VOLUME (1+2+3)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: CREATE FINAL MERGED VOLUME (1+2+3)")
    print("=" * 70)

    step4_start = time.time()

    # Compute combined transforms for V3
    # V3's position in final merged space = V3→V2 transform + V2→V1 transform
    print("\n  Computing combined transforms (V3→V2 + V2→V1)...")
    combined_offset_x = xz_results_v3['offset_x'] + xz_results_v2['offset_x']
    combined_offset_z = xz_results_v3['offset_z'] + xz_results_v2['offset_z']
    combined_y_shift = y_results_v3['y_shift'] + y_results_v2['y_shift']

    print(f"    V3→V2: dx={xz_results_v3['offset_x']}, dz={xz_results_v3['offset_z']}, dy={y_results_v3['y_shift']:.1f}")
    print(f"    V2→V1: dx={xz_results_v2['offset_x']}, dz={xz_results_v2['offset_z']}, dy={y_results_v2['y_shift']:.1f}")
    print(f"    Combined: dx={combined_offset_x}, dz={combined_offset_z}, dy={combined_y_shift:.1f}")

    # Apply COMBINED transforms to ORIGINAL volume_3
    print("\n  Applying combined transforms to volume_3...")
    combined_xz_results = {
        'offset_x': combined_offset_x,
        'offset_z': combined_offset_z,
        'confidence': xz_results_v3['confidence']
    }

    combined_y_results = {
        'y_shift': combined_y_shift,
        'contour_y_offset': combined_y_shift,
        'ncc_y_offset': combined_y_shift
    }

    # Apply combined transforms to original volume_3
    volume_3_aligned_final = apply_all_transformations_to_volume(
        volume_3,
        combined_xz_results,
        combined_y_results,
        step3_results=None  # No rotation
    )
    print(f"  ✓ Volume 3 transformed (combined): {volume_3_aligned_final.shape}")

    # Merge using the fully transformed volume
    final_merged, merge_metadata_123 = merge_two_volumes(
        merged_v1_v2,
        volume_3_aligned_final,  # Use transformed volume with combined transforms
        combined_xz_results,
        combined_y_results
    )

    print(f"\n  ✓ Final merged volume shape: {final_merged.shape}")
    print(f"  ✓ Total voxels: {merge_metadata_123['total_voxels']:,}")
    print(f"  ✓ Overlap voxels: {merge_metadata_123['overlap_voxels']:,}")
    print(f"  ✓ Data loss: {merge_metadata_123['data_loss']}%")

    step4_time = time.time() - step4_start
    print(f"\n⏱️  Step 4 time: {step4_time:.2f} seconds")

    # Save final merged volume
    np.save(data_dir / 'final_merged_volume_3volumes.npy', final_merged)
    print(f"  [SAVED] final_merged_volume_3volumes.npy")

    # ========================================================================
    # STEP 5: VISUALIZATION (OPTIONAL)
    # ========================================================================
    if args.visual:
        print("\n" + "=" * 70)
        print("STEP 5: GENERATING VISUALIZATIONS")
        print("=" * 70)

        viz_start = time.time()

        # Get source labels from final merge
        source_labels = merge_metadata_123.get('source_labels', None)

        # Visualize final merged volume (multi-angle)
        print("\n[Visualization] Generating final merged volume (V1+V2+V3)...")
        visualize_3d_multiangle(
            final_merged,
            title="Three-Volume Progressive Merge (V1+V2+V3)",
            output_path=data_dir / '3d_merged_3volumes.png',
            subsample=8,
            percentile=75,
            source_labels=source_labels
        )
        print(f"  ✓ Saved: 3d_merged_3volumes.png")

        # Side-by-side comparison for final result
        print("\n[Visualization] Generating final comparison (V1 vs merged_V1+V2 vs final_V1+V2+V3)...")
        # Reuse the already transformed volume_3 (volume_3_aligned_final)
        transform_final = {
            'dy': float(combined_y_shift),
            'dx': float(combined_offset_x),
            'dz': float(combined_offset_z)
        }
        visualize_3d_comparison(
            merged_v1_v2,
            volume_3_aligned_final,  # Reuse already transformed volume
            final_merged,
            transform_final,
            output_path=data_dir / '3d_comparison_final_all3volumes.png',
            subsample=8,
            percentile=75,
            z_crop_front=100,
            z_crop_back=100
        )
        print(f"  ✓ Saved: 3d_comparison_final_all3volumes.png")

        # MIP visualizations for all volumes
        print("\n[Visualization] Generating MIP projections (XY, YZ, XZ)...")
        visualize_three_volume_mips(
            volume_1,
            volume_2_aligned,
            volume_3_aligned_final,
            merged_v1_v2,
            final_merged,
            output_dir=data_dir
        )

        viz_time = time.time() - viz_start
        print(f"\n⏱️  Visualization time: {viz_time:.2f} seconds")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    total_time = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("✅ THREE-VOLUME PROGRESSIVE ALIGNMENT COMPLETE!")
    print("=" * 70)
    print("\n⏱️  TIMING SUMMARY:")
    print("=" * 70)
    print(f"  Volume Loading:              {loading_time:>8.2f}s")
    print(f"  Step 1 (V2→V1 alignment):    {step1_time:>8.2f}s")
    print(f"  Step 2 (Merge V1+V2):        {step2_time:>8.2f}s")
    print(f"  Step 3 (V3→V2 alignment):    {step3_time:>8.2f}s")
    print(f"  Step 4 (Final merge):        {step4_time:>8.2f}s")
    if args.visual:
        print(f"  Visualization:               {viz_time:>8.2f}s")
    print(f"  {'─' * 30}")
    print(f"  TOTAL TIME:                  {total_time:>8.2f}s ({total_time/60:.1f} min)")
    print("=" * 70)

    print(f"\n📊 RESULTS:")
    print(f"  Intermediate merge (V1+V2):  {merged_v1_v2.shape}")
    print(f"  Final merge (V1+V2+V3):      {final_merged.shape}")
    print(f"  Output directory:            {data_dir}")
    print("\n📁 SAVED FILES:")
    print(f"  Alignment Results:")
    print(f"    - step1_v2_to_v1_xz_results.npy")
    print(f"    - step1_v2_to_v1_y_results.npy")
    print(f"    - step3_v3_to_v2_xz_results.npy (V3→V2 alignment)")
    print(f"    - step3_v3_to_v2_y_results.npy (V3→V2 alignment)")
    print(f"  Merged Volumes:")
    print(f"    - merged_v1_v2.npy")
    print(f"    - final_merged_volume_3volumes.npy")
    if args.visual:
        print(f"  Visualizations:")
        print(f"    - 3d_comparison_v2_to_v1.png (Step 1 alignment)")
        print(f"    - 3d_comparison_v3_to_v2.png (Step 3 alignment)")
        print(f"    - 3d_comparison_final_all3volumes.png (Final merge)")
        print(f"    - 3d_merged_3volumes.png (Multi-angle view)")


if __name__ == '__main__':
    freeze_support()  # Required for Windows multiprocessing
    main()
