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
    visualize_3d_multiangle
)

# Import step modules for alignment functions
from steps.step1_xz_alignment import perform_xz_alignment
from steps.step2_y_alignment import perform_y_alignment


def merge_two_volumes(volume_ref, volume_mov_aligned, xz_results, y_results):
    """
    Merge two aligned volumes into a single expanded volume.

    Args:
        volume_ref: Reference volume
        volume_mov_aligned: Aligned moving volume (with Y alignment applied)
        xz_results: XZ alignment results (offset_x, offset_z)
        y_results: Y alignment results (y_shift)

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

    # Use the Y-aligned volume from step2 results
    volume_mov_aligned = y_results['volume_1_y_aligned']

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
    print("  3. Align V3 → Merged(V1+V2)")
    print("  4. Create final merged volume (V1+V2+V3)")

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
    xz_results_v2 = perform_xz_alignment(volume_1, volume_2)
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

    merged_v1_v2, merge_metadata_12 = merge_two_volumes(
        volume_1,
        volume_2,
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

    # ========================================================================
    # STEP 3: ALIGN VOLUME 3 → MERGED VOLUME (1+2)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: ALIGN VOLUME 3 → MERGED VOLUME (1+2)")
    print("=" * 70)

    step3_start = time.time()

    # XZ alignment (volume 3 to merged volume)
    print("\n[3.1] XZ Alignment (Volume 3 → Merged 1+2)...")
    xz_results_v3 = perform_xz_alignment(merged_v1_v2, volume_3)
    print(f"  ✓ XZ Offset: dx={xz_results_v3['offset_x']}, dz={xz_results_v3['offset_z']}")
    print(f"  ✓ Confidence: {xz_results_v3['confidence']:.3f}")

    # Y alignment (volume 3 to merged volume)
    print("\n[3.2] Y Alignment (Volume 3 → Merged 1+2)...")
    y_results_v3 = perform_y_alignment(
        merged_v1_v2,
        xz_results_v3['volume_1_xz_aligned']
    )
    print(f"  ✓ Y Shift: {y_results_v3['y_shift']:.2f} px")
    print(f"  ✓ Contour offset: {y_results_v3['contour_y_offset']:.2f} px")
    print(f"  ✓ NCC offset: {y_results_v3['ncc_y_offset']:.2f} px")

    step3_time = time.time() - step3_start
    print(f"\n⏱️  Step 3 time: {step3_time:.2f} seconds")

    # Save intermediate results
    np.save(data_dir / 'step3_v3_to_merged_xz_results.npy', xz_results_v3)
    np.save(data_dir / 'step3_v3_to_merged_y_results.npy', y_results_v3)
    print(f"  [SAVED] step3_v3_to_merged_xz_results.npy")
    print(f"  [SAVED] step3_v3_to_merged_y_results.npy")

    # ========================================================================
    # STEP 4: CREATE FINAL MERGED VOLUME (1+2+3)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: CREATE FINAL MERGED VOLUME (1+2+3)")
    print("=" * 70)

    step4_start = time.time()

    final_merged, merge_metadata_123 = merge_two_volumes(
        merged_v1_v2,
        volume_3,
        xz_results_v3,
        y_results_v3
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

        # Visualize final merged volume
        visualize_3d_multiangle(
            final_merged,
            title="Three-Volume Progressive Merge (V1+V2+V3)",
            output_path=data_dir / '3d_merged_3volumes.png',
            subsample=8,
            percentile=75,
            source_labels=source_labels
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
    print(f"  Step 3 (V3→Merged alignment):{step3_time:>8.2f}s")
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
    print(f"  - step1_v2_to_v1_xz_results.npy")
    print(f"  - step1_v2_to_v1_y_results.npy")
    print(f"  - merged_v1_v2.npy")
    print(f"  - step3_v3_to_merged_xz_results.npy")
    print(f"  - step3_v3_to_merged_y_results.npy")
    print(f"  - final_merged_volume_3volumes.npy")
    if args.visual:
        print(f"  - 3d_merged_3volumes.png")


if __name__ == '__main__':
    freeze_support()  # Required for Windows multiprocessing
    main()
