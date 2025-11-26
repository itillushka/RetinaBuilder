"""
Averaged B-Scan Alignment Pipeline

A computationally lighter alternative to full volume alignment.
Aligns 3 averaged B-scans (each averaged from 30 central B-scans)
using only 2D contour-based methods.

Pipeline Steps:
  0. Load 3 OCT volumes
  1. Extract and average 30 central B-scans from each volume
  2. Apply harsh denoising to averaged B-scans
  3. Perform X-alignment using contour method (no cropping)
  4. Perform Y-alignment on averaged B-scans
  5. Perform rotation alignment on averaged B-scans
  6. Save aligned averaged B-scans + comprehensive visualizations

Output: 3 aligned 2D averaged B-scans (PNG + NPY) - UNCROPPED
"""

import numpy as np
import sys
from pathlib import Path
import time
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.oct_loader import OCTImageProcessor, OCTVolumeLoader

# Import our new modules
from .bscan_averaging import (
    extract_averaged_central_bscan,
    shift_bscan_2d,
    rotate_bscan_2d,
    save_averaged_bscan,
    save_averaged_bscan_npy,
    visualize_alignment_step,
    visualize_surface_comparison,
    visualize_three_averaged_bscans
)
from .averaged_alignment_steps import (
    perform_x_fine_tuning_2d,
    perform_y_alignment_2d,
    perform_rotation_alignment_2d,
    visualize_rotation_results_2d
)


def averaged_bscan_pipeline(vol1_path, vol2_path, vol3_path, output_dir,
                            n_bscans=30, visualize=True):
    """
    Main pipeline for averaged B-scan alignment.

    Args:
        vol1_path: Path to first volume directory (reference)
        vol2_path: Path to second volume directory
        vol3_path: Path to third volume directory
        output_dir: Directory to save outputs
        n_bscans: Number of central B-scans to average (default: 30)
        visualize: Whether to generate visualizations (default: True)

    Returns:
        results: Dictionary containing alignment metrics and paths to outputs
    """
    start_time = time.time()

    print("\n" + "="*80)
    print("AVERAGED B-SCAN ALIGNMENT PIPELINE")
    print("="*80)
    print(f"  Volume 1 (ref): {vol1_path}")
    print(f"  Volume 2:       {vol2_path}")
    print(f"  Volume 3:       {vol3_path}")
    print(f"  Output dir:     {output_dir}")
    print(f"  N B-scans:      {n_bscans}")
    print("="*80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'vol1_path': str(vol1_path),
        'vol2_path': str(vol2_path),
        'vol3_path': str(vol3_path),
        'n_bscans_averaged': n_bscans,
        'v2': {},
        'v3': {}
    }

    # ========================================================================
    # STEP 0: Load volumes
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 0: LOADING VOLUMES")
    print("="*80)

    processor = OCTImageProcessor(
        sidebar_width=250,
        crop_top=100,
        crop_bottom=50
    )
    loader = OCTVolumeLoader(processor)

    print("\nLoading Volume 1 (reference)...")
    volume_1 = loader.load_volume_from_directory(vol1_path)
    print(f"  Shape: {volume_1.shape}")

    print("\nLoading Volume 2...")
    volume_2 = loader.load_volume_from_directory(vol2_path)
    print(f"  Shape: {volume_2.shape}")

    print("\nLoading Volume 3...")
    volume_3 = loader.load_volume_from_directory(vol3_path)
    print(f"  Shape: {volume_3.shape}")

    # ========================================================================
    # STEP 1: Extract and average central B-scans
    # ========================================================================
    print("\n" + "="*80)
    print(f"STEP 1: EXTRACT AND AVERAGE {n_bscans} CENTRAL B-SCANS")
    print("="*80)

    print("\n1.1. Volume 1 (reference)...")
    avg_v1_raw, z_range_v1 = extract_averaged_central_bscan(volume_1, n_bscans)
    results['v1_z_range'] = z_range_v1

    print("\n1.2. Volume 2...")
    avg_v2_raw, z_range_v2 = extract_averaged_central_bscan(volume_2, n_bscans)
    results['v2']['z_range'] = z_range_v2

    print("\n1.3. Volume 3...")
    avg_v3_raw, z_range_v3 = extract_averaged_central_bscan(volume_3, n_bscans)
    results['v3']['z_range'] = z_range_v3

    # Free memory - no longer need volumes
    del volume_1, volume_2, volume_3
    print("\n  Volumes freed from memory")

    # Visualize initial RAW averaged B-scans (before denoising)
    if visualize:
        print("\nVisualizing initial RAW averaged B-scans (before denoising)...")
        visualize_three_averaged_bscans(
            avg_v1_raw, avg_v2_raw, avg_v3_raw,
            output_dir / 'step1_initial_averaged_bscans_raw.png',
            title=f"Initial Averaged B-scans - RAW ({n_bscans} central scans)"
        )

    # ========================================================================
    # STEP 2: Apply harsh denoising to averaged B-scans
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: APPLY HARSH DENOISING TO AVERAGED B-SCANS")
    print("="*80)
    print("  Using aggressive denoising pipeline (same as volume alignment):")
    print("  - Non-local means denoising (h=30)")
    print("  - Bilateral filter (sigma=180)")
    print("  - Median blur (19x19)")
    print("  - Otsu thresholding (40%)")
    print("  - CLAHE enhancement")

    from helpers.rotation_alignment import preprocess_oct_for_visualization

    print("\n2.1. Denoising Volume 1...")
    avg_v1 = preprocess_oct_for_visualization(avg_v1_raw)
    print(f"  Denoised: {avg_v1.shape}, dtype={avg_v1.dtype}")

    print("\n2.2. Denoising Volume 2...")
    avg_v2 = preprocess_oct_for_visualization(avg_v2_raw)
    print(f"  Denoised: {avg_v2.shape}, dtype={avg_v2.dtype}")

    print("\n2.3. Denoising Volume 3...")
    avg_v3 = preprocess_oct_for_visualization(avg_v3_raw)
    print(f"  Denoised: {avg_v3.shape}, dtype={avg_v3.dtype}")

    # Visualize DENOISED averaged B-scans
    if visualize:
        print("\nVisualizing DENOISED averaged B-scans...")
        visualize_three_averaged_bscans(
            avg_v1, avg_v2, avg_v3,
            output_dir / 'step2_averaged_bscans_denoised.png',
            title=f"Averaged B-scans - DENOISED ({n_bscans} central scans)"
        )

    print("\n  [INFO] All subsequent alignment will use DENOISED versions")

    # ========================================================================
    # STEP 3: X-alignment using contour method on averaged B-scans
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: X-AXIS ALIGNMENT (CONTOUR-BASED)")
    print("="*80)

    print("\n3.1. X-alignment for Volume 2...")
    avg_v2_before_x = avg_v2.copy()
    x_results_v2 = perform_x_fine_tuning_2d(avg_v1, avg_v2, search_range=600, verbose=True)
    x_offset_v2 = x_results_v2['x_fine_tune']

    if abs(x_offset_v2) > 0:
        avg_v2 = shift_bscan_2d(avg_v2, dx=x_offset_v2, dy=0)
        print(f"\n  Applied X alignment: {x_offset_v2:+d} px")
    else:
        print(f"\n  No X alignment needed")

    results['v2']['x_offset'] = int(x_offset_v2)
    results['v2']['x_alignment'] = x_results_v2

    if visualize and abs(x_offset_v2) > 0:
        visualize_alignment_step(
            avg_v1, avg_v2_before_x, avg_v2,
            step_name="X-Alignment (Volume 2)",
            output_path=output_dir / 'step3_x_alignment_v2.png',
            transform_description=f"X offset: {x_offset_v2:+d} px",
            show_difference=True
        )

    print("\n3.2. X-alignment for Volume 3 → Volume 2...")
    avg_v3_before_x = avg_v3.copy()
    x_results_v3 = perform_x_fine_tuning_2d(avg_v2, avg_v3, search_range=600, verbose=True)
    x_offset_v3 = x_results_v3['x_fine_tune']

    if abs(x_offset_v3) > 0:
        avg_v3 = shift_bscan_2d(avg_v3, dx=x_offset_v3, dy=0)
        print(f"\n  Applied X alignment: {x_offset_v3:+d} px")
    else:
        print(f"\n  No X alignment needed")

    results['v3']['x_offset'] = int(x_offset_v3)
    results['v3']['x_alignment'] = x_results_v3

    if visualize and abs(x_offset_v3) > 0:
        visualize_alignment_step(
            avg_v2, avg_v3_before_x, avg_v3,
            step_name="X-Alignment (Volume 3 → V2)",
            output_path=output_dir / 'step3_x_alignment_v3.png',
            transform_description=f"X offset: {x_offset_v3:+d} px",
            show_difference=True
        )

    # ========================================================================
    # STEP 4: Y-alignment on averaged B-scans
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Y-ALIGNMENT ON AVERAGED B-SCANS")
    print("="*80)

    print("\n4.1. Y-aligning Volume 2...")
    avg_v2_before_y = avg_v2.copy()
    y_results_v2 = perform_y_alignment_2d(avg_v1, avg_v2, verbose=True)
    y_shift_v2 = y_results_v2['y_shift']
    avg_v2 = shift_bscan_2d(avg_v2, dx=0, dy=y_shift_v2)

    results['v2']['y_shift'] = float(y_shift_v2)
    results['v2']['y_alignment'] = y_results_v2

    if visualize:
        # Standard visualization
        visualize_alignment_step(
            avg_v1, avg_v2_before_y, avg_v2,
            step_name="Y-Alignment (Volume 2)",
            output_path=output_dir / 'step4_y_alignment_v2.png',
            transform_description=f"Y shift: {y_shift_v2:+.1f} px",
            show_difference=True
        )

        # Surface visualization
        from helpers.rotation_alignment import preprocess_oct_for_visualization, detect_contour_surface
        bscan_ref_proc = preprocess_oct_for_visualization(avg_v1)
        bscan_aligned_proc = preprocess_oct_for_visualization(avg_v2)
        surface_ref = detect_contour_surface(bscan_ref_proc)
        surface_aligned = detect_contour_surface(bscan_aligned_proc)

        visualize_surface_comparison(
            avg_v1, avg_v2,
            surface_ref, surface_aligned,
            step_name="Y-Alignment (Volume 2)",
            output_path=output_dir / 'step4_y_alignment_v2_surfaces.png',
            transform_description=f"Y shift: {y_shift_v2:+.1f} px"
        )

    print("\n4.2. Y-aligning Volume 3 → Volume 2 (not V1)...")
    avg_v3_before_y = avg_v3.copy()
    y_results_v3 = perform_y_alignment_2d(avg_v2, avg_v3, verbose=True)
    y_shift_v3 = y_results_v3['y_shift']
    avg_v3 = shift_bscan_2d(avg_v3, dx=0, dy=y_shift_v3)

    results['v3']['y_shift'] = float(y_shift_v3)
    results['v3']['y_alignment'] = y_results_v3

    if visualize:
        # Standard visualization (compare to V2, not V1)
        visualize_alignment_step(
            avg_v2, avg_v3_before_y, avg_v3,
            step_name="Y-Alignment (Volume 3 → V2)",
            output_path=output_dir / 'step4_y_alignment_v3.png',
            transform_description=f"Y shift: {y_shift_v3:+.1f} px",
            show_difference=True
        )

        # Surface visualization (compare to V2, not V1)
        bscan_v2_proc = preprocess_oct_for_visualization(avg_v2)
        bscan_aligned_proc = preprocess_oct_for_visualization(avg_v3)
        surface_v2 = detect_contour_surface(bscan_v2_proc)
        surface_aligned = detect_contour_surface(bscan_aligned_proc)

        visualize_surface_comparison(
            avg_v2, avg_v3,
            surface_v2, surface_aligned,
            step_name="Y-Alignment (Volume 3 → V2)",
            output_path=output_dir / 'step4_y_alignment_v3_surfaces.png',
            transform_description=f"Y shift: {y_shift_v3:+.1f} px"
        )

    # ========================================================================
    # STEP 5: Rotation alignment on averaged B-scans
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: ROTATION ALIGNMENT ON AVERAGED B-SCANS")
    print("="*80)

    print("\n5.1. Rotation aligning Volume 2...")
    avg_v2_before_rot = avg_v2.copy()
    rot_results_v2 = perform_rotation_alignment_2d(
        avg_v1, avg_v2,
        coarse_range=25,
        coarse_step=1,
        fine_range=3,
        fine_step=0.5,
        verbose=True
    )
    rotation_angle_v2 = rot_results_v2['rotation_angle']

    # Apply rotation if significant
    if abs(rotation_angle_v2) > 0.5:
        avg_v2 = rotate_bscan_2d(avg_v2, rotation_angle_v2)
        print(f"  Applied rotation: {rotation_angle_v2:+.2f}°")
    else:
        print(f"  Rotation too small ({rotation_angle_v2:+.2f}°), skipped")

    results['v2']['rotation_angle'] = float(rotation_angle_v2)
    results['v2']['rotation_results'] = rot_results_v2

    if visualize:
        # Standard visualization
        visualize_alignment_step(
            avg_v1, avg_v2_before_rot, avg_v2,
            step_name="Rotation Alignment (Volume 2)",
            output_path=output_dir / 'step5_rotation_v2.png',
            transform_description=f"Rotation: {rotation_angle_v2:+.2f}°",
            show_difference=True
        )

        # Detailed rotation visualization
        visualize_rotation_results_2d(
            avg_v1, avg_v2_before_rot, rot_results_v2,
            output_dir / 'step5_rotation_v2_detailed.png',
            bscan_name="Volume 2"
        )

    print("\n5.2. Rotation aligning Volume 3 → Volume 2 (not V1)...")
    avg_v3_before_rot = avg_v3.copy()
    rot_results_v3 = perform_rotation_alignment_2d(
        avg_v2, avg_v3,
        coarse_range=25,
        coarse_step=1,
        fine_range=3,
        fine_step=0.5,
        verbose=True
    )
    rotation_angle_v3 = rot_results_v3['rotation_angle']

    # Apply rotation if significant
    if abs(rotation_angle_v3) > 0.5:
        avg_v3 = rotate_bscan_2d(avg_v3, rotation_angle_v3)
        print(f"  Applied rotation: {rotation_angle_v3:+.2f}°")
    else:
        print(f"  Rotation too small ({rotation_angle_v3:+.2f}°), skipped")

    results['v3']['rotation_angle'] = float(rotation_angle_v3)
    results['v3']['rotation_results'] = rot_results_v3

    if visualize:
        # Standard visualization (compare to V2, not V1)
        visualize_alignment_step(
            avg_v2, avg_v3_before_rot, avg_v3,
            step_name="Rotation Alignment (Volume 3 → V2)",
            output_path=output_dir / 'step5_rotation_v3.png',
            transform_description=f"Rotation: {rotation_angle_v3:+.2f}°",
            show_difference=True
        )

        # Detailed rotation visualization (compare to V2, not V1)
        visualize_rotation_results_2d(
            avg_v2, avg_v3_before_rot, rot_results_v3,
            output_dir / 'step5_rotation_v3_detailed.png',
            bscan_name="Volume 3 → V2"
        )

    # ========================================================================
    # STEP 6: Apply final Y and rotation transforms to uncropped versions for panorama
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: PREPARING UNCROPPED B-SCANS FOR PANORAMA")
    print("="*80)

    print("\n6.1. Applying Y-shift and rotation to uncropped V2...")
    avg_v2_for_panorama = shift_bscan_2d(avg_v2_for_panorama, dx=0, dy=y_shift_v2)
    if abs(rotation_angle_v2) > 0.5:
        avg_v2_for_panorama = rotate_bscan_2d(avg_v2_for_panorama, rotation_angle_v2)
    print(f"  Applied: Y={y_shift_v2:+.1f}px, Rotation={rotation_angle_v2:+.2f}°")

    print("\n6.2. Applying Y-shift and rotation to uncropped V3...")
    avg_v3_for_panorama = shift_bscan_2d(avg_v3_for_panorama, dx=0, dy=y_shift_v3)
    if abs(rotation_angle_v3) > 0.5:
        avg_v3_for_panorama = rotate_bscan_2d(avg_v3_for_panorama, rotation_angle_v3)
    print(f"  Applied: Y={y_shift_v3:+.1f}px, Rotation={rotation_angle_v3:+.2f}°")

    # ========================================================================
    # STEP 7: Create panorama from aligned B-scans
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: CREATING PANORAMA")
    print("="*80)

    print("\nStitching 3 aligned B-scans into panorama...")
    panorama, panorama_positions = create_panorama_from_aligned_bscans(
        avg_v1_for_panorama,
        avg_v2_for_panorama,
        avg_v3_for_panorama,
        x_offset_v2=x_total_v2,
        x_offset_v3=x_total_v3,
        y_shift_v2=y_shift_v2,
        y_shift_v3=y_shift_v3
    )

    print(f"  Panorama size: {panorama.shape} (H x W)")
    print(f"  V1 position: x={panorama_positions['v1']['x']}, y={panorama_positions['v1']['y']}")
    print(f"  V2 position: x={panorama_positions['v2']['x']}, y={panorama_positions['v2']['y']}")
    print(f"  V3 position: x={panorama_positions['v3']['x']}, y={panorama_positions['v3']['y']}")

    results['panorama_positions'] = panorama_positions

    # Save panorama
    print("\nSaving panorama...")
    save_averaged_bscan(panorama, output_dir / 'final_panorama.png')
    save_averaged_bscan_npy(panorama, output_dir / 'final_panorama.npy')

    # ========================================================================
    # STEP 8: Save final aligned averaged B-scans (cropped versions)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 8: SAVING FINAL ALIGNED AVERAGED B-SCANS (CROPPED)")
    print("="*80)

    # Save as PNG
    print("\nSaving PNG images...")
    save_averaged_bscan(avg_v1, output_dir / 'final_averaged_v1.png')
    save_averaged_bscan(avg_v2, output_dir / 'final_averaged_v2_aligned.png')
    save_averaged_bscan(avg_v3, output_dir / 'final_averaged_v3_aligned.png')

    # Save as NPY
    print("\nSaving NPY arrays...")
    save_averaged_bscan_npy(avg_v1, output_dir / 'final_averaged_v1.npy')
    save_averaged_bscan_npy(avg_v2, output_dir / 'final_averaged_v2_aligned.npy')
    save_averaged_bscan_npy(avg_v3, output_dir / 'final_averaged_v3_aligned.npy')

    # Final comparison visualization
    if visualize:
        print("\nCreating final comparison visualization...")
        visualize_three_averaged_bscans(
            avg_v1, avg_v2, avg_v3,
            output_dir / 'final_aligned_comparison.png',
            title=f"Final Aligned Averaged B-scans ({n_bscans} central scans averaged)"
        )

    # ========================================================================
    # Summary
    # ========================================================================
    elapsed_time = time.time() - start_time
    results['elapsed_time_seconds'] = elapsed_time

    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")
    print(f"\nAlignment Summary:")
    print(f"  Volume 2:")
    print(f"    X offset (contour):   {x_offset_v2:+d} px")
    print(f"    Y shift:              {y_shift_v2:+.1f} px")
    print(f"    Rotation:             {rotation_angle_v2:+.2f}°")
    print(f"  Volume 3 → V2:")
    print(f"    X offset (contour):   {x_offset_v3:+d} px")
    print(f"    Y shift:              {y_shift_v3:+.1f} px")
    print(f"    Rotation:             {rotation_angle_v3:+.2f}°")

    print(f"\nOutput files saved to: {output_dir}")
    print("="*80 + "\n")

    # Save results as JSON
    # Clean up results for JSON serialization
    def make_json_serializable(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return None  # Skip arrays
        else:
            return obj

    results_json = make_json_serializable(results)

    # Remove numpy arrays and non-serializable items
    for key in ['v2', 'v3']:
        if 'y_alignment' in results_json[key]:
            y_align = results_json[key]['y_alignment']
            y_align.pop('surface_ref', None)
            y_align.pop('surface_mov', None)
        if 'rotation_results' in results_json[key]:
            rot_res = results_json[key]['rotation_results']
            rot_res.pop('coarse_results', None)
            rot_res.pop('fine_results', None)

    with open(output_dir / 'alignment_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved: {output_dir / 'alignment_results.json'}")

    return results


def main():
    """Example usage of the averaged B-scan alignment pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Averaged B-Scan Alignment Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Using patient ID (automatic volume discovery)
  python averaged_bscan_pipeline.py --patient EM001
  python averaged_bscan_pipeline.py --patient F001_IP --visual

  # Using explicit volume paths
  python averaged_bscan_pipeline.py \\
    --vol1 ../../oct_data/emmetropes/EM001/Volume_1 \\
    --vol2 ../../oct_data/emmetropes/EM001/Volume_2 \\
    --vol3 ../../oct_data/emmetropes/EM001/Volume_3 \\
    --output ./output_averaged_alignment
        """
    )

    # Option 1: Use patient ID (like three_volume_alignment.py)
    parser.add_argument('--patient', type=str, default=None,
                       help='Patient ID to search for (e.g., EM001, F001_IP)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override OCT data directory (default: ../oct_data)')

    # Option 2: Use explicit paths
    parser.add_argument('--vol1', help='Path to volume 1 (reference)')
    parser.add_argument('--vol2', help='Path to volume 2')
    parser.add_argument('--vol3', help='Path to volume 3')

    # Common options
    parser.add_argument('--output', help='Output directory (auto-generated if using --patient)')
    parser.add_argument('--n-bscans', type=int, default=30,
                       help='Number of central B-scans to average (default: 30)')
    parser.add_argument('--no-visual', action='store_true',
                       help='Disable visualizations (faster)')

    args = parser.parse_args()

    # Determine volume paths
    if args.patient:
        # Option 1: Search for patient volumes
        # Path structure: averaged_bscan_alignment/ -> scripts_modular_parallel/ -> RetinaBuilder/ -> oct_data/
        scripts_dir = Path(__file__).parent.parent  # scripts_modular_parallel
        retina_builder_dir = scripts_dir.parent      # RetinaBuilder

        if args.data_dir:
            oct_data_dir = Path(args.data_dir)
        else:
            # Try oct_data first (lowercase), fall back to OCT_DATA
            oct_data_dir = retina_builder_dir / 'oct_data'
            if not oct_data_dir.exists():
                oct_data_dir = retina_builder_dir / 'OCT_DATA'

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

        print(f"Found {len(patient_vols)} volumes for patient {args.patient}")
        print(f"  Volume 1: {patient_vols[0].name}")
        print(f"  Volume 2: {patient_vols[1].name}")
        print(f"  Volume 3: {patient_vols[2].name}")

        vol1_path = str(patient_vols[0])
        vol2_path = str(patient_vols[1])
        vol3_path = str(patient_vols[2])

        # Auto-generate output directory
        if not args.output:
            patient_safe = args.patient.replace('_', '').replace('/', '').replace('\\', '')
            output_dir = Path(__file__).parent / f'results_avg_{patient_safe.lower()}'
        else:
            output_dir = args.output

    else:
        # Option 2: Use explicit paths
        if not (args.vol1 and args.vol2 and args.vol3):
            parser.error("Must provide either --patient or all of --vol1, --vol2, --vol3")

        vol1_path = args.vol1
        vol2_path = args.vol2
        vol3_path = args.vol3

        if not args.output:
            parser.error("--output is required when using explicit volume paths")
        output_dir = args.output

    results = averaged_bscan_pipeline(
        vol1_path=vol1_path,
        vol2_path=vol2_path,
        vol3_path=vol3_path,
        output_dir=output_dir,
        n_bscans=args.n_bscans,
        visualize=not args.no_visual
    )

    return results


if __name__ == '__main__':
    main()
