"""
OCT Volume Alignment Pipeline - Modular Version

Main orchestrator that coordinates all alignment steps.
Each step is implemented in a separate module for better organization and maintainability.

Usage:
    python alignment_pipeline.py --step 1        # Run Step 1 only
    python alignment_pipeline.py --steps 1 2 3   # Run Steps 1, 2, 3
    python alignment_pipeline.py --all           # Run all steps
    python alignment_pipeline.py --visual-only   # Regenerate visualizations only
"""

import numpy as np
import argparse
from pathlib import Path
import time
from multiprocessing import Process, freeze_support

# Set up parent directory path
parent_dir = Path(__file__).parent.parent

# Import step modules
from steps import (
    step1_xz_alignment,
    step2_y_alignment,
    step3_rotation_z
    # step3_5_rotation_x removed - no longer used
)

# Import helper modules
from helpers import (
    apply_all_transformations_to_volume,
    generate_3d_visualizations,
    OCTImageProcessor,
    OCTVolumeLoader
)
from helpers.step_visualization import visualize_all_steps


def _visualize_all_steps_process(volume_0, step1_results, step2_results, step3_results, data_dir):
    """
    Wrapper for visualize_all_steps to run in separate process.

    Each process gets its own matplotlib instance, avoiding threading issues.
    """
    import matplotlib
    matplotlib.use('Agg')  # Set backend in this process
    visualize_all_steps(volume_0, step1_results, step2_results, step3_results, data_dir)


def _generate_visualizations_process(volume_0, volume_1, step1_results, step2_results, step3_results, data_dir):
    """
    Wrapper for generate_visualizations to run in separate process.

    Each process gets its own matplotlib instance, avoiding threading issues.
    """
    import matplotlib
    matplotlib.use('Agg')  # Set backend in this process
    generate_visualizations(volume_0, volume_1, step1_results, step2_results, step3_results, data_dir)


def main():
    """Main pipeline orchestrator."""

    parser = argparse.ArgumentParser(
        description='OCT Volume Alignment Pipeline - Modular Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default data (F001_IP)
  python alignment_pipeline.py --all

  # Run with EM005 data
  python alignment_pipeline.py --all --patient EM005

  # Run with EM005 and visualization
  python alignment_pipeline.py --all --visual --patient EM005

  # Run specific steps
  python alignment_pipeline.py --step 1 --patient EM005
  python alignment_pipeline.py --steps 1 2 3 --patient EM005

  # Regenerate visualizations only
  python alignment_pipeline.py --visual-only --patient EM005
        """
    )

    # Step selection arguments
    parser.add_argument('--step', type=int, choices=[1, 2, 3],
                        help='Run a single step')
    parser.add_argument('--steps', type=int, nargs='+', choices=[1, 2, 3],
                        help='Run multiple steps in sequence')
    parser.add_argument('--all', action='store_true',
                        help='Run all steps (1, 2, 3, 3.5)')

    # Visualization options
    parser.add_argument('--visual', action='store_true',
                        help='Generate 3D visualizations after pipeline completes')
    parser.add_argument('--visual-only', action='store_true',
                        help='Only regenerate 3D visualizations from saved results')

    # Data selection options
    parser.add_argument('--patient', type=str, default='F001_IP',
                        help='Patient ID to search for (e.g., F001_IP, EM005)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override OCT data directory (default: ../oct_data)')

    args = parser.parse_args()

    # Determine which steps to run
    if args.step:
        steps_to_run = [args.step]
    elif args.steps:
        steps_to_run = sorted(set(args.steps))
    elif args.all:
        steps_to_run = [1, 2, 3]
    elif args.visual_only:
        steps_to_run = []
    else:
        parser.print_help()
        return

    # Set up paths
    if args.data_dir:
        oct_data_dir = Path(args.data_dir)
    else:
        # Try oct_data first (lowercase), fall back to OCT_DATA
        oct_data_dir = parent_dir / 'oct_data'
        if not oct_data_dir.exists():
            oct_data_dir = parent_dir / 'OCT_DATA'

    # Create patient-specific results directory
    patient_safe = args.patient.replace('_', '').replace('/', '').replace('\\', '')
    data_dir = Path(__file__).parent / f'results_{patient_safe.lower()}'
    data_dir.mkdir(exist_ok=True)

    # Start total pipeline timer
    pipeline_start = time.time()
    timing_results = {}

    print("="*70)
    print("OCT VOLUME ALIGNMENT PIPELINE - MODULAR ARCHITECTURE (PARALLEL)")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"OCT data directory: {oct_data_dir}")

    # Handle visualization-only mode
    if args.visual_only:
        handle_visual_only_mode(oct_data_dir, data_dir, args.patient)
        return

    # Load OCT volumes
    print("\n" + "="*70)
    print("LOADING OCT VOLUMES")
    print("="*70)

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

    if len(patient_vols) < 2:
        raise ValueError(f"Need at least 2 volumes for patient {args.patient}, found {len(patient_vols)}")

    print(f"  Found {len(patient_vols)} volumes for patient {args.patient}")
    print(f"  Volume 0: {patient_vols[0].name}")
    print(f"  Volume 1: {patient_vols[1].name}")

    volume_0 = loader.load_volume_from_directory(str(patient_vols[0]))
    volume_1 = loader.load_volume_from_directory(str(patient_vols[1]))

    print(f"  V0 shape: {volume_0.shape}")
    print(f"  V1 shape: {volume_1.shape}")

    loading_time = time.time() - loading_start
    timing_results['loading'] = loading_time
    print(f"\n⏱️  Volume loading time: {loading_time:.2f} seconds")

    # Initialize results
    step1_results = None
    step2_results = None
    step3_results = None

    # Execute steps with timing
    for step_num in steps_to_run:
        if step_num == 1:
            step1_start = time.time()
            step1_results = step1_xz_alignment(volume_0, volume_1, data_dir)
            step1_time = time.time() - step1_start
            timing_results['step1'] = step1_time

            np.save(data_dir / 'step1_results.npy', step1_results)
            print(f"  [SAVED] step1_results.npy")
            print(f"⏱️  Step 1 time: {step1_time:.2f} seconds")

        elif step_num == 2:
            if step1_results is None:
                step1_results = np.load(data_dir / 'step1_results.npy', allow_pickle=True).item()

            step2_start = time.time()
            step2_results = step2_y_alignment(step1_results, data_dir)
            step2_time = time.time() - step2_start
            timing_results['step2'] = step2_time

            np.save(data_dir / 'step2_results.npy', step2_results)
            print(f"  [SAVED] step2_results.npy")
            print(f"⏱️  Step 2 time: {step2_time:.2f} seconds")

        elif step_num == 3:
            if step1_results is None:
                step1_results = np.load(data_dir / 'step1_results.npy', allow_pickle=True).item()
            if step2_results is None:
                step2_results = np.load(data_dir / 'step2_results.npy', allow_pickle=True).item()

            # Run Step 3 (Z-rotation)
            step3_start = time.time()
            step3_results = step3_rotation_z(step1_results, step2_results, data_dir, visualize=args.visual)
            step3_time = time.time() - step3_start
            timing_results['step3'] = step3_time

            # Step 3.5 (X-rotation) REMOVED - was causing incorrect alignment

            np.save(data_dir / 'step3_results.npy', step3_results)
            print(f"  [SAVED] step3_results.npy")
            print(f"⏱️  Step 3 time: {step3_time:.2f} seconds")

    # Generate visualizations if requested (IN PARALLEL using multiprocessing)
    if args.visual and step1_results and step2_results:
        viz_start = time.time()
        print("\n🎨 Generating visualizations in PARALLEL (multiprocessing)...")

        # Create separate processes for each visualization task
        # Each process has its own Python interpreter and matplotlib instance
        process1 = Process(
            target=_visualize_all_steps_process,
            args=(volume_0, step1_results, step2_results, step3_results, data_dir)
        )
        process2 = Process(
            target=_generate_visualizations_process,
            args=(volume_0, volume_1, step1_results, step2_results, step3_results, data_dir)
        )

        # Start both processes
        process1.start()
        process2.start()

        # Wait for both to complete
        process1.join()
        process2.join()

        viz_time = time.time() - viz_start
        timing_results['visualization'] = viz_time
        print(f"⏱️  Visualization time: {viz_time:.2f} seconds")

    # Calculate total time
    total_time = time.time() - pipeline_start
    timing_results['total'] = total_time

    # Print timing summary
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print("\n⏱️  TIMING SUMMARY:")
    print("="*70)
    if 'loading' in timing_results:
        print(f"  Volume Loading:    {timing_results['loading']:>8.2f}s")
    if 'step1' in timing_results:
        print(f"  Step 1 (XZ):       {timing_results['step1']:>8.2f}s")
    if 'step2' in timing_results:
        print(f"  Step 2 (Y):        {timing_results['step2']:>8.2f}s")
    if 'step3' in timing_results:
        print(f"  Step 3 (Rotation): {timing_results['step3']:>8.2f}s")
    if 'visualization' in timing_results:
        print(f"  Visualization:     {timing_results['visualization']:>8.2f}s")
    print(f"  {'─'*30}")
    print(f"  TOTAL TIME:        {total_time:>8.2f}s ({total_time/60:.1f} min)")
    print("="*70)

    # Save timing results as human-readable text file
    timing_file = data_dir / 'timing_results.txt'
    with open(timing_file, 'w', encoding='utf-8') as f:
        f.write("OCT ALIGNMENT PIPELINE - TIMING RESULTS\n")
        f.write("="*70 + "\n\n")
        if 'loading' in timing_results:
            f.write(f"Volume Loading:    {timing_results['loading']:>8.2f}s\n")
        if 'step1' in timing_results:
            f.write(f"Step 1 (XZ):       {timing_results['step1']:>8.2f}s\n")
        if 'step2' in timing_results:
            f.write(f"Step 2 (Y):        {timing_results['step2']:>8.2f}s\n")
        if 'step3' in timing_results:
            f.write(f"Step 3 (Rotation): {timing_results['step3']:>8.2f}s\n")
        if 'visualization' in timing_results:
            f.write(f"Visualization:     {timing_results['visualization']:>8.2f}s\n")
        f.write("─"*30 + "\n")
        f.write(f"TOTAL TIME:        {total_time:>8.2f}s ({total_time/60:.1f} min)\n")
        f.write("\n" + "="*70 + "\n")
    print(f"\n[SAVED] timing_results.txt")


def handle_visual_only_mode(oct_data_dir, data_dir, patient_id):
    """Handle visualization-only mode."""
    print("="*70)
    print("3D VISUALIZATION ONLY MODE")
    print("="*70)
    print("Loading existing alignment results...")

    # Check if required results exist
    required_files = [
        data_dir / 'step1_results.npy',
        data_dir / 'step2_results.npy',
        data_dir / 'step3_results.npy'
    ]

    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print(f"\n❌ Error: Missing required files:")
        for f in missing_files:
            print(f"   - {f.name}")
        print("\n   Run the full pipeline first: python alignment_pipeline.py --all --patient", patient_id)
        return

    # Load OCT volumes
    print("\nLoading OCT volumes...")
    processor = OCTImageProcessor(sidebar_width=250, crop_top=100, crop_bottom=50)
    loader = OCTVolumeLoader(processor)

    # Find volume directories
    bmp_dirs = []
    for bmp_file in oct_data_dir.rglob('*.bmp'):
        vol_dir = bmp_file.parent
        if vol_dir not in bmp_dirs:
            bmp_dirs.append(vol_dir)

    patient_vols = sorted([v for v in bmp_dirs if patient_id in str(v)])

    if len(patient_vols) < 2:
        print(f"\n❌ Error: Need at least 2 {patient_id} volumes")
        return

    volume_0 = loader.load_volume_from_directory(str(patient_vols[0]))
    volume_1 = loader.load_volume_from_directory(str(patient_vols[1]))
    print(f"  [OK] Loaded volume_0: {volume_0.shape}")
    print(f"  [OK] Loaded volume_1: {volume_1.shape}")

    # Load step results
    step1_results = np.load(data_dir / 'step1_results.npy', allow_pickle=True).item()
    step2_results = np.load(data_dir / 'step2_results.npy', allow_pickle=True).item()
    step3_results = np.load(data_dir / 'step3_results.npy', allow_pickle=True).item()
    print(f"  [OK] Loaded alignment results")

    # Generate visualizations IN PARALLEL using multiprocessing
    print("\n🎨 Generating visualizations in PARALLEL (multiprocessing)...")

    # Create separate processes for each visualization task
    process1 = Process(
        target=_visualize_all_steps_process,
        args=(volume_0, step1_results, step2_results, step3_results, data_dir)
    )
    process2 = Process(
        target=_generate_visualizations_process,
        args=(volume_0, volume_1, step1_results, step2_results, step3_results, data_dir)
    )

    # Start both processes
    process1.start()
    process2.start()

    # Wait for both to complete
    process1.join()
    process2.join()

    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)


def generate_visualizations(volume_0, volume_1, step1_results, step2_results, step3_results, data_dir):
    """Generate 3D visualizations with full transformation applied (PARALLEL)."""

    print("\n  Applying all transformations to volume_1 for visualization...")

    # Apply ALL transformations (XZ, Y, rotation) to get properly aligned volume
    # This is necessary because Step 3 applies rotation which must be in the data
    volume_1_aligned = apply_all_transformations_to_volume(
        volume_1,
        step1_results,
        step2_results,
        step3_results
    )

    # Generate 3D visualizations with the fully transformed volume
    generate_3d_visualizations(volume_0, step1_results, step2_results, data_dir, step3_results=step3_results, volume_1_aligned=volume_1_aligned)


if __name__ == '__main__':
    freeze_support()  # Required for Windows multiprocessing
    main()
