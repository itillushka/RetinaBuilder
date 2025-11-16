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


def main():
    """Main pipeline orchestrator."""

    parser = argparse.ArgumentParser(
        description='OCT Volume Alignment Pipeline - Modular Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run specific step
  python alignment_pipeline.py --step 1
  python alignment_pipeline.py --step 2
  python alignment_pipeline.py --step 3

  # Run multiple steps
  python alignment_pipeline.py --steps 1 2 3

  # Run all steps
  python alignment_pipeline.py --all

  # Run with visualization
  python alignment_pipeline.py --all --visual

  # Regenerate visualizations only
  python alignment_pipeline.py --visual-only
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
    oct_data_dir = parent_dir / 'OCT_DATA'
    data_dir = Path(__file__).parent / 'results'  # Save results in scripts_modular/results
    data_dir.mkdir(exist_ok=True)

    print("="*70)
    print("OCT VOLUME ALIGNMENT PIPELINE - MODULAR ARCHITECTURE")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"OCT data directory: {oct_data_dir}")

    # Handle visualization-only mode
    if args.visual_only:
        handle_visual_only_mode(oct_data_dir, data_dir)
        return

    # Load OCT volumes
    print("\n" + "="*70)
    print("LOADING OCT VOLUMES")
    print("="*70)

    processor = OCTImageProcessor(sidebar_width=250, crop_top=100, crop_bottom=50)
    loader = OCTVolumeLoader(processor)

    # Find volume directories
    bmp_dirs = []
    for bmp_file in oct_data_dir.rglob('*.bmp'):
        vol_dir = bmp_file.parent
        if vol_dir not in bmp_dirs:
            bmp_dirs.append(vol_dir)

    f001_vols = sorted([v for v in bmp_dirs if 'F001_IP' in str(v)])

    if len(f001_vols) < 2:
        raise ValueError("Need at least 2 F001 volumes")

    print(f"  Volume 0: {f001_vols[0].name}")
    print(f"  Volume 1: {f001_vols[1].name}")

    volume_0 = loader.load_volume_from_directory(str(f001_vols[0]))
    volume_1 = loader.load_volume_from_directory(str(f001_vols[1]))

    print(f"  V0 shape: {volume_0.shape}")
    print(f"  V1 shape: {volume_1.shape}")

    # Initialize results
    step1_results = None
    step2_results = None
    step3_results = None

    # Execute steps
    for step_num in steps_to_run:
        if step_num == 1:
            step1_results = step1_xz_alignment(volume_0, volume_1, data_dir)
            np.save(data_dir / 'step1_results.npy', step1_results)
            print(f"  [SAVED] step1_results.npy")

        elif step_num == 2:
            if step1_results is None:
                step1_results = np.load(data_dir / 'step1_results.npy', allow_pickle=True).item()
            step2_results = step2_y_alignment(step1_results, data_dir)
            np.save(data_dir / 'step2_results.npy', step2_results)
            print(f"  [SAVED] step2_results.npy")

        elif step_num == 3:
            if step1_results is None:
                step1_results = np.load(data_dir / 'step1_results.npy', allow_pickle=True).item()
            if step2_results is None:
                step2_results = np.load(data_dir / 'step2_results.npy', allow_pickle=True).item()

            # Run Step 3 (Z-rotation)
            step3_results = step3_rotation_z(step1_results, step2_results, data_dir)

            # Step 3.5 (X-rotation) REMOVED - was causing incorrect alignment

            np.save(data_dir / 'step3_results.npy', step3_results)
            print(f"  [SAVED] step3_results.npy")

    # Generate visualizations if requested
    if args.visual and step1_results and step2_results:
        # Generate step-by-step YZ views
        visualize_all_steps(volume_0, step1_results, step2_results, step3_results, data_dir)

        # Generate 3D visualizations
        generate_visualizations(volume_0, volume_1, step1_results, step2_results, step3_results, data_dir)

    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)


def handle_visual_only_mode(oct_data_dir, data_dir):
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
        print("\n   Run the full pipeline first: python alignment_pipeline.py --all")
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

    f001_vols = sorted([v for v in bmp_dirs if 'F001_IP' in str(v)])

    if len(f001_vols) < 2:
        print("\n❌ Error: Need at least 2 F001_IP volumes")
        return

    volume_0 = loader.load_volume_from_directory(str(f001_vols[0]))
    volume_1 = loader.load_volume_from_directory(str(f001_vols[1]))
    print(f"  [OK] Loaded volume_0: {volume_0.shape}")
    print(f"  [OK] Loaded volume_1: {volume_1.shape}")

    # Load step results
    step1_results = np.load(data_dir / 'step1_results.npy', allow_pickle=True).item()
    step2_results = np.load(data_dir / 'step2_results.npy', allow_pickle=True).item()
    step3_results = np.load(data_dir / 'step3_results.npy', allow_pickle=True).item()
    print(f"  [OK] Loaded alignment results")

    # Generate step-by-step YZ views
    visualize_all_steps(volume_0, step1_results, step2_results, step3_results, data_dir)

    # Generate 3D visualizations
    generate_visualizations(volume_0, volume_1, step1_results, step2_results, step3_results, data_dir)

    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)


def generate_visualizations(volume_0, volume_1, step1_results, step2_results, step3_results, data_dir):
    """Generate 3D visualizations."""

    # Apply all transformations to original volume_1
    # Step 3.5 (X-rotation) removed as per user request
    volume_1_aligned = apply_all_transformations_to_volume(
        volume_1,
        step1_results,
        step2_results,
        step3_results,
        step3_5_results=None,  # Step 3.5 disabled
        step4_results=None
    )

    # Generate 3D visualizations
    generate_3d_visualizations(volume_0, step1_results, step2_results, data_dir, step3_results=step3_results, volume_1_aligned=volume_1_aligned)


if __name__ == '__main__':
    main()
