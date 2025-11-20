"""
Three-Volume Sequential Alignment Pipeline

Aligns 3 OCT volumes in a chain: Center -> Right -> Far Right
Uses EM005 data from emmetropes dataset.

Process:
1. Load 3 volumes (center, right, far_right)
2. Align volume 2 (right) to volume 1 (center)
3. Align volume 3 (far_right) to volume 2 (right)
4. Merge all three into a single canvas

This is a simplified test script to validate the pairwise sequential alignment approach.

Usage:
    python three_volume_alignment.py --all
    python three_volume_alignment.py --visual
"""

import numpy as np
import argparse
from pathlib import Path
from scipy import ndimage
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import step modules - USE SAME FUNCTIONS AS alignment_pipeline.py
from steps.step1_xz_alignment import step1_xz_alignment
from steps.step2_y_alignment import step2_y_alignment
from steps.step3_rotation_z import step3_rotation_z

# Import helper modules
from helpers.oct_loader import OCTImageProcessor, OCTVolumeLoader
from helpers.visualization import visualize_multi_volume_panorama


class ThreeVolumeAligner:
    """Aligns 3 volumes in a horizontal chain: center -> right -> far_right."""

    def __init__(self, data_dir: Path, output_dir: Path):
        """
        Initialize aligner.

        Args:
            data_dir: Path to EM005 data directory
            output_dir: Path to save results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Storage for volumes and step results
        self.volumes = {}  # Original volumes (never modified)
        self.step_results = {}  # Store step results for each pair

        print("="*70)
        print("THREE-VOLUME SEQUENTIAL ALIGNMENT")
        print("="*70)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")

    def load_volumes(self):
        """Load 3 volumes from EM005 directory."""
        print("\n" + "="*70)
        print("LOADING VOLUMES")
        print("="*70)

        # Initialize loader
        processor = OCTImageProcessor(
            sidebar_width=250,
            crop_top=100,
            crop_bottom=50
        )
        loader = OCTVolumeLoader(processor)

        # Find all volume directories in EM005
        vol_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and 'EM005' in d.name
        ])

        if len(vol_dirs) < 3:
            raise ValueError(f"Need at least 3 volumes, found {len(vol_dirs)}")

        # Load first 3 volumes (center, right, far_right based on cross pattern)
        volume_names = ['center', 'right', 'far_right']

        for i, (vol_dir, name) in enumerate(zip(vol_dirs[:3], volume_names)):
            print(f"\nLoading volume {i+1} ({name}): {vol_dir.name}")

            volume = loader.load_volume_from_directory(str(vol_dir))

            if volume is None:
                raise RuntimeError(f"Failed to load volume from {vol_dir}")

            self.volumes[i+1] = volume
            print(f"  Shape: {volume.shape}")
            print(f"  Range: [{volume.min():.1f}, {volume.max():.1f}]")

        print(f"\nLoaded {len(self.volumes)} volumes successfully")

    def align_pair(self, ref_id: int, mov_id: int, pair_name: str):
        """
        Align a pair of volumes using EXACT same functions as alignment_pipeline.py

        Args:
            ref_id: Reference volume ID
            mov_id: Moving volume ID
            pair_name: Name for this pair (e.g., "vol2_to_vol1")

        Returns:
            Tuple of (step1_results, step2_results, step3_results)
        """
        print(f"\n{'='*70}")
        print(f"ALIGNING VOLUME {mov_id} -> VOLUME {ref_id}")
        print(f"{'='*70}")

        ref_volume = self.volumes[ref_id]
        mov_volume = self.volumes[mov_id]

        # Create temporary output directory for this pair
        pair_dir = self.output_dir / pair_name
        pair_dir.mkdir(exist_ok=True)

        # Step 1: XZ Alignment - EXACT SAME AS alignment_pipeline.py
        step1_results = step1_xz_alignment(ref_volume, mov_volume, pair_dir)
        np.save(pair_dir / 'step1_results.npy', step1_results)
        print(f"  [SAVED] {pair_name}/step1_results.npy")

        # Step 2: Y Alignment - EXACT SAME AS alignment_pipeline.py
        step2_results = step2_y_alignment(step1_results, pair_dir)
        np.save(pair_dir / 'step2_results.npy', step2_results)
        print(f"  [SAVED] {pair_name}/step2_results.npy")

        # Step 3: Rotation Alignment - EXACT SAME AS alignment_pipeline.py
        step3_results = step3_rotation_z(step1_results, step2_results, pair_dir)
        np.save(pair_dir / 'step3_results.npy', step3_results)
        print(f"  [SAVED] {pair_name}/step3_results.npy")

        # Store results
        self.step_results[mov_id] = {
            'step1': step1_results,
            'step2': step2_results,
            'step3': step3_results,
            'reference_id': ref_id
        }

        # Extract key metrics for summary
        print(f"\nSummary for volume {mov_id}:")
        print(f"  XZ offset: dx={step1_results['offset_x']:.1f}, dz={step1_results['offset_z']:.1f}")
        print(f"  Y-shift: {step2_results['y_shift']:.1f} px")
        print(f"  Rotation: {step3_results.get('rotation_angle', 0.0):.2f}°")
        print(f"  Confidence: {step1_results['confidence']:.2f}")

        return step1_results, step2_results, step3_results

    def extract_transforms_from_results(self):
        """Extract transform parameters from step results."""
        print("\n" + "="*70)
        print("EXTRACTING TRANSFORMS")
        print("="*70)

        transforms = {}

        for vol_id, results in self.step_results.items():
            step1 = results['step1']
            step2 = results['step2']
            step3 = results['step3']

            transforms[vol_id] = {
                'dx': float(step1['offset_x']),
                'dy': float(step2['y_shift']),
                'dz': float(step1['offset_z']),
                'rotation_z': float(step3.get('rotation_angle', 0.0)),
                'confidence': float(step1['confidence']),
                'reference_id': results['reference_id']
            }

            print(f"  Volume {vol_id}: dx={transforms[vol_id]['dx']:.1f}, "
                  f"dy={transforms[vol_id]['dy']:.1f}, dz={transforms[vol_id]['dz']:.1f}, "
                  f"rot={transforms[vol_id]['rotation_z']:.2f}°")

        return transforms

    def merge_volumes(self):
        """
        Merge all volumes into a single canvas.
        NOTE: This is a simplified merge for testing.
        Proper implementation should apply transforms from step results.
        """
        print("\n" + "="*70)
        print("MERGING VOLUMES (SIMPLIFIED)")
        print("="*70)
        print("Note: Full merge implementation with proper transform application needed")

        # For now, just extract the aligned volumes from step results
        # Volume 1 is reference (no transform)
        vol1 = self.volumes[1]

        # Volume 2: get aligned volume from step3 results
        vol2_results = self.step_results[2]
        vol2_aligned = vol2_results['step3']['overlap_v1_rotated']

        # Volume 3: get aligned volume from step3 results
        vol3_results = self.step_results[3]
        vol3_aligned = vol3_results['step3']['overlap_v1_rotated']

        print(f"Volume 1 (reference): {vol1.shape}")
        print(f"Volume 2 (aligned): {vol2_aligned.shape}")
        print(f"Volume 3 (aligned): {vol3_aligned.shape}")

        # Simple placeholder canvas
        canvas_shape = (
            max(vol1.shape[0], vol2_aligned.shape[0], vol3_aligned.shape[0]),
            vol1.shape[1] + vol2_aligned.shape[1] + vol3_aligned.shape[1],
            max(vol1.shape[2], vol2_aligned.shape[2], vol3_aligned.shape[2])
        )

        canvas = np.zeros(canvas_shape, dtype=np.float32)
        source_labels = np.zeros(canvas_shape, dtype=np.uint8)

        print(f"\nCanvas shape: {canvas_shape}")

        return canvas, source_labels

    def generate_visualization(self, canvas, source_labels):
        """Generate 3D visualization of merged result."""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATION")
        print("="*70)

        # Save merged volume
        np.save(self.output_dir / 'three_volume_merged.npy', canvas)
        print(f"Saved: three_volume_merged.npy")

        # Generate 3D panorama visualization
        visualize_multi_volume_panorama(
            merged_volume=canvas,
            source_labels=source_labels,
            output_dir=self.output_dir,
            subsample=8,
            percentile=75
        )

        print("Generated 3D visualization")

    def run(self, visualize=True):
        """Run the full alignment pipeline - EXACT same as alignment_pipeline.py but for 3 volumes."""

        # Step 1: Load volumes
        self.load_volumes()

        # Step 2: Align volume 2 to volume 1 (EXACT SAME as alignment_pipeline.py)
        print("\n" + "="*70)
        print("PAIR 1: VOLUME 2 -> VOLUME 1")
        print("="*70)
        self.align_pair(ref_id=1, mov_id=2, pair_name='vol2_to_vol1')

        # Step 3: Align volume 3 to volume 2 (SAME process, different pair)
        print("\n" + "="*70)
        print("PAIR 2: VOLUME 3 -> VOLUME 2")
        print("="*70)
        self.align_pair(ref_id=2, mov_id=3, pair_name='vol3_to_vol2')

        # Step 4: Extract transforms
        transforms = self.extract_transforms_from_results()

        # Step 5: Merge volumes (placeholder for now)
        canvas, source_labels = self.merge_volumes()

        # Save transforms for reference
        np.save(self.output_dir / 'transforms.npy', transforms)
        print(f"\n[SAVED] transforms.npy")

        print("\n" + "="*70)
        print("ALIGNMENT COMPLETE!")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")
        print("\nNote: Each pair uses EXACT same functions as alignment_pipeline.py")
        print("      Vol 2->1 results should be identical to running alignment_pipeline")

        return canvas, source_labels


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Three-Volume Sequential Alignment (EM005)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--all', action='store_true',
                       help='Run full alignment pipeline')
    parser.add_argument('--visual', action='store_true',
                       help='Generate visualizations (included with --all)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory')

    args = parser.parse_args()

    if not args.all:
        parser.print_help()
        return

    # Set up paths
    parent_dir = Path(__file__).parent.parent

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = parent_dir / 'oct_data' / 'emmetropes' / 'EM005'

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / 'results_em005'

    # Run aligner
    aligner = ThreeVolumeAligner(data_dir, output_dir)
    aligner.run(visualize=True)


if __name__ == '__main__':
    main()
