#!/usr/bin/env python3
"""
Multi-Volume Panoramic OCT Stitcher

Main orchestrator for 9-volume panoramic stitching with the following features:

- Cross/Plus pattern layout (1 center + 4 neighbors + 4 extensions)
- B-scan subsampling (z_stride=2) for memory optimization
- Parallel processing at each dependency level
- Progressive canvas merging
- Global coordinate tracking
- Comprehensive visualization

Architecture:
1. Load configuration (volume_layout.json)
2. Build dependency graph
3. Load volumes with subsampling
4. Process levels sequentially, volumes in parallel:
   - Level 1: Volumes 2,4,6,8 → Volume 1 (parallel)
   - Level 2: Volumes 3,5,7,9 → Neighbors (parallel)
5. Progressive canvas merging
6. Generate panoramic visualization

Author: OCT Panoramic Stitching System
"""

import numpy as np
from pathlib import Path
import argparse
import logging
import time
from typing import Dict, List, Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our modular components
from volume_graph import VolumeGraph
from helpers.coordinate_system import GlobalCoordinateSystem, Transform3D
from helpers.parallel_executor import create_executor, AlignmentTask
from helpers.canvas_merger import ProgressiveCanvasMerger
from helpers.oct_loader import OCTImageProcessor, OCTVolumeLoader

# Import alignment step functions
from steps.step1_xz_alignment import perform_xz_alignment
from steps.step2_y_alignment import perform_y_alignment
from steps.step3_rotation_z import perform_z_rotation_alignment
from steps.step3_5_rotation_x import perform_x_rotation_alignment

# Import visualization
from helpers.visualization import generate_3d_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiVolumeStitcher:
    """
    Orchestrates multi-volume panoramic stitching.
    """

    def __init__(self,
                 oct_data_dir: Path,
                 output_dir: Path,
                 config_path: Optional[Path] = None,
                 parallel: bool = True,
                 max_workers: Optional[int] = None):
        """
        Initialize the stitcher.

        Args:
            oct_data_dir: Root directory containing folder_1, folder_2, etc.
            output_dir: Directory for results
            config_path: Path to volume_layout.json (default: auto-detect)
            parallel: Use parallel processing
            max_workers: Maximum worker processes (None = auto)
        """
        self.oct_data_dir = Path(oct_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load configuration and build graph
        self.graph = VolumeGraph(config_path)
        self.graph.print_graph_summary()

        # Get subsampling config
        subsample_config = self.graph.get_subsampling_config()
        self.z_stride = subsample_config.get('z_stride', 1) if subsample_config.get('enabled', False) else 1

        # Initialize coordinate system
        self.coord_system = GlobalCoordinateSystem(z_stride=self.z_stride)

        # Create executor
        self.executor = create_executor(parallel=parallel, max_workers=max_workers)

        # Storage for loaded volumes
        self.volumes: Dict[int, np.ndarray] = {}

        # Storage for aligned volumes
        self.aligned_volumes: Dict[int, np.ndarray] = {}

        # Canvas merger
        self.merger: Optional[ProgressiveCanvasMerger] = None

        logger.info(f"\nMulti-Volume Stitcher initialized")
        logger.info(f"  OCT data directory: {self.oct_data_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Z-stride (subsampling): {self.z_stride}")
        logger.info(f"  Parallel processing: {parallel}")

    def load_all_volumes(self):
        """Load all volumes from disk with subsampling."""
        logger.info("\n" + "="*70)
        logger.info("LOADING VOLUMES")
        logger.info("="*70)

        # Initialize image processor and volume loader
        processor = OCTImageProcessor(
            sidebar_width=250,
            crop_top=100,
            crop_bottom=50
        )
        loader = OCTVolumeLoader(processor)

        # Load each volume
        for vol_id in sorted(self.graph.nodes.keys()):
            folder_name = self.graph.get_volume_folder(vol_id)
            folder_path = self.oct_data_dir / folder_name

            if not folder_path.exists():
                raise FileNotFoundError(f"Volume folder not found: {folder_path}")

            logger.info(f"\nLoading volume {vol_id} from {folder_name}...")

            # Load with subsampling
            volume = loader.load_volume_from_directory(
                str(folder_path),
                z_stride=self.z_stride
            )

            if volume is None:
                raise RuntimeError(f"Failed to load volume {vol_id}")

            self.volumes[vol_id] = volume

            # Register in coordinate system
            original_depth = volume.shape[2] * self.z_stride if self.z_stride > 1 else volume.shape[2]
            self.coord_system.register_volume(
                volume_id=vol_id,
                shape=volume.shape,
                reference_id=self.graph.nodes[vol_id].neighbor,
                original_depth=original_depth
            )

            logger.info(f"  ✓ Volume {vol_id} loaded: {volume.shape}")

        logger.info(f"\n✓ All {len(self.volumes)} volumes loaded successfully")

    def align_volume_pair(self,
                         ref_volume: np.ndarray,
                         mov_volume: np.ndarray,
                         params: Dict) -> Tuple[Transform3D, np.ndarray, Dict]:
        """
        Align a single volume pair using the full alignment pipeline.

        This is the worker function called by the parallel executor.

        Args:
            ref_volume: Reference volume
            mov_volume: Moving volume
            params: Parameters (contains volume IDs, etc.)

        Returns:
            Tuple of (transform, aligned_volume, metrics)
        """
        ref_id = params['reference_id']
        mov_id = params['moving_id']

        logger.info(f"\n{'='*70}")
        logger.info(f"ALIGNING VOLUME {mov_id} → {ref_id}")
        logger.info(f"{'='*70}")

        # Step 1: XZ Alignment
        logger.info("\nStep 1: XZ Alignment (Phase Correlation)")
        step1_results = perform_xz_alignment(ref_volume, mov_volume)

        # Step 2: Y Alignment
        logger.info("\nStep 2: Y Alignment (Center of Mass)")
        step2_results = perform_y_alignment(
            ref_volume,
            step1_results['volume_1_xz_aligned']
        )

        # Get fully aligned volume after XZ + Y
        volume_aligned = step2_results['volume_1_y_aligned']

        # Step 3: Z-Rotation Alignment
        logger.info("\nStep 3: Z-Rotation Alignment (In-Plane)")
        step3_results = perform_z_rotation_alignment(
            ref_volume,
            volume_aligned,
            visualize=False  # Disable visualization in parallel mode
        )

        # Update aligned volume
        if step3_results['volume_1_rotated'] is not None:
            volume_aligned = step3_results['volume_1_rotated']

        # Step 3.5: X-Rotation Alignment (optional - can be disabled for speed)
        # logger.info("\nStep 3.5: X-Rotation Alignment (Sagittal)")
        # step3_5_results = perform_x_rotation_alignment(
        #     ref_volume,
        #     volume_aligned,
        #     visualize=False
        # )
        # if step3_5_results['volume_1_rotated'] is not None:
        #     volume_aligned = step3_5_results['volume_1_rotated']

        # Build comprehensive transform
        transform = Transform3D(
            dx=float(step1_results['offset_x']),
            dy=float(step2_results['y_shift']),
            dz=float(step1_results['offset_z']),
            rotation_z=float(step3_results.get('rotation_angle', 0.0)),
            rotation_x=0.0,  # step3_5_results.get('rotation_angle', 0.0) if using X-rotation
            confidence=float(step1_results['confidence']),
            method='multi_step_pipeline'
        )

        # Metrics
        metrics = {
            'ncc_xz': step1_results['confidence'],
            'y_shift': step2_results['y_shift'],
            'rotation_z': step3_results.get('rotation_angle', 0.0),
            'ncc_after_rotation': step3_results.get('ncc_after', 0.0)
        }

        logger.info(f"\n✓ Alignment complete: {transform}")

        return transform, volume_aligned, metrics

    def process_all_levels(self):
        """Process all dependency levels sequentially, volumes in parallel."""
        logger.info("\n" + "="*70)
        logger.info("PROCESSING ALIGNMENT LEVELS")
        logger.info("="*70)

        # Initialize reference volume (volume 1)
        self.aligned_volumes[1] = self.volumes[1]

        # Get execution levels
        levels = self.graph.get_execution_levels()

        # Process each level (skip level 0 which is the reference)
        for level_idx, vol_ids in enumerate(levels):
            if level_idx == 0:
                continue  # Skip reference volume

            logger.info(f"\n{'='*70}")
            logger.info(f"PROCESSING LEVEL {level_idx}")
            logger.info(f"{'='*70}")

            # Get processing pairs for this level
            pairs = self.graph.get_processing_pairs(level_idx)

            logger.info(f"Volumes to process: {vol_ids}")
            logger.info(f"Processing pairs: {pairs}")

            # Create alignment tasks
            tasks = []
            for ref_id, mov_id in pairs:
                task = AlignmentTask(
                    reference_id=ref_id,
                    moving_id=mov_id,
                    reference_volume=self.aligned_volumes[ref_id],  # Use aligned reference
                    moving_volume=self.volumes[mov_id],
                    level=level_idx,
                    params={
                        'reference_id': ref_id,
                        'moving_id': mov_id
                    }
                )
                tasks.append(task)

            # Execute in parallel
            results = self.executor.execute_level(
                tasks,
                self.align_volume_pair,
                show_progress=True
            )

            # Process results
            for result in results:
                if result.success:
                    # Store aligned volume
                    self.aligned_volumes[result.moving_id] = result.aligned_volume

                    # Update coordinate system
                    self.coord_system.set_transform(
                        volume_id=result.moving_id,
                        transform=result.transform,
                        reference_id=result.reference_id
                    )

                    logger.info(f"✓ Volume {result.moving_id} aligned successfully")
                else:
                    logger.error(f"✗ Volume {result.moving_id} alignment FAILED: {result.error}")
                    raise RuntimeError(f"Alignment failed for volume {result.moving_id}")

        logger.info("\n✓ All levels processed successfully")

    def merge_panorama(self):
        """Merge all aligned volumes into a single panoramic canvas."""
        logger.info("\n" + "="*70)
        logger.info("MERGING PANORAMIC CANVAS")
        logger.info("="*70)

        # Get global canvas size
        canvas_size = self.coord_system.get_global_canvas_size()
        logger.info(f"Required canvas size: {canvas_size}")

        # Initialize merger
        self.merger = ProgressiveCanvasMerger(
            initial_shape=canvas_size,
            merge_strategy='average',
            track_sources=True
        )

        # Add volumes in order
        for vol_id in sorted(self.aligned_volumes.keys()):
            volume = self.aligned_volumes[vol_id]
            offset = self.coord_system.get_volume_offset_in_canvas(vol_id)

            logger.info(f"\nAdding volume {vol_id} to canvas...")
            logger.info(f"  Offset: {offset}")

            self.merger.add_volume(volume, offset, vol_id)

        # Finalize
        canvas, metadata = self.merger.finalize()

        logger.info(f"\n✓ Panoramic canvas created: {canvas.shape}")

        return canvas, metadata

    def generate_visualizations(self, canvas: np.ndarray):
        """Generate visualizations of the panoramic volume."""
        logger.info("\n" + "="*70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*70)

        # Use existing visualization functions
        # For now, save the merged volume
        output_path = self.output_dir / 'panorama_merged_9volumes.npy'
        np.save(output_path, canvas)
        logger.info(f"✓ Saved merged panorama: {output_path}")

        # TODO: Generate 3D visualizations using visualization_3d module
        # This would require adapting the generate_3d_visualizations function
        # to work with multi-volume merged canvas

        logger.info("\n✓ Visualizations complete")

    def run(self):
        """Execute the complete panoramic stitching pipeline."""
        start_time = time.time()

        logger.info("\n" + "="*70)
        logger.info("MULTI-VOLUME PANORAMIC STITCHER")
        logger.info("="*70)

        try:
            # Step 1: Load volumes
            self.load_all_volumes()

            # Step 2: Process alignment levels
            self.process_all_levels()

            # Step 3: Merge panorama
            canvas, metadata = self.merge_panorama()

            # Step 4: Generate visualizations
            self.generate_visualizations(canvas)

            # Print summary
            total_time = time.time() - start_time
            logger.info("\n" + "="*70)
            logger.info("PANORAMIC STITCHING COMPLETE!")
            logger.info("="*70)
            logger.info(f"Total execution time: {total_time:.1f}s")
            logger.info(f"Volumes processed: {len(self.volumes)}")
            logger.info(f"Final panorama size: {canvas.shape}")
            logger.info(f"Coverage: {metadata.coverage_percent:.1f}%")
            logger.info(f"Output directory: {self.output_dir}")

            return True

        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Multi-Volume Panoramic OCT Stitcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (parallel, z_stride=2)
  python multi_volume_stitcher.py --oct-data ../OCT_DATA --output ./results

  # Run sequentially (for debugging)
  python multi_volume_stitcher.py --oct-data ../OCT_DATA --output ./results --no-parallel

  # Use custom number of workers
  python multi_volume_stitcher.py --oct-data ../OCT_DATA --output ./results --workers 2
        """
    )

    parser.add_argument('--oct-data', type=str, required=True,
                       help='Root directory containing folder_1, folder_2, etc.')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to volume_layout.json (default: auto-detect)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing (sequential mode)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Maximum worker processes (default: auto)')

    args = parser.parse_args()

    # Create stitcher
    stitcher = MultiVolumeStitcher(
        oct_data_dir=Path(args.oct_data),
        output_dir=Path(args.output),
        config_path=Path(args.config) if args.config else None,
        parallel=not args.no_parallel,
        max_workers=args.workers
    )

    # Run pipeline
    success = stitcher.run()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
