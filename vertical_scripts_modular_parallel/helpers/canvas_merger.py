"""
Progressive Canvas Merger

Manages progressive merging of multiple OCT volumes into a single panoramic canvas.

Features:
- Incremental canvas expansion as volumes are added
- Multiple merge strategies (average, max, additive)
- Source volume tracking for visualization
- Memory-efficient implementation
- Metadata tracking (coverage, overlap, data loss)

Author: OCT Panoramic Stitching System
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CanvasMetadata:
    """Metadata about the merged canvas."""

    # Canvas dimensions
    shape: Tuple[int, int, int]

    # Number of volumes merged
    num_volumes: int = 0

    # Coverage statistics
    total_voxels: int = 0
    nonzero_voxels: int = 0
    coverage_percent: float = 0.0

    # Overlap statistics
    overlap_voxels: int = 0
    overlap_percent: float = 0.0

    # Data loss (voxels that were cropped/lost during alignment)
    data_loss_percent: float = 0.0

    # Volume contributions
    volume_contributions: Dict[int, int] = None  # volume_id -> num_voxels

    def __post_init__(self):
        if self.volume_contributions is None:
            self.volume_contributions = {}


class ProgressiveCanvasMerger:
    """
    Progressively builds a panoramic canvas by merging volumes incrementally.

    The canvas expands as needed to accommodate new volumes based on their
    global coordinates from the coordinate system.
    """

    def __init__(self,
                 initial_shape: Tuple[int, int, int],
                 merge_strategy: str = 'average',
                 track_sources: bool = True):
        """
        Initialize the canvas merger.

        Args:
            initial_shape: Initial canvas size (height, width, depth)
            merge_strategy: How to merge overlapping regions
                          'average': Average intensities
                          'max': Maximum intensity
                          'additive': Sum intensities (for visualization)
            track_sources: Whether to track which volume contributed each voxel
        """
        self.merge_strategy = merge_strategy
        self.track_sources = track_sources

        # Initialize canvas
        self.canvas = np.zeros(initial_shape, dtype=np.float32)

        # Count matrix: tracks how many volumes contributed to each voxel
        self.count_matrix = np.zeros(initial_shape, dtype=np.uint8)

        # Source labels: tracks which volume each voxel came from
        # 0 = background, 1-9 = volume IDs, 255 = overlap
        if track_sources:
            self.source_labels = np.zeros(initial_shape, dtype=np.uint8)
        else:
            self.source_labels = None

        # Metadata
        self.metadata = CanvasMetadata(shape=initial_shape)

        # Current canvas bounds (for potential future optimization)
        self.min_bounds = np.array([0, 0, 0])
        self.max_bounds = np.array(initial_shape)

        logger.info(f"Canvas initialized: shape={initial_shape}, strategy={merge_strategy}")

    def add_volume(self,
                  volume: np.ndarray,
                  offset: Tuple[int, int, int],
                  volume_id: int):
        """
        Add a volume to the canvas at the specified offset.

        Args:
            volume: Volume data (height, width, depth)
            offset: Position in canvas (y_offset, x_offset, z_offset)
            volume_id: Identifier for source tracking
        """
        y_off, x_off, z_off = offset
        vol_h, vol_w, vol_d = volume.shape

        logger.info(f"Adding volume {volume_id} at offset {offset}")

        # Check if volume fits in current canvas
        required_h = y_off + vol_h
        required_w = x_off + vol_w
        required_d = z_off + vol_d

        current_h, current_w, current_d = self.canvas.shape

        # Expand canvas if needed
        if (required_h > current_h or required_w > current_w or required_d > current_d):
            new_h = max(required_h, current_h)
            new_w = max(required_w, current_w)
            new_d = max(required_d, current_d)

            logger.info(f"  Expanding canvas: {self.canvas.shape} → ({new_h}, {new_w}, {new_d})")
            self._expand_canvas((new_h, new_w, new_d))

        # Define regions
        canvas_region = (
            slice(y_off, y_off + vol_h),
            slice(x_off, x_off + vol_w),
            slice(z_off, z_off + vol_d)
        )

        # Get existing data in this region
        existing_data = self.canvas[canvas_region]
        existing_counts = self.count_matrix[canvas_region]

        # Create mask for non-zero voxels in new volume
        volume_mask = volume > 0

        # Merge based on strategy
        if self.merge_strategy == 'average':
            # Weighted average: (existing_sum + new_value) / (count + 1)
            self.canvas[canvas_region] = np.where(
                volume_mask,
                (existing_data * existing_counts + volume) / (existing_counts + 1),
                existing_data
            )

        elif self.merge_strategy == 'max':
            # Maximum intensity
            self.canvas[canvas_region] = np.maximum(existing_data, volume)

        elif self.merge_strategy == 'additive':
            # Simple addition (good for visualization)
            self.canvas[canvas_region] = existing_data + volume

        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")

        # Update count matrix
        self.count_matrix[canvas_region] = np.where(
            volume_mask,
            existing_counts + 1,
            existing_counts
        )

        # Update source labels
        if self.source_labels is not None:
            existing_labels = self.source_labels[canvas_region]

            # Mark overlaps (where count > 1) as 255
            # Otherwise mark with volume_id
            new_labels = np.where(
                volume_mask,
                np.where(
                    existing_counts > 0,
                    255,  # Overlap
                    volume_id  # Single source
                ),
                existing_labels
            )

            self.source_labels[canvas_region] = new_labels

        # Update metadata
        self.metadata.num_volumes += 1
        voxel_contribution = np.sum(volume_mask)
        self.metadata.volume_contributions[volume_id] = voxel_contribution

        logger.info(f"  Added {voxel_contribution:,} voxels from volume {volume_id}")

    def _expand_canvas(self, new_shape: Tuple[int, int, int]):
        """
        Expand the canvas to a new size.

        Args:
            new_shape: New canvas dimensions
        """
        new_h, new_w, new_d = new_shape

        # Create new arrays
        new_canvas = np.zeros(new_shape, dtype=np.float32)
        new_count_matrix = np.zeros(new_shape, dtype=np.uint8)

        # Copy existing data
        old_h, old_w, old_d = self.canvas.shape
        new_canvas[:old_h, :old_w, :old_d] = self.canvas
        new_count_matrix[:old_h, :old_w, :old_d] = self.count_matrix

        # Update canvas
        self.canvas = new_canvas
        self.count_matrix = new_count_matrix

        # Update source labels if tracked
        if self.source_labels is not None:
            new_labels = np.zeros(new_shape, dtype=np.uint8)
            new_labels[:old_h, :old_w, :old_d] = self.source_labels
            self.source_labels = new_labels

        # Update metadata
        self.metadata.shape = new_shape
        self.max_bounds = np.array(new_shape)

    def finalize(self) -> Tuple[np.ndarray, CanvasMetadata]:
        """
        Finalize the canvas and compute final statistics.

        Returns:
            Tuple of (merged_canvas, metadata)
        """
        logger.info("\nFinalizing canvas...")

        # Compute statistics
        self.metadata.total_voxels = np.prod(self.canvas.shape)
        self.metadata.nonzero_voxels = np.sum(self.canvas > 0)
        self.metadata.coverage_percent = (
            100.0 * self.metadata.nonzero_voxels / self.metadata.total_voxels
        )

        # Overlap statistics (voxels contributed by multiple volumes)
        self.metadata.overlap_voxels = np.sum(self.count_matrix > 1)
        if self.metadata.nonzero_voxels > 0:
            self.metadata.overlap_percent = (
                100.0 * self.metadata.overlap_voxels / self.metadata.nonzero_voxels
            )

        logger.info(f"  Total voxels: {self.metadata.total_voxels:,}")
        logger.info(f"  Non-zero voxels: {self.metadata.nonzero_voxels:,} ({self.metadata.coverage_percent:.1f}%)")
        logger.info(f"  Overlap voxels: {self.metadata.overlap_voxels:,} ({self.metadata.overlap_percent:.1f}% of data)")

        # Volume contributions
        logger.info("\n  Volume contributions:")
        for vol_id in sorted(self.metadata.volume_contributions.keys()):
            voxels = self.metadata.volume_contributions[vol_id]
            percent = 100.0 * voxels / self.metadata.nonzero_voxels if self.metadata.nonzero_voxels > 0 else 0
            logger.info(f"    Volume {vol_id}: {voxels:,} voxels ({percent:.1f}%)")

        return self.canvas, self.metadata

    def get_canvas(self) -> np.ndarray:
        """Get the current canvas."""
        return self.canvas

    def get_source_labels(self) -> Optional[np.ndarray]:
        """Get the source label map."""
        return self.source_labels

    def get_overlap_map(self) -> np.ndarray:
        """
        Get a binary map showing overlap regions.

        Returns:
            Boolean array where True indicates overlap (count > 1)
        """
        return self.count_matrix > 1

    def crop_to_content(self, padding: int = 0) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
        """
        Crop the canvas to remove empty space around the content.

        Args:
            padding: Extra padding to keep around content (in voxels)

        Returns:
            Tuple of (cropped_canvas, crop_slices)
        """
        # Find non-zero bounding box
        nonzero_coords = np.argwhere(self.canvas > 0)

        if len(nonzero_coords) == 0:
            logger.warning("Canvas is empty, cannot crop")
            return self.canvas, (slice(None), slice(None), slice(None))

        min_coords = nonzero_coords.min(axis=0)
        max_coords = nonzero_coords.max(axis=0)

        # Add padding
        min_coords = np.maximum(min_coords - padding, 0)
        max_coords = np.minimum(max_coords + padding + 1, self.canvas.shape)

        # Create crop slices
        crop_slices = tuple(slice(int(min_c), int(max_c)) for min_c, max_c in zip(min_coords, max_coords))

        # Crop
        cropped_canvas = self.canvas[crop_slices]

        logger.info(f"Cropped canvas: {self.canvas.shape} → {cropped_canvas.shape}")

        return cropped_canvas, crop_slices


def main():
    """Test the canvas merger."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Create merger
    merger = ProgressiveCanvasMerger(
        initial_shape=(500, 1000, 200),
        merge_strategy='average',
        track_sources=True
    )

    # Create dummy volumes
    vol1 = np.random.rand(500, 1000, 180).astype(np.float32) * 100
    vol2 = np.random.rand(500, 1000, 180).astype(np.float32) * 100

    # Add volumes
    merger.add_volume(vol1, offset=(0, 0, 0), volume_id=1)
    merger.add_volume(vol2, offset=(0, 50, 0), volume_id=2)  # 50-pixel X offset

    # Finalize
    canvas, metadata = merger.finalize()

    print("\n" + "="*70)
    print("CANVAS MERGE SUMMARY")
    print("="*70)
    print(f"Final shape: {metadata.shape}")
    print(f"Volumes merged: {metadata.num_volumes}")
    print(f"Coverage: {metadata.coverage_percent:.1f}%")
    print(f"Overlap: {metadata.overlap_percent:.1f}%")

    # Test cropping
    print("\nTesting content cropping...")
    cropped, crop_slices = merger.crop_to_content(padding=10)
    print(f"Cropped shape: {cropped.shape}")
    print(f"Crop slices: {crop_slices}")


if __name__ == "__main__":
    main()
