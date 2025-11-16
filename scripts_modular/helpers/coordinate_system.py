"""
Global Coordinate System Manager

Manages coordinate transformations for multi-volume panoramic stitching.
Tracks cumulative transformations and handles coordinate conversions between
local volume space and global panorama space.

Key Features:
- Global reference frame based on volume 1 (center)
- Cumulative transformation tracking
- Z-stride (subsampling) awareness
- Coordinate conversion utilities
- Bounding box calculations for progressive canvas expansion

Author: OCT Panoramic Stitching System
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Transform3D:
    """Represents a 3D affine transformation."""

    # Translation (voxels)
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0

    # Rotation (degrees)
    rotation_z: float = 0.0  # In-plane rotation (XZ plane)
    rotation_x: float = 0.0  # Sagittal rotation (YZ plane)

    # Rotation center (for rotation around overlap center)
    rotation_center_z: Optional[Tuple[float, float, float]] = None  # (y, x, z) for Z-rotation
    rotation_center_x: Optional[Tuple[float, float, float]] = None  # (y, x, z) for X-rotation

    # Metadata
    confidence: float = 0.0
    method: str = "unknown"

    def __str__(self):
        return (f"Transform3D(dx={self.dx:.1f}, dy={self.dy:.1f}, dz={self.dz:.1f}, "
                f"rot_z={self.rotation_z:.2f}°, rot_x={self.rotation_x:.2f}°)")


@dataclass
class VolumeCoordinates:
    """Stores coordinate information for a single volume."""

    volume_id: int

    # Local volume dimensions
    shape: Tuple[int, int, int]  # (height, width, depth)

    # Subsampling info
    z_stride: int = 1
    original_depth: Optional[int] = None  # Original depth before subsampling

    # Transformation relative to reference volume
    transform: Transform3D = field(default_factory=Transform3D)

    # Cumulative transformation chain (from volume 1 to this volume)
    transform_chain: List[Transform3D] = field(default_factory=list)

    # Bounding box in global coordinates (min_y, min_x, min_z, max_y, max_x, max_z)
    global_bbox: Optional[Tuple[int, int, int, int, int, int]] = None

    # Reference volume this was aligned to
    reference_volume_id: Optional[int] = None

    def __str__(self):
        return (f"VolumeCoordinates(id={self.volume_id}, shape={self.shape}, "
                f"z_stride={self.z_stride}, global_bbox={self.global_bbox})")


class GlobalCoordinateSystem:
    """
    Manages global coordinate system for multi-volume panoramic stitching.

    The global coordinate system is defined by volume 1 (center reference).
    All other volumes are positioned relative to this global frame.
    """

    def __init__(self, z_stride: int = 1):
        """
        Initialize the global coordinate system.

        Args:
            z_stride: B-scan subsampling factor (1=no subsampling, 2=every 2nd, etc.)
        """
        self.z_stride = z_stride
        self.volumes: Dict[int, VolumeCoordinates] = {}

        # Global bounding box (updated as volumes are added)
        self.global_bbox: Optional[Tuple[int, int, int, int, int, int]] = None

        logger.info(f"Global coordinate system initialized (z_stride={z_stride})")

    def register_volume(self,
                       volume_id: int,
                       shape: Tuple[int, int, int],
                       reference_id: Optional[int] = None,
                       original_depth: Optional[int] = None):
        """
        Register a new volume in the coordinate system.

        Args:
            volume_id: Unique volume identifier
            shape: Volume shape (height, width, depth) after subsampling
            reference_id: ID of reference volume this was aligned to
            original_depth: Original depth before subsampling (for z_stride > 1)
        """
        coords = VolumeCoordinates(
            volume_id=volume_id,
            shape=shape,
            z_stride=self.z_stride,
            original_depth=original_depth or shape[2],
            reference_volume_id=reference_id
        )

        # Volume 1 is the global reference (identity transform)
        if volume_id == 1:
            coords.global_bbox = (0, 0, 0, shape[0], shape[1], shape[2])
            self.global_bbox = coords.global_bbox
            logger.info(f"Registered reference volume {volume_id}: shape={shape}")
        else:
            logger.info(f"Registered volume {volume_id}: shape={shape}, reference={reference_id}")

        self.volumes[volume_id] = coords

    def set_transform(self,
                     volume_id: int,
                     transform: Transform3D,
                     reference_id: int):
        """
        Set the transformation for a volume relative to its reference.

        This computes the cumulative transformation and updates the global bounding box.

        Args:
            volume_id: Volume to update
            transform: Transformation relative to reference volume
            reference_id: Reference volume ID
        """
        if volume_id not in self.volumes:
            raise ValueError(f"Volume {volume_id} not registered")

        if reference_id not in self.volumes:
            raise ValueError(f"Reference volume {reference_id} not registered")

        coords = self.volumes[volume_id]
        coords.transform = transform
        coords.reference_volume_id = reference_id

        # Build cumulative transformation chain
        coords.transform_chain = self._build_transform_chain(volume_id)

        # Compute global bounding box
        coords.global_bbox = self._compute_global_bbox(volume_id)

        # Update global bounding box
        self._update_global_bbox(coords.global_bbox)

        logger.info(f"Set transform for volume {volume_id} → {reference_id}")
        logger.debug(f"  {transform}")
        logger.debug(f"  Global bbox: {coords.global_bbox}")

    def _build_transform_chain(self, volume_id: int) -> List[Transform3D]:
        """
        Build the chain of transformations from volume 1 to the target volume.

        Args:
            volume_id: Target volume ID

        Returns:
            List of Transform3D objects from reference to target
        """
        if volume_id == 1:
            return []  # Reference volume has identity transform

        chain = []
        current_id = volume_id

        # Walk backwards to volume 1
        while current_id != 1:
            coords = self.volumes[current_id]
            if coords.reference_volume_id is None:
                raise ValueError(f"Volume {current_id} has no reference volume")

            chain.append(coords.transform)
            current_id = coords.reference_volume_id

        # Reverse to get forward direction (1 → ... → volume_id)
        return list(reversed(chain))

    def _compute_global_bbox(self, volume_id: int) -> Tuple[int, int, int, int, int, int]:
        """
        Compute the global bounding box for a volume based on its cumulative transform.

        Args:
            volume_id: Volume ID

        Returns:
            Bounding box (min_y, min_x, min_z, max_y, max_x, max_z) in global coordinates
        """
        coords = self.volumes[volume_id]
        h, w, d = coords.shape

        if volume_id == 1:
            # Reference volume at origin
            return (0, 0, 0, h, w, d)

        # Get cumulative translation (sum of all translations in chain)
        total_dx = 0.0
        total_dy = 0.0
        total_dz = 0.0

        for transform in coords.transform_chain:
            total_dx += transform.dx
            total_dy += transform.dy
            total_dz += transform.dz

        # Compute bounding box with translation applied
        # Note: We're using simple translation-based bbox estimation here.
        # For rotated volumes, this is an approximation (axis-aligned bbox).
        # The actual merged volume will handle rotation precisely.

        min_y = int(np.floor(total_dy))
        min_x = int(np.floor(total_dx))
        min_z = int(np.floor(total_dz))

        max_y = int(np.ceil(total_dy + h))
        max_x = int(np.ceil(total_dx + w))
        max_z = int(np.ceil(total_dz + d))

        return (min_y, min_x, min_z, max_y, max_x, max_z)

    def _update_global_bbox(self, volume_bbox: Tuple[int, int, int, int, int, int]):
        """
        Update the global bounding box to include a new volume.

        Args:
            volume_bbox: Bounding box of the new volume
        """
        if self.global_bbox is None:
            self.global_bbox = volume_bbox
        else:
            min_y = min(self.global_bbox[0], volume_bbox[0])
            min_x = min(self.global_bbox[1], volume_bbox[1])
            min_z = min(self.global_bbox[2], volume_bbox[2])
            max_y = max(self.global_bbox[3], volume_bbox[3])
            max_x = max(self.global_bbox[4], volume_bbox[4])
            max_z = max(self.global_bbox[5], volume_bbox[5])

            self.global_bbox = (min_y, min_x, min_z, max_y, max_x, max_z)

    def get_global_canvas_size(self) -> Tuple[int, int, int]:
        """
        Get the size of the global canvas needed to hold all volumes.

        Returns:
            Canvas size (height, width, depth)
        """
        if self.global_bbox is None:
            return (0, 0, 0)

        min_y, min_x, min_z, max_y, max_x, max_z = self.global_bbox

        height = max_y - min_y
        width = max_x - min_x
        depth = max_z - min_z

        return (height, width, depth)

    def get_volume_offset_in_canvas(self, volume_id: int) -> Tuple[int, int, int]:
        """
        Get the offset for placing a volume in the global canvas.

        Args:
            volume_id: Volume ID

        Returns:
            Offset (y_offset, x_offset, z_offset) relative to global canvas origin
        """
        if volume_id not in self.volumes:
            raise ValueError(f"Volume {volume_id} not registered")

        if self.global_bbox is None:
            return (0, 0, 0)

        coords = self.volumes[volume_id]
        if coords.global_bbox is None:
            return (0, 0, 0)

        # Offset is the volume's min position minus the global min position
        global_min_y, global_min_x, global_min_z = self.global_bbox[:3]
        vol_min_y, vol_min_x, vol_min_z = coords.global_bbox[:3]

        y_offset = vol_min_y - global_min_y
        x_offset = vol_min_x - global_min_x
        z_offset = vol_min_z - global_min_z

        return (y_offset, x_offset, z_offset)

    def get_cumulative_transform(self, volume_id: int) -> Transform3D:
        """
        Get the cumulative transformation from volume 1 to the target volume.

        Args:
            volume_id: Target volume ID

        Returns:
            Cumulative Transform3D object
        """
        if volume_id not in self.volumes:
            raise ValueError(f"Volume {volume_id} not registered")

        coords = self.volumes[volume_id]

        # Sum all transformations in chain
        cumulative = Transform3D()

        for transform in coords.transform_chain:
            cumulative.dx += transform.dx
            cumulative.dy += transform.dy
            cumulative.dz += transform.dz
            # Note: Rotation composition is more complex and depends on application order
            # For now, we store the last rotation (should be applied in sequence)
            if transform.rotation_z != 0:
                cumulative.rotation_z = transform.rotation_z
                cumulative.rotation_center_z = transform.rotation_center_z
            if transform.rotation_x != 0:
                cumulative.rotation_x = transform.rotation_x
                cumulative.rotation_center_x = transform.rotation_center_x

        return cumulative

    def print_summary(self):
        """Print summary of the coordinate system."""
        print("\n" + "="*70)
        print("GLOBAL COORDINATE SYSTEM SUMMARY")
        print("="*70)

        print(f"\nTotal registered volumes: {len(self.volumes)}")
        print(f"Z-stride (subsampling): {self.z_stride}")

        if self.global_bbox is not None:
            canvas_size = self.get_global_canvas_size()
            print(f"\nGlobal canvas size: {canvas_size}")
            print(f"Global bounding box: {self.global_bbox}")

        print("\nVolume Details:")
        for vol_id in sorted(self.volumes.keys()):
            coords = self.volumes[vol_id]
            print(f"\n  Volume {vol_id}:")
            print(f"    Shape: {coords.shape}")
            print(f"    Reference: {coords.reference_volume_id}")
            if coords.global_bbox:
                print(f"    Global bbox: {coords.global_bbox}")
                offset = self.get_volume_offset_in_canvas(vol_id)
                print(f"    Canvas offset: {offset}")
            if coords.transform_chain:
                cumulative = self.get_cumulative_transform(vol_id)
                print(f"    Cumulative transform: {cumulative}")

        print("\n" + "="*70)


def main():
    """Test the coordinate system functionality."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Create coordinate system with z_stride=2
    coords_sys = GlobalCoordinateSystem(z_stride=2)

    # Register volumes
    coords_sys.register_volume(1, shape=(500, 1000, 180))  # Reference (360/2 = 180 B-scans)
    coords_sys.register_volume(2, shape=(500, 1000, 180), reference_id=1, original_depth=360)

    # Set transform for volume 2
    transform_2 = Transform3D(
        dx=50.0, dy=10.0, dz=0.0,
        rotation_z=5.0,
        confidence=0.85,
        method="phase_correlation"
    )
    coords_sys.set_transform(2, transform_2, reference_id=1)

    # Print summary
    coords_sys.print_summary()

    # Test canvas size
    canvas_size = coords_sys.get_global_canvas_size()
    print(f"\nRequired canvas size: {canvas_size}")


if __name__ == "__main__":
    main()
