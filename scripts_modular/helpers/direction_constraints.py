"""
Direction Constraints for Cross-Pattern Multi-Volume Alignment

Provides functions to constrain volume alignment offsets to maintain
a cross/plus pattern layout instead of arbitrary 3D grid placement.

The cross pattern:
              7 (far_up)
              ↑
              6 (up)
              ↑
5 ← 4 ← ← ← ← 1 → → → → 2 → 3
(far_left) (left) (center) (right) (far_right)
              ↓
              8 (down)
              ↓
              9 (far_down)
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# Direction vectors in (X, Z) coordinate space
# Y-axis (depth) is always free for retinal layer alignment
DIRECTION_VECTORS = {
    'right': np.array([1.0, 0.0]),       # +X only
    'far_right': np.array([1.0, 0.0]),   # +X only
    'left': np.array([-1.0, 0.0]),       # -X only
    'far_left': np.array([-1.0, 0.0]),   # -X only
    'up': np.array([0.0, 1.0]),          # +Z only
    'far_up': np.array([0.0, 1.0]),      # +Z only
    'down': np.array([0.0, -1.0]),       # -Z only
    'far_down': np.array([0.0, -1.0]),   # -Z only
    'center': None                        # No constraint for center
}


# Rotation axes for position-aware rotation
# (axis1, axis2) for ndimage.rotate()
ROTATION_AXES = {
    'right': (0, 1),       # Z-rotation (YX plane) - for horizontal volumes
    'far_right': (0, 1),   # Z-rotation (YX plane)
    'left': (0, 1),        # Z-rotation (YX plane)
    'far_left': (0, 1),    # Z-rotation (YX plane)
    'up': (0, 2),          # X-rotation (YZ plane) - for vertical volumes
    'far_up': (0, 2),      # X-rotation (YZ plane)
    'down': (0, 2),        # X-rotation (YZ plane)
    'far_down': (0, 2),    # X-rotation (YZ plane)
    'center': (0, 1)       # Default to Z-rotation
}


def constrain_offset_to_direction(
    offset_x: float,
    offset_z: float,
    position: str,
    tolerance: float = 0.1
) -> Tuple[float, float]:
    """
    Constrain XZ offset to expected direction for cross-pattern alignment.

    Projects the offset onto the canonical direction for the given position,
    with a tolerance for perpendicular movement.

    Args:
        offset_x: Unconstrained X offset (pixels)
        offset_z: Unconstrained Z offset (pixels)
        position: Volume position ('right', 'left', 'up', 'down', etc.)
        tolerance: Fraction of perpendicular movement allowed (0.0-1.0)
                  0.0 = strict (no perpendicular movement)
                  1.0 = no constraint
                  0.1 = allow 10% perpendicular (recommended)

    Returns:
        Tuple of (constrained_dx, constrained_dz)

    Example:
        # Right volume with offset (+50, +20) should move mostly in +X
        dx, dz = constrain_offset_to_direction(50, 20, 'right', tolerance=0.1)
        # Result: dx=~50, dz=~5 (20 reduced by 75% to 10% of 50)
    """
    direction = DIRECTION_VECTORS.get(position)

    if direction is None:
        # Center or unknown position - no constraint
        logger.debug(f"No constraint for position '{position}'")
        return offset_x, offset_z

    # Create offset vector
    offset_vector = np.array([offset_x, offset_z])

    # Calculate magnitude along expected direction
    parallel_magnitude = np.dot(offset_vector, direction)

    # Project onto direction (parallel component)
    parallel_vector = parallel_magnitude * direction

    # Calculate perpendicular component
    perpendicular_vector = offset_vector - parallel_vector

    # Allow tolerance fraction of parallel magnitude in perpendicular direction
    max_perpendicular = abs(parallel_magnitude) * tolerance
    perpendicular_magnitude = np.linalg.norm(perpendicular_vector)

    if perpendicular_magnitude > max_perpendicular:
        # Scale down perpendicular component
        scale_factor = max_perpendicular / perpendicular_magnitude
        perpendicular_vector *= scale_factor

        logger.info(
            f"Position '{position}': Constrained perpendicular offset "
            f"from {perpendicular_magnitude:.1f} to {max_perpendicular:.1f} px "
            f"({tolerance*100:.0f}% of parallel {abs(parallel_magnitude):.1f} px)"
        )

    # Combine parallel and constrained perpendicular
    constrained_vector = parallel_vector + perpendicular_vector

    # Log constraint application
    original_mag = np.linalg.norm(offset_vector)
    constrained_mag = np.linalg.norm(constrained_vector)
    constraint_applied = abs(constrained_mag - original_mag) / (original_mag + 1e-6)

    if constraint_applied > 0.1:
        logger.warning(
            f"Position '{position}': Large constraint applied "
            f"({constraint_applied*100:.1f}% change). "
            f"Original: ({offset_x:.1f}, {offset_z:.1f}), "
            f"Constrained: ({constrained_vector[0]:.1f}, {constrained_vector[1]:.1f})"
        )
    else:
        logger.debug(
            f"Position '{position}': Constraint applied "
            f"({constraint_applied*100:.1f}% change). "
            f"Original: ({offset_x:.1f}, {offset_z:.1f}), "
            f"Constrained: ({constrained_vector[0]:.1f}, {constrained_vector[1]:.1f})"
        )

    return float(constrained_vector[0]), float(constrained_vector[1])


def get_rotation_axes_for_position(position: Optional[str]) -> Tuple[int, int]:
    """
    Get rotation axes for ndimage.rotate() based on volume position.

    Args:
        position: Volume position ('right', 'left', 'up', 'down', etc.)

    Returns:
        Tuple of (axis1, axis2) for ndimage.rotate()
        - Horizontal volumes (left/right): (0, 1) for Z-rotation (YX plane)
        - Vertical volumes (up/down): (0, 2) for X-rotation (YZ plane)

    Physical interpretation:
        - Horizontal volumes move in X → need YX plane rotation to align layers
        - Vertical volumes move in Z → need YZ plane rotation to align across B-scans
    """
    axes = ROTATION_AXES.get(position, (0, 1))  # Default to Z-rotation

    logger.debug(f"Position '{position}': Using rotation axes {axes}")

    return axes


def validate_cross_pattern(
    volume_positions: dict,
    volume_offsets: dict,
    tolerance: float = 0.3
) -> dict:
    """
    Validate that volume offsets maintain cross-pattern geometry.

    Checks if volumes are moving in their expected directions and
    provides warnings for violations.

    Args:
        volume_positions: Dict mapping volume_id -> position string
        volume_offsets: Dict mapping volume_id -> (dx, dy, dz) offset tuple
        tolerance: Tolerance for direction violations (0.0-1.0)

    Returns:
        Dict with validation results and warnings
    """
    warnings = []
    violations = []

    for vol_id, position in volume_positions.items():
        if vol_id not in volume_offsets:
            continue

        dx, dy, dz = volume_offsets[vol_id]
        direction = DIRECTION_VECTORS.get(position)

        if direction is None:
            continue  # Skip center

        # Calculate offset components
        offset_xz = np.array([dx, dz])
        parallel = np.dot(offset_xz, direction)
        perpendicular = np.linalg.norm(offset_xz - parallel * direction)

        # Check if perpendicular is too large
        if perpendicular > abs(parallel) * tolerance:
            violation = {
                'volume_id': vol_id,
                'position': position,
                'parallel': parallel,
                'perpendicular': perpendicular,
                'ratio': perpendicular / (abs(parallel) + 1e-6)
            }
            violations.append(violation)
            warnings.append(
                f"Volume {vol_id} ({position}): Perpendicular offset {perpendicular:.1f} "
                f"is {violation['ratio']*100:.0f}% of parallel {abs(parallel):.1f}"
            )

    return {
        'valid': len(violations) == 0,
        'violations': violations,
        'warnings': warnings
    }
