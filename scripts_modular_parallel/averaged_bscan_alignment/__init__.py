"""
Averaged B-Scan Alignment Pipeline

A computationally lighter alternative to full volume alignment that operates
on averaged B-scans instead of full 3D volumes.
"""

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
    perform_y_alignment_2d,
    perform_rotation_alignment_2d,
    visualize_rotation_results_2d
)

from .averaged_bscan_pipeline import averaged_bscan_pipeline

__all__ = [
    # Averaging and manipulation
    'extract_averaged_central_bscan',
    'shift_bscan_2d',
    'rotate_bscan_2d',
    'save_averaged_bscan',
    'save_averaged_bscan_npy',

    # Visualization
    'visualize_alignment_step',
    'visualize_surface_comparison',
    'visualize_three_averaged_bscans',
    'visualize_rotation_results_2d',

    # Alignment
    'perform_y_alignment_2d',
    'perform_rotation_alignment_2d',

    # Main pipeline
    'averaged_bscan_pipeline',
]
