#!/usr/bin/env python3
"""
OCT Grid Stitcher

Clean, modern implementation for stitching 4×2 grid of OCT volumes.
Each volume is 6mm × 6mm with proper voxel spacing.

Approach:
1. Load volumes with correct spatial dimensions
2. Register adjacent volumes using phase correlation
3. Stitch row-by-row, then merge rows
4. Output final stitched volume with correct physical dimensions

Author: OCT Stitching v2.0
"""

import numpy as np
import glob
import os
import re
import logging
from scipy import signal
from scipy.ndimage import shift as nd_shift
from typing import List, Tuple, Dict, Optional
import pyvista as pv
import cv2

from oct_volumetric_viewer import OCTImageProcessor, OCTVolumeLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VolumeInfo:
    """Information about a single OCT volume."""
    def __init__(self, path: str, volume_id: int, grid_pos: Tuple[int, int]):
        self.path = path
        self.volume_id = volume_id
        self.grid_x, self.grid_y = grid_pos
        self.volume = None
        self.loaded = False

    def __repr__(self):
        return f"Vol{self.volume_id}[{self.grid_x},{self.grid_y}]"


class OCTGridStitcher:
    """
    Stitches 4×2 grid of OCT volumes with proper physical spacing.
    """

    def __init__(self, scan_area_mm: float = 6.0, sidebar_width: int = 250, crop_top: int = 50,
                 use_surf: bool = True):
        """
        Initialize grid stitcher.

        Args:
            scan_area_mm: Physical size of each volume (default: 6mm × 6mm)
            sidebar_width: Sidebar pixels to remove from left
            crop_top: Top pixels to crop
            use_surf: Use SURF feature-based registration (default: True, fallback to phase correlation)
        """
        self.scan_area_mm = scan_area_mm
        self.processor = OCTImageProcessor(sidebar_width=sidebar_width, crop_top=crop_top)
        self.loader = OCTVolumeLoader(self.processor)
        self.volumes: List[VolumeInfo] = []
        self.use_surf = use_surf

    def discover_volumes(self, data_root: str) -> List[VolumeInfo]:
        """
        Discover all OCT volumes and organize into 4×2 grid.

        Args:
            data_root: Root directory with volume folders

        Returns:
            List of VolumeInfo objects
        """
        logger.info(f"Discovering volumes in {data_root}")

        # Find all volume directories
        pattern = os.path.join(data_root, "F001_IP_*_Retina_3D_L_6mm_1536x360_2")
        volume_dirs = sorted(glob.glob(pattern))

        logger.info(f"Found {len(volume_dirs)} volumes")

        # Organize into grid based on timestamp order
        # Assumption: scanned left-to-right, bottom-to-top
        volumes = []
        for i, vol_dir in enumerate(volume_dirs):
            # Calculate grid position (4 columns × 2 rows)
            grid_x = i % 4  # Column (0-3)
            grid_y = i // 4  # Row (0-1)

            vol_info = VolumeInfo(vol_dir, i, (grid_x, grid_y))
            volumes.append(vol_info)

            logger.info(f"Volume {i}: Grid[{grid_x},{grid_y}] - {os.path.basename(vol_dir)}")

        self.volumes = volumes
        return volumes

    def load_volume(self, vol_info: VolumeInfo) -> np.ndarray:
        """
        Load a single volume.

        Args:
            vol_info: Volume information

        Returns:
            Loaded volume array
        """
        logger.info(f"Loading {vol_info}...")
        volume = self.loader.load_volume_from_directory(vol_info.path)
        vol_info.volume = volume
        vol_info.loaded = True
        logger.info(f"Loaded {vol_info}: shape {volume.shape}")
        return volume

    def register_volumes_surf_single_direction(self, vol1: np.ndarray, vol2: np.ndarray,
                                               side: str) -> Optional[Dict]:
        """
        Register two volumes using SURF feature-based matching for a specific side.

        Args:
            vol1: Reference volume
            vol2: Volume to align
            side: Which side of vol1 to match with vol2: 'right', 'left', 'top', 'bottom'

        Returns:
            Registration result with offset and transformation matrix, or None if failed
        """
        logger.info(f"SURF registration (vol2 on {side} of vol1)...")

        h1, w1, d1 = vol1.shape
        h2, w2, d2 = vol2.shape

        # Extract overlap regions (50% as per actual overlap)
        if side == 'right':
            # Vol2 is to the right of vol1
            crop_size = int(w1 * 0.50)
            region1 = vol1[:, -crop_size:, :]  # Right 50% of vol1
            region2 = vol2[:, :crop_size, :]    # Left 50% of vol2
            logger.info(f"Testing RIGHT: {crop_size}px width")
        elif side == 'left':
            # Vol2 is to the left of vol1
            crop_size = int(w1 * 0.50)
            region1 = vol1[:, :crop_size, :]   # Left 50% of vol1
            region2 = vol2[:, -crop_size:, :]   # Right 50% of vol2
            logger.info(f"Testing LEFT: {crop_size}px width")
        elif side == 'top':
            # Vol2 is above vol1
            crop_size = int(h1 * 0.50)
            region1 = vol1[-crop_size:, :, :]  # Top 50% of vol1
            region2 = vol2[:crop_size, :, :]    # Bottom 50% of vol2
            logger.info(f"Testing TOP: {crop_size}px height")
        else:  # bottom
            # Vol2 is below vol1
            crop_size = int(h1 * 0.50)
            region1 = vol1[:crop_size, :, :]   # Bottom 50% of vol1
            region2 = vol2[-crop_size:, :, :]   # Top 50% of vol2
            logger.info(f"Testing BOTTOM: {crop_size}px height")

        # Use middle slice
        slice_idx = d1 // 2
        slice1 = region1[:, :, slice_idx]
        slice2 = region2[:, :, slice_idx]

        # Normalize to 8-bit for SURF
        slice1_8bit = cv2.normalize(slice1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        slice2_8bit = cv2.normalize(slice2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        # Apply histogram equalization for better feature detection
        slice1_8bit = cv2.equalizeHist(slice1_8bit)
        slice2_8bit = cv2.equalizeHist(slice2_8bit)

        try:
            # Detect ORB features (patent-free alternative to SURF)
            orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
            kp1, des1 = orb.detectAndCompute(slice1_8bit, None)
            kp2, des2 = orb.detectAndCompute(slice2_8bit, None)

            logger.info(f"ORB detected {len(kp1)} features in vol1, {len(kp2)} in vol2")

            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                logger.warning("Insufficient ORB features detected, falling back to phase correlation")
                return None

            # Match features with BFMatcher (better for ORB binary descriptors)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            logger.info(f"Found {len(good_matches)} good matches after ratio test")

            if len(good_matches) < 4:
                logger.warning("Insufficient good matches, falling back to phase correlation")
                return None

            # Estimate transformation with RANSAC
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is None:
                logger.warning("Homography estimation failed, falling back to phase correlation")
                return None

            # Count inliers
            inliers = np.sum(mask)
            logger.info(f"RANSAC inliers: {inliers}/{len(good_matches)}")

            # Extract translation from homography
            offset_x = M[0, 2]
            offset_y = M[1, 2]

            # Calculate confidence score
            match_ratio = inliers / len(good_matches) if len(good_matches) > 0 else 0

            result = {
                'offset_x': int(offset_x),
                'offset_y': int(offset_y),
                'offset_z': 0,
                'confidence': float(inliers),
                'match_ratio': float(match_ratio),
                'num_matches': len(good_matches),
                'num_inliers': int(inliers),
                'transformation_matrix': M,
                'side': side,
                'method': 'ORB'
            }

            logger.info(f"ORB {side}: offset=({int(offset_x)}, {int(offset_y)}), "
                       f"inliers={inliers}/{len(good_matches)} ({match_ratio:.1%})")

            return result

        except Exception as e:
            logger.error(f"ORB registration {side} failed: {e}")
            return None

    def register_volumes_surf(self, vol1: np.ndarray, vol2: np.ndarray,
                             direction: str = 'horizontal') -> Optional[Dict]:
        """
        Register two volumes using ORB by trying all 4 possible sides.
        Returns the result with highest confidence.

        Args:
            vol1: Reference volume
            vol2: Volume to align
            direction: Hint for expected direction ('horizontal' or 'vertical')

        Returns:
            Best registration result, or None if all failed
        """
        logger.info("=" * 60)
        logger.info("ORB 4-DIRECTIONAL REGISTRATION")
        logger.info("=" * 60)

        # Try all 4 sides
        sides = ['right', 'left', 'top', 'bottom']
        results = []

        for side in sides:
            result = self.register_volumes_surf_single_direction(vol1, vol2, side)
            if result is not None:
                results.append(result)

        if not results:
            logger.warning("All 4 directions failed in ORB registration")
            return None

        # Pick best result based on number of inliers
        best_result = max(results, key=lambda r: r['num_inliers'])

        logger.info("=" * 60)
        logger.info(f"BEST MATCH: {best_result['side'].upper()} side")
        logger.info(f"  Inliers: {best_result['num_inliers']}/{best_result['num_matches']}")
        logger.info(f"  Match ratio: {best_result['match_ratio']:.1%}")
        logger.info(f"  Offset: ({best_result['offset_x']}, {best_result['offset_y']})")
        logger.info("=" * 60)

        # Convert side to direction for compatibility
        if best_result['side'] in ['right', 'left']:
            best_result['direction'] = 'horizontal'
        else:
            best_result['direction'] = 'vertical'

        return best_result

    def register_volumes(self, vol1: np.ndarray, vol2: np.ndarray,
                        direction: str = 'horizontal') -> Dict:
        """
        Register two volumes using ORB (default) or phase correlation fallback.

        Args:
            vol1: Reference volume
            vol2: Volume to align
            direction: 'horizontal' (left-right) or 'vertical' (bottom-top)

        Returns:
            Registration result with offset
        """
        # Try ORB first if enabled
        if self.use_surf:  # Variable name kept for compatibility
            result = self.register_volumes_surf(vol1, vol2, direction)
            if result is not None:
                return result
            logger.info("ORB failed, falling back to phase correlation")

        # Fallback to phase correlation
        return self.register_volumes_phase_correlation(vol1, vol2, direction)

    def register_volumes_phase_correlation(self, vol1: np.ndarray, vol2: np.ndarray,
                                          direction: str = 'horizontal') -> Dict:
        """
        Register two volumes using phase correlation on overlap regions only.

        OPTIMIZATION: Use only central 25% of each volume where overlap occurs.
        This reduces data by 75% and focuses on actual overlap region.

        Args:
            vol1: Reference volume
            vol2: Volume to align
            direction: 'horizontal' (left-right) or 'vertical' (bottom-top)

        Returns:
            Registration result with offset
        """
        logger.info(f"Phase correlation registration ({direction})...")

        h1, w1, d1 = vol1.shape
        h2, w2, d2 = vol2.shape

        # Extract overlap regions (50% as per actual overlap)
        if direction == 'horizontal':
            # Vol1: rightmost 50% (where it meets vol2)
            # Vol2: leftmost 50% (where it meets vol1)
            crop_size = int(w1 * 0.50)
            region1 = vol1[:, -crop_size:, :]  # Right 50% of vol1
            region2 = vol2[:, :crop_size, :]    # Left 50% of vol2
            logger.info(f"Using overlap regions: {crop_size}px width (50% of {w1}px)")
        else:  # vertical
            # Vol1: topmost 50% (where it meets vol2 above)
            # Vol2: bottommost 50% (where it meets vol1 below)
            crop_size = int(h1 * 0.50)
            region1 = vol1[-crop_size:, :, :]  # Top 50% of vol1
            region2 = vol2[:crop_size, :, :]    # Bottom 50% of vol2
            logger.info(f"Using overlap regions: {crop_size}px height (50% of {h1}px)")

        # Use just 1 middle slice for fastest registration
        # (Can increase to 3 if accuracy needed)
        slice_indices = [d1 // 2]  # Just the center slice

        offsets_x = []
        offsets_y = []
        confidences = []

        for slice_idx in slice_indices:
            if slice_idx < min(region1.shape[2], region2.shape[2]):
                slice1 = region1[:, :, slice_idx]
                slice2 = region2[:, :, slice_idx]

                # Normalize
                slice1_norm = (slice1 - slice1.mean()) / (slice1.std() + 1e-8)
                slice2_norm = (slice2 - slice2.mean()) / (slice2.std() + 1e-8)

                # Phase correlation (much faster on smaller regions!)
                correlation = signal.correlate2d(slice1_norm, slice2_norm, mode='same')

                # Find peak
                peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
                center_y, center_x = np.array(correlation.shape) // 2

                offset_y = peak_y - center_y
                offset_x = peak_x - center_x
                confidence = correlation.max() / (correlation.std() + 1e-8)

                offsets_x.append(offset_x)
                offsets_y.append(offset_y)
                confidences.append(confidence)

        # Average results
        avg_offset_x = int(np.median(offsets_x))
        avg_offset_y = int(np.median(offsets_y))
        avg_confidence = np.mean(confidences)

        result = {
            'offset_x': avg_offset_x,
            'offset_y': avg_offset_y,
            'offset_z': 0,
            'confidence': avg_confidence,
            'direction': direction,
            'method': 'phase_correlation'
        }

        logger.info(f"Phase correlation complete: offset=({avg_offset_x}, {avg_offset_y}), confidence={avg_confidence:.2f}")

        return result

    def stitch_horizontal(self, vol_left: np.ndarray, vol_right: np.ndarray) -> np.ndarray:
        """
        Stitch two volumes horizontally (left + right).

        Args:
            vol_left: Left volume
            vol_right: Right volume

        Returns:
            Stitched volume
        """
        logger.info("Stitching horizontal pair...")

        # Register
        registration = self.register_volumes(vol_left, vol_right, 'horizontal')

        h1, w1, d1 = vol_left.shape
        h2, w2, d2 = vol_right.shape

        # Expected overlap based on 6mm scans
        # Actual overlap is ~50% between adjacent volumes
        expected_overlap = int(w1 * 0.50)  # 50% overlap

        # Calculate output size
        output_width = w1 + w2 - expected_overlap
        output_height = max(h1, h2)
        output_depth = max(d1, d2)

        logger.info(f"Output size: {output_height} × {output_width} × {output_depth}")

        # Create canvas
        stitched = np.zeros((output_height, output_width, output_depth), dtype=np.float32)

        # Place left volume
        stitched[:h1, :w1, :d1] = vol_left

        # Place right volume with blending in overlap region
        x_start = w1 - expected_overlap

        # Blend in overlap region
        for x in range(expected_overlap):
            alpha = x / expected_overlap  # 0 to 1
            x_left = w1 - expected_overlap + x
            x_right = x

            if x_right < w2:
                stitched[:h2, x_left, :d2] = (
                    (1 - alpha) * stitched[:h2, x_left, :d2] +
                    alpha * vol_right[:, x_right, :]
                )

        # Place remaining part of right volume
        if expected_overlap < w2:
            stitched[:h2, x_start+expected_overlap:x_start+w2, :d2] = \
                vol_right[:, expected_overlap:, :]

        logger.info(f"Horizontal stitch complete: {stitched.shape}")

        return stitched

    def stitch_vertical(self, vol_bottom: np.ndarray, vol_top: np.ndarray) -> np.ndarray:
        """
        Stitch two volumes vertically (bottom + top).

        Args:
            vol_bottom: Bottom volume
            vol_top: Top volume

        Returns:
            Stitched volume
        """
        logger.info("Stitching vertical pair...")

        # Register
        registration = self.register_volumes(vol_bottom, vol_top, 'vertical')

        h1, w1, d1 = vol_bottom.shape
        h2, w2, d2 = vol_top.shape

        # Expected overlap in Y direction
        # Actual overlap is ~50% between adjacent volumes
        expected_overlap = int(h1 * 0.50)  # 50% overlap

        # Calculate output size
        output_height = h1 + h2 - expected_overlap
        output_width = max(w1, w2)
        output_depth = max(d1, d2)

        logger.info(f"Output size: {output_height} × {output_width} × {output_depth}")

        # Create canvas
        stitched = np.zeros((output_height, output_width, output_depth), dtype=np.float32)

        # Place bottom volume
        stitched[:h1, :w1, :d1] = vol_bottom

        # Place top volume with blending
        y_start = h1 - expected_overlap

        # Blend in overlap region
        for y in range(expected_overlap):
            alpha = y / expected_overlap
            y_bottom = h1 - expected_overlap + y
            y_top = y

            if y_top < h2:
                stitched[y_bottom, :w2, :d2] = (
                    (1 - alpha) * stitched[y_bottom, :w2, :d2] +
                    alpha * vol_top[y_top, :, :]
                )

        # Place remaining part of top volume
        if expected_overlap < h2:
            stitched[y_start+expected_overlap:y_start+h2, :w2, :d2] = \
                vol_top[expected_overlap:, :, :]

        logger.info(f"Vertical stitch complete: {stitched.shape}")

        return stitched

    def stitch_grid(self, num_volumes: int = 8) -> np.ndarray:
        """
        Stitch full 4×2 grid of volumes.

        Args:
            num_volumes: Number of volumes to stitch (default: 8 for full grid)

        Returns:
            Final stitched volume
        """
        if len(self.volumes) < num_volumes:
            raise ValueError(f"Need {num_volumes} volumes, found {len(self.volumes)}")

        logger.info(f"Starting grid stitching ({num_volumes} volumes)...")

        # Load all volumes
        for vol_info in self.volumes[:num_volumes]:
            self.load_volume(vol_info)

        # Stitch bottom row (volumes 0-3)
        logger.info("=" * 60)
        logger.info("STITCHING BOTTOM ROW (Y=0)")
        logger.info("=" * 60)

        bottom_row = self.volumes[0].volume
        for i in range(1, 4):
            bottom_row = self.stitch_horizontal(bottom_row, self.volumes[i].volume)

        logger.info(f"Bottom row complete: {bottom_row.shape}")

        # Stitch top row (volumes 4-7)
        logger.info("=" * 60)
        logger.info("STITCHING TOP ROW (Y=1)")
        logger.info("=" * 60)

        top_row = self.volumes[4].volume
        for i in range(5, 8):
            top_row = self.stitch_horizontal(top_row, self.volumes[i].volume)

        logger.info(f"Top row complete: {top_row.shape}")

        # Stitch rows vertically
        logger.info("=" * 60)
        logger.info("MERGING ROWS")
        logger.info("=" * 60)

        final_volume = self.stitch_vertical(bottom_row, top_row)

        logger.info(f"FINAL STITCHED VOLUME: {final_volume.shape}")

        # Calculate physical dimensions
        # 4 volumes horizontally with 50% overlap: 4 × 6mm - 3 × 3mm overlaps = 15mm
        # 2 volumes vertically with 50% overlap: 2 × 6mm - 1 × 3mm overlap = 9mm
        horizontal_mm = 4 * self.scan_area_mm - 3 * (self.scan_area_mm * 0.5)
        vertical_mm = 2 * self.scan_area_mm - 1 * (self.scan_area_mm * 0.5)

        logger.info(f"Physical coverage: ~{horizontal_mm:.1f}mm × {vertical_mm:.1f}mm")

        return final_volume


def main():
    """Main function for testing grid stitcher."""
    import argparse

    parser = argparse.ArgumentParser(description='OCT Grid Stitcher')
    parser.add_argument('--data-dir', required=True, help='Directory containing OCT volumes')
    parser.add_argument('--num-volumes', type=int, default=2, choices=[2, 4, 8],
                       help='Number of volumes to stitch (2=test, 4=row, 8=full grid)')
    parser.add_argument('--output', help='Output path for stitched volume (.vtk or .npz)')
    parser.add_argument('--visualize', action='store_true', help='Visualize result')
    parser.add_argument('--method', choices=['orb', 'phase', 'auto'], default='auto',
                       help='Registration method: orb (ORB only), phase (phase correlation only), auto (ORB with fallback)')

    args = parser.parse_args()

    # Initialize stitcher (use_surf variable name kept for compatibility)
    use_surf = args.method in ['orb', 'auto']
    stitcher = OCTGridStitcher(scan_area_mm=6.0, sidebar_width=250, crop_top=50, use_surf=use_surf)

    # Discover volumes
    stitcher.discover_volumes(args.data_dir)

    # Stitch
    if args.num_volumes == 2:
        logger.info("Test mode: stitching 2 volumes only")
        stitcher.load_volume(stitcher.volumes[0])
        stitcher.load_volume(stitcher.volumes[1])
        final_volume = stitcher.stitch_horizontal(
            stitcher.volumes[0].volume,
            stitcher.volumes[1].volume
        )
    elif args.num_volumes == 4:
        logger.info("Row mode: stitching bottom row (4 volumes)")
        for i in range(4):
            stitcher.load_volume(stitcher.volumes[i])

        result = stitcher.volumes[0].volume
        for i in range(1, 4):
            result = stitcher.stitch_horizontal(result, stitcher.volumes[i].volume)
        final_volume = result
    else:
        final_volume = stitcher.stitch_grid(num_volumes=8)

    # Save
    if args.output:
        if args.output.endswith('.vtk'):
            # Save as VTK
            grid = pv.ImageData()
            grid.dimensions = (final_volume.shape[1], final_volume.shape[0], final_volume.shape[2])
            grid.spacing = (0.00390625, 0.00390625, 0.016666666666666666)
            grid.point_data['intensity'] = final_volume.transpose(1, 0, 2).flatten(order='F')
            grid.save(args.output)
            logger.info(f"Saved to {args.output}")
        else:
            # Save as numpy
            np.savez_compressed(args.output, volume=final_volume)
            logger.info(f"Saved to {args.output}")

    # Visualize
    if args.visualize:
        from oct_volumetric_viewer import OCTVolumetricViewer

        # Calculate physical size based on number of volumes (50% overlap)
        if args.num_volumes == 2:
            scan_size = 2 * 6.0 - 1 * 3.0  # 2 volumes - 1 overlap = 9mm
        elif args.num_volumes == 4:
            scan_size = 4 * 6.0 - 3 * 3.0  # 4 volumes - 3 overlaps = 15mm
        else:
            scan_size = 4 * 6.0 - 3 * 3.0  # 4×2 grid width = 15mm

        viewer = OCTVolumetricViewer(final_volume, scan_area_mm=scan_size)
        viewer.render_volume()


if __name__ == "__main__":
    main()
