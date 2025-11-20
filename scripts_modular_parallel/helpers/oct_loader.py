#!/usr/bin/env python3
"""
OCT Volumetric Viewer with PyVista

Proper volumetric reconstruction of OCT data with correct spatial scaling.
Uses PyVista for true 3D volume rendering without browser limitations.

Features:
- Correct voxel spacing based on physical dimensions (6mm × 6mm scan area)
- True volumetric rendering (not scattered points)
- GPU-accelerated visualization
- Multiple rendering modes (volume, surface, slices)
- No browser memory limits
- Export capabilities

Physical Specifications:
- Scan area: 6mm × 6mm
- B-scans: 360 (covering depth dimension)
- A-scans per B-scan: 1536 (covering width dimension)
- Voxel spacing: ~3.9μm × depth × 16.7μm

Author: OCT Volumetric Specialist
"""

import os
import glob
import numpy as np
from PIL import Image
import re
from typing import Optional, Tuple
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

# Lazy import for pyvista (only needed for VTK visualization, not alignment)
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OCTImageProcessor:
    """
    Handles preprocessing of individual OCT images.
    Removes unwanted elements like sidebars and black regions.
    """

    def __init__(self, sidebar_width: int = 250, crop_top: int = 100, crop_bottom: int = 50):
        """
        Initialize the image processor.

        Args:
            sidebar_width: Width of sidebar to remove from LEFT side (pixels)
            crop_top: Number of pixels to crop from top (remove text/numbers)
            crop_bottom: Number of pixels to crop from bottom (remove text/numbers)
        """
        self.sidebar_width = sidebar_width
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom

    def load_image(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess a single OCT image (OPTIMIZED).

        Optimizations:
        - Convert to grayscale in Pillow (faster than np.mean)
        - Crop in Pillow before array conversion (smaller memory footprint)
        - Convert to numpy only once at the end

        Args:
            file_path: Path to the BMP file

        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image
            img = Image.open(file_path)

            # Convert to grayscale BEFORE array conversion (2x faster than np.mean)
            if img.mode != 'L':
                img = img.convert('L')

            # Crop in Pillow BEFORE array conversion (works on smaller data)
            width, height = img.size

            # Calculate crop box
            left = self.sidebar_width
            top = self.crop_top
            right = width
            bottom = height - self.crop_bottom

            # Apply crop
            img = img.crop((left, top, right, bottom))

            # Convert to numpy array ONCE (already grayscale, already cropped)
            img_array = np.array(img, dtype=np.float32)

            return img_array

        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            return None


class OCTVolumeLoader:
    """
    Loads and reconstructs 3D OCT volumes from BMP files with proper spatial scaling.
    """

    def __init__(self, processor: OCTImageProcessor):
        """
        Initialize the volume loader.

        Args:
            processor: Image processor for individual slices
        """
        self.processor = processor

    def load_volume_from_directory(self, data_dir: str, z_stride: int = 1) -> Optional[np.ndarray]:
        """
        Load BMP files from directory and create 3D volume with optional B-scan subsampling.

        Args:
            data_dir: Directory containing BMP files
            z_stride: Load every Nth B-scan (1=all, 2=every 2nd, etc.)
                     Used for memory optimization in multi-volume panoramic stitching

        Returns:
            3D numpy array representing the OCT volume
        """
        try:
            # Find all BMP files
            bmp_files = glob.glob(os.path.join(data_dir, "*.bmp"))
            if not bmp_files:
                logger.error(f"No BMP files found in {data_dir}")
                return None

            # Sort files numerically
            def extract_number(filename):
                numbers = re.findall(r'\d+', os.path.basename(filename))
                return int(numbers[-1]) if numbers else 0

            bmp_files.sort(key=extract_number)

            # Apply subsampling if z_stride > 1
            if z_stride > 1:
                bmp_files = bmp_files[::z_stride]
                logger.info(f"Found {len(bmp_files) * z_stride} BMP files, loading every {z_stride}th B-scan → {len(bmp_files)} B-scans")
            else:
                logger.info(f"Found {len(bmp_files)} BMP files")

            # Load first image to get dimensions
            first_img = self.processor.load_image(bmp_files[0])
            if first_img is None:
                logger.error("Failed to load first image")
                return None

            height, width = first_img.shape
            num_slices = len(bmp_files)

            # Initialize volume array
            volume = np.zeros((height, width, num_slices), dtype=np.float32)

            # PARALLEL LOADING: Load all images using thread pool (I/O-bound)
            # Use more threads than CPU cores for I/O-bound operations
            # SSD can handle 16-24 concurrent reads efficiently
            num_workers = min(24, cpu_count() * 3)  # 3x CPU cores for I/O
            logger.info(f"Loading {num_slices} B-scans in PARALLEL (using {num_workers} threads)...")

            def load_single_image(file_idx_tuple):
                """Worker function to load a single BMP file."""
                idx, file_path = file_idx_tuple
                try:
                    img = self.processor.load_image(file_path)
                    return (idx, img)
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
                    return (idx, None)

            # Load images in parallel
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                file_idx_pairs = list(enumerate(bmp_files))
                results = list(executor.map(load_single_image, file_idx_pairs))

            # Place loaded images into volume array
            loaded_count = 0
            for idx, img in results:
                if img is not None:
                    volume[:, :, idx] = img
                    loaded_count += 1
                else:
                    logger.warning(f"Skipped corrupted file at index {idx}")

            logger.info(f"Successfully loaded {loaded_count}/{num_slices} scans with shape {volume.shape} (z_stride={z_stride})")
            return volume

        except Exception as e:
            logger.error(f"Error loading volume: {e}")
            return None


class OCTVolumetricViewer:
    """
    PyVista-based volumetric viewer with proper physical dimensions.
    """

    def __init__(self, volume: np.ndarray, scan_area_mm: float = 6.0):
        """
        Initialize the volumetric viewer.

        Args:
            volume: 3D numpy array of OCT data (height, width, depth)
            scan_area_mm: Physical scan area in millimeters (default: 6mm × 6mm)
        """
        self.volume = volume
        self.scan_area_mm = scan_area_mm
        self.height, self.width, self.depth = volume.shape

        # Calculate proper voxel spacing in millimeters
        self.spacing = self.calculate_voxel_spacing()

        logger.info(f"Volume dimensions: {self.height} × {self.width} × {self.depth}")
        logger.info(f"Voxel spacing (mm): {self.spacing}")

    def calculate_voxel_spacing(self) -> Tuple[float, float, float]:
        """
        Calculate proper voxel spacing based on physical dimensions.

        Returns:
            Tuple of (x_spacing, y_spacing, z_spacing) in millimeters
        """
        # X-axis: lateral width (A-scans) - spans scan_area_mm
        x_spacing = self.scan_area_mm / self.width  # ~3.9 μm

        # Z-axis: scan depth (B-scans) - spans scan_area_mm
        z_spacing = self.scan_area_mm / self.depth  # ~16.7 μm

        # Y-axis: tissue depth - preserve pixel aspect ratio
        # Assume similar resolution to X-axis for now
        y_spacing = x_spacing

        return (x_spacing, y_spacing, z_spacing)

    def create_vtk_volume(self) -> pv.ImageData:
        """
        Create PyVista ImageData with proper spacing and orientation.

        Returns:
            PyVista ImageData object with correct voxel spacing
        """
        # Create ImageData with correct spacing
        grid = pv.ImageData()

        # Set dimensions (PyVista uses (nx, ny, nz) order)
        grid.dimensions = (self.width, self.height, self.depth)

        # Set spacing in millimeters
        grid.spacing = self.spacing

        # Set origin (start at 0,0,0)
        grid.origin = (0, 0, 0)

        # Add scalar data - need to reshape and transpose for VTK ordering
        # VTK expects data in Fortran order (column-major)
        scalars = self.volume.transpose(1, 0, 2).flatten(order='F')
        grid.point_data['intensity'] = scalars

        logger.info(f"Created VTK volume: {grid.dimensions}, spacing: {grid.spacing}")

        return grid

    def render_volume(self, opacity_transfer='sigmoid', cmap='hot'):
        """
        Render the volume with proper volumetric rendering.

        Args:
            opacity_transfer: Opacity transfer function ('sigmoid', 'linear', 'geom')
            cmap: Colormap for intensity values
        """
        grid = self.create_vtk_volume()

        # Create plotter
        plotter = pv.Plotter()

        # Add volume with smart opacity mapping
        plotter.add_volume(
            grid,
            scalars='intensity',
            cmap=cmap,
            opacity=opacity_transfer,
            shade=True,
            show_scalar_bar=True
        )

        # Add scale bar
        plotter.add_scalar_bar(
            title='OCT Intensity',
            n_labels=5,
            vertical=True
        )

        # Set proper camera view
        plotter.camera_position = 'xy'
        plotter.camera.zoom(1.2)

        # Add title with physical dimensions
        plotter.add_text(
            f'OCT Volume: {self.scan_area_mm}mm × {self.scan_area_mm}mm scan area\n'
            f'Voxel spacing: {self.spacing[0]:.4f} × {self.spacing[1]:.4f} × {self.spacing[2]:.4f} mm',
            position='upper_left',
            font_size=10
        )

        plotter.show()

    def render_slices(self):
        """
        Render orthogonal slice views (axial, sagittal, coronal).
        """
        grid = self.create_vtk_volume()

        # Create plotter with multiple viewports
        plotter = pv.Plotter(shape=(2, 2))

        # Axial slice (XY plane) - top left
        plotter.subplot(0, 0)
        plotter.add_mesh(
            grid.slice(normal='z', origin=grid.center),
            cmap='hot',
            show_scalar_bar=False
        )
        plotter.add_text('Axial View (XY)', font_size=10)
        plotter.camera_position = 'xy'

        # Sagittal slice (YZ plane) - top right
        plotter.subplot(0, 1)
        plotter.add_mesh(
            grid.slice(normal='x', origin=grid.center),
            cmap='hot',
            show_scalar_bar=False
        )
        plotter.add_text('Sagittal View (YZ)', font_size=10)
        plotter.camera_position = 'yz'

        # Coronal slice (XZ plane) - bottom left
        plotter.subplot(1, 0)
        plotter.add_mesh(
            grid.slice(normal='y', origin=grid.center),
            cmap='hot',
            show_scalar_bar=False
        )
        plotter.add_text('Coronal View (XZ)', font_size=10)
        plotter.camera_position = 'xz'

        # 3D overview - bottom right
        plotter.subplot(1, 1)
        plotter.add_mesh(
            grid.outline(),
            color='white',
            line_width=2
        )
        # Add some slices for context
        plotter.add_mesh(grid.slice('x'), cmap='hot', opacity=0.5)
        plotter.add_mesh(grid.slice('y'), cmap='hot', opacity=0.5)
        plotter.add_mesh(grid.slice('z'), cmap='hot', opacity=0.5)
        plotter.add_text('3D Overview', font_size=10)

        plotter.show()

    def render_surface(self, threshold_percentile: float = 70):
        """
        Render isosurface at specified intensity threshold.

        Args:
            threshold_percentile: Intensity percentile for surface extraction
        """
        grid = self.create_vtk_volume()

        # Calculate threshold value
        threshold = np.percentile(self.volume, threshold_percentile)

        # Extract surface
        surface = grid.contour([threshold], scalars='intensity')

        # Create plotter
        plotter = pv.Plotter()

        # Add surface
        plotter.add_mesh(
            surface,
            color='tan',
            smooth_shading=True,
            show_scalar_bar=True
        )

        # Add outline for context
        plotter.add_mesh(grid.outline(), color='black', line_width=2)

        plotter.add_text(
            f'OCT Surface (>{threshold_percentile}th percentile)\n'
            f'Physical size: {self.scan_area_mm}mm × {self.scan_area_mm}mm',
            position='upper_left',
            font_size=10
        )

        plotter.show()

    def export_vtk(self, output_path: str):
        """
        Export volume to VTK format for external analysis.

        Args:
            output_path: Path to save VTK file
        """
        grid = self.create_vtk_volume()
        grid.save(output_path)
        logger.info(f"Exported volume to {output_path}")


def main():
    """Main function to run the volumetric OCT viewer."""
    parser = argparse.ArgumentParser(description='OCT Volumetric Viewer with PyVista')
    parser.add_argument('--data-dir', required=True, help='Directory containing BMP files')
    parser.add_argument('--sidebar-width', type=int, default=250, help='Sidebar width to remove (left side)')
    parser.add_argument('--crop-top', type=int, default=100, help='Pixels to crop from top')
    parser.add_argument('--crop-bottom', type=int, default=50, help='Pixels to crop from bottom')
    parser.add_argument('--scan-area', type=float, default=6.0, help='Physical scan area in mm (default: 6mm)')
    parser.add_argument('--mode', choices=['volume', 'slices', 'surface', 'all'], default='volume',
                       help='Visualization mode')
    parser.add_argument('--export', help='Export volume to VTK file')

    args = parser.parse_args()

    print(f"Loading OCT volume from: {args.data_dir}")
    print(f"Physical scan area: {args.scan_area}mm × {args.scan_area}mm")
    print(f"Processing: sidebar={args.sidebar_width}px (left), crop_top={args.crop_top}px, crop_bottom={args.crop_bottom}px")

    # Initialize components
    processor = OCTImageProcessor(sidebar_width=args.sidebar_width, crop_top=args.crop_top, crop_bottom=args.crop_bottom)
    loader = OCTVolumeLoader(processor)

    # Load volume
    print("\nLoading OCT volume...")
    volume = loader.load_volume_from_directory(args.data_dir)

    if volume is None:
        print("Error: Failed to load OCT volume")
        return

    # Create viewer
    print("\nInitializing volumetric viewer...")
    viewer = OCTVolumetricViewer(volume, scan_area_mm=args.scan_area)

    # Export if requested
    if args.export:
        viewer.export_vtk(args.export)

    # Render based on mode
    print(f"\nRendering {args.mode} view...")

    if args.mode == 'volume' or args.mode == 'all':
        viewer.render_volume()

    if args.mode == 'slices' or args.mode == 'all':
        viewer.render_slices()

    if args.mode == 'surface' or args.mode == 'all':
        viewer.render_surface()


if __name__ == "__main__":
    main()