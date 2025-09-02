#!/usr/bin/env python3
"""
OCT 3D Viewer - Interactive 3D visualization of OCT scan data
Reads BMP images from oct_data folder and creates rotatable 3D volume
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
import glob
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class OCT3DViewer:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.volume = None
        self.scan_files = []
        self.physical_params = {
            'width_mm': 6.0,  # 6mm scan width
            'a_scans': 1536,  # A-scans per B-scan
            'b_scans': 360,   # Total B-scans
        }
        
    def load_oct_data(self):
        """Load all BMP files and create 3D volume"""
        print("Loading OCT data...")
        
        # Find all scan files
        scan_pattern = os.path.join(self.data_folder, "*scan*.bmp")
        self.scan_files = sorted(glob.glob(scan_pattern))
        
        if not self.scan_files:
            raise FileNotFoundError(f"No BMP files found in {self.data_folder}")
            
        print(f"Found {len(self.scan_files)} scan files")
        
        # Load first image to get dimensions
        first_img = Image.open(self.scan_files[0])
        first_array = np.array(first_img)
        
        # Convert to grayscale if needed
        if len(first_array.shape) == 3:
            first_array = np.mean(first_array, axis=2)
            
        height, width = first_array.shape
        print(f"Image dimensions: {width} x {height}")
        
        # Initialize volume array
        self.volume = np.zeros((len(self.scan_files), height, width), dtype=np.float32)
        
        # Load all images
        for i, file_path in enumerate(self.scan_files):
            img = Image.open(file_path)
            img_array = np.array(img)
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                img_array = np.mean(img_array, axis=2)
            
            self.volume[i] = img_array.astype(np.float32)
            
            if (i + 1) % 50 == 0:
                print(f"Loaded {i + 1}/{len(self.scan_files)} images")
        
        print(f"Volume shape: {self.volume.shape}")
        return self.volume
    
    def create_interactive_3d_view(self):
        """Create fast 3D visualization using volume rendering"""
        if self.volume is None:
            self.load_oct_data()
            
        print("Creating interactive 3D visualization...")
        
        # Heavy downsampling for performance (every 16th pixel)
        volume_small = self.volume[::16, ::8, ::16]
        print(f"Downsampled volume shape: {volume_small.shape}")
        
        # Limit number of points to maximum 10000 for performance
        max_points = 10000
        total_points = np.prod(volume_small.shape)
        
        if total_points > max_points:
            step = int(np.cbrt(total_points / max_points))
            volume_small = volume_small[::step, ::step, ::step]
            print(f"Further downsampled to: {volume_small.shape}")
        
        # Create coordinate arrays
        z_coords, y_coords, x_coords = np.meshgrid(
            np.arange(volume_small.shape[0]),
            np.arange(volume_small.shape[1]),
            np.arange(volume_small.shape[2]),
            indexing='ij'
        )
        
        # Flatten and filter high intensity points only
        values = volume_small.flatten()
        threshold = np.percentile(values, 90)  # Only top 10% intensities
        mask = values > threshold
        
        x_plot = x_coords.flatten()[mask]
        y_plot = y_coords.flatten()[mask]
        z_plot = z_coords.flatten()[mask]
        values_plot = values[mask]
        
        print(f"Plotting {len(values_plot)} points")
        
        # Create simple 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=x_plot,
            y=y_plot,
            z=z_plot,
            mode='markers',
            marker=dict(
                size=3,
                color=values_plot,
                colorscale='Greys',
                opacity=0.8,
                colorbar=dict(title="Intensity")
            ),
            name='OCT Data'
        ))
        
        # Simple layout
        fig.update_layout(
            title='OCT 3D Volume (Downsampled)',
            scene=dict(
                xaxis_title='Width',
                yaxis_title='Depth',
                zaxis_title='B-scan',
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def create_simple_surface_view(self):
        """Create simple 3D surface view - most reliable option"""
        if self.volume is None:
            self.load_oct_data()
            
        print("Creating simple 3D surface view...")
        
        # Take middle B-scan as surface
        mid_scan = self.volume.shape[0] // 2
        surface_data = self.volume[mid_scan]
        
        # Downsample for performance
        surface_small = surface_data[::4, ::4]
        
        fig = go.Figure(data=[go.Surface(
            z=surface_small,
            colorscale='Greys',
            name=f'B-scan {mid_scan}'
        )])
        
        fig.update_layout(
            title=f'OCT B-scan {mid_scan} - 3D Surface View',
            scene=dict(
                xaxis_title='A-scan',
                yaxis_title='Depth',
                zaxis_title='Intensity',
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_cross_sections_view(self):
        """Create interactive cross-sections with sliders"""
        if self.volume is None:
            self.load_oct_data()
            
        print("Creating cross-sections view...")
        
        # Create subplot figure with 2x2 layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('B-scan (XY plane)', 'En-face (XZ plane)', 
                          'C-scan (YZ plane)', '3D Surface'),
            specs=[[{"type": "image"}, {"type": "image"}],
                   [{"type": "image"}, {"type": "surface"}]]
        )
        
        # Get middle slices for initial display
        mid_z = self.volume.shape[0] // 2
        mid_y = self.volume.shape[1] // 2  
        mid_x = self.volume.shape[2] // 2
        
        # B-scan (XY plane) - traditional OCT view
        fig.add_trace(
            go.Heatmap(z=self.volume[mid_z], colorscale='gray'),
            row=1, col=1
        )
        
        # En-face view (XZ plane)
        fig.add_trace(
            go.Heatmap(z=self.volume[:, mid_y, :], colorscale='gray'),
            row=1, col=2
        )
        
        # C-scan (YZ plane)
        fig.add_trace(
            go.Heatmap(z=self.volume[:, :, mid_x], colorscale='gray'),
            row=2, col=1
        )
        
        # 3D surface of a B-scan
        x_surf = np.arange(self.volume.shape[2])
        y_surf = np.arange(self.volume.shape[1])
        fig.add_trace(
            go.Surface(z=self.volume[mid_z], colorscale='gray'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='OCT Multi-planar Reconstruction',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def show_matplotlib_viewer(self):
        """Create matplotlib-based viewer with sliders"""
        if self.volume is None:
            self.load_oct_data()
            
        print("Creating matplotlib viewer...")
        
        # Create the figure and subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.subplots_adjust(bottom=0.25)
        
        # Initial slice positions
        initial_z = self.volume.shape[0] // 2
        initial_y = self.volume.shape[1] // 2
        initial_x = self.volume.shape[2] // 2
        
        # Display initial images
        im1 = axes[0,0].imshow(self.volume[initial_z], cmap='gray', aspect='auto')
        axes[0,0].set_title('B-scan (XY plane)')
        axes[0,0].set_xlabel('A-scan')
        axes[0,0].set_ylabel('Depth')
        
        im2 = axes[0,1].imshow(self.volume[:, initial_y, :], cmap='gray', aspect='auto')
        axes[0,1].set_title('En-face (XZ plane)')
        axes[0,1].set_xlabel('A-scan')
        axes[0,1].set_ylabel('B-scan')
        
        im3 = axes[1,0].imshow(self.volume[:, :, initial_x], cmap='gray', aspect='auto')
        axes[1,0].set_title('C-scan (YZ plane)')
        axes[1,0].set_xlabel('Depth')
        axes[1,0].set_ylabel('B-scan')
        
        # 3D-like visualization
        axes[1,1].remove()  # Remove the 4th subplot
        ax3d = fig.add_subplot(2, 2, 4, projection='3d')
        
        # Show volume outline
        x_range = np.arange(0, self.volume.shape[2], 20)
        y_range = np.arange(0, self.volume.shape[1], 20)
        z_range = np.arange(0, self.volume.shape[0], 20)
        
        # Sample points from volume
        sample_volume = self.volume[::20, ::20, ::20]
        threshold = np.percentile(sample_volume, 90)
        z_pts, y_pts, x_pts = np.where(sample_volume > threshold)
        
        ax3d.scatter(x_pts*20, y_pts*20, z_pts*20, 
                    c=sample_volume[z_pts, y_pts, x_pts], 
                    cmap='gray', alpha=0.1, s=1)
        ax3d.set_title('3D Volume Sample')
        ax3d.set_xlabel('Width')
        ax3d.set_ylabel('Depth')
        ax3d.set_zlabel('B-scan')
        
        # Add sliders
        ax_z = plt.axes([0.1, 0.15, 0.3, 0.03])
        ax_y = plt.axes([0.1, 0.1, 0.3, 0.03])
        ax_x = plt.axes([0.1, 0.05, 0.3, 0.03])
        
        slider_z = Slider(ax_z, 'B-scan', 0, self.volume.shape[0]-1, 
                         valinit=initial_z, valfmt='%d')
        slider_y = Slider(ax_y, 'Depth', 0, self.volume.shape[1]-1, 
                         valinit=initial_y, valfmt='%d')
        slider_x = Slider(ax_x, 'A-scan', 0, self.volume.shape[2]-1, 
                         valinit=initial_x, valfmt='%d')
        
        # Update function for sliders
        def update(val):
            z_idx = int(slider_z.val)
            y_idx = int(slider_y.val)
            x_idx = int(slider_x.val)
            
            im1.set_array(self.volume[z_idx])
            im2.set_array(self.volume[:, y_idx, :])
            im3.set_array(self.volume[:, :, x_idx])
            
            fig.canvas.draw()
        
        slider_z.on_changed(update)
        slider_y.on_changed(update)
        slider_x.on_changed(update)
        
        plt.suptitle('OCT 3D Viewer - Interactive Cross-sections', fontsize=14)
        plt.show()

def main():
    # Set up the data folder path
    data_folder = "/home/aristarx/Diploma/RetinaBuilder/oct_data/F001_IP_20250604_175814_Retina_3D_L_6mm_1536x360_2"
    
    # Create viewer instance
    viewer = OCT3DViewer(data_folder)
    
    try:
        print("OCT 3D Viewer")
        print("=============")
        print("Loading OCT scan data...")
        
        # Load the data
        volume = viewer.load_oct_data()
        print(f"Successfully loaded volume with shape: {volume.shape}")
        
        # Choose visualization method
        print("\nSelect visualization method:")
        print("1. Simple 3D surface view (Fast & Reliable)")
        print("2. 3D volume scatter plot (May be slow)")
        print("3. Cross-sections with sliders (Matplotlib)")
        print("4. Multi-planar reconstruction (Plotly)")
        print("5. All visualizations")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1' or choice == '5':
            print("\nCreating simple 3D surface view...")
            fig_surface = viewer.create_simple_surface_view()
            fig_surface.show()
        
        if choice == '2' or choice == '5':
            print("\nCreating 3D volume visualization...")
            fig_3d = viewer.create_interactive_3d_view()
            fig_3d.show()
        
        if choice == '3' or choice == '5':
            print("\nCreating matplotlib cross-sections viewer...")
            viewer.show_matplotlib_viewer()
            
        if choice == '4' or choice == '5':
            print("\nCreating multi-planar reconstruction...")
            fig_mpr = viewer.create_cross_sections_view()
            fig_mpr.show()
            
        print("\nVisualization complete!")
        print("You can now interact with the 3D views:")
        print("- Rotate by dragging")
        print("- Zoom with mouse wheel")  
        print("- Pan by shift+drag")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the oct_data folder exists and contains BMP files.")

if __name__ == "__main__":
    main()