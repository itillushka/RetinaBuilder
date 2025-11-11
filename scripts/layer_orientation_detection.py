#!/usr/bin/env python3
"""
Layer Orientation Detection for OCT B-scan Rotation Alignment

This module directly measures the orientation of retinal layers in OCT B-scans
to determine the optimal rotation angle. Unlike correlation-based metrics,
this approach measures actual anatomical structure orientation.

Methods:
1. Hough Transform: Detects dominant line angles in edge image
2. Gradient-based: Detects ILM/RPE boundaries and measures their slope

Author: RetinaBuilder Pipeline
"""

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from skimage import filters, feature, transform
import matplotlib.pyplot as plt


class LayerOrientationDetector:
    """
    Detects retinal layer orientation in OCT B-scans for rotation alignment.
    """

    @staticmethod
    def detect_orientation_hough(bscan, mask=None, angle_range=(-20, 20),
                                  angle_step=0.5, num_peaks=10,
                                  sigma=2.0, visualize=False):
        """
        Detect layer orientation using Hough line transform.

        This method:
        1. Applies Canny edge detection to find layer boundaries
        2. Uses Hough transform to find dominant line angles
        3. Returns the mean angle of detected lines

        Args:
            bscan: 2D array (Y, X) - OCT B-scan image
            mask: Optional 2D boolean array - region to analyze
            angle_range: Tuple (min_deg, max_deg) - angles to test
            angle_step: Float - angular resolution in degrees
            num_peaks: Int - number of Hough peaks to average
            sigma: Float - Gaussian sigma for Canny edge detection
            visualize: Bool - if True, return visualization data

        Returns:
            angle: Float - detected layer tilt in degrees (+ = counterclockwise)
            confidence: Float - detection confidence (0-1)
            vis_data: Dict - visualization data (only if visualize=True)
        """
        # Apply mask if provided
        if mask is not None:
            bscan_masked = bscan.copy()
            bscan_masked[~mask] = 0
        else:
            bscan_masked = bscan

        # Normalize image
        bscan_norm = (bscan_masked - bscan_masked.min()) / (bscan_masked.max() - bscan_masked.min() + 1e-8)

        # Apply Canny edge detection
        edges = feature.canny(bscan_norm, sigma=sigma)

        # If mask provided, only keep edges in masked region
        if mask is not None:
            edges = edges & mask

        # Define angles to test
        tested_angles = np.deg2rad(np.arange(angle_range[0], angle_range[1] + angle_step, angle_step))

        # Hough line transform
        hough_space, angles_rad, distances = transform.hough_line(edges, theta=tested_angles)

        # Find peaks in Hough space
        hough_peaks, angles_peaks, dists_peaks = transform.hough_line_peaks(
            hough_space, angles_rad, distances,
            num_peaks=num_peaks,
            threshold=0.3 * hough_space.max()
        )

        if len(angles_peaks) == 0:
            # No lines detected
            return 0.0, 0.0, None

        # Convert angles to degrees
        # Note: Hough returns angle perpendicular to line normal
        # For horizontal layers, we want 0°
        angles_deg = np.rad2deg(angles_peaks)

        # Adjust angles: Hough angle is measured from horizontal
        # We want layer tilt angle (deviation from horizontal)
        # If layer is tilted +10°, Hough will detect lines at -80° (perpendicular)
        # Actually, let's check: horizontal lines have theta=0 or 180 in Hough
        # Tilted lines have theta = tilt angle

        # Calculate mean angle (this is the layer tilt)
        mean_angle = np.mean(angles_deg)

        # Calculate confidence based on:
        # 1. Number of detected lines
        # 2. Consistency (low std deviation)
        # 3. Hough peak strength
        std_angle = np.std(angles_deg)
        consistency = np.exp(-std_angle / 10.0)  # High when std is low
        coverage = min(len(angles_peaks) / num_peaks, 1.0)
        peak_strength = np.mean(hough_peaks) / (hough_space.max() + 1e-8)

        confidence = consistency * coverage * peak_strength

        # Prepare visualization data
        vis_data = None
        if visualize:
            vis_data = {
                'edges': edges,
                'hough_space': hough_space,
                'angles_rad': angles_rad,
                'distances': distances,
                'peak_angles_deg': angles_deg,
                'peak_distances': dists_peaks,
                'peak_strengths': hough_peaks
            }

        return mean_angle, confidence, vis_data

    @staticmethod
    def detect_orientation_gradient(bscan, mask=None, num_samples=20,
                                     poly_degree=2, visualize=False):
        """
        Detect layer orientation by finding ILM/RPE boundaries using gradients.

        This method:
        1. Calculates vertical gradient (perpendicular to layers)
        2. Samples A-scans across the B-scan
        3. Finds ILM (first peak) and RPE (strongest peak) in each A-scan
        4. Fits polynomial to detected boundaries
        5. Measures slope angle

        Args:
            bscan: 2D array (Y, X) - OCT B-scan image
            mask: Optional 2D boolean array - region to analyze
            num_samples: Int - number of A-scans to sample
            poly_degree: Int - polynomial degree for curve fitting
            visualize: Bool - if True, return visualization data

        Returns:
            angle: Float - detected layer tilt in degrees
            confidence: Float - detection confidence (0-1)
            vis_data: Dict - visualization data (only if visualize=True)
        """
        H, W = bscan.shape

        # Determine sampling positions
        if mask is not None:
            # Find X positions that have sufficient mask coverage
            mask_coverage = mask.sum(axis=0) / H
            valid_x = np.where(mask_coverage > 0.5)[0]
            if len(valid_x) < num_samples:
                valid_x = np.where(mask_coverage > 0.2)[0]
            if len(valid_x) == 0:
                return 0.0, 0.0, None
            x_positions = np.linspace(valid_x[0], valid_x[-1], num_samples, dtype=int)
        else:
            x_positions = np.linspace(0, W - 1, num_samples, dtype=int)

        # Calculate vertical gradient
        gradient_y = np.abs(ndimage.sobel(bscan, axis=0))

        # Detect layer boundaries
        ilm_points = []  # (x, y) tuples for ILM
        rpe_points = []  # (x, y) tuples for RPE

        for x in x_positions:
            # Extract A-scan and gradient profile
            ascan = bscan[:, x]
            grad_profile = gradient_y[:, x]

            # Apply mask if provided
            if mask is not None:
                valid_y = np.where(mask[:, x])[0]
                if len(valid_y) < 10:
                    continue
                y_min, y_max = valid_y[0], valid_y[-1]
                grad_profile_masked = grad_profile[y_min:y_max]
                ascan_masked = ascan[y_min:y_max]
            else:
                y_min = 0
                grad_profile_masked = grad_profile
                ascan_masked = ascan

            # Skip if profile too short
            if len(grad_profile_masked) < 20:
                continue

            # Find peaks in gradient (layer boundaries)
            prominence_threshold = np.percentile(grad_profile_masked, 70)
            peaks, properties = find_peaks(
                grad_profile_masked,
                prominence=prominence_threshold,
                distance=10
            )

            if len(peaks) == 0:
                continue

            # ILM: First strong peak from top
            ilm_y = y_min + peaks[0]
            ilm_points.append((x, ilm_y))

            # RPE: Peak with strongest signal in A-scan
            # (RPE is hyperreflective)
            peak_intensities = [ascan_masked[p] for p in peaks]
            rpe_idx = np.argmax(peak_intensities)
            rpe_y = y_min + peaks[rpe_idx]
            rpe_points.append((x, rpe_y))

        # Need at least 3 points to fit
        if len(ilm_points) < 3 or len(rpe_points) < 3:
            return 0.0, 0.0, None

        # Convert to arrays
        ilm_points = np.array(ilm_points)
        rpe_points = np.array(rpe_points)

        # Fit polynomial curves
        ilm_coeffs = np.polyfit(ilm_points[:, 0], ilm_points[:, 1], deg=poly_degree)
        rpe_coeffs = np.polyfit(rpe_points[:, 0], rpe_points[:, 1], deg=poly_degree)

        # Calculate fitted curves
        x_fit = np.linspace(ilm_points[:, 0].min(), ilm_points[:, 0].max(), 100)
        ilm_fit = np.polyval(ilm_coeffs, x_fit)
        rpe_fit = np.polyval(rpe_coeffs, x_fit)

        # Measure slope at center of B-scan
        x_center = W // 2

        # Derivative of polynomial at center
        ilm_slope = np.polyval(np.polyder(ilm_coeffs), x_center)
        rpe_slope = np.polyval(np.polyder(rpe_coeffs), x_center)

        # Convert slope to angle
        ilm_angle = np.rad2deg(np.arctan(ilm_slope))
        rpe_angle = np.rad2deg(np.arctan(rpe_slope))

        # Average the two layer angles
        mean_angle = (ilm_angle + rpe_angle) / 2.0

        # Confidence based on:
        # 1. Agreement between ILM and RPE
        # 2. Number of detected points
        # 3. Fit quality (residuals)
        angle_agreement = np.exp(-abs(ilm_angle - rpe_angle) / 5.0)
        point_coverage = min(len(ilm_points) / num_samples, 1.0)

        ilm_residuals = np.std(ilm_points[:, 1] - np.polyval(ilm_coeffs, ilm_points[:, 0]))
        rpe_residuals = np.std(rpe_points[:, 1] - np.polyval(rpe_coeffs, rpe_points[:, 0]))
        fit_quality = np.exp(-(ilm_residuals + rpe_residuals) / 20.0)

        confidence = angle_agreement * point_coverage * fit_quality

        # Visualization data
        vis_data = None
        if visualize:
            vis_data = {
                'gradient_y': gradient_y,
                'ilm_points': ilm_points,
                'rpe_points': rpe_points,
                'ilm_fit_x': x_fit,
                'ilm_fit_y': ilm_fit,
                'rpe_fit_x': x_fit,
                'rpe_fit_y': rpe_fit,
                'ilm_angle': ilm_angle,
                'rpe_angle': rpe_angle
            }

        return mean_angle, confidence, vis_data

    @staticmethod
    def detect_orientation(bscan, mask=None, method='hough', **kwargs):
        """
        Detect layer orientation using specified method.

        Args:
            bscan: 2D array (Y, X) - OCT B-scan image
            mask: Optional 2D boolean array - region to analyze
            method: String - 'hough' or 'gradient'
            **kwargs: Additional arguments for specific method

        Returns:
            angle: Float - detected layer tilt in degrees
            confidence: Float - detection confidence (0-1)
            vis_data: Dict or None - visualization data
        """
        if method == 'hough':
            return LayerOrientationDetector.detect_orientation_hough(bscan, mask, **kwargs)
        elif method == 'gradient':
            return LayerOrientationDetector.detect_orientation_gradient(bscan, mask, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'hough' or 'gradient'")


def visualize_detection(bscan, angle, confidence, vis_data, method='hough',
                        save_path=None, title=None):
    """
    Visualize layer orientation detection results.

    Args:
        bscan: 2D array - original B-scan
        angle: Float - detected angle
        confidence: Float - confidence score
        vis_data: Dict - visualization data from detector
        method: String - detection method used
        save_path: Optional string - path to save figure
        title: Optional string - figure title
    """
    if method == 'hough':
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Original B-scan
        axes[0].imshow(bscan, cmap='gray', aspect='auto')
        axes[0].set_title('Original B-scan')
        axes[0].set_xlabel('X (lateral)')
        axes[0].set_ylabel('Y (depth)')

        # Edge detection
        axes[1].imshow(vis_data['edges'], cmap='gray', aspect='auto')
        axes[1].set_title('Canny Edge Detection')
        axes[1].set_xlabel('X (lateral)')
        axes[1].set_ylabel('Y (depth)')

        # Hough space
        angles_deg = np.rad2deg(vis_data['angles_rad'])
        im = axes[2].imshow(
            vis_data['hough_space'],
            cmap='hot',
            aspect='auto',
            extent=[angles_deg[0], angles_deg[-1],
                   vis_data['distances'][-1], vis_data['distances'][0]]
        )
        axes[2].set_title(f'Hough Transform\nDetected: {angle:.2f}° (conf: {confidence:.3f})')
        axes[2].set_xlabel('Angle (degrees)')
        axes[2].set_ylabel('Distance (pixels)')
        axes[2].axvline(angle, color='cyan', linestyle='--', linewidth=2, label=f'Mean: {angle:.2f}°')

        # Mark detected peaks
        for peak_angle in vis_data['peak_angles_deg']:
            axes[2].axvline(peak_angle, color='yellow', linestyle=':', alpha=0.5)

        axes[2].legend()
        plt.colorbar(im, ax=axes[2])

    elif method == 'gradient':
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Original B-scan with detected boundaries
        axes[0].imshow(bscan, cmap='gray', aspect='auto')
        axes[0].plot(vis_data['ilm_points'][:, 0], vis_data['ilm_points'][:, 1],
                    'r.', markersize=8, label='ILM points')
        axes[0].plot(vis_data['rpe_points'][:, 0], vis_data['rpe_points'][:, 1],
                    'b.', markersize=8, label='RPE points')
        axes[0].plot(vis_data['ilm_fit_x'], vis_data['ilm_fit_y'],
                    'r-', linewidth=2, label=f'ILM fit ({vis_data["ilm_angle"]:.2f}°)')
        axes[0].plot(vis_data['rpe_fit_x'], vis_data['rpe_fit_y'],
                    'b-', linewidth=2, label=f'RPE fit ({vis_data["rpe_angle"]:.2f}°)')
        axes[0].set_title(f'Detected Layer Boundaries\nMean: {angle:.2f}° (conf: {confidence:.3f})')
        axes[0].set_xlabel('X (lateral)')
        axes[0].set_ylabel('Y (depth)')
        axes[0].legend()

        # Gradient image
        axes[1].imshow(vis_data['gradient_y'], cmap='hot', aspect='auto')
        axes[1].set_title('Vertical Gradient (Layer Boundaries)')
        axes[1].set_xlabel('X (lateral)')
        axes[1].set_ylabel('Y (depth)')

        # Angle comparison
        axes[2].bar(['ILM', 'RPE', 'Mean'],
                   [vis_data['ilm_angle'], vis_data['rpe_angle'], angle],
                   color=['red', 'blue', 'green'])
        axes[2].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[2].set_ylabel('Angle (degrees)')
        axes[2].set_title('Layer Orientation Angles')
        axes[2].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")

    plt.close()


if __name__ == '__main__':
    print("Layer Orientation Detection Module")
    print("===================================")
    print()
    print("This module provides methods to detect retinal layer orientation in OCT B-scans.")
    print()
    print("Usage:")
    print("  from layer_orientation_detection import LayerOrientationDetector")
    print()
    print("  # Hough transform method (fast, direct)")
    print("  angle, conf, vis = LayerOrientationDetector.detect_orientation_hough(")
    print("      bscan, mask=mask, visualize=True)")
    print()
    print("  # Gradient-based method (more precise, layer-specific)")
    print("  angle, conf, vis = LayerOrientationDetector.detect_orientation_gradient(")
    print("      bscan, mask=mask, visualize=True)")
    print()
    print("See test_layer_orientation.py for examples.")
