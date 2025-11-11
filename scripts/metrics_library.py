#!/usr/bin/env python3
"""
Metrics Library for Rotation Alignment

Collection of metrics for evaluating OCT B-scan alignment quality.
Includes both standard and custom metrics designed to detect parallel retinal layers.
"""

import numpy as np
from scipy import ndimage
from scipy.fft import fft2, fftshift
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information
from skimage.transform import radon, hough_line, hough_line_peaks
from skimage.feature import canny


class RotationMetrics:
    """Collection of metrics for rotation alignment evaluation"""

    @staticmethod
    def calculate_ncc(img1, img2, mask=None):
        """
        Normalized Cross-Correlation (NCC)

        Standard metric for image registration.
        Measures pixel-by-pixel intensity correlation.

        Range: [-1, 1] where 1 = perfect match
        Higher is better.

        Good for: Pixel intensity matching
        Bad for: Detecting parallel structures that don't perfectly overlap
        """
        if mask is not None:
            img1 = img1[mask]
            img2 = img2[mask]

        # Normalize
        img1_norm = (img1 - img1.mean()) / (img1.std() + 1e-8)
        img2_norm = (img2 - img2.mean()) / (img2.std() + 1e-8)

        # Calculate correlation
        ncc = np.mean(img1_norm * img2_norm)

        return float(ncc)

    @staticmethod
    def calculate_ssim(img1, img2):
        """
        Structural Similarity Index (SSIM)

        Measures structural similarity (luminance, contrast, structure).
        Better than NCC for capturing structural patterns.

        Range: [0, 1] where 1 = perfect match
        Higher is better.

        Good for: Structural pattern matching
        May work for: Layer alignment
        """
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
        if data_range == 0:
            return 0.0

        return float(ssim(img1, img2, data_range=data_range))

    @staticmethod
    def calculate_layer_gradient_correlation(img1, img2, mask=None):
        """
        Layer Gradient Correlation (LGC) - CUSTOM METRIC

        Measures correlation between horizontal gradients (perpendicular to layers).
        When retinal layers are parallel, horizontal gradients align strongly.

        Range: [-1, 1] where 1 = perfect gradient alignment
        Higher is better.

        Good for: Detecting parallel horizontal structures (retinal layers)
        Theory: Parallel layers have consistent vertical gradients
        """
        # Calculate horizontal gradients (Y-direction, perpendicular to layers)
        gy1 = ndimage.sobel(img1, axis=0)
        gy2 = ndimage.sobel(img2, axis=0)

        if mask is not None:
            gy1 = gy1[mask]
            gy2 = gy2[mask]

        # Normalize
        gy1_norm = (gy1 - gy1.mean()) / (gy1.std() + 1e-8)
        gy2_norm = (gy2 - gy2.mean()) / (gy2.std() + 1e-8)

        # Calculate correlation
        lgc = np.mean(gy1_norm * gy2_norm)

        return float(lgc)

    @staticmethod
    def calculate_edge_orientation_alignment(img1, img2, mask=None):
        """
        Edge Orientation Alignment (EOA) - CUSTOM METRIC

        Measures how well edge orientations match between images.
        Parallel layers have consistent horizontal edge orientations.

        Range: [0, 1] where 1 = perfect orientation match
        Higher is better.

        Good for: Detecting consistent edge angles (parallel layers)
        Theory: When layers are parallel, edge angles are similar
        """
        # Calculate gradients in both directions
        gx1 = ndimage.sobel(img1, axis=1)
        gy1 = ndimage.sobel(img1, axis=0)
        gx2 = ndimage.sobel(img2, axis=1)
        gy2 = ndimage.sobel(img2, axis=0)

        # Calculate edge orientations
        theta1 = np.arctan2(gy1, gx1)
        theta2 = np.arctan2(gy2, gx2)

        # Calculate edge magnitudes
        mag1 = np.sqrt(gx1**2 + gy1**2)
        mag2 = np.sqrt(gx2**2 + gy2**2)

        # Focus on strong edges
        threshold = np.percentile(np.concatenate([mag1.ravel(), mag2.ravel()]), 70)
        edge_mask = (mag1 > threshold) & (mag2 > threshold)

        if mask is not None:
            edge_mask = edge_mask & mask

        if edge_mask.sum() < 100:  # Need at least 100 edge pixels
            return 0.0

        # Calculate orientation difference (circular)
        theta_diff = np.abs(theta1[edge_mask] - theta2[edge_mask])
        theta_diff = np.minimum(theta_diff, 2*np.pi - theta_diff)

        # Score: inverse of mean angular difference
        eoa = 1.0 - np.mean(theta_diff) / np.pi

        return float(eoa)

    @staticmethod
    def calculate_horizontal_frequency_peak(img1, img2):
        """
        Horizontal Frequency Peak (HFP) - CUSTOM METRIC

        Measures alignment of horizontal frequency components in Fourier domain.
        Parallel horizontal layers create strong horizontal frequencies.

        Range: [-1, 1] where 1 = perfect frequency alignment
        Higher is better.

        Good for: Detecting horizontal periodicity (layer spacing)
        Theory: Parallel layers have matching horizontal frequency components
        """
        # FFT of both images
        fft_img1 = fftshift(fft2(img1))
        fft_img2 = fftshift(fft2(img2))

        # Power spectrum
        power1 = np.abs(fft_img1)**2
        power2 = np.abs(fft_img2)**2

        # Extract horizontal frequency band (around ky=0, exclude DC)
        h, w = power1.shape
        center_y = h // 2
        band_width = max(5, h // 20)  # Adaptive band width

        # Avoid DC component
        horiz_band1 = power1[center_y-band_width:center_y+band_width, :]
        horiz_band2 = power2[center_y-band_width:center_y+band_width, :]

        # Exclude DC column (center)
        center_x = w // 2
        dc_width = 5
        horiz_band1[:, center_x-dc_width:center_x+dc_width] = 0
        horiz_band2[:, center_x-dc_width:center_x+dc_width] = 0

        # Normalize and correlate
        horiz_band1_norm = (horiz_band1 - horiz_band1.mean()) / (horiz_band1.std() + 1e-8)
        horiz_band2_norm = (horiz_band2 - horiz_band2.mean()) / (horiz_band2.std() + 1e-8)

        hfp = np.mean(horiz_band1_norm * horiz_band2_norm)

        return float(hfp)

    @staticmethod
    def calculate_gradient_magnitude_correlation(img1, img2, mask=None):
        """
        Gradient Magnitude Correlation (GMC) - CUSTOM METRIC

        Measures correlation between gradient magnitudes (edge strength).
        Detects edge alignment without directionality bias.

        Range: [-1, 1] where 1 = perfect magnitude correlation
        Higher is better.

        Good for: Detecting edge structure similarity
        Theory: Parallel layers have similar edge strength patterns
        """
        # Calculate gradients
        gx1 = ndimage.sobel(img1, axis=1)
        gy1 = ndimage.sobel(img1, axis=0)
        gx2 = ndimage.sobel(img2, axis=1)
        gy2 = ndimage.sobel(img2, axis=0)

        # Calculate gradient magnitudes
        mag1 = np.sqrt(gx1**2 + gy1**2)
        mag2 = np.sqrt(gx2**2 + gy2**2)

        if mask is not None:
            mag1 = mag1[mask]
            mag2 = mag2[mask]

        # Normalize and correlate
        mag1_norm = (mag1 - mag1.mean()) / (mag1.std() + 1e-8)
        mag2_norm = (mag2 - mag2.mean()) / (mag2.std() + 1e-8)

        gmc = np.mean(mag1_norm * mag2_norm)

        return float(gmc)

    @staticmethod
    def calculate_nmi(img1, img2):
        """
        Normalized Mutual Information (NMI)

        Information-theoretic metric measuring statistical dependency.
        Based on joint probability distribution.

        Range: [0, 1+] where higher = more mutual information
        Higher is better.

        Good for: General image similarity
        May work for: Complex structural relationships
        """
        # Normalize to 0-255 range for histogram
        img1_norm = ((img1 - img1.min()) / (img1.max() - img1.min() + 1e-8) * 255).astype(int)
        img2_norm = ((img2 - img2.min()) / (img2.max() - img2.min() + 1e-8) * 255).astype(int)

        # Clip to valid range
        img1_norm = np.clip(img1_norm, 0, 255)
        img2_norm = np.clip(img2_norm, 0, 255)

        nmi = normalized_mutual_information(img1_norm.ravel(), img2_norm.ravel())

        return float(nmi)

    @staticmethod
    def calculate_mse(img1, img2):
        """
        Mean Squared Error (MSE)

        Simple pixel-wise difference metric.

        Range: [0, +inf] where 0 = perfect match
        LOWER is better (opposite of other metrics).

        Good for: Pixel intensity matching
        Bad for: Structural alignment
        """
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)

        return float(mse)

    @staticmethod
    def calculate_radon_transform_alignment(img1, img2):
        """
        Radon Transform Alignment (RTA) - NEW PARALLEL-LAYER METRIC

        Uses Radon transform to detect dominant line angles in both images.
        When layers are parallel and horizontal, both images have strongest
        projections at 0° (horizontal).

        Range: [0, 1] where 1 = perfect horizontal alignment
        Higher is better.

        Theory: Parallel horizontal layers create strong Radon peaks at 0°
        """
        # Normalize images to 0-255
        img1_norm = ((img1 - img1.min()) / (img1.max() - img1.min() + 1e-8) * 255).astype(np.uint8)
        img2_norm = ((img2 - img2.min()) / (img2.max() - img2.min() + 1e-8) * 255).astype(np.uint8)

        # Radon transform at angles around horizontal (0°)
        # Test angles from -45° to +45° around horizontal
        theta = np.linspace(-45, 45, 91)

        radon1 = radon(img1_norm, theta=theta)
        radon2 = radon(img2_norm, theta=theta)

        # Find peak strength at each angle
        peak1 = np.max(radon1, axis=0)  # Max projection for each angle
        peak2 = np.max(radon2, axis=0)

        # Normalize peaks
        peak1_norm = (peak1 - peak1.mean()) / (peak1.std() + 1e-8)
        peak2_norm = (peak2 - peak2.mean()) / (peak2.std() + 1e-8)

        # Correlate the peak profiles
        # High correlation means both images have similar angular distribution
        correlation = np.corrcoef(peak1_norm, peak2_norm)[0, 1]

        # Bonus: Check if both peak at horizontal (0° is at index 45)
        horizontal_idx = 45
        horizontal_weight1 = peak1[horizontal_idx] / (np.max(peak1) + 1e-8)
        horizontal_weight2 = peak2[horizontal_idx] / (np.max(peak2) + 1e-8)

        # Combined score: correlation + horizontal emphasis
        rta = 0.7 * correlation + 0.3 * min(horizontal_weight1, horizontal_weight2)

        return float(np.clip(rta, 0, 1))

    @staticmethod
    def calculate_row_wise_correlation_profile(img1, img2, mask=None):
        """
        Row-Wise Correlation Profile (RCP) - NEW PARALLEL-LAYER METRIC

        Calculates correlation for each row independently and measures uniformity.
        When layers are parallel, corresponding rows correlate uniformly.

        Range: [0, 1] where 1 = perfect uniform row correlation
        Higher is better.

        Theory: Parallel layers have consistent row-by-row correlation
        """
        H, W = img1.shape

        # Calculate correlation for each row
        row_correlations = []

        for row_idx in range(H):
            row1 = img1[row_idx, :]
            row2 = img2[row_idx, :]

            # Skip if mask excludes this row
            if mask is not None:
                row_mask = mask[row_idx, :]
                if row_mask.sum() < W * 0.3:  # Skip if < 30% valid
                    continue
                row1 = row1[row_mask]
                row2 = row2[row_mask]

            # Skip if row is too uniform
            if row1.std() < 1 or row2.std() < 1:
                continue

            # Calculate correlation
            row1_norm = (row1 - row1.mean()) / (row1.std() + 1e-8)
            row2_norm = (row2 - row2.mean()) / (row2.std() + 1e-8)
            corr = np.mean(row1_norm * row2_norm)

            row_correlations.append(corr)

        if len(row_correlations) < 10:  # Need at least 10 rows
            return 0.0

        row_correlations = np.array(row_correlations)

        # Score: mean correlation with penalty for high variance
        # High mean + low variance = uniformly high correlation = parallel layers
        mean_corr = np.mean(row_correlations)
        std_corr = np.std(row_correlations)

        # Normalize: high mean, low std is best
        rcp = mean_corr * (1 - std_corr)

        return float(np.clip(rcp, 0, 1))

    @staticmethod
    def calculate_layer_spacing_consistency(img1, img2, num_samples=20):
        """
        Layer Spacing Consistency (LSC) - NEW PARALLEL-LAYER METRIC

        Detects layer peaks in vertical profiles at multiple X positions.
        When layers are parallel, peak spacing is consistent across image.

        Range: [0, 1] where 1 = perfect layer spacing match
        Higher is better.

        Theory: Parallel layers have uniform vertical spacing
        """
        H, W = img1.shape

        # Sample vertical profiles at multiple X positions
        x_positions = np.linspace(W // 4, 3 * W // 4, num_samples).astype(int)

        spacing_similarity_scores = []

        for x in x_positions:
            # Extract vertical profiles
            profile1 = img1[:, x]
            profile2 = img2[:, x]

            # Find peaks (layer boundaries)
            peaks1, _ = find_peaks(profile1, distance=10, prominence=20)
            peaks2, _ = find_peaks(profile2, distance=10, prominence=20)

            if len(peaks1) < 2 or len(peaks2) < 2:
                continue

            # Calculate spacing between consecutive peaks
            spacing1 = np.diff(peaks1)
            spacing2 = np.diff(peaks2)

            # Compare spacing distributions
            # If layers are parallel, spacing patterns should match
            if len(spacing1) > 0 and len(spacing2) > 0:
                # Normalize spacings
                spacing1_norm = spacing1 / (np.mean(spacing1) + 1e-8)
                spacing2_norm = spacing2 / (np.mean(spacing2) + 1e-8)

                # Calculate similarity (inverse of difference)
                min_len = min(len(spacing1_norm), len(spacing2_norm))
                if min_len > 0:
                    diff = np.abs(spacing1_norm[:min_len] - spacing2_norm[:min_len])
                    similarity = 1.0 - np.mean(diff)
                    spacing_similarity_scores.append(similarity)

        if len(spacing_similarity_scores) < 5:  # Need at least 5 samples
            return 0.0

        # Average similarity across all sampled positions
        lsc = np.mean(spacing_similarity_scores)

        return float(np.clip(lsc, 0, 1))

    @staticmethod
    def calculate_horizontal_line_detection(img1, img2):
        """
        Horizontal Line Detection (HLD) - NEW PARALLEL-LAYER METRIC

        Uses Hough transform to detect horizontal lines in both images.
        When layers are parallel, both images have strong horizontal lines.

        Range: [0, 1] where 1 = perfect horizontal line match
        Higher is better.

        Theory: Parallel layers maximize horizontal line detection
        """
        # Edge detection
        edges1 = canny(img1, sigma=2)
        edges2 = canny(img2, sigma=2)

        # Hough transform to detect lines
        # Focus on angles near horizontal (0° ± 15°)
        tested_angles = np.deg2rad(np.arange(-15, 16, 1))
        h1, theta1, d1 = hough_line(edges1, theta=tested_angles)
        h2, theta2, d2 = hough_line(edges2, theta=tested_angles)

        # Find peaks in Hough space
        try:
            _, angles1, _ = hough_line_peaks(h1, theta1, d1, num_peaks=10)
            _, angles2, _ = hough_line_peaks(h2, theta2, d2, num_peaks=10)
        except:
            return 0.0

        if len(angles1) == 0 or len(angles2) == 0:
            return 0.0

        # Convert angles to degrees
        angles1_deg = np.rad2deg(angles1)
        angles2_deg = np.rad2deg(angles2)

        # Count how many lines are near horizontal (±5°)
        horizontal_threshold = 5
        horizontal_count1 = np.sum(np.abs(angles1_deg) < horizontal_threshold)
        horizontal_count2 = np.sum(np.abs(angles2_deg) < horizontal_threshold)

        # Score based on:
        # 1. Percentage of horizontal lines in both images
        # 2. Similarity in angle distributions
        horizontal_ratio1 = horizontal_count1 / (len(angles1) + 1e-8)
        horizontal_ratio2 = horizontal_count2 / (len(angles2) + 1e-8)

        # Average horizontal ratio
        avg_horizontal = (horizontal_ratio1 + horizontal_ratio2) / 2

        # Angle distribution similarity
        angle_diff = np.abs(np.mean(angles1_deg) - np.mean(angles2_deg))
        angle_similarity = max(0, 1 - angle_diff / 15)  # Normalize by max tested angle

        # Combined score
        hld = 0.6 * avg_horizontal + 0.4 * angle_similarity

        return float(np.clip(hld, 0, 1))

    @staticmethod
    def calculate_layer_contour_smoothness(img1, img2, num_layers=5):
        """
        Layer Contour Smoothness (LCS) - NEW METHOD FOR TRUE PARALLEL LAYER DETECTION

        Detects retinal layer boundaries and measures their horizontal smoothness.
        When layers are parallel, boundaries are smooth, nearly horizontal curves.

        Range: [0, 1] where 1 = perfectly smooth parallel layers
        Higher is better.

        Theory: Retinal layers are naturally smooth structures. Rotation at wrong
        angle creates jagged/irregular layer boundaries. This metric directly
        measures layer structure, not just intensity correlation.
        """
        def extract_layer_smoothness(img):
            H, W = img.shape

            # Detect layer boundaries using vertical gradient
            vertical_gradient = np.abs(ndimage.sobel(img, axis=0))

            # Sample at multiple X positions
            x_positions = np.linspace(W//4, 3*W//4, 50).astype(int)
            layer_y_coords = []

            for x in x_positions:
                profile = vertical_gradient[:, x]

                # Skip if profile is empty or too uniform
                if profile.max() < 1 or len(profile[profile > 0]) < 10:
                    continue

                # Find peaks (layer boundaries)
                prominence_threshold = np.percentile(profile[profile > 0], 60) if len(profile[profile > 0]) > 0 else 10
                peaks, properties = find_peaks(profile,
                                              distance=20,  # Min distance between layers
                                              prominence=prominence_threshold)

                if len(peaks) > 0:
                    # Take top N peaks by prominence
                    if len(peaks) > num_layers:
                        prominences = properties['prominences']
                        top_indices = np.argsort(prominences)[-num_layers:]
                        peaks = peaks[top_indices]
                        peaks = np.sort(peaks)  # Sort by Y position

                    layer_y_coords.append(peaks)

            if len(layer_y_coords) < 10:
                return 0.0

            # For each detected layer, fit a polynomial curve
            smoothness_scores = []
            for layer_idx in range(min(num_layers, min(len(coords) for coords in layer_y_coords))):
                # Extract Y coordinates for this layer across X
                y_coords = []
                x_coords = []
                for i, peaks in enumerate(layer_y_coords):
                    if layer_idx < len(peaks):
                        y_coords.append(peaks[layer_idx])
                        x_coords.append(x_positions[i])

                if len(y_coords) < 5:
                    continue

                # Fit polynomial (degree 3 for natural curvature)
                try:
                    coeffs = np.polyfit(x_coords, y_coords, deg=3)
                    fitted_curve = np.polyval(coeffs, x_coords)

                    # Measure residual (deviation from smooth curve)
                    residual = np.std(y_coords - fitted_curve)
                    max_residual = H * 0.1  # Normalize by image height
                    smoothness = 1.0 / (1.0 + residual / max_residual)
                    smoothness_scores.append(smoothness)
                except:
                    continue

            if len(smoothness_scores) == 0:
                return 0.0

            return np.mean(smoothness_scores)

        # Calculate smoothness for both images
        smoothness1 = extract_layer_smoothness(img1)
        smoothness2 = extract_layer_smoothness(img2)

        # Both should have smooth layers when properly aligned
        lcs = (smoothness1 + smoothness2) / 2

        return float(np.clip(lcs, 0, 1))

    @staticmethod
    def calculate_directional_gradient_histogram(img1, img2, bins=36):
        """
        Directional Gradient Histogram (DGH) - NEW METHOD FOR HORIZONTAL STRUCTURE DETECTION

        Measures concentration of edge orientations around horizontal (0°).
        When layers are parallel, gradients cluster around horizontal.

        Range: [0, 1] where 1 = perfect horizontal alignment
        Higher is better.

        Theory: Parallel retinal layers create strong horizontal edges.
        This metric directly measures the "horizontalness" of structures.
        """
        def calculate_gradient_concentration(img):
            # Calculate gradients
            gx = ndimage.sobel(img, axis=1)
            gy = ndimage.sobel(img, axis=0)

            # Calculate angles and magnitudes
            angles = np.arctan2(gy, gx)  # Range: -π to π
            magnitudes = np.sqrt(gx**2 + gy**2)

            # Focus on strong edges
            threshold = np.percentile(magnitudes[magnitudes > 0], 70)
            strong_edge_mask = magnitudes > threshold

            if strong_edge_mask.sum() < 100:
                return 0.0

            strong_angles = angles[strong_edge_mask]
            strong_magnitudes = magnitudes[strong_edge_mask]

            # Create weighted histogram
            hist, bin_edges = np.histogram(strong_angles, bins=bins,
                                          range=(-np.pi, np.pi),
                                          weights=strong_magnitudes)

            # Normalize histogram
            hist = hist / (np.sum(hist) + 1e-8)

            # Measure concentration around horizontal (0° and ±180°)
            # 0° is at bin index bins//2, ±π at bins 0 and bins-1
            horizontal_indices = [0, bins//2, bins-1]
            horizontal_weight = sum(hist[i] for i in horizontal_indices if i < len(hist))

            # Also calculate entropy (lower = more concentrated)
            entropy = -np.sum(hist * np.log(hist + 1e-8))
            max_entropy = np.log(bins)
            concentration = 1.0 - entropy / max_entropy

            # Combined score
            score = horizontal_weight * 2.0 * concentration  # Weight horizontal more

            return score

        # Calculate for both images
        conc1 = calculate_gradient_concentration(img1)
        conc2 = calculate_gradient_concentration(img2)

        # Both should have horizontal concentration
        dgh = (conc1 + conc2) / 2

        return float(np.clip(dgh, 0, 1))

    @staticmethod
    def calculate_vertical_profile_correlation(img1, img2, num_samples=50):
        """
        Vertical Profile Correlation (VPC) - NEW METHOD FOR CURVED LAYER ALIGNMENT

        Correlates vertical intensity profiles at multiple X positions.
        When layers are parallel (even if curved), vertical profiles correlate consistently.

        Range: [0, 1] where 1 = perfect consistent vertical correlation
        Higher is better.

        Theory: Unlike RCP (row-wise), VPC samples perpendicular to layers.
        Handles curved layers naturally as each vertical profile captures local structure.
        """
        H, W = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])

        # Sample at evenly spaced X positions
        x_positions = np.linspace(W//8, 7*W//8, num_samples).astype(int)

        correlations = []
        valid_profiles = 0

        for x in x_positions:
            if x >= img1.shape[1] or x >= img2.shape[1]:
                continue

            profile1 = img1[:min(H, img1.shape[0]), x]
            profile2 = img2[:min(H, img2.shape[0]), x]

            # Skip if profiles are too uniform
            if profile1.std() < 5 or profile2.std() < 5:
                continue

            # Normalize profiles
            p1_norm = (profile1 - profile1.mean()) / (profile1.std() + 1e-8)
            p2_norm = (profile2 - profile2.mean()) / (profile2.std() + 1e-8)

            # Calculate correlation
            try:
                corr = np.corrcoef(p1_norm, p2_norm)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
                    valid_profiles += 1
            except:
                continue

        if len(correlations) < 10:
            return 0.0

        correlations = np.array(correlations)

        # Score = high mean correlation + low variance (consistency)
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)

        # Normalize mean_corr from [-1, 1] to [0, 1]
        mean_corr_norm = (mean_corr + 1) / 2

        # Penalize high variance
        consistency = 1.0 - std_corr

        # Combined score
        vpc = mean_corr_norm * consistency

        return float(np.clip(vpc, 0, 1))

    @staticmethod
    def evaluate_all(img1, img2, mask=None):
        """
        Calculate all metrics at once.

        Args:
            img1: Reference image (2D numpy array)
            img2: Image to compare (2D numpy array)
            mask: Optional binary mask (2D boolean array)

        Returns:
            Dictionary with all metric scores
        """
        return {
            # Original 8 metrics
            'ncc': RotationMetrics.calculate_ncc(img1, img2, mask),
            'ssim': RotationMetrics.calculate_ssim(img1, img2),
            'lgc': RotationMetrics.calculate_layer_gradient_correlation(img1, img2, mask),
            'eoa': RotationMetrics.calculate_edge_orientation_alignment(img1, img2, mask),
            'hfp': RotationMetrics.calculate_horizontal_frequency_peak(img1, img2),
            'nmi': RotationMetrics.calculate_nmi(img1, img2),
            'gmc': RotationMetrics.calculate_gradient_magnitude_correlation(img1, img2, mask),
            'mse': RotationMetrics.calculate_mse(img1, img2),
            # New parallel-layer metrics
            'rta': RotationMetrics.calculate_radon_transform_alignment(img1, img2),
            'rcp': RotationMetrics.calculate_row_wise_correlation_profile(img1, img2, mask),
            'lsc': RotationMetrics.calculate_layer_spacing_consistency(img1, img2),
            'hld': RotationMetrics.calculate_horizontal_line_detection(img1, img2),
            # Latest metrics for true parallel layer detection
            'lcs': RotationMetrics.calculate_layer_contour_smoothness(img1, img2),
            'dgh': RotationMetrics.calculate_directional_gradient_histogram(img1, img2),
            'vpc': RotationMetrics.calculate_vertical_profile_correlation(img1, img2)
        }


if __name__ == '__main__':
    print("="*70)
    print("ROTATION METRICS LIBRARY - 15 METRICS")
    print("="*70)
    print("\nOriginal 8 metrics:")
    print("  1. NCC  - Normalized Cross-Correlation (standard)")
    print("  2. SSIM - Structural Similarity Index (standard)")
    print("  3. LGC  - Layer Gradient Correlation (custom)")
    print("  4. EOA  - Edge Orientation Alignment (custom)")
    print("  5. HFP  - Horizontal Frequency Peak (custom)")
    print("  6. NMI  - Normalized Mutual Information (standard)")
    print("  7. GMC  - Gradient Magnitude Correlation (custom)")
    print("  8. MSE  - Mean Squared Error (standard, lower=better)")
    print("\nFirst-gen Parallel-Layer Metrics:")
    print("  9. RTA  - Radon Transform Alignment")
    print(" 10. RCP  - Row-Wise Correlation Profile")
    print(" 11. LSC  - Layer Spacing Consistency")
    print(" 12. HLD  - Horizontal Line Detection")
    print("\nLatest True Parallel-Layer Metrics:")
    print(" 13. LCS  - Layer Contour Smoothness (detects smooth horizontal layers)")
    print(" 14. DGH  - Directional Gradient Histogram (measures horizontalness)")
    print(" 15. VPC  - Vertical Profile Correlation (handles curved layers)")
    print("\nNew metrics (13-15) specifically address RCP failure modes.")
    print("="*70)
