"""
Аналіз різних метрик для rotation alignment
Порівняємо SSIM, NCC, MSE, Mutual Information
"""

import numpy as np
from scipy.ndimage import rotate, shift
from scipy.fft import fft2
from skimage.metrics import structural_similarity as ssim, normalized_mutual_information


def calculate_ncc(img1, img2):
    """
    Normalized Cross-Correlation - класична метрика для image registration

    NCC = sum((img1 - mean1) * (img2 - mean2)) / (std1 * std2 * N)

    Діапазон: -1 до 1 (1 = ідеальний збіг)
    """
    img1_norm = (img1 - img1.mean()) / (img1.std() + 1e-8)
    img2_norm = (img2 - img2.mean()) / (img2.std() + 1e-8)

    ncc = np.mean(img1_norm * img2_norm)
    return float(ncc)


def calculate_mse(img1, img2):
    """
    Mean Squared Error - проста але ефективна метрика

    MSE = mean((img1 - img2)^2)

    Діапазон: 0 до +inf (0 = ідеальний збіг)
    МЕНШЕ = КРАЩЕ!
    """
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    return float(mse)


def calculate_mi(img1, img2, bins=50):
    """
    Mutual Information - інформаційна метрика

    MI вимірює статистичну залежність між зображеннями

    Діапазон: 0 до +inf (вище = краще)
    """
    # Normalize to 0-1 range
    img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
    img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)

    # Calculate mutual information using sklearn
    mi = normalized_mutual_information(
        (img1_norm * (bins-1)).astype(int).ravel(),
        (img2_norm * (bins-1)).astype(int).ravel()
    )

    return float(mi)


def calculate_edge_overlap(img1, img2, threshold_percentile=90):
    """
    Edge Overlap - порівняння країв (важливо для retinal layers!)

    Знаходить краї в обох зображеннях і порівнює їх overlap
    """
    from scipy.ndimage import sobel

    # Calculate edges using Sobel
    edges1 = np.sqrt(sobel(img1, axis=0)**2 + sobel(img1, axis=1)**2)
    edges2 = np.sqrt(sobel(img2, axis=0)**2 + sobel(img2, axis=1)**2)

    # Threshold to binary
    threshold1 = np.percentile(edges1, threshold_percentile)
    threshold2 = np.percentile(edges2, threshold_percentile)

    edges1_binary = edges1 > threshold1
    edges2_binary = edges2 > threshold2

    # Calculate overlap (Dice coefficient)
    intersection = np.sum(edges1_binary & edges2_binary)
    total = np.sum(edges1_binary) + np.sum(edges2_binary)

    if total > 0:
        dice = 2.0 * intersection / total
    else:
        dice = 0.0

    return float(dice)


def evaluate_all_metrics(bscan_ref, bscan_mov):
    """
    Обчислити всі метрики одразу

    Returns:
        dict з всіма метриками
    """
    return {
        'ssim': ssim(bscan_ref, bscan_mov,
                    data_range=max(bscan_ref.max() - bscan_ref.min(),
                                  bscan_mov.max() - bscan_mov.min())),
        'ncc': calculate_ncc(bscan_ref, bscan_mov),
        'mse': calculate_mse(bscan_ref, bscan_mov),
        'mi': calculate_mi(bscan_ref, bscan_mov),
        'edge_overlap': calculate_edge_overlap(bscan_ref, bscan_mov)
    }


def find_best_rotation_multi_metric(bscan_ref, bscan_mov,
                                     phase_corr_func,
                                     angle_range=30, step=2, max_y_shift=50):
    """
    Знайти найкращий кут використовуючи КІЛЬКА метрик

    Повертає результати для кожної метрики окремо
    """
    angles_to_test = range(-angle_range, angle_range + 1, step)

    results = []

    print(f"Testing {len(angles_to_test)} angles with multiple metrics...")
    print(f"Metrics: SSIM, NCC, MSE, MI, Edge Overlap\n")

    for i, angle in enumerate(angles_to_test):
        if i % 5 == 0:
            print(f"  Progress: {i}/{len(angles_to_test)} angles...")

        # Rotate B-scan
        bscan_rotated = rotate(
            bscan_mov, angle, axes=(0, 1),
            reshape=False, order=1, mode='constant', cval=0
        )

        # Find Y-offset
        y_offset, confidence, _ = phase_corr_func(
            bscan_ref, bscan_rotated, max_shift=max_y_shift
        )

        # Apply Y-offset
        bscan_aligned = shift(
            bscan_rotated, shift=(y_offset, 0),
            order=1, mode='constant', cval=0
        )

        # Calculate ALL metrics
        metrics = evaluate_all_metrics(bscan_ref, bscan_aligned)

        results.append({
            'angle': angle,
            'y_offset': y_offset,
            'confidence': confidence,
            **metrics  # Unpack all metrics
        })

    print("\n✓ Multi-metric evaluation complete!")

    # Find best angle for each metric
    best_angles = {
        'ssim': max(results, key=lambda x: x['ssim']),
        'ncc': max(results, key=lambda x: x['ncc']),
        'mse': min(results, key=lambda x: x['mse']),  # LOWER is better for MSE!
        'mi': max(results, key=lambda x: x['mi']),
        'edge_overlap': max(results, key=lambda x: x['edge_overlap'])
    }

    return results, best_angles


if __name__ == "__main__":
    print("Rotation Metrics Analysis Module")
    print("="*60)
    print("Available metrics:")
    print("  1. SSIM - Structural Similarity (0-1, higher better)")
    print("  2. NCC - Normalized Cross-Correlation (-1 to 1, higher better)")
    print("  3. MSE - Mean Squared Error (0+, LOWER better)")
    print("  4. MI - Mutual Information (0+, higher better)")
    print("  5. Edge Overlap - Dice coefficient for edges (0-1, higher better)")
