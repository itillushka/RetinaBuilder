"""
Script to integrate BOE2020 vessel segmentation into our pipeline.
This can be run standalone or integrated into the Jupyter notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import cv2

# Add BOE2020 to path
sys.path.insert(0, '/home/aristarx/Diploma/BOE2020-OCTA-vessel-segmentation')

from manager import ModelManager
import tensorflow as tf

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def prepare_image_for_boe2020(enface_mip, target_size=(304, 304)):
    """
    Prepare MIP en-face image for BOE2020 model.

    Args:
        enface_mip: MIP en-face projection (W, Z)
        target_size: Target size for model input

    Returns:
        img_prepared: Prepared image for model (H, W, 1)
        img_original: Original normalized image
    """
    # Normalize to 0-255
    img_norm = ((enface_mip - enface_mip.min()) /
                (enface_mip.max() - enface_mip.min()) * 255).astype(np.uint8)

    # Transpose to (Z, W) for standard image orientation
    img_norm = img_norm.T

    # Resize to model input size
    img_resized = cv2.resize(img_norm, target_size, interpolation=cv2.INTER_LINEAR)

    # Add channel dimension: (H, W) -> (H, W, 1)
    img_prepared = img_resized[:, :, np.newaxis]

    return img_prepared, img_norm


def save_image_for_boe2020(img_prepared, output_path):
    """
    Save prepared image to disk for BOE2020 processing.

    Args:
        img_prepared: Prepared image (H, W, 1)
        output_path: Path to save image
    """
    # Remove channel dimension for saving
    img_to_save = img_prepared[:, :, 0]
    cv2.imwrite(str(output_path), img_to_save)
    print(f"Saved image to {output_path}")


def segment_vessels_boe2020(image_dir, model_ckpt_dir, output_dir):
    """
    Run BOE2020 vessel segmentation on images in a directory.

    Args:
        image_dir: Directory containing input images
        model_ckpt_dir: Path to pretrained model checkpoint
        output_dir: Directory to save segmentation outputs

    Returns:
        manager: ModelManager object (for further use)
    """
    # Initialize model
    # iUNET with L=3 layers, F=64 feature maps, 4 iterations during training
    # We can use more iterations (e.g., 6) for potentially better results
    manager = ModelManager(
        name='iUNET',
        n_iterations=6,  # Can be higher than training (4)
        num_layers=3,
        feature_maps_root=64
    )

    # Run segmentation
    manager.run_on_images(
        image_dir,
        model_ckpt_dir,
        get_intermediate_outputs=True,
        show_outputs=False,  # Set to True to visualize during processing
        path_to_save_dir=output_dir
    )

    return manager


def load_segmentation_result(result_path):
    """
    Load segmentation result from BOE2020 output.

    Args:
        result_path: Path to segmentation result image

    Returns:
        segmentation: Binary vessel mask
    """
    seg = cv2.imread(str(result_path), cv2.IMREAD_GRAYSCALE)

    # Convert to binary (BOE2020 outputs are typically 0-255)
    vessel_mask = seg > 127

    return vessel_mask


if __name__ == "__main__":
    # Load MIP from Phase 1
    data_dir = Path('../notebooks/data')
    enface_mip = np.load(data_dir / 'enface_mip_volume0.npy')

    print(f"Loaded MIP shape: {enface_mip.shape}")

    # Prepare image for BOE2020
    img_prepared, img_original = prepare_image_for_boe2020(enface_mip)
    print(f"Prepared image shape: {img_prepared.shape}")

    # Save prepared image
    boe_input_dir = Path('../notebooks/data/boe2020_input')
    boe_input_dir.mkdir(exist_ok=True, parents=True)

    input_image_path = boe_input_dir / 'volume0_mip.png'
    save_image_for_boe2020(img_prepared, input_image_path)

    # Run BOE2020 segmentation
    model_ckpt_dir = '/home/aristarx/Diploma/BOE2020-OCTA-vessel-segmentation/pretrained_models/iUNET/L_3_F_64_loss_i-bce-topo_C1_2_C2_2_C3_4_0.01_0.001_0.0001_iters_4'
    output_dir = Path('../notebooks/data/boe2020_output')
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\nRunning BOE2020 vessel segmentation...")
    segment_vessels_boe2020(str(boe_input_dir), model_ckpt_dir, str(output_dir))

    print("\n✓ BOE2020 segmentation complete!")
    print(f"Results saved to: {output_dir}")
