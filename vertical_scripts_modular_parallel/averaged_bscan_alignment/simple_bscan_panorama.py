"""
Simple B-Scan Panorama Stitching

A simplified script that:
1. Loads 3 OCT volumes
2. Averages central B-scans
3. Stitches them into a panorama using various OpenCV methods

No denoising - raw images preserve texture for better feature detection.
No manual alignment - lets the panorama libraries handle everything automatically.
"""

import numpy as np
import cv2
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.oct_loader import OCTImageProcessor, OCTVolumeLoader
from helpers.rotation_alignment import preprocess_oct_for_visualization
from bscan_averaging import extract_averaged_central_bscan


def prepare_bscans_for_stitching(vol1_path, vol2_path, vol3_path, n_bscans=30):
    """
    Load volumes and average central B-scans (no denoising).

    Args:
        vol1_path: Path to first volume directory
        vol2_path: Path to second volume directory
        vol3_path: Path to third volume directory
        n_bscans: Number of central B-scans to average (default: 30)

    Returns:
        List of 3 raw averaged B-scans ready for stitching
    """
    print("\n" + "="*80)
    print("PREPARING B-SCANS FOR PANORAMA STITCHING")
    print("="*80)

    # ========================================================================
    # STEP 1: Load volumes
    # ========================================================================
    print("\nSTEP 1: Loading volumes...")
    processor = OCTImageProcessor(
        sidebar_width=250,
        crop_top=100,
        crop_bottom=50
    )
    loader = OCTVolumeLoader(processor)

    print("  Loading Volume 1...")
    volume_1 = loader.load_volume_from_directory(vol1_path)
    print(f"    Shape: {volume_1.shape}")

    print("  Loading Volume 2...")
    volume_2 = loader.load_volume_from_directory(vol2_path)
    print(f"    Shape: {volume_2.shape}")

    print("  Loading Volume 3...")
    volume_3 = loader.load_volume_from_directory(vol3_path)
    print(f"    Shape: {volume_3.shape}")

    # ========================================================================
    # STEP 2: Extract and average central B-scans
    # ========================================================================
    print(f"\nSTEP 2: Extracting and averaging {n_bscans} central B-scans...")

    print("  Volume 1...")
    avg_v1_raw, _ = extract_averaged_central_bscan(volume_1, n_bscans)

    print("  Volume 2...")
    avg_v2_raw, _ = extract_averaged_central_bscan(volume_2, n_bscans)

    print("  Volume 3...")
    avg_v3_raw, _ = extract_averaged_central_bscan(volume_3, n_bscans)

    # Free memory
    del volume_1, volume_2, volume_3
    print("  Volumes freed from memory")

    # ========================================================================
    # STEP 3: Skip denoising (use raw images for better feature detection)
    # ========================================================================
    print("\nSTEP 3: Using RAW averaged images (no denoising)")
    print("  This preserves texture and features needed for stitching")

    print("\n" + "="*80)
    print("PREPARATION COMPLETE - 3 raw B-scans ready for stitching")
    print("="*80)

    return [avg_v1_raw, avg_v2_raw, avg_v3_raw]


def normalize_to_8bit(images):
    """Convert images to 8-bit for OpenCV stitching."""
    images_8bit = []
    for img in images:
        if img.dtype != np.uint8:
            # Normalize to 0-255
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img_norm = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_norm = np.zeros_like(img, dtype=np.uint8)
        else:
            img_norm = img
        images_8bit.append(img_norm)
    return images_8bit


def stitch_with_opencv_stitcher(images, mode='panorama', output_dir=None):
    """
    Stitch images using OpenCV's built-in Stitcher class.

    Args:
        images: List of 3 images to stitch
        mode: 'panorama' or 'scans' mode
        output_dir: Directory to save debug visualizations

    Returns:
        status: Status code (0 = success)
        panorama: Stitched panorama (or None if failed)
    """
    print(f"\n{'='*80}")
    print(f"TRYING: OpenCV Stitcher ({mode.upper()} mode)")
    print(f"{'='*80}")

    # Convert to 8-bit
    images_8bit = normalize_to_8bit(images)

    # Convert grayscale to BGR (OpenCV Stitcher works better with color)
    images_bgr = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in images_8bit]

    # Create stitcher
    if mode.lower() == 'panorama':
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    else:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)

    # Try stitching
    print("  Stitching images...")
    try:
        status, panorama = stitcher.stitch(images_bgr)

        if status == cv2.Stitcher_OK:
            print(f"  SUCCESS! Panorama created")
            print(f"  Panorama shape: {panorama.shape}")

            # Convert back to grayscale
            panorama_gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

            if output_dir:
                output_path = output_dir / f'panorama_{mode}_mode.png'
                cv2.imwrite(str(output_path), panorama_gray)
                print(f"  Saved: {output_path}")

            return status, panorama_gray
        else:
            error_messages = {
                1: "ERR_NEED_MORE_IMGS",
                2: "ERR_HOMOGRAPHY_EST_FAIL",
                3: "ERR_CAMERA_PARAMS_ADJUST_FAIL"
            }
            error_msg = error_messages.get(status, f"Unknown error ({status})")
            print(f"  FAILED: {error_msg}")
            return status, None

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return -1, None


def stitch_with_features(images, output_dir=None):
    """
    Manual stitching using SIFT features + homography + blending.

    Args:
        images: List of 3 images to stitch
        output_dir: Directory to save debug visualizations

    Returns:
        panorama: Stitched panorama (or None if failed)
    """
    print(f"\n{'='*80}")
    print("TRYING: Manual Feature-Based Stitching (SIFT)")
    print(f"{'='*80}")

    # Convert to 8-bit
    images_8bit = normalize_to_8bit(images)

    try:
        # Create SIFT detector
        sift = cv2.SIFT_create()

        # Detect features in all images
        keypoints_list = []
        descriptors_list = []

        for i, img in enumerate(images_8bit):
            print(f"  Detecting features in image {i+1}...")
            kp, des = sift.detectAndCompute(img, None)
            keypoints_list.append(kp)
            descriptors_list.append(des)
            print(f"    Found {len(kp)} keypoints")

        # Match features between adjacent images
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Match image 1 and 2
        print("\n  Matching images 1 and 2...")
        matches_12 = matcher.knnMatch(descriptors_list[0], descriptors_list[1], k=2)

        # Apply Lowe's ratio test
        good_12 = []
        for m, n in matches_12:
            if m.distance < 0.7 * n.distance:
                good_12.append(m)
        print(f"    {len(good_12)} good matches")

        # Match image 2 and 3
        print("  Matching images 2 and 3...")
        matches_23 = matcher.knnMatch(descriptors_list[1], descriptors_list[2], k=2)

        good_23 = []
        for m, n in matches_23:
            if m.distance < 0.7 * n.distance:
                good_23.append(m)
        print(f"    {len(good_23)} good matches")

        if len(good_12) < 10 or len(good_23) < 10:
            print("  FAILED: Not enough good matches (need at least 10)")
            return None

        # Visualize matches if output_dir provided
        if output_dir:
            print("\n  Visualizing matches...")

            # Draw matches 1-2
            match_img_12 = cv2.drawMatches(
                images_8bit[0], keypoints_list[0],
                images_8bit[1], keypoints_list[1],
                good_12[:50], None,  # Show top 50 matches
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imwrite(str(output_dir / 'matches_1_2.png'), match_img_12)

            # Draw matches 2-3
            match_img_23 = cv2.drawMatches(
                images_8bit[1], keypoints_list[1],
                images_8bit[2], keypoints_list[2],
                good_23[:50], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imwrite(str(output_dir / 'matches_2_3.png'), match_img_23)
            print(f"    Saved match visualizations")

        # Find homographies
        print("\n  Computing homographies...")

        # Homography to align image 2 to image 1
        src_pts_12 = np.float32([keypoints_list[0][m.queryIdx].pt for m in good_12]).reshape(-1, 1, 2)
        dst_pts_12 = np.float32([keypoints_list[1][m.trainIdx].pt for m in good_12]).reshape(-1, 1, 2)
        H_12, mask_12 = cv2.findHomography(dst_pts_12, src_pts_12, cv2.RANSAC, 5.0)
        print(f"    H(2→1): {np.sum(mask_12)} inliers / {len(good_12)} matches")

        # Homography to align image 3 to image 2
        src_pts_23 = np.float32([keypoints_list[1][m.queryIdx].pt for m in good_23]).reshape(-1, 1, 2)
        dst_pts_23 = np.float32([keypoints_list[2][m.trainIdx].pt for m in good_23]).reshape(-1, 1, 2)
        H_23, mask_23 = cv2.findHomography(dst_pts_23, src_pts_23, cv2.RANSAC, 5.0)
        print(f"    H(3→2): {np.sum(mask_23)} inliers / {len(good_23)} matches")

        # Compose homography: H(3→1) = H(2→1) @ H(3→2)
        H_13 = H_12 @ H_23

        # Determine output panorama size
        h1, w1 = images_8bit[0].shape
        h2, w2 = images_8bit[1].shape
        h3, w3 = images_8bit[2].shape

        # Get corners of all images in reference frame (image 1)
        corners_1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        corners_2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        corners_3 = np.float32([[0, 0], [w3, 0], [w3, h3], [0, h3]]).reshape(-1, 1, 2)

        corners_2_warped = cv2.perspectiveTransform(corners_2, H_12)
        corners_3_warped = cv2.perspectiveTransform(corners_3, H_13)

        all_corners = np.concatenate([corners_1, corners_2_warped, corners_3_warped], axis=0)

        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # Translation to shift everything to positive coordinates
        translation = np.array([[1, 0, -x_min],
                                [0, 1, -y_min],
                                [0, 0, 1]])

        print(f"\n  Creating panorama canvas: {x_max - x_min} x {y_max - y_min} pixels")

        # Warp images
        panorama_width = x_max - x_min
        panorama_height = y_max - y_min

        img1_warped = cv2.warpPerspective(images_8bit[0], translation,
                                          (panorama_width, panorama_height))
        img2_warped = cv2.warpPerspective(images_8bit[1], translation @ H_12,
                                          (panorama_width, panorama_height))
        img3_warped = cv2.warpPerspective(images_8bit[2], translation @ H_13,
                                          (panorama_width, panorama_height))

        # Simple blending: take maximum (works for dark backgrounds)
        print("  Blending images...")
        panorama = np.maximum(img1_warped, np.maximum(img2_warped, img3_warped))

        print(f"  SUCCESS! Panorama created")
        print(f"  Panorama shape: {panorama.shape}")

        if output_dir:
            output_path = output_dir / 'panorama_manual_sift.png'
            cv2.imwrite(str(output_path), panorama)
            print(f"  Saved: {output_path}")

        return panorama

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def visualize_input_images(images, output_path):
    """Visualize the 3 input images before stitching."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img, cmap='gray', aspect='auto')
        ax.set_title(f'Input Image {i+1}\nShape: {img.shape}', fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Input Images for Panorama Stitching', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nInput visualization saved: {output_path}")


def main():
    """Main function for simple panorama stitching."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Simple B-Scan Panorama Stitching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Using patient ID
  python simple_bscan_panorama.py --patient EM001

  # Using explicit volume paths
  python simple_bscan_panorama.py \\
    --vol1 ../../oct_data/emmetropes/EM001/Volume_1 \\
    --vol2 ../../oct_data/emmetropes/EM001/Volume_2 \\
    --vol3 ../../oct_data/emmetropes/EM001/Volume_3 \\
    --output ./panorama_results
        """
    )

    # Option 1: Use patient ID
    parser.add_argument('--patient', type=str, default=None,
                       help='Patient ID to search for (e.g., EM001, EM005)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override OCT data directory')

    # Option 2: Use explicit paths
    parser.add_argument('--vol1', help='Path to volume 1')
    parser.add_argument('--vol2', help='Path to volume 2')
    parser.add_argument('--vol3', help='Path to volume 3')

    # Common options
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--n-bscans', type=int, default=30,
                       help='Number of central B-scans to average (default: 30)')

    args = parser.parse_args()

    # Determine volume paths
    if args.patient:
        # Search for patient volumes
        scripts_dir = Path(__file__).parent.parent
        retina_builder_dir = scripts_dir.parent

        if args.data_dir:
            oct_data_dir = Path(args.data_dir)
        else:
            oct_data_dir = retina_builder_dir / 'oct_data'
            if not oct_data_dir.exists():
                oct_data_dir = retina_builder_dir / 'OCT_DATA'

        # Find volume directories
        bmp_dirs = []
        for bmp_file in oct_data_dir.rglob('*.bmp'):
            vol_dir = bmp_file.parent
            if vol_dir not in bmp_dirs:
                bmp_dirs.append(vol_dir)

        # Filter by patient ID
        patient_vols = sorted([v for v in bmp_dirs if args.patient in str(v)])

        if len(patient_vols) < 3:
            raise ValueError(f"Need at least 3 volumes for patient {args.patient}, found {len(patient_vols)}")

        print(f"\nFound {len(patient_vols)} volumes for patient {args.patient}")
        for i, vol in enumerate(patient_vols[:3]):
            print(f"  Volume {i+1}: {vol.name}")

        vol1_path = str(patient_vols[0])
        vol2_path = str(patient_vols[1])
        vol3_path = str(patient_vols[2])

        # Auto-generate output directory
        if not args.output:
            patient_safe = args.patient.replace('_', '').replace('/', '').replace('\\', '')
            output_dir = Path(__file__).parent / f'panorama_results_{patient_safe.lower()}'
        else:
            output_dir = Path(args.output)
    else:
        # Use explicit paths
        if not (args.vol1 and args.vol2 and args.vol3):
            parser.error("Must provide either --patient or all of --vol1, --vol2, --vol3")

        vol1_path = args.vol1
        vol2_path = args.vol2
        vol3_path = args.vol3

        if not args.output:
            parser.error("--output is required when using explicit volume paths")
        output_dir = Path(args.output)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # ========================================================================
    # PREPARE B-SCANS (Load, Average, Denoise)
    # ========================================================================
    images = prepare_bscans_for_stitching(
        vol1_path, vol2_path, vol3_path,
        n_bscans=args.n_bscans
    )

    # Visualize input images
    visualize_input_images(images, output_dir / 'input_images.png')

    # ========================================================================
    # TRY DIFFERENT STITCHING METHODS
    # ========================================================================
    print("\n" + "="*80)
    print("ATTEMPTING PANORAMA STITCHING WITH MULTIPLE METHODS")
    print("="*80)

    results = {}

    # Method 1: OpenCV Stitcher (PANORAMA mode)
    status, panorama = stitch_with_opencv_stitcher(images, mode='panorama', output_dir=output_dir)
    results['opencv_panorama'] = {
        'status': status,
        'success': status == 0,
        'panorama': panorama
    }

    # Method 2: OpenCV Stitcher (SCANS mode)
    status, panorama = stitch_with_opencv_stitcher(images, mode='scans', output_dir=output_dir)
    results['opencv_scans'] = {
        'status': status,
        'success': status == 0,
        'panorama': panorama
    }

    # Method 3: Manual SIFT stitching
    panorama = stitch_with_features(images, output_dir=output_dir)
    results['manual_sift'] = {
        'status': 0 if panorama is not None else -1,
        'success': panorama is not None,
        'panorama': panorama
    }

    # ========================================================================
    # SUMMARY
    # ========================================================================
    elapsed_time = time.time() - start_time

    print("\n" + "="*80)
    print("STITCHING RESULTS SUMMARY")
    print("="*80)
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    print(f"\nMethod Results:")
    print(f"  OpenCV Stitcher (PANORAMA): {'SUCCESS' if results['opencv_panorama']['success'] else 'FAILED'}")
    print(f"  OpenCV Stitcher (SCANS):    {'SUCCESS' if results['opencv_scans']['success'] else 'FAILED'}")
    print(f"  Manual SIFT Stitching:      {'SUCCESS' if results['manual_sift']['success'] else 'FAILED'}")

    # Find the best result
    successful_methods = [name for name, res in results.items() if res['success']]

    if successful_methods:
        print(f"\nSuccessful methods: {', '.join(successful_methods)}")
        print(f"\nOutput files saved to: {output_dir}")
    else:
        print("\nWARNING: All stitching methods failed!")
        print("\nPossible reasons:")
        print("  1. Images don't have enough overlapping regions")
        print("  2. OCT images lack distinctive features for matching")
        print("  3. Images are too different (brightness, contrast, etc.)")
        print("\nSuggestions:")
        print("  - Try the full alignment pipeline (averaged_bscan_pipeline.py)")
        print("  - Adjust preprocessing parameters")
        print("  - Ensure volumes are from the same scanning session")

    print("="*80 + "\n")

    return results


if __name__ == '__main__':
    main()
