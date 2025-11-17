"""
Test Y-offset visualization with manual adjustments.

Generates 3D visualizations with different Y-offsets to verify alignment effect.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from helpers import generate_3d_visualizations, OCTImageProcessor, OCTVolumeLoader

# Load existing results
data_dir = Path(__file__).parent / 'results'
step1_results = np.load(data_dir / 'step1_results.npy', allow_pickle=True).item()
step2_results = np.load(data_dir / 'step2_results.npy', allow_pickle=True).item()

# Load volumes
parent_dir = Path(__file__).parent.parent
oct_data_dir = parent_dir / 'OCT_DATA'

processor = OCTImageProcessor(sidebar_width=250, crop_top=100, crop_bottom=50)
loader = OCTVolumeLoader(processor)

# Find volume directories
bmp_dirs = []
for bmp_file in oct_data_dir.rglob('*.bmp'):
    vol_dir = bmp_file.parent
    if vol_dir not in bmp_dirs:
        bmp_dirs.append(vol_dir)

f001_vols = sorted([v for v in bmp_dirs if 'F001_IP' in str(v)])
volume_0 = loader.load_volume_from_directory(str(f001_vols[0]))
volume_1 = loader.load_volume_from_directory(str(f001_vols[1]))

print("Testing different Y-offsets:")
print(f"Original Step 2 Y-shift: {step2_results['y_shift']:.2f} px")

# Test offsets: original, +50, -50
test_offsets = [
    (0, "original"),
    (50, "plus_50px"),
    (-50, "minus_50px")
]

for offset_adjustment, label in test_offsets:
    print(f"\n{'='*70}")
    print(f"Testing: {label} (adjustment: {offset_adjustment:+d} px)")
    print(f"{'='*70}")

    # Create modified step2_results with adjusted Y-shift
    step2_modified = step2_results.copy()
    step2_modified['y_shift'] = step2_results['y_shift'] + offset_adjustment

    print(f"Applied Y-shift: {step2_modified['y_shift']:.2f} px")

    # Create output directory for this test
    test_dir = data_dir / f'test_y_offset_{label}'
    test_dir.mkdir(exist_ok=True)

    # Generate visualization with modified offset
    generate_3d_visualizations(
        volume_0,
        step1_results,
        step2_modified,
        test_dir,
        step3_results=None,
        volume_1_aligned=None
    )

    print(f"\nVisualization saved to: {test_dir}")

print(f"\n{'='*70}")
print("All test visualizations complete!")
print(f"{'='*70}")
print("\nCompare the results in:")
for _, label in test_offsets:
    print(f"  - results/test_y_offset_{label}/")
