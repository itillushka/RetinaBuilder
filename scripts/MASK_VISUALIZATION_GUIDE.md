# Mask Verification Guide

## Purpose

The mask visualization shows **exactly what regions are being compared** during rotation alignment, so you can verify the masking is working correctly.

## Output File

**`step3_mask_verification.png`** - Generated automatically before rotation search begins

Location: `notebooks/data/step3_mask_verification.png`

## Understanding the Visualization

The visualization has **4 rows × 3 columns** (12 panels total):

### Row 1: Original Volumes (Before Masking)

| Panel | Content | What to Look For |
|-------|---------|------------------|
| 1 | V0 En-face MIP (Reference) | Gray intensity image, should show vessel patterns |
| 2 | V1 En-face MIP (Before rotation) | Gray intensity image, similar vessel patterns |
| 3 | Overlay (Red+Green) | Check alignment quality before rotation |

### Row 2: Masks (What Gets Compared)

| Panel | Content | What to Look For |
|-------|---------|------------------|
| 4 | V0 Mask | White = included, Black = excluded. Should match tissue regions. |
| 5 | V1 Mask | White = included. Should be similar to V0 mask. |
| 6 | **Combined Mask (CRITICAL)** | **Hot (white/yellow) = final comparison region**. This is what NCC uses! |

**Key Metrics Shown:**
- Number of valid voxels
- Percentage of total volume
- **Combined mask is the intersection (V0 & V1)**

### Row 3: B-scans (Cross-sectional View)

| Panel | Content | What to Look For |
|-------|---------|------------------|
| 7 | V0 B-scan at Z=center | Should show retinal layers clearly |
| 8 | V1 B-scan at Z=center | Should show similar retinal structure |
| 9 | B-scan Overlay (Red+Green) | Check vertical alignment from Step 2 |

### Row 4: Masked B-scans (What NCC Actually Uses)

| Panel | Content | What to Look For |
|-------|---------|------------------|
| 10 | V0 B-scan MASKED | Black regions excluded, should keep retinal tissue |
| 11 | V1 B-scan MASKED | Should match V0 masked pattern |
| 12 | **FINAL Combined Mask Region** | **This is exactly what NCC compares!** Red+Green overlay of only valid regions |

## Printed Statistics

When Step 3 runs, you'll see:

```
  Mask Statistics:
    Volume 0 valid voxels: 45,234,567 (78.3%)
    Volume 1 valid voxels: 44,891,234 (77.7%)
    Combined valid voxels: 42,567,890 (73.8%)
    Overlap efficiency: 94.5%
```

**What These Mean:**

- **Volume 0 valid voxels**: How much of V0 overlap region has tissue (not background)
- **Volume 1 valid voxels**: How much of V1 overlap region has tissue
- **Combined valid voxels**: How much tissue is present in BOTH volumes (this is what's compared)
- **Overlap efficiency**: % of smaller volume that overlaps with larger volume

## What to Check

### ✅ Good Masking:

1. **Combined mask covers most tissue regions** (70-90% typical)
2. **Black regions are mostly background/corners**
3. **Retinal layers are included** in white regions
4. **Symmetric masking** - V0 and V1 masks look similar
5. **Overlap efficiency > 85%** - most tissue is being used

### ⚠️ Potential Issues:

1. **Combined mask < 50%** - too much exclusion, may need to adjust threshold
2. **Asymmetric masks** - V0 and V1 very different, check alignment quality
3. **Tissue excluded** - retinal layers appearing black in masked B-scans
4. **Overlap efficiency < 70%** - poor overlap, may need better XZ/Y alignment

## Adjusting the Mask (If Needed)

If masking looks wrong, you can adjust the threshold in `rotation_alignment.py`:

```python
def create_overlap_mask(volume, percentile=10):  # <-- Change this
    """Create mask for valid tissue regions."""
    threshold = np.percentile(volume[volume > 0], percentile)
    mask = volume > threshold
    return mask
```

**Percentile = 10 (default):**
- Excludes bottom 10% of intensities
- Removes background and noise
- **Good for most OCT volumes**

**Lower percentile (e.g., 5):**
- More inclusive, keeps weaker signals
- Use if tissue is being excluded

**Higher percentile (e.g., 20):**
- More exclusive, focuses on bright tissue
- Use if too much noise/background included

## Typical Results

### Well-Aligned Volumes
```
Combined valid voxels: 42M (75%)
Overlap efficiency: 95%
```
- Most tissue included
- High efficiency
- Good starting point for rotation

### Poorly-Aligned Volumes
```
Combined valid voxels: 15M (28%)
Overlap efficiency: 45%
```
- Low overlap
- May indicate problems with Steps 1-2
- Rotation search may be unreliable

## Example Interpretation

**Good Mask Visualization:**
```
Row 1: Vessels visible, reasonable alignment
Row 2: Combined mask covers 75% of volume
Row 3: B-scans show retinal layers
Row 4: Final mask includes all major tissue structures
```
→ **Proceed with rotation search, good quality input**

**Problem Mask Visualization:**
```
Row 1: Misaligned vessels
Row 2: Combined mask only 30%
Row 3: B-scans vertically offset
Row 4: Final mask missing tissue
```
→ **Re-check Steps 1-2 alignment before rotation**

## Files Generated

After Step 3 completes:

```
notebooks/data/
├── step3_mask_verification.png   # THIS FILE - check BEFORE rotation
├── step3_rotation_search.png     # Angle optimization
└── step3_rotation_comparison.png # Before/after results
```

## Quick Checklist

Before trusting rotation results, verify in `step3_mask_verification.png`:

- [ ] Combined mask (Row 2, Panel 6) is white/yellow in tissue regions
- [ ] Combined mask percentage > 60%
- [ ] Final B-scan overlay (Row 4, Panel 12) shows clear tissue
- [ ] Mask statistics show overlap efficiency > 80%
- [ ] No major retinal structures excluded (black in Row 4)

If all checkboxes pass → **Masking is correct**, proceed with confidence!

---

**Generated automatically when running:**
```bash
python alignment_pipeline.py --step 3
# or
python alignment_pipeline.py --all
```
