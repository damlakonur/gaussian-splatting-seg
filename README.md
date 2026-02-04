# Semantic Training Setup Guide

Quick step-by-step guide for setting up semantic segmentation training with ScanNet++ dataset.

## Steps

### 1. Generate Class Remapping File

```bash
cd gaussian-splatting-seg
python utils/semantic_utils.py <scene_id> --save
```

Example:
```bash
python utils/semantic_utils.py f36e3e1e53 --save
```

This will:
- Scan all masks in the scene
- Find unique class indices
- Create `/home/dkonur/scannetpp/metadata/class_remap_{scene_id}.json`
- **Print the required `NUM_SEMANTIC_CHANNELS` value**

### 2. Update NUM_SEMANTIC_CHANNELS in config.h

Edit `submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h`:

```c
#define NUM_SEMANTIC_CHANNELS 21  // Use the number from step 1
```

**Important:** This number must match the output from step 1.

### 3. Recompile CUDA Rasterizer

```bash
cd submodules/diff-gaussian-rasterization
pip install -e .
cd ../..
```

### 4. Configure Training Script

Edit `train_joint_scannet.sh`:

```bash
SCENE_ID="f36e3e1e53"
SEG_MASKS_DIR="/home/dkonur/scannetpp/semantic_2d_output/${SCENE_ID}"
SEG_MASKS_SUBDIR="${SCENE_ID}_4x"
COLORED_MASKS_SUBDIR="${SCENE_ID}_4x"
PALETTE_PATH="/home/dkonur/scannetpp/metadata/semantic_palette.txt"
SEMANTIC_REMAP_PATH="/home/dkonur/scannetpp/metadata/class_remap_${SCENE_ID}.json"
```

### 5. Run Training

```bash
bash train_joint_scannet.sh
```

## Verification

During training startup, you should see:
```
✓ Semantic palette loaded successfully. Number of classes: 2878
✓ Remap loaded: 21 classes (global <-> local)
⚠ Make sure NUM_SEMANTIC_CHANNELS = 21 in config.h
✓ Semantic remap loaded successfully.
```

## Output Files

Training will generate:
- `output/scannetpp_joint_{scene_id}/semantic_test/` - Predicted semantic maps
- `output/scannetpp_joint_{scene_id}/semantic_comparison/` - Side-by-side comparisons (RGB | Pred | GT)
- `output/scannetpp_joint_{scene_id}/test/` - RGB renders

## Notes

- Masks contain global class indices from `semantic_classes.txt` (2878 total classes)
- Each scene uses a sparse subset of these (e.g., 21 classes)
- Remapping converts global sparse indices → local consecutive indices (0, 1, ..., N-1)
- This reduces memory and makes training more efficient

