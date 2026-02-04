# CUDA Module Rebuild and Testing Guide

## Rebuilding the CUDA Extension

After modifying any CUDA files in `submodules/diff-gaussian-rasterization/`, you must rebuild the extension.

```bash
cd gaussian-splatting-seg/submodules/diff-gaussian-rasterization
rm -rf build
python setup.py build_ext --inplace
pip install -e .
```

## Testing Semantic Features

### Forward Pass Test

Test that semantic features are correctly rendered without running full training.

**Command:**
```bash
cd gaussian-splatting-seg
python test_semantic_forward.py \
    --source_path /path/to/dataset \
    --model_path /path/to/stage1/output \
    --iteration 5000
```

**Example:**
```bash
python test_semantic_forward.py \
    --source_path /home/mipnerf360/bicycle/ \
    --model_path /home/dkonur/sem3dgs/gaussian-splatting/output/bicycle_2_5000 \
    --iteration 5000
```

**What it tests:**
- Loads a pre-trained Stage 1 model
- Initializes semantic features
- Runs forward pass rendering
- Verifies semantic map output shape and values

**Expected output:**
```
Loading trained model at iteration 5000
Semantic features initialized with shape: torch.Size([N, 10])
Forward pass successful
Semantic map shape: torch.Size([10, H, W])
Semantic map device: cuda:0
Semantic map dtype: torch.float32
Semantic map mean: X.XXXX
Semantic map std: X.XXXX
Semantic map min: X.XXXX
Semantic map max: X.XXXX
```

