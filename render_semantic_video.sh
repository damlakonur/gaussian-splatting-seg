#!/bin/bash

# Render semantic video from joint RGB + Semantic trained model

SCENE_ID="f36e3e1e53"
DATA_ROOT="/home/dkonur/scannetpp/data"
MODEL_PATH="./output/scannetpp_joint_${SCENE_ID}_30000k_21classes/${SCENE_ID}"
OUTPUT_DIR="./output/scannetpp_joint_${SCENE_ID}_30000k_21classes/${SCENE_ID}/video"

# GT masks for comparison
GT_MASKS_DIR="/home/dkonur/scannetpp/semantic_2d_output/${SCENE_ID}"
SEG_SUBDIR="${SCENE_ID}_4x"

# Semantic remap (REQUIRED for correct GT mask loading!)
SEMANTIC_REMAP_PATH="/home/dkonur/scannetpp/metadata/class_remap_${SCENE_ID}.json"

# Semantic palette (REQUIRED for correct coloring - same as training!)
PALETTE_PATH="/home/dkonur/scannetpp/metadata/semantic_palette.txt"

# Dataset settings (should match training)
IMAGE_SUBDIR="resized_undistorted_images_4x"
MASK_SUBDIR="resized_undistorted_masks_4x"
TRANSFORM_FILE="transforms_undistorted_4x.json"

# Render test views with semantic comparison
python render_video_scannet.py \
    --model_path ${MODEL_PATH} \
    --data_root ${DATA_ROOT} \
    --scene_id ${SCENE_ID} \
    --output_dir ${OUTPUT_DIR} \
    --path_type test \
    --fps 5 \
    --render_semantic \
    --gt_masks_dir ${GT_MASKS_DIR} \
    --seg_subdir ${SEG_SUBDIR} \
    --semantic_remap_path ${SEMANTIC_REMAP_PATH} \
    --palette_path ${PALETTE_PATH} \
    --image_subdir ${IMAGE_SUBDIR} \
    --mask_subdir ${MASK_SUBDIR} \
    --transform_file ${TRANSFORM_FILE}

echo ""
echo "Rendered videos:"
echo "  RGB: ${OUTPUT_DIR}/test_rgb.mp4"
echo "  Semantic: ${OUTPUT_DIR}/test_semantic.mp4"
echo "  Comparison (RGB | Pred Sem | GT Sem): ${OUTPUT_DIR}/test_comparison.mp4"

