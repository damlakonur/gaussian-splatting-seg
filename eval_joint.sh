#!/bin/bash

# Example evaluation script for joint RGB + Semantic training

SCENE_ID="3e8bba0176"
DATA_ROOT="/home/dkonur/scannetpp/data"
GT_MASKS_DIR="/home/dkonur/scannetpp/semantic_2d_output_100/${SCENE_ID}_4x"
SEG_SUBDIR="${SCENE_ID}_4x"
NUM_ITERATIONS=7000
NUM_CLASSES=28

# Model output directory
# MODEL_PATH="./output/scannetpp_joint_onerformer_${SCENE_ID}_${NUM_ITERATIONS}k_${NUM_CLASSES}classes"
MODEL_PATH="/home/dkonur/sem3dgs/gaussian-splatting-seg/output/scannetpp_joint_3e8bba0176_7000k_28classes_softmax"

# Image settings (should match training)
IMAGE_SUBDIR="resized_undistorted_images_4x"
MASK_SUBDIR="resized_undistorted_masks_4x"
TRANSFORM_FILE="transforms_undistorted_4x.json"

# Semantic class remap (CRITICAL for correct mIoU!)
SEMANTIC_REMAP_PATH="/home/dkonur/scannetpp/metadata/class_remap_${SCENE_ID}_100.json"

python eval_joint_scannet_remapped.py \
    --model_paths ${MODEL_PATH} \
    --data_root ${DATA_ROOT} \
    --scene_id ${SCENE_ID} \
    --gt_masks_dir ${GT_MASKS_DIR} \
    --seg_subdir ${SEG_SUBDIR} \
    --num_classes ${NUM_CLASSES} \
    --image_subdir ${IMAGE_SUBDIR} \
    --mask_subdir ${MASK_SUBDIR} \
    --transform_file ${TRANSFORM_FILE} \
    --semantic_remap_path ${SEMANTIC_REMAP_PATH}

