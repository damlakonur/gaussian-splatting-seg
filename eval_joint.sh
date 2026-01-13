#!/bin/bash

# Example evaluation script for joint RGB + Semantic training

SCENE_ID="f36e3e1e53"
DATA_ROOT="/home/dkonur/scannetpp/data"
GT_MASKS_DIR="/home/dkonur/scannetpp/semantic_2d_output/${SCENE_ID}"
SEG_SUBDIR="${SCENE_ID}_4x"
NUM_CLASSES=21

# Model output directory
MODEL_PATH="./output/scannetpp_joint_${SCENE_ID}_30000k_21classes/${SCENE_ID}"

# Image settings (should match training)
IMAGE_SUBDIR="resized_undistorted_images_4x"
MASK_SUBDIR="resized_undistorted_masks_4x"
TRANSFORM_FILE="transforms_undistorted_4x.json"

# Semantic class remap (CRITICAL for correct mIoU!)
SEMANTIC_REMAP_PATH="/home/dkonur/scannetpp/metadata/class_remap_${SCENE_ID}.json"

python eval_joint_scannet.py \
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

