#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
SCENE_ID="7831862f02"
NUM_ITERATIONS=30000
NUM_SEMANTIC_CHANNELS=14
# SEG_MASKS_DIR="/home/dkonur/scannetpp/semantic_output_oneformer_remapped/"
SEG_MASKS_DIR="/home/dkonur/scannetpp/semantic_2d_output_100/"
SEG_MASKS_SUBDIR="${SCENE_ID}_4x"
COLORED_MASKS_SUBDIR="${SCENE_ID}_4x"  # Same directory as seg masks, colored ones have _viz.png suffix
OUTPUT_ROOT="./output/scannetpp_joint_onerformer_${SCENE_ID}_${NUM_ITERATIONS}k_${NUM_SEMANTIC_CHANNELS}classes_original"
DATA_ROOT="/home/dkonur/scannetpp/data"
PALETTE_PATH="/home/dkonur/scannetpp/metadata/semantic_palette_100.txt"
# SEMANTIC_REMAP_PATH="/home/dkonur/scannetpp/metadata/class_remap_${SCENE_ID}_100_oneformer_remapped.json"
SEMANTIC_REMAP_PATH="/home/dkonur/scannetpp/metadata/class_remap_${SCENE_ID}_100.json"

IMAGE_SUBDIR="resized_undistorted_images_4x"
MASK_SUBDIR="resized_undistorted_masks_4x"
TRANSFORM_FILE="transforms_undistorted_4x.json"
python train_joint_scannet.py \
    --data_root ${DATA_ROOT} \
    --scene_id ${SCENE_ID} \
    --output_root ${OUTPUT_ROOT} \
    --image_subdir ${IMAGE_SUBDIR} \
    --mask_subdir ${MASK_SUBDIR} \
    --transform_file ${TRANSFORM_FILE} \
    --seg_masks_dir ${SEG_MASKS_DIR} \
    --seg_masks_subdir ${SEG_MASKS_SUBDIR} \
    --palette_path ${PALETTE_PATH} \
    --colored_masks_subdir ${COLORED_MASKS_SUBDIR} \
    --semantic_remap_path ${SEMANTIC_REMAP_PATH} \
    --iterations ${NUM_ITERATIONS} \
    --save_iterations 7000 ${NUM_ITERATIONS} \
    --test_every 1000 \
    --num_semantic_channels ${NUM_SEMANTIC_CHANNELS}