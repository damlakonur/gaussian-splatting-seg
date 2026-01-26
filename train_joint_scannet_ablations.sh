#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
SCENE_ID="7831862f02"
NUM_ITERATIONS=30000
NUM_SEMANTIC_CHANNELS=11
SEG_MASKS_DIR="/home/scannetpp/semantic_output_oneformer_remapped"
SEG_MASKS_SUBDIR="${SCENE_ID}_4x"
COLORED_MASKS_SUBDIR="${SCENE_ID}_4x"  # Same directory as seg masks, colored ones have _viz.png suffix
OUTPUT_ROOT="./output/scannetpp_joint_${SCENE_ID}_${NUM_ITERATIONS}k_${NUM_SEMANTIC_CHANNELS}classes"
DATA_ROOT="/home/scannetpp/data"
PALETTE_PATH="/home/scannetpp/metadata/semantic_palette_100.txt" 
SEMANTIC_REMAP_PATH="/home/scannetpp/metadata/class_remap_${SCENE_ID}_100_oneformer_remapped.json"

IMAGE_SUBDIR="resized_undistorted_images_4x"
MASK_SUBDIR="resized_undistorted_masks_4x"
TRANSFORM_FILE="transforms_undistorted_4x.json"

# Softmax
echo "------------------------------------------------"
echo "TEST 1: SOFTMAX"
echo "------------------------------------------------"
python train_joint_scannet.py \
    --data_root ${DATA_ROOT} \
    --scene_id ${SCENE_ID} \
    --output_root "./output/scannetpp_joint_${SCENE_ID}_${NUM_ITERATIONS}k_${NUM_SEMANTIC_CHANNELS}classes_softmax" \
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
    --num_semantic_channels ${NUM_SEMANTIC_CHANNELS} \
    --semantic_weight 1.0 \
    --semantic_softmax

# Entropy weight=0.01, warmup=5k
echo "------------------------------------------------"
echo "TEST 2: ENTROPY (weight=0.01, warmup=5k)"
echo "------------------------------------------------"
python train_joint_scannet.py \
    --data_root ${DATA_ROOT} \
    --scene_id ${SCENE_ID} \
    --output_root "./output/scannetpp_joint_${SCENE_ID}_${NUM_ITERATIONS}k_${NUM_SEMANTIC_CHANNELS}classes_entropy_w0.01" \
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
    --num_semantic_channels ${NUM_SEMANTIC_CHANNELS} \
    --semantic_weight 1.0 \
    --entropy_weight 0.01 \
    --entropy_warmup_iters 5000

# Entropy weight=0.05, warmup=5k
echo "------------------------------------------------"
echo "TEST 3: ENTROPY (weight=0.05, warmup=5k)"
echo "------------------------------------------------"
python train_joint_scannet.py \
    --data_root ${DATA_ROOT} \
    --scene_id ${SCENE_ID} \
    --output_root "./output/scannetpp_joint_${SCENE_ID}_${NUM_ITERATIONS}k_${NUM_SEMANTIC_CHANNELS}classes_entropy_w0.05" \
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
    --num_semantic_channels ${NUM_SEMANTIC_CHANNELS} \
    --semantic_weight 1.0 \
    --entropy_weight 0.05 \
    --entropy_warmup_iters 5000

echo "------------------------------------------------"
echo "ALL TESTS COMPLETE"
echo "------------------------------------------------"