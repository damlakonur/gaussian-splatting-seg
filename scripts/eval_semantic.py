#!/usr/bin/env python3
"""
Evaluate semantic segmentation quality.
Computes mIoU, pixel accuracy, and per-class metrics.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from gaussian_renderer import render


def load_gt_mask(seg_masks_dir, image_name):
    """Load ground truth segmentation mask."""
    seg_path = Path(seg_masks_dir)
    
    for pattern in [f"{image_name}_seg.npy", f"{image_name}.npy"]:
        mask_file = seg_path / pattern
        if mask_file.exists():
            return np.load(mask_file)
    return None


def compute_iou(pred, gt, num_classes):
    """
    Compute IoU for each class.
    
    Args:
        pred: [H, W] predicted class indices
        gt: [H, W] ground truth class indices
        num_classes: number of classes
    
    Returns:
        per_class_iou: [num_classes] IoU for each class
        valid_classes: [num_classes] bool mask for classes present in GT
    """
    per_class_iou = np.zeros(num_classes)
    valid_classes = np.zeros(num_classes, dtype=bool)
    
    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c = (gt == c)
        
        intersection = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        
        if union > 0:
            per_class_iou[c] = intersection / union
            valid_classes[c] = True
    
    return per_class_iou, valid_classes


def compute_metrics(pred, gt, num_classes):
    """
    Compute all metrics for a single image.
    
    Returns:
        dict with pixel_acc, per_class_iou, valid_classes
    """
    # Pixel accuracy
    correct = (pred == gt).sum()
    total = pred.size
    pixel_acc = correct / total
    
    # Per-class IoU
    per_class_iou, valid_classes = compute_iou(pred, gt, num_classes)
    
    # Confusion matrix for additional metrics
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for c_pred in range(num_classes):
        for c_gt in range(num_classes):
            confusion[c_pred, c_gt] = np.logical_and(pred == c_pred, gt == c_gt).sum()
    
    return {
        'pixel_acc': pixel_acc,
        'per_class_iou': per_class_iou,
        'valid_classes': valid_classes,
        'confusion': confusion,
        'correct': correct,
        'total': total
    }


def main():
    parser = ArgumentParser(description="Evaluate semantic segmentation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to Stage 2 model directory")
    parser.add_argument("--source_path", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--seg_masks", type=str, required=True,
                       help="Path to ground truth segmentation masks")
    parser.add_argument("--iteration", type=int, required=True,
                       help="Iteration to load")
    parser.add_argument("--resolution", type=int, default=8,
                       help="Resolution factor (must match training)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for detailed results (optional)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Semantic Segmentation Evaluation")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Iteration: {args.iteration}")
    print(f"GT Masks: {args.seg_masks}")
    
    # Load model
    model_args = Namespace(
        sh_degree=3,
        source_path=args.source_path,
        model_path=args.model_path,
        images="images",
        depths="",
        resolution=args.resolution,
        white_background=False,
        data_device="cuda",
        eval=False,
        train_test_exp=False
    )
    
    pipe_args = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False
    )
    
    print("\nLoading model...")
    gaussians = GaussianModel(model_args.sh_degree, "default")
    scene = Scene(model_args, gaussians, load_iteration=args.iteration, shuffle=False)
    
    num_classes = gaussians.get_semantic_features.shape[1]
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    print(f"Semantic classes: {num_classes}")
    
    # Setup
    cameras = scene.getTrainCameras()
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    
    # Accumulate metrics
    all_pixel_acc = []
    all_per_class_iou = []
    all_valid_classes = []
    total_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_correct = 0
    total_pixels = 0
    
    # Track pixels per class
    gt_pixels_per_class = np.zeros(num_classes, dtype=np.int64)
    pred_pixels_per_class = np.zeros(num_classes, dtype=np.int64)
    
    print(f"\nEvaluating {len(cameras)} images...")
    
    for cam in tqdm(cameras):
        image_name = os.path.splitext(os.path.basename(cam.image_name))[0]
        
        # Load GT mask
        gt_mask = load_gt_mask(args.seg_masks, image_name)
        if gt_mask is None:
            continue
        
        # Render
        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipe_args, background,
                               use_trained_exp=False, separate_sh=False)
        
        semantic = render_pkg["semantic"]  # [C, H, W]
        
        # Get predictions
        pred_classes = semantic.argmax(dim=0).cpu().numpy()  # [H, W]
        H, W = pred_classes.shape
        
        # Resize GT to match prediction
        gt_resized = np.array(
            Image.fromarray(gt_mask.astype(np.uint8)).resize((W, H), Image.NEAREST)
        )
        gt_resized = np.clip(gt_resized, 0, num_classes - 1)
        
        # Compute metrics
        metrics = compute_metrics(pred_classes, gt_resized, num_classes)
        
        all_pixel_acc.append(metrics['pixel_acc'])
        all_per_class_iou.append(metrics['per_class_iou'])
        all_valid_classes.append(metrics['valid_classes'])
        total_confusion += metrics['confusion']
        total_correct += metrics['correct']
        total_pixels += metrics['total']
        
        # Count pixels per class
        for c in range(num_classes):
            gt_pixels_per_class[c] += (gt_resized == c).sum()
            pred_pixels_per_class[c] += (pred_classes == c).sum()
    
    # Aggregate results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Overall pixel accuracy
    overall_pixel_acc = total_correct / total_pixels
    mean_pixel_acc = np.mean(all_pixel_acc)
    
    print(f"\n📊 Pixel Accuracy:")
    print(f"   Overall: {overall_pixel_acc*100:.2f}%")
    print(f"   Mean per image: {mean_pixel_acc*100:.2f}%")
    
    # Per-class IoU
    all_per_class_iou = np.array(all_per_class_iou)
    all_valid_classes = np.array(all_valid_classes)
    
    # Mean IoU per class (only over images where class is present)
    mean_per_class_iou = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    
    for c in range(num_classes):
        valid_mask = all_valid_classes[:, c]
        if valid_mask.sum() > 0:
            mean_per_class_iou[c] = all_per_class_iou[valid_mask, c].mean()
            class_counts[c] = valid_mask.sum()
    
    print(f"\n📊 Per-Class IoU and Pixel Counts:")
    print(f"   {'Class':<8} {'IoU':<12} {'GT Pixels':<15} {'Pred Pixels':<15} {'GT %':<10} {'Images':<8}")
    print(f"   {'-'*70}")
    for c in range(num_classes):
        gt_pct = gt_pixels_per_class[c] / total_pixels * 100
        if class_counts[c] > 0:
            print(f"   {c:<8} {mean_per_class_iou[c]*100:>6.2f}%     {gt_pixels_per_class[c]:<15,} {pred_pixels_per_class[c]:<15,} {gt_pct:>6.2f}%    {int(class_counts[c]):<8}")
        else:
            print(f"   {c:<8} {'N/A':<12} {gt_pixels_per_class[c]:<15,} {pred_pixels_per_class[c]:<15,} {gt_pct:>6.2f}%    {'0':<8}")
    
    # mIoU (mean over classes that are present)
    valid_classes_global = class_counts > 0
    mIoU = mean_per_class_iou[valid_classes_global].mean()
    
    print(f"\n📊 Mean IoU (mIoU): {mIoU*100:.2f}%")
    print(f"   (computed over {valid_classes_global.sum()} valid classes)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Pixel Accuracy: {overall_pixel_acc*100:.2f}%")
    print(f"  mIoU:           {mIoU*100:.2f}%")
    print(f"  Images:         {len(all_pixel_acc)}")
    print(f"  Classes:        {int(valid_classes_global.sum())} / {num_classes}")
    print("="*70)
    
    # Save detailed results
    if args.output:
        with open(args.output, 'w') as f:
            f.write("Semantic Segmentation Evaluation Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Iteration: {args.iteration}\n")
            f.write(f"Images evaluated: {len(all_pixel_acc)}\n")
            f.write(f"Total pixels: {total_pixels:,}\n\n")
            f.write(f"Pixel Accuracy: {overall_pixel_acc*100:.2f}%\n")
            f.write(f"mIoU: {mIoU*100:.2f}%\n\n")
            f.write("Per-Class Results:\n")
            f.write(f"{'Class':<8} {'IoU':<12} {'GT Pixels':<15} {'Pred Pixels':<15} {'GT %':<10}\n")
            f.write("-"*60 + "\n")
            for c in range(num_classes):
                gt_pct = gt_pixels_per_class[c] / total_pixels * 100
                if class_counts[c] > 0:
                    f.write(f"{c:<8} {mean_per_class_iou[c]*100:>6.2f}%     {gt_pixels_per_class[c]:<15,} {pred_pixels_per_class[c]:<15,} {gt_pct:>6.2f}%\n")
                else:
                    f.write(f"{c:<8} {'N/A':<12} {gt_pixels_per_class[c]:<15,} {pred_pixels_per_class[c]:<15,} {gt_pct:>6.2f}%\n")
        print(f"\n✓ Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()

