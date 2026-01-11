#!/usr/bin/env python3
"""
Evaluate joint optimization model on both RGB quality and semantic segmentation.
Computes PSNR, SSIM, L1 for RGB and mIoU, pixel accuracy for semantics.
Uses TEST cameras for proper evaluation.
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

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
from utils.loss_utils import ssim


def load_gt_mask(seg_masks_dir, image_name):
    """Load ground truth segmentation mask."""
    if seg_masks_dir is None:
        return None
    
    seg_path = Path(seg_masks_dir)
    if not seg_path.exists():
        return None
    
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


def compute_semantic_metrics(pred, gt, num_classes):
    """Compute semantic segmentation metrics for a single image."""
    # Pixel accuracy
    correct = (pred == gt).sum()
    total = pred.size
    pixel_acc = correct / total
    
    # Per-class IoU
    per_class_iou, valid_classes = compute_iou(pred, gt, num_classes)
    
    return {
        'pixel_acc': pixel_acc,
        'per_class_iou': per_class_iou,
        'valid_classes': valid_classes,
        'correct': correct,
        'total': total
    }


def main():
    parser = ArgumentParser(description="Evaluate joint optimization model (RGB + Semantics)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model directory")
    parser.add_argument("--source_path", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--iteration", type=int, required=True,
                       help="Iteration to load")
    parser.add_argument("--seg_masks", type=str, default=None,
                       help="Path to ground truth segmentation masks (optional)")
    parser.add_argument("--resolution", type=int, default=8,
                       help="Resolution factor for loading dataset (1=full, 2=1/2, 4=1/4, 8=1/8, -1=auto)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for detailed results (optional)")
    parser.add_argument("--use_train", action="store_true",
                       help="Use train cameras instead of test")
    parser.add_argument("--skip_train", type=int, default=8,
                       help="When using train cameras, evaluate every Nth camera (default: 8)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Joint Optimization Evaluation (RGB + Semantics)")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Iteration: {args.iteration}")
    print(f"Dataset: {args.source_path}")
    print(f"Resolution: {args.resolution} (images loaded at this resolution)")
    if args.seg_masks:
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
        eval=True,  # Enable test camera split (every 8th camera for LLFF datasets)
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
    
    # Get cameras
    test_cameras = scene.getTestCameras()
    train_cameras = scene.getTrainCameras()
    
    if len(test_cameras) == 0:
        if not args.use_train:
            print(f"\n✗ ERROR: No test cameras found in dataset!")
            print(f"   Available: {len(train_cameras)} train cameras")
            print(f"   Use --use_train to evaluate on training set instead")
            return
        else:
            print(f"\n⚠ WARNING: Using train cameras as requested (--use_train)")
    
    if args.use_train:
        # Use subset of train cameras (every Nth camera)
        cameras = train_cameras[::args.skip_train]
        print(f"\nUsing {len(cameras)} TRAIN cameras (every {args.skip_train}th camera)")
        print(f"⚠ Note: This is evaluation on training set, not true test performance")
    else:
        cameras = test_cameras
        print(f"\nUsing {len(cameras)} TEST cameras")
    
    if len(cameras) == 0:
        print("\n✗ ERROR: No cameras available for evaluation!")
        print(f"   Train cameras: {len(train_cameras)}")
        print(f"   Test cameras: {len(test_cameras)}")
        return
    
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    
    # RGB metrics accumulators
    psnr_scores = []
    ssim_scores = []
    l1_scores = []
    
    # Semantic metrics accumulators
    all_pixel_acc = []
    all_per_class_iou = []
    all_valid_classes = []
    total_correct = 0
    total_pixels = 0
    
    # Track pixels per class
    gt_pixels_per_class = np.zeros(num_classes, dtype=np.int64)
    pred_pixels_per_class = np.zeros(num_classes, dtype=np.int64)
    
    print(f"\nEvaluating {len(cameras)} images...")
    
    for idx, cam in enumerate(tqdm(cameras)):
        image_name = os.path.splitext(os.path.basename(cam.image_name))[0]
        
        # Render (images already loaded at correct resolution via --resolution)
        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipe_args, background,
                               use_trained_exp=False, separate_sh=False)
        
        # =====================================================================
        # RGB EVALUATION
        # =====================================================================
        rendered_rgb = render_pkg["render"]  # [3, H, W]
        gt_rgb = cam.original_image.cuda()   # [3, H, W]
        
        # Clamp to [0, 1]
        rendered_rgb = torch.clamp(rendered_rgb, 0.0, 1.0)
        gt_rgb = torch.clamp(gt_rgb, 0.0, 1.0)
        
        # Compute RGB metrics
        psnr_val = psnr(rendered_rgb, gt_rgb).mean().item()
        ssim_val = ssim(rendered_rgb, gt_rgb).item()
        l1_val = torch.abs(rendered_rgb - gt_rgb).mean().item()
        
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)
        l1_scores.append(l1_val)
        
        # =====================================================================
        # SEMANTIC EVALUATION
        # =====================================================================
        if args.seg_masks:
            gt_mask = load_gt_mask(args.seg_masks, image_name)
            if gt_mask is not None:
                semantic = render_pkg["semantic"]  # [C, H, W]
                pred_classes = semantic.argmax(dim=0).cpu().numpy()  # [H, W]
                H, W = pred_classes.shape
                
                # Resize GT to match prediction
                gt_resized = np.array(
                    Image.fromarray(gt_mask.astype(np.uint8)).resize((W, H), Image.NEAREST)
                )
                gt_resized = np.clip(gt_resized, 0, num_classes - 1)
                
                # Compute semantic metrics
                sem_metrics = compute_semantic_metrics(pred_classes, gt_resized, num_classes)
                
                all_pixel_acc.append(sem_metrics['pixel_acc'])
                all_per_class_iou.append(sem_metrics['per_class_iou'])
                all_valid_classes.append(sem_metrics['valid_classes'])
                total_correct += sem_metrics['correct']
                total_pixels += sem_metrics['total']
                
                # Count pixels per class
                for c in range(num_classes):
                    gt_pixels_per_class[c] += (gt_resized == c).sum()
                    pred_pixels_per_class[c] += (pred_classes == c).sum()
    
    # =========================================================================
    # PRINT RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # RGB Results
    print("\n📷 RGB Quality:")
    print(f"   PSNR:  {np.mean(psnr_scores):.2f} dB  (±{np.std(psnr_scores):.2f})")
    print(f"   SSIM:  {np.mean(ssim_scores):.4f}  (±{np.std(ssim_scores):.4f})")
    print(f"   L1:    {np.mean(l1_scores):.4f}  (±{np.std(l1_scores):.4f})")
    
    # Semantic Results
    if args.seg_masks and len(all_pixel_acc) > 0:
        print("\n🎨 Semantic Segmentation:")
        
        # Overall pixel accuracy
        overall_pixel_acc = total_correct / total_pixels
        mean_pixel_acc = np.mean(all_pixel_acc)
        
        print(f"   Pixel Accuracy (overall): {overall_pixel_acc*100:.2f}%")
        print(f"   Pixel Accuracy (mean):    {mean_pixel_acc*100:.2f}%")
        
        # Per-class IoU
        all_per_class_iou = np.array(all_per_class_iou)
        all_valid_classes = np.array(all_valid_classes)
        
        mean_per_class_iou = np.zeros(num_classes)
        class_counts = np.zeros(num_classes)
        
        for c in range(num_classes):
            valid_mask = all_valid_classes[:, c]
            if valid_mask.sum() > 0:
                mean_per_class_iou[c] = all_per_class_iou[valid_mask, c].mean()
                class_counts[c] = valid_mask.sum()
        
        # mIoU
        valid_classes_global = class_counts > 0
        mIoU = mean_per_class_iou[valid_classes_global].mean()
        
        print(f"\n   mIoU: {mIoU*100:.2f}%")
        print(f"   (computed over {valid_classes_global.sum()} valid classes)")
        
        print(f"\n   Per-Class IoU:")
        print(f"   {'Class':<8} {'IoU':<12} {'Images':<10} {'GT %':<10}")
        print(f"   {'-'*40}")
        for c in range(num_classes):
            gt_pct = gt_pixels_per_class[c] / total_pixels * 100 if total_pixels > 0 else 0
            if class_counts[c] > 0:
                print(f"   {c:<8} {mean_per_class_iou[c]*100:>6.2f}%     {int(class_counts[c]):<10} {gt_pct:>6.2f}%")
            else:
                print(f"   {c:<8} {'N/A':<12} {'0':<10} {gt_pct:>6.2f}%")
    elif args.seg_masks:
        print("\n🎨 Semantic Segmentation: No masks found")
    else:
        print("\n🎨 Semantic Segmentation: Skipped (no --seg_masks provided)")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Dataset:        {os.path.basename(args.source_path)}")
    camera_type = f"train (every {args.skip_train}th)" if args.use_train else "test"
    print(f"  Cameras:        {len(cameras)} {camera_type}")
    print(f"  Iteration:      {args.iteration}")
    print(f"  Gaussians:      {gaussians.get_xyz.shape[0]:,}")
    print(f"\n  RGB Metrics:")
    print(f"    PSNR:         {np.mean(psnr_scores):.2f} dB")
    print(f"    SSIM:         {np.mean(ssim_scores):.4f}")
    print(f"    L1:           {np.mean(l1_scores):.4f}")
    
    if args.seg_masks and len(all_pixel_acc) > 0:
        print(f"\n  Semantic Metrics:")
        print(f"    Pixel Acc:    {overall_pixel_acc*100:.2f}%")
        print(f"    mIoU:         {mIoU*100:.2f}%")
        print(f"    Classes:      {int(valid_classes_global.sum())} / {num_classes}")
    
    print("="*80)
    
    # Save detailed results
    if args.output:
        with open(args.output, 'w') as f:
            f.write("Joint Optimization Evaluation Results\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Iteration: {args.iteration}\n")
            f.write(f"Dataset: {args.source_path}\n")
            camera_type = f"train (every {args.skip_train}th)" if args.use_train else "test"
            f.write(f"Cameras: {len(cameras)} {camera_type}\n")
            f.write(f"Gaussians: {gaussians.get_xyz.shape[0]:,}\n\n")
            
            f.write("RGB Metrics:\n")
            f.write(f"  PSNR: {np.mean(psnr_scores):.2f} dB (±{np.std(psnr_scores):.2f})\n")
            f.write(f"  SSIM: {np.mean(ssim_scores):.4f} (±{np.std(ssim_scores):.4f})\n")
            f.write(f"  L1:   {np.mean(l1_scores):.4f} (±{np.std(l1_scores):.4f})\n\n")
            
            if args.seg_masks and len(all_pixel_acc) > 0:
                f.write("Semantic Metrics:\n")
                f.write(f"  Pixel Accuracy: {overall_pixel_acc*100:.2f}%\n")
                f.write(f"  mIoU: {mIoU*100:.2f}%\n")
                f.write(f"  Valid Classes: {int(valid_classes_global.sum())} / {num_classes}\n\n")
                
                f.write("Per-Class IoU:\n")
                f.write(f"{'Class':<8} {'IoU':<12} {'Images':<10} {'GT Pixels':<15} {'Pred Pixels':<15}\n")
                f.write("-"*60 + "\n")
                for c in range(num_classes):
                    if class_counts[c] > 0:
                        f.write(f"{c:<8} {mean_per_class_iou[c]*100:>6.2f}%     {int(class_counts[c]):<10} {gt_pixels_per_class[c]:<15,} {pred_pixels_per_class[c]:<15,}\n")
                    else:
                        f.write(f"{c:<8} {'N/A':<12} {'0':<10} {gt_pixels_per_class[c]:<15,} {pred_pixels_per_class[c]:<15,}\n")
        
        print(f"\n✓ Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()


