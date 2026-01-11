#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

# Color palette for semantic visualization (10 classes)
SEMANTIC_COLORS = np.array([
    [230, 25, 75],    # 0: red
    [60, 180, 75],    # 1: green
    [255, 225, 25],   # 2: yellow
    [0, 130, 200],    # 3: blue
    [245, 130, 48],   # 4: orange
    [145, 30, 180],   # 5: purple
    [70, 240, 240],   # 6: cyan
    [240, 50, 230],   # 7: magenta
    [210, 245, 60],   # 8: lime
    [250, 190, 212],  # 9: pink
], dtype=np.uint8)

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def load_semantic_mask(seg_masks_dir, image_name):
    """Load ground truth segmentation mask for an image."""
    if seg_masks_dir is None:
        return None
    
    seg_path = Path(seg_masks_dir)
    if not seg_path.exists():
        return None
    
    # Try different naming patterns
    base_name = os.path.splitext(image_name)[0]
    for pattern in [f"{base_name}_seg.npy", f"{base_name}.npy"]:
        mask_file = seg_path / pattern
        if mask_file.exists():
            return np.load(mask_file)
    return None


def color_to_class(color_image, num_classes=10):
    """Convert colored semantic image back to class indices."""
    H, W = color_image.shape[:2]
    class_map = np.zeros((H, W), dtype=np.int32)
    
    for class_id in range(min(num_classes, len(SEMANTIC_COLORS))):
        color = SEMANTIC_COLORS[class_id]
        mask = np.all(color_image == color, axis=2)
        class_map[mask] = class_id
    
    return class_map


def compute_iou(pred, gt, num_classes):
    """Compute IoU for each class."""
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


def compute_semantic_metrics(pred_classes, gt_classes, num_classes=10):
    """Compute semantic segmentation metrics for a single image."""
    # Pixel accuracy
    correct = (pred_classes == gt_classes).sum()
    total = pred_classes.size
    pixel_acc = correct / total
    
    # Per-class IoU
    per_class_iou, valid_classes = compute_iou(pred_classes, gt_classes, num_classes)
    
    # mIoU (over classes present in this image)
    if valid_classes.sum() > 0:
        miou = per_class_iou[valid_classes].mean()
    else:
        miou = 0.0
    
    return {
        'pixel_acc': pixel_acc,
        'miou': miou,
        'per_class_iou': per_class_iou,
        'valid_classes': valid_classes
    }


def readSemanticImages(semantic_dir, gt_masks_dir, image_names, camera_mapping, num_classes=10):
    """Read rendered semantic images and corresponding GT masks.
    
    Args:
        camera_mapping: dict mapping render filenames (e.g., "00000.png") to original camera names
    """
    pred_classes_list = []
    gt_classes_list = []
    valid_names = []
    
    for fname in image_names:
        # Load rendered semantic (colored image)
        semantic_file = semantic_dir / fname
        if not semantic_file.exists():
            continue
        
        semantic_img = np.array(Image.open(semantic_file))
        pred_classes = color_to_class(semantic_img, num_classes)
        
        # Get original camera name from mapping (if available)
        if camera_mapping and fname in camera_mapping:
            original_name = camera_mapping[fname]
        else:
            # Fallback to render filename
            original_name = fname
        
        # Load GT mask using original camera name
        gt_mask = load_semantic_mask(gt_masks_dir, original_name)
        if gt_mask is None:
            continue
        
        # Resize GT to match prediction if needed
        H, W = pred_classes.shape
        if gt_mask.shape[0] != H or gt_mask.shape[1] != W:
            gt_mask = np.array(
                Image.fromarray(gt_mask.astype(np.uint8)).resize((W, H), Image.NEAREST)
            )
        
        # Clamp to valid range
        gt_mask = np.clip(gt_mask, 0, num_classes - 1)
        
        pred_classes_list.append(pred_classes)
        gt_classes_list.append(gt_mask)
        valid_names.append(fname)
    
    return pred_classes_list, gt_classes_list, valid_names

def evaluate(model_paths, seg_masks_dir=None, num_classes=10):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                # Skip files, only process directories (e.g., "ours_7000", "ours_30000")
                method_dir = test_dir / method
                if not method_dir.is_dir():
                    continue
                
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                semantic_dir = method_dir / "semantic"
                
                # Check if required directories exist
                if not renders_dir.exists() or not gt_dir.exists():
                    print(f"  Skipping {method}: missing renders or gt directory")
                    continue
                
                renders, gts, image_names = readImages(renders_dir, gt_dir)
                
                if len(renders) == 0:
                    print(f"  Skipping {method}: no images found")
                    continue

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="RGB metrics"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

                # =====================================================================
                # SEMANTIC METRICS (if semantic dir and GT masks exist)
                # =====================================================================
                if seg_masks_dir and semantic_dir.exists():
                    print("  Computing semantic metrics...")
                    
                    # Load camera name mapping (render filename -> original camera name)
                    mapping_file = method_dir / "camera_mapping.json"
                    camera_mapping = {}
                    if mapping_file.exists():
                        with open(mapping_file, 'r') as f:
                            camera_mapping = json.load(f)
                    else:
                        print("  ⚠ Warning: No camera_mapping.json found, using render filenames")
                    
                    pred_list, gt_list, valid_names = readSemanticImages(
                        semantic_dir, seg_masks_dir, image_names, camera_mapping, num_classes
                    )
                    
                    if len(pred_list) > 0:
                        all_pixel_acc = []
                        all_miou = []
                        all_per_class_iou = []
                        all_valid_classes = []
                        
                        for pred, gt in tqdm(zip(pred_list, gt_list), desc="Semantic metrics", total=len(pred_list)):
                            metrics = compute_semantic_metrics(pred, gt, num_classes)
                            all_pixel_acc.append(metrics['pixel_acc'])
                            all_miou.append(metrics['miou'])
                            all_per_class_iou.append(metrics['per_class_iou'])
                            all_valid_classes.append(metrics['valid_classes'])
                        
                        # Aggregate results
                        mean_pixel_acc = np.mean(all_pixel_acc)
                        mean_miou = np.mean(all_miou)
                        
                        # Compute global mIoU (across all images)
                        all_per_class_iou = np.array(all_per_class_iou)
                        all_valid_classes = np.array(all_valid_classes)
                        
                        global_per_class_iou = np.zeros(num_classes)
                        class_counts = np.zeros(num_classes)
                        for c in range(num_classes):
                            valid_mask = all_valid_classes[:, c]
                            if valid_mask.sum() > 0:
                                global_per_class_iou[c] = all_per_class_iou[valid_mask, c].mean()
                                class_counts[c] = valid_mask.sum()
                        
                        valid_global = class_counts > 0
                        global_miou = global_per_class_iou[valid_global].mean() if valid_global.sum() > 0 else 0.0
                        
                        print("  Pixel Acc: {:>10.2f}%".format(mean_pixel_acc * 100))
                        print("  mIoU:      {:>10.2f}%".format(global_miou * 100))
                        print("  Classes:   {:>10d} / {}".format(int(valid_global.sum()), num_classes))
                        
                        full_dict[scene_dir][method].update({
                            "Pixel_Acc": mean_pixel_acc,
                            "mIoU": global_miou,
                            "Valid_Classes": int(valid_global.sum())
                        })
                        per_view_dict[scene_dir][method].update({
                            "Pixel_Acc": {name: pa for pa, name in zip(all_pixel_acc, valid_names)},
                            "mIoU": {name: mi for mi, name in zip(all_miou, valid_names)}
                        })
                    else:
                        print("  Semantic: No valid masks found")
                elif seg_masks_dir and not semantic_dir.exists():
                    print("  Semantic: No semantic renders found (run render.py with --semantic)")
                
                print("")

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print(f"Unable to compute metrics for model {scene_dir}: {e}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation script for RGB and semantic metrics")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[],
                       help="Path(s) to model directories")
    parser.add_argument('--seg_masks', type=str, default=None,
                       help="Path to ground truth segmentation masks (enables semantic metrics)")
    parser.add_argument('--num_classes', type=int, default=10,
                       help="Number of semantic classes (default: 10)")
    args = parser.parse_args()
    evaluate(args.model_paths, args.seg_masks, args.num_classes)
