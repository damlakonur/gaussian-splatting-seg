#!/usr/bin/env python3
"""
Render semantic segmentation from trained Stage 2 model.
Outputs colored segmentation images with GT comparison.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from gaussian_renderer import render

# Color palette for visualization (10 classes) - distinct colors
COLORS = np.array([
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


def class_to_color(class_map, num_classes=10):
    """Convert class indices to colored image.
    
    Args:
        class_map: [H, W] numpy array of class indices
        num_classes: number of classes
    
    Returns:
        [H, W, 3] numpy array (RGB)
    """
    H, W = class_map.shape
    color_image = np.zeros((H, W, 3), dtype=np.uint8)
    
    for class_id in range(min(num_classes, len(COLORS))):
        mask = class_map == class_id
        color_image[mask] = COLORS[class_id]
    
    return color_image


def load_gt_mask(seg_masks_dir, image_name):
    """Load ground truth segmentation mask."""
    seg_path = Path(seg_masks_dir)
    
    for pattern in [f"{image_name}_seg.npy", f"{image_name}.npy"]:
        mask_file = seg_path / pattern
        if mask_file.exists():
            return np.load(mask_file)
    
    return None


def create_comparison(rgb, pred_color, gt_color, image_name):
    """Create side-by-side comparison image.
    
    Layout: [RGB | Predicted | GT]
    """
    H, W = pred_color.shape[:2]
    
    # Resize RGB to match semantic size
    rgb_resized = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))
    
    # Create comparison image
    gap = 4  # gap between images
    comparison = np.ones((H + 30, W * 3 + gap * 2, 3), dtype=np.uint8) * 255
    
    # Place images
    comparison[30:30+H, 0:W] = rgb_resized
    comparison[30:30+H, W+gap:2*W+gap] = pred_color
    comparison[30:30+H, 2*W+2*gap:3*W+2*gap] = gt_color
    
    # Convert to PIL for text
    img = Image.fromarray(comparison)
    draw = ImageDraw.Draw(img)
    
    # Add labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text((W//2 - 20, 5), "RGB", fill=(0, 0, 0), font=font)
    draw.text((W + gap + W//2 - 40, 5), "Predicted", fill=(0, 0, 0), font=font)
    draw.text((2*W + 2*gap + W//2 - 50, 5), "Ground Truth", fill=(0, 0, 0), font=font)
    
    return np.array(img)


def main():
    parser = ArgumentParser(description="Render semantic segmentation with GT comparison")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to Stage 2 model directory")
    parser.add_argument("--source_path", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--seg_masks", type=str, required=True,
                       help="Path to ground truth segmentation masks")
    parser.add_argument("--iteration", type=int, required=True,
                       help="Iteration to load")
    parser.add_argument("--output", type=str, default="semantic_renders",
                       help="Output directory")
    parser.add_argument("--resolution", type=int, default=8,
                       help="Resolution factor (must match training)")
    parser.add_argument("--num_images", type=int, default=10,
                       help="Number of images to render (0 = all)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Semantic Segmentation Renderer (with GT comparison)")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Iteration: {args.iteration}")
    print(f"GT Masks: {args.seg_masks}")
    print(f"Output: {args.output}")
    
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
    
    # Create output directories
    output_dir = Path(args.output)
    comparison_dir = output_dir / "comparison"
    semantic_dir = output_dir / "semantic"
    gt_dir = output_dir / "gt"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Render
    num_images = args.num_images if args.num_images > 0 else len(cameras)
    num_images = min(num_images, len(cameras))
    
    print(f"\nRendering {num_images} images...")
    
    for i in tqdm(range(num_images)):
        cam = cameras[i]
        image_name = os.path.splitext(os.path.basename(cam.image_name))[0]
        
        # Load GT mask
        gt_mask = load_gt_mask(args.seg_masks, image_name)
        if gt_mask is None:
            continue
        
        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipe_args, background,
                               use_trained_exp=False, separate_sh=False)
        
        # Get outputs
        rgb = render_pkg["render"]  # [3, H, W]
        semantic = render_pkg["semantic"]  # [C, H, W]
        
        # Get predicted classes
        C, H_pred, W_pred = semantic.shape
        pred_classes = semantic.argmax(dim=0).cpu().numpy()  # [H, W]
        
        # Resize GT mask to match prediction size
        gt_mask_resized = np.array(
            Image.fromarray(gt_mask.astype(np.uint8)).resize(
                (W_pred, H_pred), Image.NEAREST
            )
        )
        
        # Clamp GT to valid range
        gt_mask_resized = np.clip(gt_mask_resized, 0, num_classes - 1)
        
        # Convert to colors (same palette!)
        pred_color = class_to_color(pred_classes, num_classes)
        gt_color = class_to_color(gt_mask_resized, num_classes)
        
        # RGB to numpy
        rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Create comparison
        comparison = create_comparison(rgb_np, pred_color, gt_color, image_name)
        
        # Save all
        Image.fromarray(comparison).save(comparison_dir / f"{image_name}.png")
        Image.fromarray(pred_color).save(semantic_dir / f"{image_name}.png")
        Image.fromarray(gt_color).save(gt_dir / f"{image_name}.png")
    
    print(f"\n✓ Saved {num_images} images to {output_dir}/")
    print(f"  - Comparison (RGB|Pred|GT): {comparison_dir}/")
    print(f"  - Predicted semantic: {semantic_dir}/")
    print(f"  - GT semantic: {gt_dir}/")


if __name__ == "__main__":
    main()

