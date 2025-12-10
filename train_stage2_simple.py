#!/usr/bin/env python3
"""
Stage 2: Semantic Feature Optimization
=======================================
Loads Stage 1 checkpoint and trains ONLY semantic features.
All other Gaussian parameters (position, color, opacity, scale, rotation) are FROZEN.

Usage:
    python train_stage2_simple.py \
        --stage1_model /path/to/stage1/output \
        --iteration 7000 \
        --source_path /path/to/dataset \
        --seg_masks /path/to/seg_masks_10class \
        --output output/stage2 \
        --resolution 8
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from random import randint

from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim


def load_segmentation_mask(seg_masks_dir, image_name):
    """Load segmentation mask for given image name"""
    seg_masks_path = Path(seg_masks_dir)
    
    # Try different naming patterns
    for pattern in [f"{image_name}_seg.npy", f"{image_name}.npy"]:
        mask_file = seg_masks_path / pattern
        if mask_file.exists():
            mask = np.load(mask_file)
            return torch.from_numpy(mask).long()
    
    return None


def train_stage2(args):
    """Stage 2 training: optimize ONLY semantic features"""
    
    print("\n" + "="*80)
    print("STAGE 2: SEMANTIC FEATURE OPTIMIZATION")
    print("="*80)
    print(f"Stage 1 model:  {args.stage1_model}")
    print(f"Iteration:      {args.iteration}")
    print(f"Source:         {args.source_path}")
    print(f"Seg masks:      {args.seg_masks}")
    print(f"Output:         {args.output}")
    print(f"Resolution:     {args.resolution}")
    print("="*80)
    
    # =========================================================================
    # 1. LOAD STAGE 1 MODEL
    # =========================================================================
    print(f"\n[1/4] Loading Stage 1 checkpoint...")
    
    model_params = Namespace(
        sh_degree=3,
        source_path=args.source_path,
        model_path=args.stage1_model,
        images="images",
        depths="",
        resolution=args.resolution,
        white_background=False,
        data_device="cuda",
        eval=False,
        train_test_exp=False
    )
    
    pipe_params = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False
    )
    
    gaussians = GaussianModel(model_params.sh_degree, "default")
    scene = Scene(model_params, gaussians, load_iteration=args.iteration, shuffle=False)
    
    num_gaussians = gaussians.get_xyz.shape[0]
    print(f"  ✓ Loaded {num_gaussians} Gaussians")
    print(f"  ✓ Training cameras: {len(scene.getTrainCameras())}")
    
    # =========================================================================
    # 2. FREEZE EVERYTHING EXCEPT SEMANTIC FEATURES
    # =========================================================================
    print(f"\n[2/4] Setting up optimization...")
    
    # Explicitly freeze all non-semantic parameters
    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)
    
    # Enable gradients ONLY for semantic features
    gaussians._semantic_features.requires_grad_(True)
    
    # Optimizer for semantic features only
    optimizer = torch.optim.Adam([
        {'params': gaussians._semantic_features, 'lr': args.semantic_lr}
    ])
    
    print(f"  ✓ Semantic LR: {args.semantic_lr}")
    print(f"  ✓ FROZEN: xyz, features_dc, features_rest, opacity, scaling, rotation")
    print(f"  ✓ TRAINABLE: semantic_features only")
    
    # =========================================================================
    # 3. SETUP TRAINING
    # =========================================================================
    train_cameras = scene.getTrainCameras()
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    
    # Verify seg masks exist
    sample_cam = train_cameras[0]
    sample_name = os.path.splitext(os.path.basename(sample_cam.image_name))[0]
    sample_mask = load_segmentation_mask(args.seg_masks, sample_name)
    
    if sample_mask is None:
        print(f"\n✗ ERROR: Cannot find segmentation mask!")
        print(f"  Looking for: {args.seg_masks}/{sample_name}_seg.npy or {sample_name}.npy")
        sys.exit(1)
    
    print(f"\n  Sample mask shape: {sample_mask.shape}")
    print(f"  Sample mask classes: {torch.unique(sample_mask).tolist()}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # =========================================================================
    # 4. TRAINING LOOP
    # =========================================================================
    print(f"\n[3/4] Training for {args.iterations} iterations...")
    
    progress_bar = tqdm(range(1, args.iterations + 1), desc="Training")
    losses = []
    initial_sem_feats = None  # For tracking changes
    
    for iteration in progress_bar:
        # Debug on first iteration
        if iteration == 1:
            print(f"\n  [DEBUG] BEFORE 1st render:")
            print(f"  [DEBUG] Semantic features: mean={gaussians._semantic_features.mean():.6f}, std={gaussians._semantic_features.std():.6f}")
            print(f"  [DEBUG] Semantic features: min={gaussians._semantic_features.min():.6f}, max={gaussians._semantic_features.max():.6f}")
            print(f"  [DEBUG] requires_grad={gaussians._semantic_features.requires_grad}")
        
        # Sample random camera
        viewpoint_cam = train_cameras[randint(0, len(train_cameras) - 1)]
        
        # Load segmentation mask
        image_name = os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]
        gt_semantic = load_segmentation_mask(args.seg_masks, image_name)
        
        if gt_semantic is None:
            continue
        
        gt_semantic = gt_semantic.cuda()
        
        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe_params, background,
                           use_trained_exp=False, separate_sh=False)
        
        rendered_semantic = render_pkg["semantic"]  # [C, H, W]
        
        # Match sizes (rendered vs GT mask)
        C, H_render, W_render = rendered_semantic.shape
        H_gt, W_gt = gt_semantic.shape
        
        if H_gt != H_render or W_gt != W_render:
            # Resize GT mask to match rendered
            gt_semantic = F.interpolate(
                gt_semantic.unsqueeze(0).unsqueeze(0).float(),
                size=(H_render, W_render),
                mode='nearest'
            ).squeeze().long()
        
        # Debug: Check for NaN in rendered semantic
        nan_count = torch.isnan(rendered_semantic).sum().item()
        if nan_count > 0 and iteration == 1:
            print(f"\n  [DEBUG] NaN found in rendered_semantic: {nan_count} values")
            print(f"  [DEBUG] Semantic features: min={gaussians._semantic_features.min():.4f}, max={gaussians._semantic_features.max():.4f}")
        
        # Replace NaN with 0 (background pixels with no Gaussian coverage)
        rendered_semantic = torch.nan_to_num(rendered_semantic, nan=0.0)
        
        # Get number of classes from rendered semantic
        num_classes = rendered_semantic.shape[0]
        
        # Clamp GT to valid class range [0, num_classes-1]
        gt_semantic = gt_semantic.clamp(0, num_classes - 1)
        
        # Cross-entropy loss (standard for classification)
        # CE applies softmax internally, so it works with any scale of inputs
        semantic_loss = F.cross_entropy(
            rendered_semantic.unsqueeze(0),  # [1, C, H, W]
            gt_semantic.unsqueeze(0)         # [1, H, W]
        )
        
        # Skip if loss is NaN
        if torch.isnan(semantic_loss):
            continue
        
        # Backward
        optimizer.zero_grad()
        semantic_loss.backward()
        optimizer.step()
        
        # Track loss
        loss_val = semantic_loss.item()
        losses.append(loss_val)
        
        # Debug: Check if semantic features are actually changing
        if iteration == 1:
            initial_sem_feats = gaussians._semantic_features.data.clone()
            print(f"\n  [DEBUG] Initial semantic features: mean={initial_sem_feats.mean():.4f}, std={initial_sem_feats.std():.4f}")
            print(f"  [DEBUG] Gradient norm: {gaussians._semantic_features.grad.norm().item():.6f}")
        
        if iteration == 100:
            current_sem_feats = gaussians._semantic_features.data
            diff = (current_sem_feats - initial_sem_feats).abs().mean()
            print(f"\n  [DEBUG] After 100 iters: feature change = {diff:.6f}")
            if diff < 1e-6:
                print(f"  ⚠ WARNING: Semantic features NOT changing! Check gradients.")
        
        # Update progress bar
        if iteration % 10 == 0:
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            progress_bar.set_postfix({'Loss': f'{loss_val:.4f}', 'Avg': f'{avg_loss:.4f}'})
        
        # Save checkpoint
        if iteration % args.save_interval == 0:
            save_dir = os.path.join(args.output, "point_cloud", f"iteration_{iteration}")
            os.makedirs(save_dir, exist_ok=True)
            gaussians.save_ply(os.path.join(save_dir, "point_cloud.ply"))
            tqdm.write(f"  ✓ Checkpoint saved at iteration {iteration}")
        
        # Memory cleanup
        del gt_semantic, rendered_semantic
        if iteration % 100 == 0:
            torch.cuda.empty_cache()
    
    # =========================================================================
    # 5. SAVE FINAL MODEL
    # =========================================================================
    print(f"\n[4/4] Saving final model...")
    save_dir = os.path.join(args.output, "point_cloud", f"iteration_{args.iterations}")
    os.makedirs(save_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(save_dir, "point_cloud.ply"))
    
    # Print summary
    print("\n" + "="*80)
    print("✓ STAGE 2 TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTraining summary:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Best loss:    {min(losses):.4f}")
    print(f"\nModel saved to: {save_dir}/point_cloud.ply")


if __name__ == "__main__":
    parser = ArgumentParser(description="Stage 2: Semantic Feature Optimization")
    
    # Required
    parser.add_argument("--stage1_model", type=str, required=True,
                       help="Path to Stage 1 model directory")
    parser.add_argument("--iteration", type=int, required=True,
                       help="Stage 1 iteration to load")
    parser.add_argument("--source_path", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--seg_masks", type=str, required=True,
                       help="Path to segmentation masks (.npy files)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for Stage 2 model")
    
    # Optional
    parser.add_argument("--iterations", type=int, default=5000,
                       help="Number of training iterations (default: 5000)")
    parser.add_argument("--semantic_lr", type=float, default=0.001,
                       help="Learning rate for semantic features (default: 0.001)")
    parser.add_argument("--save_interval", type=int, default=1000,
                       help="Save checkpoint every N iterations (default: 1000)")
    parser.add_argument("--resolution", type=int, default=-1,
                       help="Resolution factor (must match Stage 1, e.g., 8)")
    
    args = parser.parse_args()
    
    # Validate Stage 1 checkpoint exists
    ply_path = os.path.join(args.stage1_model, "point_cloud", 
                            f"iteration_{args.iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"✗ ERROR: Stage 1 checkpoint not found: {ply_path}")
        print(f"\nAvailable iterations:")
        pc_dir = os.path.join(args.stage1_model, "point_cloud")
        if os.path.exists(pc_dir):
            for item in sorted(os.listdir(pc_dir)):
                if item.startswith("iteration_"):
                    print(f"  {item}")
        sys.exit(1)
    
    # Validate seg masks directory exists
    if not os.path.exists(args.seg_masks):
        print(f"✗ ERROR: Segmentation masks not found: {args.seg_masks}")
        sys.exit(1)
    
    train_stage2(args)
