#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Joint Optimization Training for ScanNet++ Dataset
# Combines RGB reconstruction with semantic feature learning
#

from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple, List, Dict, Union, Callable
import os
import sys
import gc
import json
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from scene.cameras import Camera, MiniCam
from gaussian_renderer import render
from utils.scannet_dataset import ScannetppDataset
from utils.cuda_utils import GPUCacheLoader, move_to_device
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr, tensor2image
from utils.metrics import AverageMeter


try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False


def load_semantic_palette(palette_path: str = None) -> np.ndarray:
    """
    Load semantic color palette from file.
    
    Args:
        palette_path: Path to semantic_palette.txt (one RGB triplet per line)
    
    Returns:
        colors: [N, 3] numpy array of RGB colors
    """
    if palette_path is None or not os.path.exists(palette_path):
        # Fallback: generate random colors
        return None
    
    colors = []
    with open(palette_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                rgb = [int(x) for x in line.split()]
                if len(rgb) == 3:
                    colors.append(rgb)
    
    return np.array(colors, dtype=np.uint8)


# Global palette and remap caches
_SEMANTIC_PALETTE = None
_SEMANTIC_REMAP = None  # Maps global class indices to local consecutive indices
_SEMANTIC_REMAP_INVERSE = None  # Maps local indices back to global indices


def load_semantic_remap(remap_path: str = None) -> Dict[int, int]:
    """
    Load semantic class remapping from JSON file.
    Maps global sparse class indices to consecutive local indices (0, 1, 2, ..., N-1).
    
    Args:
        remap_path: Path to class_remap_{scene_id}.json
    
    Returns:
        remap_dict: Dictionary mapping global_index -> local_index
    """
    global _SEMANTIC_REMAP, _SEMANTIC_REMAP_INVERSE
    
    if remap_path is None or not os.path.exists(remap_path):
        return None
    
    with open(remap_path, 'r') as f:
        data = json.load(f)
    
    # Convert string keys to integers: global_index -> local_index
    remap_dict = {int(k): int(v) for k, v in data['global_to_local'].items()}
    _SEMANTIC_REMAP = remap_dict
    
    # Also load inverse mapping: local_index -> global_index (for visualization)
    inverse_dict = {int(k): int(v) for k, v in data['local_to_global'].items()}
    _SEMANTIC_REMAP_INVERSE = inverse_dict
    
    print(f"  Loaded remap: {len(remap_dict)} classes (global <-> local)")
    
    return remap_dict


def get_semantic_colormap(palette_path: str = None) -> np.ndarray:
    """
    Get semantic colormap, loading from file if provided.
    
    Args:
        palette_path: Path to semantic_palette.txt (REQUIRED)
    
    Returns:
        colors: [N, 3] numpy array of RGB colors
    
    Raises:
        ValueError: If palette_path is not provided or cannot be loaded
    """
    global _SEMANTIC_PALETTE
    
    if palette_path is not None:
        _SEMANTIC_PALETTE = load_semantic_palette(palette_path)
    
    if _SEMANTIC_PALETTE is not None:
        return _SEMANTIC_PALETTE
    
    # No fallback - require palette file
    raise ValueError(
        "Semantic palette must be loaded from file. "
        "Please provide palette_path argument pointing to semantic_palette.txt. "
        "No fallback colormap is available."
    )


def colorize_semantic_map(labels: np.ndarray, palette_path: str = None, local_to_global: bool = True) -> np.ndarray:
    """
    Convert semantic labels to RGB image using colormap from palette file.
    
    Args:
        labels: [H, W] integer class labels (local indices if using remap)
        palette_path: Path to semantic_palette.txt
        local_to_global: If True, convert local indices to global before colorizing
    
    Returns:
        colored: [H, W, 3] RGB image
    """
    colormap = get_semantic_colormap(palette_path)
    num_colors = len(colormap)
    
    # Convert local indices to global indices if remap is available
    global _SEMANTIC_REMAP_INVERSE
    if local_to_global and _SEMANTIC_REMAP_INVERSE is not None:
        # Create a lookup array for fast conversion
        max_local = max(_SEMANTIC_REMAP_INVERSE.keys()) + 1
        local_to_global_arr = np.zeros(max_local, dtype=np.int64)
        for local_idx, global_idx in _SEMANTIC_REMAP_INVERSE.items():
            local_to_global_arr[local_idx] = global_idx
        
        # Convert labels from local to global indices
        labels_clipped = np.clip(labels, 0, max_local - 1)
        labels_global = local_to_global_arr[labels_clipped]
    else:
        labels_global = labels
    # color = semantic_palette[sem_id % len(semantic_palette)]
    labels_global = labels_global % num_colors
    colored = colormap[labels_global]
    return colored


def compute_miou(pred_semantic, gt_semantic, num_classes):
    """
    Compute mean IoU for semantic segmentation.
    
    Args:
        pred_semantic: [C, H, W] logits
        gt_semantic: [H, W] integer class labels
        num_classes: number of classes
    
    Returns:
        mIoU: mean IoU across all classes
        per_class_iou: IoU for each class
    """
    # Get predictions by taking argmax
    pred_labels = torch.argmax(pred_semantic, dim=0)  # [H, W]
    
    # Convert to numpy for easier computation
    pred_np = pred_labels.cpu().numpy()
    gt_np = gt_semantic.cpu().numpy()
    
    per_class_iou = np.zeros(num_classes)
    valid_classes = np.zeros(num_classes, dtype=bool)
    
    for c in range(num_classes):
        pred_c = (pred_np == c)
        gt_c = (gt_np == c)
        
        intersection = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        
        if union > 0:
            per_class_iou[c] = intersection / union
            valid_classes[c] = True
    
    # Compute mean IoU (only over classes present in GT)
    if valid_classes.sum() > 0:
        miou = per_class_iou[valid_classes].mean()
    else:
        miou = 0.0
    
    return miou, per_class_iou


def load_segmentation_mask(seg_masks_dir: str, image_name: str, scene_id: str = None, seg_subdir: str = None):
    """
    Load segmentation mask for given image name.
    Supports image files (.png, .jpg) where pixel values are class indices.
    
    Args:
        seg_masks_dir: Root directory for segmentation masks (may already include scene_id)
        image_name: Image filename (e.g., 'DSC04151.JPG')
        scene_id: Optional scene ID (not used if seg_masks_dir already includes it)
        seg_subdir: Optional subdirectory (e.g., 'f36e3e1e53_4x')
    """
    if seg_masks_dir is None:
        return None
    
    seg_masks_path = Path(seg_masks_dir)
    if not seg_masks_path.exists():
        return None
    
    # Build search paths based on provided structure
    search_paths = []
    # Priority 1: seg_masks_dir/seg_subdir/filename (if seg_masks_dir already has scene_id)
    if seg_subdir:
        base = seg_masks_path  / seg_subdir
        search_paths.extend([
            base / f"{image_name}.png",
            base / f"{image_name}.jpg",
            base / image_name,  # Already has extension like DSC04151.JPG.png
        ])
    
    # Priority 2: Full nested structure: seg_masks_dir/scene_id/seg_subdir/filename
    if scene_id and seg_subdir:
        base = seg_masks_path / scene_id / seg_subdir
        search_paths.extend([
            base / f"{image_name}.png",
            base / f"{image_name}.jpg",
            base / image_name,
        ])
    
    # Priority 3: Partial structure: seg_masks_dir/scene_id/filename
    if scene_id:
        base = seg_masks_path / scene_id / seg_subdir
        search_paths.extend([
            base / f"{image_name}.png",
            base / f"{image_name}.jpg",
            base / image_name,
        ])
    
    # Priority 4: Flat structure: seg_masks_dir/filename
    base_name = os.path.splitext(image_name)[0]  # Remove extension like .JPG
    search_paths.extend([
        seg_masks_path  / seg_subdir / f"{base_name}.png",  # DSC04151.JPG.png
        seg_masks_path / f"{base_name}.png",   # DSC04151.png
        seg_masks_path / f"{base_name}_seg.npy",
        seg_masks_path / f"{base_name}.npy",
    ])
    
    for mask_file in search_paths:
        if mask_file.exists():
            ext = mask_file.suffix.lower()
            
            if ext == '.npy':
                mask = np.load(mask_file)
            elif ext in ['.png', '.jpg', '.jpeg']:
                mask_img = Image.open(mask_file)
                
                # Convert to grayscale if RGB
                if mask_img.mode == 'RGB' or mask_img.mode == 'RGBA':
                    mask_img = mask_img.convert('L')
                
                mask = np.array(mask_img, dtype=np.int64)
            else:
                continue
            
            # Apply semantic remapping if available (global sparse indices -> local consecutive indices)
            global _SEMANTIC_REMAP
            if _SEMANTIC_REMAP is not None:
                remapped_mask = np.zeros_like(mask, dtype=np.int64)
                for global_idx, local_idx in _SEMANTIC_REMAP.items():
                    remapped_mask[mask == global_idx] = local_idx
                # Unmapped indices default to 0 (first class in the remap)
                mask = remapped_mask
            
            return torch.from_numpy(mask).long()
    
    return None


def load_colored_semantic_mask(seg_masks_dir: str, image_name: str, scene_id: str = None, colored_subdir: str = None) -> np.ndarray:
    """
    Load pre-colored semantic mask visualization (RGB image).
    
    Args:
        seg_masks_dir: Root directory for segmentation masks (already includes scene_id)
        image_name: Image filename (e.g., 'DSC04151.JPG')
        scene_id: Optional scene ID (not used if seg_masks_dir already includes it)
        colored_subdir: Subdirectory with colored masks (e.g., 'f36e3e1e53_4x')
    
    Returns:
        RGB image as numpy array [H, W, 3] or None if not found
    """
    if seg_masks_dir is None or colored_subdir is None:
        return None
    
    seg_masks_path = Path(seg_masks_dir)
    if not seg_masks_path.exists():
        return None
    
    # Build search paths for colored visualizations
    # seg_masks_dir already includes scene_id, so just add colored_subdir
    search_paths = []
    
    if colored_subdir:
        base = seg_masks_path / colored_subdir
        search_paths.extend([
            base / f"{image_name}_viz.png",  # DSC04151.JPG_viz.png
            base / f"{image_name}_colored.png",
        ])
    
    base_name = os.path.splitext(image_name)[0]
    if colored_subdir:
        base = seg_masks_path / colored_subdir
        search_paths.extend([
            base / f"{base_name}_viz.png",  # DSC04151_viz.png
            base / f"{base_name}_colored.png",
        ])
    
    for mask_file in search_paths:
        if mask_file.exists():
            colored_img = Image.open(mask_file).convert('RGB')
            return np.array(colored_img, dtype=np.uint8)
    
    return None


def get_dataloader(
    root_dir: str,
    scene_id: str,
    preload_images: bool = True,
    preload_device: str = "cpu",
    max_images: int = -1,
    image_subdir: str = None,
    mask_subdir: str = None,
    transform_file: str = None,
) -> Tuple[DataLoader, ScannetppDataset, ScannetppDataset]:
    train_dataset = ScannetppDataset(
        root_dir=root_dir,
        scene_id=scene_id,
        split="train",
        preload_images=preload_images,
        max_images=max_images,
        image_subdir=image_subdir,
        mask_subdir=mask_subdir,
        transform_file=transform_file,
    )
    test_dataset = ScannetppDataset(
        root_dir=root_dir,
        scene_id=scene_id,
        split="test",
        preload_images=preload_images,
        image_subdir=image_subdir,
        mask_subdir=mask_subdir,
        transform_file=transform_file,
    )
    if preload_images and preload_device == "cuda":
        train_loader = GPUCacheLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            device="cuda",
            collate_fn=train_dataset.collate_fn,
            verbose=True,
        )
    else:
        num_workers = 0 if preload_images else 4
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
        )
    return train_loader, train_dataset, test_dataset


def data_to_camera(data: Dict[str, torch.Tensor]) -> MiniCam:
    return MiniCam(
        width=data["image_width"],
        height=data["image_height"],
        fovx=data["fovx"],
        fovy=data["fovy"],
        znear=data["znear"],
        zfar=data["zfar"],
        world_view_transform=data["world_view_transform"],
        full_proj_transform=data["full_proj_transform"],
        image_name=data["image_name"],
    )


def training(
    data_root: str,
    scene_id: str,
    output_path: str,
    model_params: ModelParams,
    opt_params: OptimizationParams,
    pipeline_params: PipelineParams,
    test_every: int,
    save_iterations: List[int],
    checkpoint_iterations: List[int],
    max_images: int = -1,
    image_subdir: str = None,
    mask_subdir: str = None,
    transform_file: str = None,
    semantic_weight: float = 1.0,
    seg_masks_dir: str = None,
    seg_masks_subdir: str = None,
    palette_path: str = None,
    colored_masks_subdir: str = None,
    semantic_remap_path: str = None,
    num_semantic_channels: int = 21,
    entropy_weight: float = 0.0, 
    entropy_warmup_iters: int = 0, 
    semantic_softmax: bool = False
):
    """
    Joint optimization training for ScanNet++ dataset.
    Optimizes both geometry/appearance AND semantic features together.
    
    Args:
        seg_masks_dir: Root directory for segmentation masks
        seg_masks_subdir: Subdirectory for class index masks (e.g., 'f36e3e1e53_4x')
        palette_path: Path to semantic_palette.txt for colorizing semantic maps
        colored_masks_subdir: Subdirectory for pre-colored GT visualizations (e.g., '39e6ee46df')
        semantic_remap_path: Path to class_remap_{scene_id}.json for remapping global sparse indices to local consecutive indices
    """
    train_loader, train_dataset, test_dataset = get_dataloader(
        data_root,
        scene_id,
        preload_images=False,
        preload_device=model_params.data_device,
        max_images=max_images,
        image_subdir=image_subdir,
        mask_subdir=mask_subdir,
        transform_file=transform_file,
    )

    print(f"Scene: {scene_id}")
    print(f"Number of training cameras: {len(train_dataset)}")
    print(f"Number of test cameras: {len(test_dataset)}")
    print(f"Semantic weight: {semantic_weight}")
    if seg_masks_dir:
        print(f"Segmentation masks directory: {seg_masks_dir}")
        if seg_masks_subdir:
            print(f"Segmentation masks subdirectory: {seg_masks_subdir}")
    if palette_path:
        print(f"Semantic palette path: {palette_path}")
        # Pre-load palette to ensure it's valid
        try:
            palette = get_semantic_colormap(palette_path)
            print(f"✓ Semantic palette loaded successfully. Number of classes: {len(palette)}")
        except Exception as e:
            print(f"✗ Warning: Failed to load semantic palette: {e}")
    if colored_masks_subdir:
        print(f"Pre-colored GT masks subdirectory: {colored_masks_subdir}")
    if semantic_remap_path:
        print(f"Semantic remap path: {semantic_remap_path}")
        # Pre-load remap to ensure it's valid
        try:
            remap_dict = load_semantic_remap(semantic_remap_path)
            if remap_dict:
                print(f"✓ Remap loaded: {len(remap_dict)} classes (global sparse → local consecutive)")
                print(f"  Make sure NUM_SEMANTIC_CHANNELS = {len(remap_dict)} in config.h")
            else:
                print(f"Warning: No remap loaded. Masks will use raw global indices")
        except Exception as e:
            print(f"✗ Warning: Failed to load semantic remap: {e}")

    scene_output_dir = os.path.join(output_path, scene_id)
    os.makedirs(scene_output_dir, exist_ok=True)
    writer = SummaryWriter(scene_output_dir)
    train_meter = AverageMeter()
    
    # Set model_path for Scene to use the output directory
    model_params.model_path = scene_output_dir
    
    gaussians = GaussianModel(model_params.sh_degree, opt_params.optimizer_type, num_semantic_channels)
    scene = Scene(model_params, gaussians, scannetpp_dataset=train_dataset)
    
    # JOINT OPTIMIZATION MODE: all parameters trainable including semantic features
    gaussians.training_setup(opt_params, joint_optimization=True)

    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    use_sparse_adam = opt_params.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE

    # Validate seg masks exist
    if seg_masks_dir:
        sample_data = train_dataset[0]
        sample_name = sample_data["image_name"]
        sample_mask = load_segmentation_mask(seg_masks_dir, sample_name, scene_id, seg_masks_subdir)
        if sample_mask is not None:
            print(f"✓ Segmentation masks found. Sample shape: {sample_mask.shape}")
            print(f"  Unique classes in sample: {torch.unique(sample_mask).tolist()}")
        else:
            print(f"✗ Warning: Could not find segmentation mask for sample image: {sample_name}")
            print(f"  Looked in: {seg_masks_dir}")

    progress_bar = tqdm(range(1, opt_params.iterations + 1), desc="Training progress")
    train_iter = iter(train_loader)
    
    ema_rgb_loss = 0.0
    ema_sem_loss = 0.0
    
    for iteration in range(1, opt_params.iterations + 1):
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data = next(train_iter)

        data = data[0]
        data = move_to_device(data, "cuda")

        gaussians.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        bg = torch.rand((3), device="cuda") if model_params.white_background else background

        viewpoint_cam = data_to_camera(data)
        render_pkg = render(viewpoint_cam, gaussians, pipeline_params, bg, 
                           use_trained_exp=model_params.train_test_exp, 
                           separate_sh=SPARSE_ADAM_AVAILABLE)
        
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        semantic_map = render_pkg["semantic"]
        depth_image = render_pkg["depth"]
        render_pkg.clear()

        gt_image = data["image"]
        alpha_mask = data["mask"]
        if alpha_mask is not None:
            image = image * alpha_mask
            
        # RGB Loss
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        rgb_loss = (1.0 - opt_params.lambda_dssim) * Ll1 + opt_params.lambda_dssim * (1.0 - ssim_value)

        # Semantic Feature Loss - Cross-Entropy for classification
        semantic_loss = 0.0
        has_semantic_loss = False
        
        if seg_masks_dir is not None:
            image_name = data["image_name"]
            gt_semantic = load_segmentation_mask(seg_masks_dir, image_name, scene_id, seg_masks_subdir)
            
            if gt_semantic is not None:
                has_semantic_loss = True
                gt_semantic = gt_semantic.cuda()
                
                # semantic_map: [C, H, W] - class logits
                # gt_semantic: [H, W] - integer class labels
                C, H_render, W_render = semantic_map.shape
                
                # Resize GT mask to match rendered size if needed
                if gt_semantic.shape[0] != H_render or gt_semantic.shape[1] != W_render:
                    gt_semantic = F.interpolate(
                        gt_semantic.unsqueeze(0).unsqueeze(0).float(),
                        size=(H_render, W_render),
                        mode='nearest'
                    ).squeeze().long()
                
                # Handle NaN values (pixels with no Gaussian coverage)
                semantic_map_clean = torch.nan_to_num(semantic_map, nan=0.0)
                
                # Clamp GT to valid class range
                num_classes = semantic_map.shape[0]
                gt_semantic = gt_semantic.clamp(0, num_classes - 1)
                
                # Modified Softmax logic
                if semantic_softmax:
                    # Log-Softmax + NLL Loss (Explicit Softmax)
                    log_probs = F.log_softmax(semantic_map_clean.unsqueeze(0), dim=1)
                    semantic_loss = F.nll_loss(log_probs, gt_semantic.unsqueeze(0))
                else:
                    # Standard Cross Entropy (Implicit Softmax)
                    semantic_loss = F.cross_entropy(
                        semantic_map_clean.unsqueeze(0), 
                        gt_semantic.unsqueeze(0)
                    )
                
                del gt_semantic, semantic_map_clean
        
        # If no semantic loss computed, detach semantic_map
        if not has_semantic_loss:
            semantic_map = semantic_map.detach()
        
        # Gaussian entropy regularization on per-Gaussian semantic logits
        entropy_loss = 0.0
        entropy_stats = None  # (mean, median, p10, p90) for logging

        # Always compute entropy stats if semantic features exist
        gaussian_logits = getattr(gaussians, "_semantic_features", None)
        if gaussian_logits is not None:
            # gaussian_logits: [N, C]
            probs = torch.softmax(gaussian_logits, dim=-1)
            eps = 1e-8
            entropy_per_gauss = -torch.sum(probs * torch.log(probs + eps), dim=-1)  # [N]

            # Stats over Gaussians
            entropy_mean = entropy_per_gauss.mean()
            entropy_median = entropy_per_gauss.median()
            p10 = torch.quantile(entropy_per_gauss, 0.10)
            p90 = torch.quantile(entropy_per_gauss, 0.90)
            
            # Save for logging
            entropy_stats = (
                entropy_mean.detach(),
                entropy_median.detach(),
                p10.detach(),
                p90.detach(),
            )

            # Only add entropy to loss when entropy_weight > 0
            if entropy_weight > 0.0:
                if entropy_warmup_iters > 0:
                    w_factor = min(1.0, float(iteration) / float(entropy_warmup_iters))
                else:
                    w_factor = 1.0
                curr_entropy_weight = entropy_weight * w_factor
                entropy_loss = curr_entropy_weight * entropy_mean
        else:
            entropy_loss = 0.0

        # Combined loss for joint optimization
        loss = rgb_loss + semantic_weight * semantic_loss + entropy_loss
        
        del depth_image
        loss.backward()

        # Extract values before cleanup
        Ll1_val = Ll1.item()
        rgb_loss_val = rgb_loss.item()
        sem_loss_val = semantic_loss.item() if torch.is_tensor(semantic_loss) else semantic_loss
        total_loss_val = loss.item()
        
        # Cleanup
        del loss, rgb_loss, Ll1, ssim_value, gt_image
        if torch.is_tensor(semantic_loss):
            del semantic_loss
        del image, semantic_map

        with torch.no_grad():
            psnr_value = psnr(render_pkg.get("render", torch.zeros(1)), 
                             data.get("image", torch.zeros(1))).mean() if False else 0.0
            
            # EMA for logging
            ema_rgb_loss = 0.4 * rgb_loss_val + 0.6 * ema_rgb_loss
            ema_sem_loss = 0.4 * sem_loss_val + 0.6 * ema_sem_loss
            
            train_meter.update({
                "l1_loss": Ll1_val,
                "rgb_loss": rgb_loss_val,
                "sem_loss": sem_loss_val,
                "total_loss": total_loss_val,
            })

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "RGB": f"{ema_rgb_loss:.5f}",
                    "Sem": f"{ema_sem_loss:.5f}",
                })
                progress_bar.update(10)

            if iteration % 100 == 0:
                train_metrics = train_meter.finalize()
                writer.add_scalar("Train/RGB_Loss", train_metrics["rgb_loss"], iteration)
                writer.add_scalar("Train/Semantic_Loss", train_metrics["sem_loss"], iteration)
                writer.add_scalar("Train/Total_Loss", train_metrics["total_loss"], iteration)
                writer.add_scalar("Train/L1_Loss", train_metrics["l1_loss"], iteration)
                writer.add_scalar("Train/Num_GS", gaussians.get_xyz.shape[0], iteration)
                writer.add_scalar("Train/Entropy_Loss", entropy_loss.item() if torch.is_tensor(entropy_loss) else entropy_loss, iteration)

                # Log Entropy Stats
                if entropy_stats is not None:
                    writer.add_scalar("Entropy/Mean", entropy_stats[0].item(), iteration)
                    writer.add_scalar("Entropy/Median", entropy_stats[1].item(), iteration)
                    writer.add_scalar("Entropy/P10", entropy_stats[2].item(), iteration)
                    writer.add_scalar("Entropy/P90", entropy_stats[3].item(), iteration)

                train_meter.reset()

            if iteration == opt_params.iterations:
                progress_bar.close()

            if iteration % test_every == 0 and not test_dataset.is_testing_scene:
                evaluate(
                    writer,
                    iteration,
                    test_dataset,
                    render,
                    gaussians,
                    (pipeline_params, background, 1., SPARSE_ADAM_AVAILABLE, None, False),
                    scene_output_dir,
                    seg_masks_dir,
                    scene_id,
                    seg_masks_subdir,
                    palette_path,
                    colored_masks_subdir,
                )

            if iteration in save_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians to {scene_output_dir}")
                scene.save(iteration, scene_output_dir)

            # Densification
            if iteration < opt_params.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt_params.densify_from_iter and iteration % opt_params.densification_interval == 0:
                    size_threshold = 20 if iteration > opt_params.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt_params.densify_grad_threshold, 0.005, 
                                                scene.cameras_extent, size_threshold, radii)

                if iteration % opt_params.opacity_reset_interval == 0 or \
                   (model_params.white_background and iteration == opt_params.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt_params.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            del viewspace_point_tensor, visibility_filter, radii

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    os.path.join(scene_output_dir, "chkpnt" + str(iteration) + ".pth"),
                )

            # Periodic memory cleanup
            if iteration % 500 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1024**3
                num_gaussians = gaussians.get_xyz.shape[0]
                tqdm.write(f"[MEM {iteration}] Alloc: {allocated:.2f}GB, Gaussians: {num_gaussians}")
    
    print("\nJoint training complete.")


def evaluate(
    writer: SummaryWriter,
    iteration: int,
    test_dataset: ScannetppDataset,
    render_func: Callable,
    gaussians: GaussianModel,
    render_params: Tuple,
    output_path: str,
    seg_masks_dir: str = None,
    scene_id: str = None,
    seg_masks_subdir: str = None,
    palette_path: str = None,
    colored_masks_subdir: str = None,
):
    """
    Evaluate on test set with both RGB and semantic metrics.
    
    Args:
        colored_masks_subdir: Optional subdirectory for pre-colored GT masks (e.g., '39e6ee46df')
    """
    test_meter = AverageMeter()
    render_start = torch.cuda.Event(enable_timing=True)
    render_end = torch.cuda.Event(enable_timing=True)
    
    miou_total = 0.0
    num_with_semantics = 0

    for i, data in enumerate(test_dataset):
        data = move_to_device(data, "cuda")
        gt_image = data["image"]
        alpha_mask = data["mask"]
        image_name = data["image_name"]

        viewpoint_cam = data_to_camera(data)

        render_start.record()
        render_pkg = render_func(viewpoint_cam, gaussians, *render_params)
        image = render_pkg["render"]
        semantic_map = render_pkg["semantic"]
        image = torch.clamp(image, 0, 1)
        render_end.record()
        torch.cuda.synchronize()
        render_time = render_start.elapsed_time(render_end)

        if alpha_mask is not None:
            Ll1 = l1_loss(image * alpha_mask, gt_image * alpha_mask)
            psnr_value = psnr(image * alpha_mask, gt_image * alpha_mask).mean()
        else:
            Ll1 = l1_loss(image, gt_image)
            psnr_value = psnr(image, gt_image).mean()

        # Load GT semantic mask once (if available)
        gt_semantic = None
        if seg_masks_dir is not None:
            gt_semantic = load_segmentation_mask(seg_masks_dir, image_name, scene_id, seg_masks_subdir)
            
            if gt_semantic is not None:
                gt_semantic = gt_semantic.cuda()
                
                # Resize if needed to match rendered semantic map
                if gt_semantic.shape[0] != semantic_map.shape[1] or gt_semantic.shape[1] != semantic_map.shape[2]:
                    gt_semantic = F.interpolate(
                        gt_semantic.unsqueeze(0).unsqueeze(0).float(),
                        size=(semantic_map.shape[1], semantic_map.shape[2]),
                        mode='nearest'
                    ).squeeze().long()
                
                # Compute mIoU
                semantic_map_clean = torch.nan_to_num(semantic_map, nan=0.0)
                num_classes = semantic_map.shape[0]
                gt_semantic_clamped = gt_semantic.clamp(0, num_classes - 1)
                
                miou, _ = compute_miou(semantic_map_clean, gt_semantic_clamped, num_classes)
                miou_total += miou
                num_with_semantics += 1

        # Save the testing image
        image_np = tensor2image(image, normalized=True)
        gt_image_np = tensor2image(gt_image, normalized=True)
        image_cat = np.concatenate((image_np, gt_image_np), axis=1)
        save_dir = os.path.join(output_path, "test")
        os.makedirs(save_dir, exist_ok=True)
        Image.fromarray(image_cat).save(os.path.join(save_dir, image_name))
        
        # Save semantic visualizations if palette is available
        if palette_path is not None:
            try:
                # Render semantic prediction map
                semantic_map_clean = torch.nan_to_num(semantic_map, nan=0.0)
                pred_labels = torch.argmax(semantic_map_clean, dim=0).cpu().numpy()  # [H, W]
                
                # Colorize prediction using palette
                pred_colored = colorize_semantic_map(pred_labels, palette_path)
                
                # Create semantic output directory
                semantic_save_dir = os.path.join(output_path, "semantic_test")
                os.makedirs(semantic_save_dir, exist_ok=True)
                
                # Save predicted semantic map
                base_name = os.path.splitext(image_name)[0]
                Image.fromarray(pred_colored).save(
                    os.path.join(semantic_save_dir, f"{base_name}_semantic.png")
                )
                
                # Save comparison if GT is available
                if gt_semantic is not None or colored_masks_subdir is not None:
                    # Try to load pre-colored GT first (faster and more accurate)
                    gt_colored = load_colored_semantic_mask(seg_masks_dir, image_name, scene_id, colored_masks_subdir)
                    
                    # If pre-colored GT not found, colorize the class index mask
                    if gt_colored is None and gt_semantic is not None:
                        gt_semantic_np = gt_semantic.cpu().numpy()
                        gt_colored = colorize_semantic_map(gt_semantic_np, palette_path)
                    
                    if gt_colored is not None:
                        # Resize GT to match prediction if needed
                        if gt_colored.shape[0] != pred_colored.shape[0] or gt_colored.shape[1] != pred_colored.shape[1]:
                            gt_colored_pil = Image.fromarray(gt_colored)
                            gt_colored_pil = gt_colored_pil.resize(
                                (pred_colored.shape[1], pred_colored.shape[0]), 
                                Image.NEAREST
                            )
                            gt_colored = np.array(gt_colored_pil)
                        
                        # Save side-by-side comparison: RGB | Predicted | Ground Truth
                        comparison = np.concatenate((image_np, pred_colored, gt_colored), axis=1)
                        comparison_save_dir = os.path.join(output_path, "semantic_comparison")
                        os.makedirs(comparison_save_dir, exist_ok=True)
                        Image.fromarray(comparison).save(
                            os.path.join(comparison_save_dir, f"{base_name}_comparison.png")
                        )
            except Exception as e:
                print(f"Warning: Failed to save semantic visualization for {image_name}: {e}")

        test_meter.update({
            "l1_loss": Ll1.item(),
            "psnr": psnr_value.item(),
            "render_time": render_time,
            "fps": 1000.0 / render_time,
        })

    metrics = test_meter.finalize()
    writer.add_scalar("Test/Render_Time", metrics["render_time"], iteration)
    writer.add_scalar("Test/FPS", metrics["fps"], iteration)
    writer.add_scalar("Test/Loss", metrics["l1_loss"], iteration)
    writer.add_scalar("Test/PSNR", metrics["psnr"], iteration)
    
    if num_with_semantics > 0:
        avg_miou = miou_total / num_with_semantics
        writer.add_scalar("Test/mIoU", avg_miou, iteration)
        print(f"Test iteration {iteration}: PSNR: {metrics['psnr']:.4f}, mIoU: {avg_miou:.4f}, FPS: {metrics['fps']:.2f}")
    else:
        print(f"Test iteration {iteration}: PSNR: {metrics['psnr']:.4f}, FPS: {metrics['fps']:.2f}")


def prepare_submission(
    test_dataset: ScannetppDataset,
    render_func: Callable,
    gaussians: GaussianModel,
    render_params: Tuple,
    output_path: str,
):
    """Render test images for submission."""
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    for i, data in enumerate(test_dataset):
        data = move_to_device(data, "cuda")
        image_name = data["image_name"]
        viewpoint_cam = data_to_camera(data)
        render_pkg = render_func(viewpoint_cam, gaussians, *render_params)
        image = render_pkg["render"]
        image = torch.clamp(image, 0, 1)
        image = tensor2image(image, normalized=True)

        Image.fromarray(image).save(os.path.join(output_path, image_name))
        print(f"Saved {image_name} to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Joint Training script for ScanNet++")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_every", type=int, default=1000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # ScanNet++ specific arguments
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory of ScanNet++ dataset")
    parser.add_argument("--output_root", type=str, required=True,
                       help="Output directory for training results")
    parser.add_argument("--scene_id", type=str, required=True,
                       help="ScanNet++ scene ID (e.g., 'f36e3e1e53')")
    parser.add_argument("--max_images", type=int, default=-1, 
                       help="Maximum number of training images to use. -1 uses all images.")
    parser.add_argument("--image_subdir", type=str, default=None, 
                       help="Custom image subdirectory (e.g., 'resized_undistorted_images_4x')")
    parser.add_argument("--mask_subdir", type=str, default=None, 
                       help="Custom mask subdirectory (e.g., 'resized_undistorted_masks_4x')")
    parser.add_argument("--transform_file", type=str, default=None, 
                       help="Custom transform file (e.g., 'transforms_undistorted_4x.json')")
    
    # Semantic segmentation arguments
    parser.add_argument("--semantic_weight", type=float, default=1.0,
                       help="Weight for semantic loss in joint optimization")
    parser.add_argument("--seg_masks_dir", type=str, default=None,
                       help="Root directory for segmentation masks (e.g., '/path/to/semantic_2d_output')")
    parser.add_argument("--seg_masks_subdir", type=str, default=None,
                       help="Subdirectory for class index masks within scene folder (e.g., 'f36e3e1e53_4x')")
    parser.add_argument("--palette_path", type=str, default=None,
                       help="Path to semantic_palette.txt for colorizing semantic maps (REQUIRED for semantic rendering)")
    parser.add_argument("--colored_masks_subdir", type=str, default=None,
                       help="Optional subdirectory for pre-colored GT visualizations (e.g., '39e6ee46df')")
    parser.add_argument("--semantic_remap_path", type=str, default=None,
                       help="Path to semantic_remap_{scene_id}.json for remapping segment IDs to class indices")
    parser.add_argument("--num_semantic_channels", type=int, default=21,
                       help="Number of semantic channels for semantic features")

    # Entropy and softmax arguments
    parser.add_argument("--entropy_weight", type=float, default=0.0,
                        help="Weight for Gaussian semantic entropy regularizer")
    parser.add_argument("--entropy_warmup_iters", type=int, default=0,
                        help="Iterations to linearly warm up entropy weight from 0")
    parser.add_argument("--semantic_softmax", action="store_true",
                        help="Apply LogSoftmax+NLL instead of CrossEntropy")
    
    args = parser.parse_args()
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        args.data_root,
        args.scene_id,
        args.output_root,
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_every,
        args.save_iterations,
        args.checkpoint_iterations,
        args.max_images,
        args.image_subdir,
        args.mask_subdir,
        args.transform_file,
        args.semantic_weight,
        args.seg_masks_dir,
        args.seg_masks_subdir,
        args.palette_path,
        args.colored_masks_subdir,
        args.semantic_remap_path,
        args.num_semantic_channels,
        args.entropy_weight,
        args.entropy_warmup_iters,
        args.semantic_softmax
    )

