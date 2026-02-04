"""
Evaluation script for joint RGB + Semantic training.
Computes RGB metrics (SSIM, PSNR, LPIPS) and semantic metrics (mIoU, Pixel Accuracy).
Re-renders test views to compute metrics.
"""

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from utils.image_utils import psnr
import json
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
from typing import Dict

# Import 3DGS components
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from utils.scannet_dataset import ScannetppDataset
from utils.cuda_utils import move_to_device

try:
    import sys
    sys.path.append('scannetpp_tools')
    from scannetpp_tools.eval.lpips.lpips import LPIPS
    LPIPS_AVAILABLE = True
    lpips_model = None
except ImportError as e:
    print(f"Warning: Could not load LPIPS: {e}")
    LPIPS_AVAILABLE = False
    lpips_model = None

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False
    
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


def compute_miou_pixacc(pred_labels, gt_labels, num_classes):
    """
    Compute mean IoU and pixel accuracy for semantic segmentation.
    
    Args:
        pred_labels: [H, W] numpy array of predicted class indices
        gt_labels: [H, W] numpy array of ground truth class indices
        num_classes: number of classes
    
    Returns:
        miou: mean IoU across all classes
        pixel_acc: overall pixel accuracy
        per_class_iou: IoU for each class
    """
    # Flatten arrays
    pred_flat = pred_labels.flatten()
    gt_flat = gt_labels.flatten()
    
    # Compute pixel accuracy
    correct = (pred_flat == gt_flat).sum()
    total = len(pred_flat)
    pixel_acc = correct / total if total > 0 else 0.0
    
    # Compute per-class IoU
    per_class_iou = np.zeros(num_classes)
    valid_classes = np.zeros(num_classes, dtype=bool)
    
    for c in range(num_classes):
        pred_c = (pred_flat == c)
        gt_c = (gt_flat == c)
        
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
    
    return miou, pixel_acc, per_class_iou

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
        base = seg_masks_path / scene_id / seg_subdir
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
                # set unmapped indicies to 255
                remapped_mask = np.full_like(mask, 255, dtype=np.int64)
                for global_idx, local_idx in _SEMANTIC_REMAP.items():
                    remapped_mask[mask == global_idx] = local_idx
                # Unmapped indices default to 0 (first class in the remap)
                mask = remapped_mask
            
            return torch.from_numpy(mask).long()
    
    return None

def load_gt_semantic(gt_masks_dir, image_name, scene_id, seg_subdir):
    """Load ground truth semantic mask with remapping applied."""
    image_name += ".png"
    
    # Try to load GT mask (remapping is applied inside load_segmentation_mask if _SEMANTIC_REMAP is set)
    gt_mask = load_segmentation_mask(gt_masks_dir, image_name, scene_id, seg_subdir)
    if gt_mask is None:
        return None
    
    return gt_mask.cpu().numpy()


def data_to_camera(data):
    """Convert data dict to MiniCam for rendering."""
    from scene.cameras import MiniCam
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


def load_model_and_dataset(scene_dir, data_root, scene_id, image_subdir=None, 
                           mask_subdir=None, transform_file=None):
    """Load trained model and test dataset."""
    # Load model parameters
    model_parser = ArgumentParser()
    ModelParams(model_parser)
    model_args = model_parser.parse_args([])
    model_args.model_path = str(scene_dir)
    model_args.source_path = str(data_root / scene_id)
    model_args.images = "images"
    model_args.resolution = -1
    model_args.white_background = False
    model_args.data_device = "cuda"
    model_args.eval = True
    
    # Load Gaussians
    gaussians = GaussianModel(model_args.sh_degree, optimizer_type="default")
    
    # Find the latest checkpoint or iteration file
    scene_dir = scene_dir / scene_id
    ckpt_files = list(scene_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No checkpoint found in {scene_dir}")
    
    # Sort by iteration number and get the latest
    latest_ckpt = sorted(ckpt_files, key=lambda x: int(x.parent.name.split('_')[-1]))[-1]
    print(f"  Loading checkpoint: {latest_ckpt}")
    
    gaussians.load_ply(str(latest_ckpt))
    
    # Load test dataset
    test_dataset = ScannetppDataset(
        root_dir=str(data_root),
        scene_id=scene_id,
        split="test",
        preload_images=False,
        image_subdir=image_subdir,
        mask_subdir=mask_subdir,
        transform_file=transform_file,
    )
    
    return gaussians, test_dataset


def evaluate(model_paths, data_root, scene_id, gt_masks_dir=None, seg_subdir=None,
             num_classes=None, image_subdir=None, mask_subdir=None, transform_file=None,
             semantic_remap_path=None, output_json=True):
    """
    Evaluate joint RGB + Semantic training results by re-rendering test views.
    
    Args:
        model_paths: List of paths to model output directories
        data_root: Root directory of ScanNet++ dataset
        scene_id: Scene ID
        gt_masks_dir: Path to ground truth semantic masks
        seg_subdir: Subdirectory for semantic masks (e.g., 'f36e3e1e53_4x')
        num_classes: Number of semantic classes
        image_subdir: Image subdirectory
        mask_subdir: Mask subdirectory
        transform_file: Transform file name
        semantic_remap_path: Path to semantic class remap JSON file
        output_json: Whether to save results to JSON
    """
    full_dict = {}
    per_view_dict = {}
    
    # Load semantic class remapping if provided
    if semantic_remap_path:
        remap = load_semantic_remap(semantic_remap_path)
        if remap:
            print(f"✓ Loaded semantic remap: {len(remap)} classes")
            print(f"  This will be applied to GT masks during evaluation")
    
    # Initialize LPIPS model if needed
    global lpips_model
    if LPIPS_AVAILABLE and lpips_model is None:
        print("Loading LPIPS model...")
        lpips_model = LPIPS(net_type='vgg').cuda()
    
    # Pipeline params
    pipeline_parser = ArgumentParser()
    pipeline_params = PipelineParams(pipeline_parser)
    pipeline = pipeline_parser.parse_args([])
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    for scene_dir in model_paths:
        try:
            scene_dir = Path(scene_dir)
            print(f"\n{'='*60}")
            print(f"Evaluating: {scene_dir}")
            print(f"{'='*60}")
            
            full_dict[str(scene_dir)] = {}
            per_view_dict[str(scene_dir)] = {}

            # Load model and dataset
            print("\n[Loading Model]")
            gaussians, test_dataset = load_model_and_dataset(
                scene_dir, Path(data_root), scene_id,
                image_subdir, mask_subdir, transform_file
            )
            
            print(f"  Gaussians: {gaussians.get_xyz.shape[0]}")
            print(f"  Test views: {len(test_dataset)}")
            print(f"  Semantic channels: {gaussians.get_semantic_features.shape[1]}")

            # Render and evaluate
            print("\n[Rendering & Evaluating]")
            ssims = []
            psnrs = []
            lpipss = []
            mious = []
            pixel_accs = []
            per_class_ious_list = []
            image_names = []
            
            for idx, data in enumerate(tqdm(test_dataset, desc="  Progress")):
                data = move_to_device(data, "cuda")
                
                # Create camera
                viewpoint_cam = data_to_camera(data)
                image_name = data["image_name"]
                image_names.append(image_name)
                
                # Render
                with torch.no_grad():
                    render_pkg = render(
                        viewpoint_cam, gaussians, pipeline, background,
                        use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE
                    )
                    render_img = torch.clamp(render_pkg["render"], 0, 1)
                    semantic_map = render_pkg["semantic"]
                
                # GT image
                gt_img = data["image"]
                
                # Compute RGB metrics
                ssims.append(ssim(render_img.unsqueeze(0), gt_img.unsqueeze(0)))
                psnrs.append(psnr(render_img.unsqueeze(0), gt_img.unsqueeze(0)))
                if LPIPS_AVAILABLE:
                    lpipss.append(lpips_model(
                        render_img.unsqueeze(0), gt_img.unsqueeze(0)
                    ))
                
                # Compute semantic metrics
                if gt_masks_dir and num_classes:
                    # Get prediction labels
                    semantic_map_clean = torch.nan_to_num(semantic_map, nan=0.0)
                    pred_labels = torch.argmax(semantic_map_clean, dim=0).cpu().numpy()
                    
                    # Load GT
                    gt_labels = load_gt_semantic(gt_masks_dir, image_name, scene_id, seg_subdir)
     
                    
                    if gt_labels is not None:
                        # Resize if needed
                        if pred_labels.shape != gt_labels.shape:
                            from scipy.ndimage import zoom
                            scale_h = gt_labels.shape[0] / pred_labels.shape[0]
                            scale_w = gt_labels.shape[1] / pred_labels.shape[1]
                            pred_labels = zoom(pred_labels, (scale_h, scale_w), order=0)
                        
                        # Compute metrics
                        miou, pixel_acc, per_class_iou = compute_miou_pixacc(
                            pred_labels, gt_labels, num_classes
                        )
                        
                        mious.append(miou)
                        pixel_accs.append(pixel_acc)
                        per_class_ious_list.append(per_class_iou)

            # Print RGB results
            mean_ssim = torch.tensor(ssims).mean().item()
            mean_psnr = torch.tensor(psnrs).mean().item()
            
            print(f"\n[RGB Results]")
            print(f"  SSIM : {mean_ssim:.4f}")
            print(f"  PSNR : {mean_psnr:.2f}")
            if LPIPS_AVAILABLE:
                mean_lpips = torch.tensor(lpipss).mean().item()
                print(f"  LPIPS: {mean_lpips:.4f}")

            # Store RGB results
            method_name = "joint_3dgs"
            full_dict[str(scene_dir)][method_name] = {
                "RGB": {
                    "SSIM": mean_ssim,
                    "PSNR": mean_psnr,
                }
            }
            if LPIPS_AVAILABLE:
                full_dict[str(scene_dir)][method_name]["RGB"]["LPIPS"] = mean_lpips
            
            per_view_dict[str(scene_dir)][method_name] = {
                "RGB": {
                    "SSIM": {name: s.item() for s, name in zip(ssims, image_names)},
                    "PSNR": {name: p.item() for p, name in zip(psnrs, image_names)},
                }
            }
            if LPIPS_AVAILABLE:
                per_view_dict[str(scene_dir)][method_name]["RGB"]["LPIPS"] = {
                    name: lp.item() for lp, name in zip(lpipss, image_names)
                }
            
            # Print semantic results
            if len(mious) > 0:
                mean_miou = np.mean(mious)
                mean_pixel_acc = np.mean(pixel_accs)
                mean_per_class_iou = np.mean(per_class_ious_list, axis=0)
                
                print(f"\n[Semantic Results] ({len(mious)} images)")
                print(f"  mIoU      : {mean_miou:.4f}")
                print(f"  Pixel Acc : {mean_pixel_acc:.4f}")
                print(f"  Per-class IoU:")
                for c, iou in enumerate(mean_per_class_iou):
                    if iou > 0:
                        print(f"    Class {c:2d}: {iou:.4f}")
                
                # Store semantic results
                full_dict[str(scene_dir)][method_name]["Semantic"] = {
                    "mIoU": mean_miou,
                    "Pixel_Accuracy": mean_pixel_acc,
                    "Per_Class_IoU": mean_per_class_iou.tolist(),
                }
                
                per_view_dict[str(scene_dir)][method_name]["Semantic"] = {
                    "mIoU": {name: m for m, name in zip(mious, image_names)},
                    "Pixel_Accuracy": {name: p for p, name in zip(pixel_accs, image_names)},
                }
            elif gt_masks_dir:
                print(f"\n[Semantic Results] No GT masks found")

            # Save to JSON
            if output_json:
                results_path = scene_dir / "results.json"
                per_view_path = scene_dir / "per_view.json"
                
                with open(results_path, 'w') as fp:
                    json.dump(full_dict[str(scene_dir)], fp, indent=2)
                print(f"\n✓ Saved results to: {results_path}")
                
                with open(per_view_path, 'w') as fp:
                    json.dump(per_view_dict[str(scene_dir)], fp, indent=2)
                print(f"✓ Saved per-view results to: {per_view_path}")

        except Exception as e:
            print(f"\n✗ Error evaluating {scene_dir}: {e}")
            import traceback
            traceback.print_exc()

    return full_dict, per_view_dict


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Joint RGB + Semantic Evaluation")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str,
                       help="Path(s) to model output directories")
    parser.add_argument('--data_root', required=True, type=str,
                       help="Root directory of ScanNet++ dataset")
    parser.add_argument('--scene_id', required=True, type=str,
                       help="Scene ID")
    parser.add_argument('--gt_masks_dir', type=str, default=None,
                       help="Path to ground truth semantic masks directory")
    parser.add_argument('--seg_subdir', type=str, default=None,
                       help="Subdirectory for semantic masks (e.g., 'f36e3e1e53_4x')")
    parser.add_argument('--num_classes', type=int, default=None,
                       help="Number of semantic classes")
    parser.add_argument('--image_subdir', type=str, default=None,
                       help="Image subdirectory (e.g., 'resized_undistorted_images_4x')")
    parser.add_argument('--mask_subdir', type=str, default=None,
                       help="Mask subdirectory")
    parser.add_argument('--transform_file', type=str, default=None,
                       help="Transform file name (e.g., 'transforms_undistorted_4x.json')")
    parser.add_argument('--semantic_remap_path', type=str, default=None,
                       help="Path to semantic class remap JSON file (required for correct mIoU)")
    parser.add_argument('--no_json', action='store_true',
                       help="Don't save results to JSON files")
    
    args = parser.parse_args()
    
    evaluate(
        args.model_paths,
        data_root=args.data_root,
        scene_id=args.scene_id,
        gt_masks_dir=args.gt_masks_dir,
        seg_subdir=args.seg_subdir,
        num_classes=args.num_classes,
        image_subdir=args.image_subdir,
        mask_subdir=args.mask_subdir,
        transform_file=args.transform_file,
        semantic_remap_path=args.semantic_remap_path,
        output_json=not args.no_json
    )

