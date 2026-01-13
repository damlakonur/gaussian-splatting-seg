"""
Render script for trained 3D Gaussian Splatting models with semantic support.
Can generate novel view videos from trained scenes with RGB and semantic maps.
"""
import os
from argparse import ArgumentParser
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

from arguments import PipelineParams
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from gaussian_renderer import render
from utils.image_utils import tensor2image


def generate_circular_camera_path(cameras, n_frames=120, radius_scale=1.0):
    """Generate a circular camera path around the scene center."""
    from utils.graphics_utils import getProjectionMatrix
    
    # Get camera centers and compute scene center
    centers = np.array([cam.camera_center.cpu().numpy() for cam in cameras])
    scene_center = centers.mean(axis=0)
    
    # Compute radius from average distance
    radius = np.linalg.norm(centers - scene_center, axis=1).mean() * radius_scale
    
    # Get average height
    avg_height = centers[:, 2].mean()
    
    # Use first camera as reference for intrinsics
    ref_cam = cameras[0]
    
    novel_cams = []
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        
        # Circular path
        cam_pos = np.array([
            scene_center[0] + radius * np.cos(theta),
            scene_center[1] + radius * np.sin(theta),
            avg_height
        ], dtype=np.float32)
        
        # Look at scene center
        forward = scene_center - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Compute right and up vectors
        world_up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(world_up, forward)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(forward, right)
        
        # Build rotation matrix (world to camera)
        R = np.stack([right, up, -forward], axis=0)
        
        # Build translation
        T = -R @ cam_pos
        
        # Create world_view_transform (4x4 matrix) and transpose it
        world_view_transform = np.eye(4, dtype=np.float32)
        world_view_transform[:3, :3] = R
        world_view_transform[:3, 3] = T
        world_view_transform = torch.from_numpy(world_view_transform).float().cuda().transpose(0, 1)
        
        # Create projection matrix
        projection_matrix = getProjectionMatrix(
            znear=ref_cam.znear,
            zfar=ref_cam.zfar,
            fovX=ref_cam.FoVx,
            fovY=ref_cam.FoVy
        ).transpose(0, 1).cuda()
        
        # Compute full projection transform
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        
        # Create camera
        cam = MiniCam(
            width=ref_cam.image_width,
            height=ref_cam.image_height,
            fovx=ref_cam.FoVx,
            fovy=ref_cam.FoVy,
            znear=ref_cam.znear,
            zfar=ref_cam.zfar,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
            image_name=f"novel_{i:04d}"
        )
        novel_cams.append(cam)
    
    return novel_cams


def generate_spiral_camera_path(cameras, n_frames=120, n_loops=2):
    """Generate a spiral camera path."""
    from utils.graphics_utils import getProjectionMatrix
    
    centers = np.array([cam.camera_center.cpu().numpy() for cam in cameras])
    scene_center = centers.mean(axis=0)
    radius = np.linalg.norm(centers - scene_center, axis=1).mean()
    
    height_min = centers[:, 2].min()
    height_max = centers[:, 2].max()
    
    ref_cam = cameras[0]
    novel_cams = []
    
    for i in range(n_frames):
        t = i / n_frames
        theta = 2 * np.pi * n_loops * t
        
        # Spiral motion
        r = radius * (1.0 - 0.3 * t)  # Gradually move closer
        height = height_min + (height_max - height_min) * t
        
        cam_pos = np.array([
            scene_center[0] + r * np.cos(theta),
            scene_center[1] + r * np.sin(theta),
            height
        ], dtype=np.float32)
        
        # Look at scene center
        forward = scene_center - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        world_up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(world_up, forward)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(forward, right)
        
        R = np.stack([right, up, -forward], axis=0)
        T = -R @ cam_pos
        
        world_view_transform = np.eye(4, dtype=np.float32)
        world_view_transform[:3, :3] = R
        world_view_transform[:3, 3] = T
        world_view_transform = torch.from_numpy(world_view_transform).float().cuda().transpose(0, 1)
        
        projection_matrix = getProjectionMatrix(
            znear=ref_cam.znear,
            zfar=ref_cam.zfar,
            fovX=ref_cam.FoVx,
            fovY=ref_cam.FoVy
        ).transpose(0, 1).cuda()
        
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        
        cam = MiniCam(
            width=ref_cam.image_width,
            height=ref_cam.image_height,
            fovx=ref_cam.FoVx,
            fovy=ref_cam.FoVy,
            znear=ref_cam.znear,
            zfar=ref_cam.zfar,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
            image_name=f"spiral_{i:04d}"
        )
        novel_cams.append(cam)
    
    return novel_cams


def load_trained_gaussians(model_path, iteration=-1):
    """Load trained Gaussian model."""
    gaussians = GaussianModel(3, "default")
    
    # Find the checkpoint to load
    if iteration == -1:
        # Find the latest iteration
        point_cloud_dir = os.path.join(model_path, "point_cloud")
        if not os.path.exists(point_cloud_dir):
            raise ValueError(f"No point_cloud directory found in {model_path}")
        
        iterations = []
        for item in os.listdir(point_cloud_dir):
            if item.startswith("iteration_"):
                iter_num = int(item.split("_")[1])
                iterations.append(iter_num)
        
        if len(iterations) == 0:
            raise ValueError(f"No saved iterations found in {point_cloud_dir}")
        
        iteration = max(iterations)
    
    # Load the PLY file
    ply_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")
    print(f"Loading checkpoint from iteration {iteration}: {ply_path}")
    
    gaussians.load_ply(ply_path)
    
    # Load exposure if it exists
    exposure_path = os.path.join(model_path, "exposure.json")
    if os.path.exists(exposure_path):
        import json
        with open(exposure_path, 'r') as f:
            exposure_dict = json.load(f)
        # Optionally load exposures if needed
        # gaussians.exposure_mapping = exposure_dict
    
    return gaussians


def colorize_semantic_map_render(semantic_logits, palette_path=None):
    """
    Convert semantic logits to colored image using the same palette as training.
    
    Args:
        semantic_logits: [C, H, W] tensor of class logits
        palette_path: Path to semantic_palette.txt
    
    Returns:
        numpy array of shape [H, W, 3] with RGB values
    """
    from train_joint_scannet import colorize_semantic_map
    
    # Handle NaN values
    semantic_clean = torch.nan_to_num(semantic_logits, nan=0.0)
    
    # Get class predictions (local indices)
    pred_labels = torch.argmax(semantic_clean, dim=0).cpu().numpy()
    
    # Use the same colorize function from training (converts local→global→color)
    colored = colorize_semantic_map(pred_labels, palette_path, local_to_global=True)
    
    return colored.astype(np.uint8)


def load_gt_semantic_for_render(gt_masks_dir, image_name, scene_id, seg_subdir, 
                                 target_size=None, palette_path=None):
    """Load ground truth semantic mask for visualization and colorize it."""
    from train_joint_scannet import load_segmentation_mask, colorize_semantic_map
    
    gt_mask = load_segmentation_mask(gt_masks_dir, image_name, scene_id, seg_subdir)
    if gt_mask is None:
        return None, None
    
    gt_labels = gt_mask.cpu().numpy()
    
    # Resize if needed
    if target_size is not None:
        from scipy.ndimage import zoom
        target_h, target_w = target_size
        if gt_labels.shape[0] != target_h or gt_labels.shape[1] != target_w:
            scale_h = target_h / gt_labels.shape[0]
            scale_w = target_w / gt_labels.shape[1]
            gt_labels = zoom(gt_labels, (scale_h, scale_w), order=0)
    
    # Colorize GT using the same palette (local→global→color)
    gt_colored = colorize_semantic_map(gt_labels, palette_path, local_to_global=True)
    
    return gt_labels, gt_colored.astype(np.uint8)


def render_cameras(gaussians, cameras, output_dir, pipeline_params, background,
                   render_semantic=False, gt_masks_dir=None, scene_id=None, 
                   seg_subdir=None, palette_path=None):
    """Render a sequence of cameras with optional semantic maps."""
    rgb_dir = os.path.join(output_dir, "renders")
    os.makedirs(rgb_dir, exist_ok=True)
    
    if render_semantic:
        sem_dir = os.path.join(output_dir, "semantic")
        comparison_dir = os.path.join(output_dir, "comparison")
        os.makedirs(sem_dir, exist_ok=True)
        os.makedirs(comparison_dir, exist_ok=True)
    
    print(f"Rendering {len(cameras)} frames...")
    if render_semantic:
        print(f"  Semantic rendering enabled")
        if palette_path:
            print(f"  Using palette: {palette_path}")
        if gt_masks_dir:
            print(f"  GT masks: {gt_masks_dir}")
    
    for cam in tqdm(cameras, desc="Rendering"):
        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipeline_params, background)
            image = render_pkg["render"]
            image = torch.clamp(image, 0, 1)
            
            # Convert RGB to numpy
            image_np = tensor2image(image, normalized=True)
            
            # Save RGB image
            Image.fromarray(image_np).save(
                os.path.join(rgb_dir, f"{cam.image_name}.png")
            )
            
            # Render semantic if available and requested
            if render_semantic and "semantic" in render_pkg:
                semantic_map = render_pkg["semantic"]
                H, W = semantic_map.shape[1], semantic_map.shape[2]
                
                # Colorize predicted semantic (uses same palette as training)
                pred_colored = colorize_semantic_map_render(semantic_map, palette_path)
                
                # Save semantic image
                Image.fromarray(pred_colored).save(
                    os.path.join(sem_dir, f"{cam.image_name}.png")
                )
                
                # Create comparison image if GT available
                if gt_masks_dir:
                    gt_labels, gt_colored = load_gt_semantic_for_render(
                        gt_masks_dir, cam.image_name, scene_id, seg_subdir,
                        target_size=(H, W), palette_path=palette_path
                    )
                    
                    if gt_colored is not None:
                        # Create side-by-side comparison: RGB | Pred Semantic | GT Semantic
                        comparison = np.concatenate([image_np, pred_colored, gt_colored], axis=1)
                        Image.fromarray(comparison).save(
                            os.path.join(comparison_dir, f"{cam.image_name}.png")
                        )
    
    print(f"✓ Rendered {len(cameras)} frames to {rgb_dir}")
    if render_semantic:
        print(f"✓ Semantic maps saved to {sem_dir}")
        if gt_masks_dir:
            print(f"✓ Comparisons saved to {comparison_dir}")
    
    return rgb_dir


def create_video(image_dir, output_path, fps=30):
    """Create video from rendered images using ffmpeg."""
    import subprocess
    
    # Try different encoders in order of preference
    # yuv420p requires even dimensions, so we use scale filter
    encoder_configs = [
        ('libx264', ['-c:v', 'libx264', '-crf', '18']),
        ('libopenh264', ['-c:v', 'libopenh264', '-b:v', '5M']),
        ('mpeg4', ['-c:v', 'mpeg4', '-q:v', '3']),
    ]
    
    base_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', f'{image_dir}/*.png',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
    ]
    
    for name, encoder_opts in encoder_configs:
        cmd = base_cmd + encoder_opts + ['-pix_fmt', 'yuv420p', output_path]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ Created video using {name}: {output_path}")
            return
        except subprocess.CalledProcessError as e:
            stderr = e.stderr if e.stderr else ""
            if "Unknown encoder" in stderr or "Encoder not found" in stderr:
                print(f"{name} not available, trying next encoder...")
                continue
            else:
                print(f"Error creating video with {name}: {stderr}")
                continue
        except FileNotFoundError:
            print("Error: ffmpeg not found. Install with: sudo apt install ffmpeg")
            print("You can manually create video from images in:", image_dir)
            return
    
    print("Error: No compatible video encoder found.")
    print("You can manually create video from images in:", image_dir)


def main():
    parser = ArgumentParser(description="Render video from trained model with semantic support")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to ScanNet++ data root")
    parser.add_argument("--scene_id", type=str, required=True,
                       help="Scene ID")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: model_path/video)")
    parser.add_argument("--iteration", type=int, default=-1,
                       help="Iteration to load (default: -1 = latest)")
    
    # Dataset paths (for matching training resolution)
    parser.add_argument("--image_subdir", type=str, default=None,
                       help="Custom image subdirectory")
    parser.add_argument("--mask_subdir", type=str, default=None,
                       help="Custom mask subdirectory")
    parser.add_argument("--transform_file", type=str, default=None,
                       help="Custom transforms file")
    
    # Camera path options
    parser.add_argument("--path_type", type=str, default="circular",
                       choices=["circular", "spiral", "train", "test"],
                       help="Type of camera path")
    parser.add_argument("--n_frames", type=int, default=120,
                       help="Number of frames to render (for circular/spiral)")
    parser.add_argument("--subsample", type=int, default=1,
                       help="Subsample factor for train/test cameras (e.g., 100 = use every 100th frame)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Video framerate")
    
    # Rendering options
    parser.add_argument("--white_background", action="store_true",
                       help="Use white background instead of black")
    parser.add_argument("--no_video", action="store_true",
                       help="Don't create video, only render images")
    
    # Semantic rendering options
    parser.add_argument("--render_semantic", action="store_true",
                       help="Render semantic maps alongside RGB")
    parser.add_argument("--gt_masks_dir", type=str, default=None,
                       help="Path to GT semantic masks for comparison (e.g., /path/to/semantic_2d_output/scene_id)")
    parser.add_argument("--seg_subdir", type=str, default=None,
                       help="Subdirectory for semantic masks (e.g., 'f36e3e1e53_4x')")
    parser.add_argument("--semantic_remap_path", type=str, default=None,
                       help="Path to semantic class remap JSON (required for correct GT mask loading)")
    parser.add_argument("--palette_path", type=str, default=None,
                       help="Path to semantic_palette.txt for colorizing semantic maps (REQUIRED for semantic rendering)")
    
    # Pipeline parameters
    pp = PipelineParams(parser)
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "video")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load semantic remap if provided (for GT mask loading)
    if args.semantic_remap_path:
        from train_joint_scannet import load_semantic_remap
        remap = load_semantic_remap(args.semantic_remap_path)
        if remap:
            print(f"✓ Loaded semantic remap: {len(remap)} classes")
    
    # Load trained Gaussians
    print(f"Loading model from: {args.model_path}")
    gaussians = load_trained_gaussians(args.model_path, iteration=args.iteration)
    
    # Check if model has semantic features
    has_semantic = hasattr(gaussians, 'get_semantic_features') and gaussians.get_semantic_features is not None
    if args.render_semantic and not has_semantic:
        print("Warning: Model does not have semantic features, disabling semantic rendering")
        args.render_semantic = False
    elif has_semantic:
        num_semantic_channels = gaussians.get_semantic_features.shape[1]
        print(f"  Model has {num_semantic_channels} semantic channels")
        if args.render_semantic:
            print("  Semantic rendering enabled")
    
    # Load dataset to get camera information
    print(f"Loading dataset for camera information...")
    from utils.scannet_dataset import ScannetppDataset
    from utils.cuda_utils import move_to_device
    
    def load_cameras_from_dataset(split):
        """Helper to load cameras from dataset."""
        ds = ScannetppDataset(
            root_dir=args.data_root,
            scene_id=args.scene_id,
            split=split,
            preload_images=False,
            return_images=False,
            image_subdir=args.image_subdir,
            mask_subdir=args.mask_subdir,
            transform_file=args.transform_file
        )
        cams = []
        for i in range(len(ds)):
            data = ds[i]
            data = move_to_device(data, "cuda")
            cam = MiniCam(
                width=data["image_width"],
                height=data["image_height"],
                fovx=data["fovx"],
                fovy=data["fovy"],
                znear=data["znear"],
                zfar=data["zfar"],
                world_view_transform=data["world_view_transform"],
                full_proj_transform=data["full_proj_transform"],
                image_name=data["image_name"]
            )
            cams.append(cam)
        return cams
    
    # Load train cameras (needed for circular/spiral paths)
    train_cameras = load_cameras_from_dataset("train")
    print(f"  Loaded {len(train_cameras)} train cameras")
    
    # Generate camera path
    if args.path_type == "train":
        print(f"Using training cameras (subsample every {args.subsample})")
        cameras = train_cameras[::args.subsample]
        print(f"  Selected {len(cameras)} cameras from {len(train_cameras)} total")
    elif args.path_type == "test":
        print(f"Using test cameras (subsample every {args.subsample})")
        test_cameras = load_cameras_from_dataset("test")
        cameras = test_cameras[::args.subsample]
        print(f"  Selected {len(cameras)} cameras from {len(test_cameras)} total")
    elif args.path_type == "circular":
        print(f"Generating circular camera path with {args.n_frames} frames")
        cameras = generate_circular_camera_path(
            train_cameras, 
            n_frames=args.n_frames
        )
    elif args.path_type == "spiral":
        print(f"Generating spiral camera path with {args.n_frames} frames")
        cameras = generate_spiral_camera_path(
            train_cameras, 
            n_frames=args.n_frames
        )
    
    # Set background color
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Create path-specific output directory
    render_output_dir = os.path.join(args.output_dir, args.path_type)
    os.makedirs(render_output_dir, exist_ok=True)
    
    # Check palette requirement for semantic rendering
    if args.render_semantic and args.palette_path is None:
        print("Warning: --palette_path is required for semantic rendering. Disabling semantic.")
        args.render_semantic = False
    
    # Render cameras with optional semantic
    render_dir = render_cameras(
        gaussians,
        cameras,
        render_output_dir,
        pp.extract(args),
        background,
        render_semantic=args.render_semantic,
        gt_masks_dir=args.gt_masks_dir,
        scene_id=args.scene_id,
        seg_subdir=args.seg_subdir,
        palette_path=args.palette_path
    )
    
    # Create videos
    if not args.no_video:
        # RGB video
        video_path = os.path.join(args.output_dir, f"{args.path_type}_rgb.mp4")
        create_video(render_dir, video_path, fps=args.fps)
        
        # Semantic video (if rendered)
        if args.render_semantic:
            sem_dir = os.path.join(render_output_dir, "semantic")
            if os.path.exists(sem_dir):
                sem_video_path = os.path.join(args.output_dir, f"{args.path_type}_semantic.mp4")
                create_video(sem_dir, sem_video_path, fps=args.fps)
            
            # Comparison video (RGB | Pred Semantic | GT Semantic)
            comparison_dir = os.path.join(render_output_dir, "comparison")
            if os.path.exists(comparison_dir) and len(os.listdir(comparison_dir)) > 0:
                comparison_video_path = os.path.join(args.output_dir, f"{args.path_type}_comparison.mp4")
                create_video(comparison_dir, comparison_video_path, fps=args.fps)
    
    print("\n✓ Done!")
    print(f"  RGB renders: {render_dir}")
    if args.render_semantic:
        print(f"  Semantic renders: {os.path.join(render_output_dir, 'semantic')}")
        if args.gt_masks_dir:
            print(f"  Comparisons: {os.path.join(render_output_dir, 'comparison')}")
    if not args.no_video:
        print(f"  RGB video: {os.path.join(args.output_dir, f'{args.path_type}_rgb.mp4')}")
        if args.render_semantic:
            print(f"  Semantic video: {os.path.join(args.output_dir, f'{args.path_type}_semantic.mp4')}")
            if args.gt_masks_dir:
                print(f"  Comparison video: {os.path.join(args.output_dir, f'{args.path_type}_comparison.mp4')}")


if __name__ == "__main__":
    main()

