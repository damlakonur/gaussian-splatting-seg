#!/usr/bin/env python3
"""
Render semantic segmentation video from trained Stage 2 model.
Uses actual camera poses from the dataset for reliable trajectories.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import subprocess
import copy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from scene.cameras import Camera, MiniCam
from gaussian_renderer import render
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

# Color palette (same as render_semantic.py)
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


def interpolate_cameras(cameras, num_frames, loop=True):
    """
    Create interpolated camera trajectory for novel view synthesis.
    Uses SLERP for rotation and linear interpolation for translation.
    Returns MiniCam objects (simpler than full Camera).
    """
    n_cams = len(cameras)
    
    # Extract camera parameters
    quaternions = []
    translations = []
    
    for cam in cameras:
        R = cam.R
        rot = Rotation.from_matrix(R)
        quaternions.append(rot.as_quat())
        translations.append(cam.T)
    
    # Create key times
    if loop:
        quaternions.append(quaternions[0])
        translations.append(translations[0])
        key_times = np.linspace(0, 1, n_cams + 1)
    else:
        key_times = np.linspace(0, 1, n_cams)
    
    quaternions = np.array(quaternions)
    translations = np.array(translations)
    
    trans_interp = interp1d(key_times, translations, axis=0, kind='cubic')
    
    # Reference camera for intrinsics
    ref_cam = cameras[0]
    znear, zfar = 0.01, 100.0
    
    interp_times = np.linspace(0, 1, num_frames, endpoint=not loop)
    interp_cameras = []
    
    for t in interp_times:
        idx = np.searchsorted(key_times, t, side='right') - 1
        idx = max(0, min(idx, len(key_times) - 2))
        
        t0, t1 = key_times[idx], key_times[idx + 1]
        alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0
        
        # SLERP for rotation
        key_rots = Rotation.from_quat(np.stack([quaternions[idx], quaternions[idx + 1]]))
        slerp = Slerp([0, 1], key_rots)
        R_interp = slerp(alpha).as_matrix()
        T_interp = trans_interp(t)
        
        # Compute view matrices (convert to torch tensors)
        world_view = torch.tensor(getWorld2View2(R_interp, T_interp)).float().transpose(0, 1)
        projection = getProjectionMatrix(
            znear=znear, zfar=zfar,
            fovX=ref_cam.FoVx, fovY=ref_cam.FoVy
        ).transpose(0, 1)
        full_proj = world_view @ projection
        
        # Create MiniCam (much simpler!)
        new_cam = MiniCam(
            width=ref_cam.image_width,
            height=ref_cam.image_height,
            fovy=ref_cam.FoVy,
            fovx=ref_cam.FoVx,
            znear=znear,
            zfar=zfar,
            world_view_transform=world_view.cuda(),
            full_proj_transform=full_proj.cuda()
        )
        interp_cameras.append(new_cam)
    
    return interp_cameras


def create_circle_trajectory(cameras, num_frames, height_offset=0.0):
    """
    Create a circular camera trajectory around the scene center.
    Simple horizontal orbit at fixed height.
    """
    # Find scene center from camera positions
    cam_positions = []
    for cam in cameras:
        pos = -cam.R.T @ cam.T
        cam_positions.append(pos)
    
    cam_positions = np.array(cam_positions)
    center = cam_positions.mean(axis=0)
    
    distances = np.linalg.norm(cam_positions - center, axis=1)
    avg_distance = distances.mean()
    avg_height = cam_positions[:, 1].mean()
    
    up = np.array([0, -1, 0])
    
    ref_cam = cameras[0]
    znear, zfar = 0.01, 100.0
    circle_cameras = []
    
    for i in range(num_frames):
        angle = (i / num_frames) * 2 * np.pi
        
        # Circle around scene center
        x = center[0] + avg_distance * np.cos(angle)
        y = avg_height + height_offset
        z = center[2] + avg_distance * np.sin(angle)
        cam_pos = np.array([x, y, z])
        
        # Look at center
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        up_new = np.cross(right, forward)
        
        R = np.stack([right, -up_new, -forward], axis=1)
        T = -R @ cam_pos
        
        # Compute view matrices
        world_view = torch.tensor(getWorld2View2(R, T)).float().transpose(0, 1)
        projection = getProjectionMatrix(
            znear=znear, zfar=zfar,
            fovX=ref_cam.FoVx, fovY=ref_cam.FoVy
        ).transpose(0, 1)
        full_proj = world_view @ projection
        
        new_cam = MiniCam(
            width=ref_cam.image_width,
            height=ref_cam.image_height,
            fovy=ref_cam.FoVy,
            fovx=ref_cam.FoVx,
            znear=znear,
            zfar=zfar,
            world_view_transform=world_view.cuda(),
            full_proj_transform=full_proj.cuda()
        )
        circle_cameras.append(new_cam)
    
    return circle_cameras


def create_spiral_trajectory(cameras, num_frames, n_loops=2, radius_scale=0.3):
    """
    Create a spiral camera trajectory around the scene center.
    Returns MiniCam objects.
    """
    # Find scene center from camera positions
    cam_positions = []
    for cam in cameras:
        pos = -cam.R.T @ cam.T
        cam_positions.append(pos)
    
    cam_positions = np.array(cam_positions)
    center = cam_positions.mean(axis=0)
    
    distances = np.linalg.norm(cam_positions - center, axis=1)
    avg_distance = distances.mean()
    
    up = np.array([0, -1, 0])
    
    ref_cam = cameras[0]
    znear, zfar = 0.01, 100.0
    spiral_cameras = []
    
    for i in range(num_frames):
        t = i / num_frames
        angle = t * 2 * np.pi * n_loops
        
        height = np.sin(t * np.pi) * avg_distance * 0.3
        radius = avg_distance * (1 + radius_scale * np.sin(t * np.pi * 2))
        
        x = center[0] + radius * np.cos(angle)
        y = center[1] + height
        z = center[2] + radius * np.sin(angle)
        cam_pos = np.array([x, y, z])
        
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        up_new = np.cross(right, forward)
        
        R = np.stack([right, -up_new, -forward], axis=1)
        T = -R @ cam_pos
        
        # Compute view matrices (convert to torch tensors)
        world_view = torch.tensor(getWorld2View2(R, T)).float().transpose(0, 1)
        projection = getProjectionMatrix(
            znear=znear, zfar=zfar,
            fovX=ref_cam.FoVx, fovY=ref_cam.FoVy
        ).transpose(0, 1)
        full_proj = world_view @ projection
        
        new_cam = MiniCam(
            width=ref_cam.image_width,
            height=ref_cam.image_height,
            fovy=ref_cam.FoVy,
            fovx=ref_cam.FoVx,
            znear=znear,
            zfar=zfar,
            world_view_transform=world_view.cuda(),
            full_proj_transform=full_proj.cuda()
        )
        spiral_cameras.append(new_cam)
    
    return spiral_cameras


def class_to_color(class_map, num_classes=10):
    """Convert class indices to colored image."""
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


def create_side_by_side(rgb, pred_color, gt_color=None):
    """Create side-by-side frame: [RGB | Predicted | GT] or [RGB | Predicted]."""
    H, W = pred_color.shape[:2]
    
    if gt_color is not None:
        # Triple: RGB | Pred | GT
        combined = np.concatenate([rgb, pred_color, gt_color], axis=1)
    else:
        # Double: RGB | Pred
        combined = np.concatenate([rgb, pred_color], axis=1)
    
    # Ensure dimensions are divisible by 2 (required for libx264)
    new_H = combined.shape[0] if combined.shape[0] % 2 == 0 else combined.shape[0] + 1
    new_W = combined.shape[1] if combined.shape[1] % 2 == 0 else combined.shape[1] + 1
    
    if new_H != combined.shape[0] or new_W != combined.shape[1]:
        padded = np.zeros((new_H, new_W, 3), dtype=np.uint8)
        padded[:combined.shape[0], :combined.shape[1]] = combined
        combined = padded
    
    return combined


def frames_to_video(frames_dir, output_path, fps=30):
    """Convert frames to video using ffmpeg."""
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr.decode()}")
        return False


def main():
    parser = ArgumentParser(description="Render semantic segmentation video")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to Stage 2 model directory")
    parser.add_argument("--source_path", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--iteration", type=int, required=True,
                       help="Iteration to load")
    parser.add_argument("--output", type=str, default="semantic_video.mp4",
                       help="Output video path")
    parser.add_argument("--seg_masks", type=str, default=None,
                       help="Path to GT segmentation masks (optional, for comparison)")
    parser.add_argument("--resolution", type=int, default=8,
                       help="Resolution factor (must match training)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Video FPS")
    parser.add_argument("--mode", type=str, default="side_by_side",
                       choices=["semantic", "rgb", "side_by_side"],
                       help="Video mode")
    parser.add_argument("--skip", type=int, default=1,
                       help="Use every Nth camera (1=all, 2=every other, etc)")
    parser.add_argument("--rgb_source", type=str, default="rendered",
                       choices=["rendered", "original"],
                       help="RGB source: 'rendered' from model, 'original' from dataset")
    parser.add_argument("--trajectory", type=str, default="gt",
                       choices=["gt", "interpolate", "spiral", "circle"],
                       help="Camera trajectory: 'gt' (ground truth), 'interpolate' (smooth between GT), 'spiral' (spiral path), 'circle' (horizontal orbit)")
    parser.add_argument("--num_frames", type=int, default=None,
                       help="Number of frames for interpolate/spiral (default: same as GT cameras)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Semantic Video Renderer")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Iteration: {args.iteration}")
    print(f"Output: {args.output}")
    print(f"Mode: {args.mode}")
    
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
    
    # Get base cameras (sorted by name for smooth trajectory)
    base_cameras = scene.getTrainCameras()
    base_cameras = sorted(base_cameras, key=lambda c: c.image_name)
    
    # Skip cameras if requested (for base trajectory)
    base_cameras = base_cameras[::args.skip]
    
    # Generate camera trajectory
    if args.trajectory == "gt":
        cameras = base_cameras
        print(f"Using {len(cameras)} GT cameras")
    elif args.trajectory == "interpolate":
        num_frames = args.num_frames or len(base_cameras) * 3
        cameras = interpolate_cameras(base_cameras, num_frames, loop=True)
        print(f"Using {len(cameras)} interpolated cameras (from {len(base_cameras)} GT)")
    elif args.trajectory == "spiral":
        num_frames = args.num_frames or 120
        cameras = create_spiral_trajectory(base_cameras, num_frames)
        print(f"Using {len(cameras)} spiral trajectory cameras")
    elif args.trajectory == "circle":
        num_frames = args.num_frames or 120
        cameras = create_circle_trajectory(base_cameras, num_frames)
        print(f"Using {len(cameras)} circular trajectory cameras")
    
    # For novel views, we can't use GT masks
    if args.trajectory != "gt" and args.seg_masks:
        print("⚠ Warning: GT masks only available for 'gt' trajectory, ignoring --seg_masks")
        args.seg_masks = None
    
    # Setup
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    
    # Create temp frames directory
    frames_dir = Path("_temp_frames")
    frames_dir.mkdir(exist_ok=True)
    
    # For novel views, original images aren't available
    if args.trajectory != "gt" and args.rgb_source == "original":
        print("⚠ Warning: Original images only available for 'gt' trajectory, using rendered RGB")
        args.rgb_source = "rendered"
    
    print(f"\nRendering {len(cameras)} frames...")
    print(f"RGB source: {args.rgb_source}")
    print(f"Trajectory: {args.trajectory}")
    if args.seg_masks:
        print(f"Including GT masks from: {args.seg_masks}")
    
    for i, cam in enumerate(tqdm(cameras)):
        # Get image name (MiniCam doesn't have this attribute)
        if hasattr(cam, 'image_name'):
            image_name = os.path.splitext(os.path.basename(cam.image_name))[0]
        else:
            image_name = f"frame_{i:04d}"
        
        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipe_args, background,
                               use_trained_exp=False, separate_sh=False)
        
        semantic = render_pkg["semantic"]  # [C, H, W]
        pred_classes = semantic.argmax(dim=0).cpu().numpy()
        H, W = pred_classes.shape
        
        # Get RGB - for MiniCam (novel views), always use rendered
        if args.rgb_source == "original" and hasattr(cam, 'original_image'):
            rgb_np = (cam.original_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rgb_np = np.array(Image.fromarray(rgb_np).resize((W, H), Image.BILINEAR))
        else:
            rgb = render_pkg["render"]  # [3, H, W]
            rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        pred_color = class_to_color(pred_classes, num_classes)
        
        # Load GT mask if provided (only works for GT trajectory)
        gt_color = None
        if args.seg_masks and hasattr(cam, 'image_name'):
            gt_mask = load_gt_mask(args.seg_masks, image_name)
            if gt_mask is not None:
                gt_resized = np.array(
                    Image.fromarray(gt_mask.astype(np.uint8)).resize((W, H), Image.NEAREST)
                )
                gt_resized = np.clip(gt_resized, 0, num_classes - 1)
                gt_color = class_to_color(gt_resized, num_classes)
        
        # Create frame based on mode
        if args.mode == "semantic":
            frame = pred_color
        elif args.mode == "rgb":
            frame = rgb_np
        else:  # side_by_side
            frame = create_side_by_side(rgb_np, pred_color, gt_color)
        
        # Save frame
        Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
    
    # Convert to video
    print(f"\nCreating video at {args.fps} FPS...")
    output_path = Path(args.output)
    
    if frames_to_video(frames_dir, output_path, args.fps):
        print(f"\n✓ Video saved to: {output_path}")
    else:
        print(f"\n✗ Failed to create video. Frames saved in {frames_dir}/")
    
    # Cleanup temp frames
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)
    
    print("Done!")


if __name__ == "__main__":
    main()

