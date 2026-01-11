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
# Modified for Joint Optimization of Geometry and Semantic Features

import os
import gc
import torch
import numpy as np
from pathlib import Path
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def compute_miou(pred_semantic, gt_semantic, num_classes):
    """
    Compute mean IoU for semantic segmentation - matching eval_semantic.py
    
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
    
    # Convert to numpy for easier computation (matching your eval_semantic.py)
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


def load_segmentation_mask(seg_masks_dir, image_name):
    """Load segmentation mask for given image name"""
    if seg_masks_dir is None:
        return None
    
    seg_masks_path = Path(seg_masks_dir)
    if not seg_masks_path.exists():
        return None
    
    # Try different naming patterns
    for pattern in [f"{image_name}_seg.npy", f"{image_name}.npy"]:
        mask_file = seg_masks_path / pattern
        if mask_file.exists():
            mask = np.load(mask_file)
            return torch.from_numpy(mask).long()
    
    return None


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, 
             checkpoint, debug_from, semantic_weight=1.0, seg_masks_dir=None):
    """
    Joint optimization training: optimizes both geometry/appearance AND semantic features together.
    
    Args:
        seg_masks_dir: Directory containing segmentation masks as .npy files
    """
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    
    # JOINT OPTIMIZATION MODE: all parameters trainable
    gaussians.training_setup(opt, joint_optimization=True)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt, joint_optimization=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_semantic_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        semantic_map = render_pkg["semantic"]
        depth_image = render_pkg["depth"]
        # Clear render_pkg dict immediately to remove extra references
        render_pkg.clear()

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # RGB Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Semantic Feature Loss - Cross-Entropy for classification
        # Use scalar 0.0 to avoid creating CUDA tensor when not needed
        semantic_loss = 0.0
        has_semantic_loss = False
        
        # Load semantic mask from seg_masks directory
        if seg_masks_dir is not None:
            image_name = os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]
            gt_semantic = load_segmentation_mask(seg_masks_dir, image_name)
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
                
                # Cross-entropy loss (standard for semantic segmentation)
                # CE applies softmax internally
                semantic_loss = F.cross_entropy(
                    semantic_map_clean.unsqueeze(0),  # [1, C, H, W]
                    gt_semantic.unsqueeze(0)          # [1, H, W]
                )
                
                # Explicitly delete GT tensor to help memory management
                del gt_semantic, semantic_map_clean
        
        # If no semantic loss computed, detach semantic_map to prevent autograd graph retention
        if not has_semantic_loss:
            semantic_map = semantic_map.detach()

        # Combined loss for joint optimization
        # semantic_loss can be float 0.0 or tensor depending on whether GT was available
        loss = rgb_loss + semantic_weight * semantic_loss

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((depth_image - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
            del mono_invdepth, depth_mask  # Clean up depth tensors
        else:
            Ll1depth = 0
        del depth_image  # Always clean up depth_image

        loss.backward()

        iter_end.record()
        
        # CRITICAL: Extract values BEFORE deleting tensors
        Ll1_val = Ll1.item()
        rgb_loss_val = rgb_loss.item()
        sem_loss_val = semantic_loss.item() if torch.is_tensor(semantic_loss) else semantic_loss
        total_loss_val = loss.item()
        
        # AGGRESSIVE MEMORY CLEANUP after backward pass
        # Delete all tensors holding computation graph references
        del loss, rgb_loss, Ll1, ssim_value, gt_image
        if torch.is_tensor(semantic_loss):
            del semantic_loss
        del image, semantic_map
        
        # NOTE: gc.collect() was removed here as it caused significant slowdown (4x slower)
        # Memory cleanup is handled by del statements above and periodic cache clearing below
        
        with torch.no_grad():
            # Progress bar - use pre-extracted values
            ema_loss_for_log = 0.4 * rgb_loss_val + 0.6 * ema_loss_for_log
            ema_semantic_loss_for_log = 0.4 * sem_loss_val + 0.6 * ema_semantic_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"RGB": f"{ema_loss_for_log:.{5}f}", "Sem": f"{ema_semantic_loss_for_log:.{5}f}", "Depth": f"{ema_Ll1depth_for_log:.{5}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save - pass scalar values instead of tensors
            training_report(tb_writer, iteration, Ll1_val, sem_loss_val, total_loss_val, l1_loss, 
                          iter_start.elapsed_time(iter_end), testing_iterations, scene, 
                          render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), 
                          dataset.train_test_exp, seg_masks_dir)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
            
            # Cleanup densification-related tensors after optimizer step
            del viewspace_point_tensor, visibility_filter, radii

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            # Cache clearing every 500 iterations to prevent memory fragmentation
            # (more frequent clearing slows down training significantly)
            if iteration % 500 == 0:
                gc.collect()  # Force Python garbage collection
                torch.cuda.empty_cache()
            
            # Memory logging every 500 iterations to track memory vs Gaussian count
            if iteration % 500 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                num_gaussians = gaussians.get_xyz.shape[0]
                tqdm.write(f"[MEM {iteration}] Alloc: {allocated:.2f}GB, Max: {max_allocated:.2f}GB, Reserved: {reserved:.2f}GB, Gaussians: {num_gaussians}")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1_val, sem_loss_val, total_loss_val, l1_loss, elapsed, 
                   testing_iterations, scene: Scene, renderFunc, renderArgs, train_test_exp,
                   seg_masks_dir=None):
    """
    Training report function.
    
    Args:
        Ll1_val: L1 loss value (scalar float)
        sem_loss_val: Semantic loss value (scalar float)
        total_loss_val: Total loss value (scalar float)
    """
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1_val, iteration)
        tb_writer.add_scalar('train_loss_patches/semantic_loss', sem_loss_val, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', total_loss_val, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, 
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                miou_test = 0.0
                num_with_semantics = 0
                
                for idx, viewpoint in enumerate(config['cameras']):
                    render_output = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_output["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    
                    # Evaluate semantic segmentation if masks available
                    if seg_masks_dir is not None:
                        image_name = os.path.splitext(os.path.basename(viewpoint.image_name))[0]
                        gt_semantic = load_segmentation_mask(seg_masks_dir, image_name)
                        
                        if gt_semantic is not None:
                            gt_semantic = gt_semantic.cuda()
                            semantic_map = render_output["semantic"]
                            
                            # Resize if needed
                            if gt_semantic.shape[0] != semantic_map.shape[1] or gt_semantic.shape[1] != semantic_map.shape[2]:
                                gt_semantic = F.interpolate(
                                    gt_semantic.unsqueeze(0).unsqueeze(0).float(),
                                    size=(semantic_map.shape[1], semantic_map.shape[2]),
                                    mode='nearest'
                                ).squeeze().long()
                            
                            semantic_map_clean = torch.nan_to_num(semantic_map, nan=0.0)
                            num_classes = semantic_map.shape[0]
                            gt_semantic = gt_semantic.clamp(0, num_classes - 1)
                            
                            miou, _ = compute_miou(semantic_map_clean, gt_semantic, num_classes)
                            miou_test += miou
                            num_with_semantics += 1
                            
                            # Visualize semantic predictions in tensorboard
                            if tb_writer and idx < 5:
                                pred_labels = torch.argmax(semantic_map_clean, dim=0)
                                # Normalize for visualization
                                pred_vis = (pred_labels.float() / (num_classes - 1)).unsqueeze(0)
                                gt_vis = (gt_semantic.float() / (num_classes - 1)).unsqueeze(0)
                                tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/semantic_pred", pred_vis[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/semantic_gt", gt_vis[None], global_step=iteration)
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                
                if num_with_semantics > 0:
                    miou_test /= num_with_semantics
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} mIoU {}".format(iteration, config['name'], l1_test, psnr_test, miou_test))
                else:
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    if num_with_semantics > 0:
                        tb_writer.add_scalar(config['name'] + '/semantic - mIoU', miou_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--semantic_weight", type=float, default=1.0, 
                       help="Weight for semantic loss in joint optimization")
    parser.add_argument("--seg_masks", type=str, default=None,
                       help="Path to segmentation masks directory (.npy files)")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, 
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, 
             args.debug_from, args.semantic_weight, args.seg_masks)

    # All done
    print("\nTraining complete.")

