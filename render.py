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

import torch
import numpy as np
import json
from PIL import Image
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

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


def class_to_color(class_map, num_classes=10):
    """Convert class indices to colored RGB image."""
    H, W = class_map.shape
    color_image = np.zeros((H, W, 3), dtype=np.uint8)
    
    for class_id in range(min(num_classes, len(SEMANTIC_COLORS))):
        mask = class_map == class_id
        color_image[mask] = SEMANTIC_COLORS[class_id]
    
    return color_image


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, render_semantic=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    # Create semantic output directories if needed
    if render_semantic:
        semantic_path = os.path.join(model_path, name, "ours_{}".format(iteration), "semantic")
        makedirs(semantic_path, exist_ok=True)
        num_classes = gaussians.get_semantic_features.shape[1]
        print(f"Rendering semantic maps with {num_classes} classes")
    
    # Save mapping from render index to original camera name (for semantic evaluation)
    camera_mapping = {}

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # Save mapping for semantic evaluation
        camera_mapping['{0:05d}.png'.format(idx)] = os.path.basename(view.image_name)
        
        render_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        # Save semantic prediction as colored image
        if render_semantic:
            semantic_map = render_pkg["semantic"]  # [C, H, W]
            pred_classes = semantic_map.argmax(dim=0).cpu().numpy()  # [H, W]
            pred_color = class_to_color(pred_classes, num_classes)
            Image.fromarray(pred_color).save(os.path.join(semantic_path, '{0:05d}'.format(idx) + ".png"))
    
    # Save camera name mapping (for semantic evaluation)
    mapping_file = os.path.join(model_path, name, "ours_{}".format(iteration), "camera_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(camera_mapping, f, indent=2)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, render_semantic: bool = False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, render_semantic)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, render_semantic)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--semantic", action="store_true", 
                       help="Also render semantic segmentation predictions (colored)")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.semantic)