#!/usr/bin/env python3
"""
Test semantic feature forward pass only (no backward, no gradients).
This verifies that the CUDA modifications work correctly.

Run: python test_semantic_forward.py --source_path /path/to/your/data
"""

import torch
import sys
import os
from argparse import ArgumentParser, Namespace

# Import from the project
from scene import Scene, GaussianModel
from gaussian_renderer import render


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_forward_pass(source_path, model_path, iteration):
    """Test the forward pass with semantic features"""
    
    print_section("TEST: Semantic Feature Forward Pass with Stage 1 Model")
    
    # Setup arguments
    model_args = Namespace(
        sh_degree=3,
        source_path=source_path,
        model_path=model_path,
        images="images",
        depths="",
        resolution=8,  # Match Stage 1 training resolution!
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
    
    # Initialize scene and Gaussians - will load PLY from iteration
    print(f"\n[1/7] Loading Stage 1 model from iteration {iteration}...")
    print(f"  Model path: {model_path}")
    print(f"  PLY file: {model_path}/point_cloud/iteration_{iteration}/point_cloud.ply")
    try:
        gaussians = GaussianModel(model_args.sh_degree, "default")
        scene = Scene(model_args, gaussians, load_iteration=iteration)
        
        print(f"✓ Stage 1 model loaded successfully")
        print(f"  Training cameras: {len(scene.getTrainCameras())}")
        print(f"  Gaussians: {gaussians.get_xyz.shape[0]} points")
        print(f"  XYZ range: [{gaussians.get_xyz.min().item():.2f}, {gaussians.get_xyz.max().item():.2f}]")
        print(f"  Opacity mean: {gaussians.get_opacity.mean().item():.4f}")
        print(f"  Scaling mean: {gaussians.get_scaling.mean().item():.4f}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check semantic features exist (should be initialized randomly if not in PLY)
    print("\n[2/6] Checking semantic feature attribute...")
    print("  (Note: Stage 1 model has no semantics - will be initialized randomly)")
    try:
        semantic_features = gaussians.get_semantic_features
        print(f"✓ Semantic features exist")
        print(f"  Shape: {semantic_features.shape}")
        print(f"  Expected: [{gaussians.get_xyz.shape[0]}, 10]")
        print(f"  Mean: {semantic_features.mean().item():.6f}")
        print(f"  Std: {semantic_features.std().item():.6f}")
        
        if semantic_features.shape[1] != 10:
            print(f"✗ ERROR: Expected 10 semantic channels, got {semantic_features.shape[1]}")
            return False
    except Exception as e:
        print(f"✗ Failed to access semantic features: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Set semantic features to known test values
    print("\n[3/6] Checking semantic features (NOT modifying - using random init)...")
    print(f"  Semantic features range: [{gaussians._semantic_features.min():.4f}, {gaussians._semantic_features.max():.4f}]")
    print(f"  Semantic features mean: {gaussians._semantic_features.mean():.4f}")
    # DON'T modify - keep random initialization to test if NaN appears
    # with torch.no_grad():
    #     gaussians._semantic_features.data.fill_(1.0)
        # print(f"✓ Set all semantic features to 1.0")
        # set even channels to 1.0 and odd channels to 0.0
        # gaussians._semantic_features.data[:, ::2] = 1.0
        # gaussians._semantic_features.data[:, 1::2] = 0.0
        # print(f"✓ Set even channels to 1.0 and odd channels to 0.0")
    
    # Get a test camera
    print("\n[4/6] Getting test camera...")
    try:
        viewpoint_cam = scene.getTrainCameras()[0]
        print(f"✓ Using camera: {viewpoint_cam.image_name}")
        print(f"  Image size: {viewpoint_cam.image_width}x{viewpoint_cam.image_height}")
    except Exception as e:
        print(f"✗ Failed to get camera: {e}")
        return False
    
    # Setup background (match training: white background)
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    
    # Run forward pass WITHOUT gradients (no backward)
    print("\n[5/6] Running forward pass (no gradients, no optimization)...")
    print("  This tests the CUDA kernels only - no loss, no GT needed")
    try:
        with torch.no_grad():  # Disable autograd - forward only!no
            render_pkg = render(
                viewpoint_cam, 
                gaussians, 
                pipe_args, 
                background,
                use_trained_exp=False,
                separate_sh=False
            )
        print("✓ Forward pass completed successfully")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check semantic output
    print("\n[6/6] Verifying semantic output...")
    try:
        if "semantic" not in render_pkg:
            print(f"✗ ERROR: 'semantic' not in render output!")
            print(f"  Available keys: {list(render_pkg.keys())}")
            return False
        
        semantic_map = render_pkg["semantic"]
        breakpoint()
        print(f"✓ Semantic output exists")
        print(f"\n  Semantic Map Properties:")
        print(f"  - Shape: {semantic_map.shape}")
        print(f"  - Expected: [10, {viewpoint_cam.image_height}, {viewpoint_cam.image_width}]")
        print(f"  - Device: {semantic_map.device}")
        print(f"  - Dtype: {semantic_map.dtype}")
        print(f"  - Mean: {semantic_map.mean().item():.6f}")
        print(f"  - Std: {semantic_map.std().item():.6f}")
        print(f"  - Min: {semantic_map.min().item():.6f}")
        print(f"  - Max: {semantic_map.max().item():.6f}")
        
        # Verify shape
        expected_shape = (10, viewpoint_cam.image_height, viewpoint_cam.image_width)
        if semantic_map.shape != expected_shape:
            print(f"✗ ERROR: Shape mismatch!")
            print(f"  Expected: {expected_shape}")
            print(f"  Got: {semantic_map.shape}")
            return False
        
        # Check per-channel statistics
        print(f"\n  Per-Channel Statistics:")
        for ch in range(10):
            ch_mean = semantic_map[ch].mean().item()
            ch_std = semantic_map[ch].std().item()
            ch_min = semantic_map[ch].min().item()
            ch_max = semantic_map[ch].max().item()
            print(f"  Channel {ch}: mean={ch_mean:.4f}, std={ch_std:.4f}, min={ch_min:.4f}, max={ch_max:.4f}")
        
        # Check for NaN or Inf
        if torch.isnan(semantic_map).any():
            print(f"✗ ERROR: NaN values detected in semantic output!")
            return False
        
        if torch.isinf(semantic_map).any():
            print(f"✗ ERROR: Inf values detected in semantic output!")
            return False
        
        print(f"\n✓ No NaN or Inf values detected")
        
        # Additional test: Set different values per channel
        print("\n[BONUS TEST] Testing per-channel variation...")
        with torch.no_grad():
            # Set each Gaussian to have different semantic values per channel
            for ch in range(10):
                gaussians._semantic_features.data[:, ch] = ch * 1.0
        
        with torch.no_grad():
            render_pkg2 = render(viewpoint_cam, gaussians, pipe_args, background)
            semantic_map2 = render_pkg2["semantic"]
        
        print(f"✓ Second render completed")
        print(f"  Channel means after variation:")
        for ch in range(10):
            print(f"  Channel {ch}: {semantic_map2[ch].mean().item():.4f}")
        
        # Check that channels are different
        channel_means = [semantic_map2[ch].mean().item() for ch in range(10)]
        if len(set([f"{m:.2f}" for m in channel_means])) > 1:
            print(f"✓ Channels show expected variation")
        else:
            print(f"⚠ Warning: All channels have similar values")
        
    except Exception as e:
        print(f"✗ Failed to verify semantic output: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check again opacity and scaling means
    print("\n[7/7] Verifying opacity and scaling means...")
    try:
        print(f"  Opacity mean: {gaussians.get_opacity.mean().item():.4f}")
        print(f"  Scaling mean: {gaussians.get_scaling.mean().item():.4f}")
    except Exception as e:
        print(f"✗ Failed to verify opacity and scaling means: {e}")
        import traceback
        traceback.print_exc()
    
    return True


if __name__ == "__main__":
    parser = ArgumentParser(description="Test semantic feature forward pass with Stage 1 model")
    parser.add_argument("--source_path", type=str, required=True, 
                       help="Path to dataset (same as used in Stage 1 training)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to Stage 1 model directory (contains point_cloud/)")
    parser.add_argument("--iteration", type=int, required=True,
                       help="Iteration number to load (e.g., 5000, 30000)")
    args = parser.parse_args()
    
    # Validate model path exists
    if not os.path.exists(args.model_path):
        print(f"\n✗ ERROR: Model path not found: {args.model_path}")
        print("Please provide a valid Stage 1 model directory.")
        sys.exit(1)
    
    # Check if PLY file exists
    ply_path = f"{args.model_path}/point_cloud/iteration_{args.iteration}/point_cloud.ply"
    if not os.path.exists(ply_path):
        print(f"\n✗ ERROR: PLY file not found: {ply_path}")
        print(f"Available iterations in {args.model_path}/point_cloud/:")
        if os.path.exists(f"{args.model_path}/point_cloud"):
            for item in os.listdir(f"{args.model_path}/point_cloud"):
                if item.startswith("iteration_"):
                    print(f"  - {item}")
        sys.exit(1)
    
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*15 + "SEMANTIC FEATURE FORWARD PASS TEST" + " "*19 + "║")
    print("╚" + "═"*68 + "╝")
    
    success = test_forward_pass(args.source_path, args.model_path, args.iteration)
    
    print_section("FINAL RESULT")
    if success:
        print("\n  ✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\n  The semantic feature forward pass is working correctly!")
        print("  Next steps:")
        print("    1. Test backward pass (gradients)")
        print("    2. Add semantic loss to train.py")
        print("    3. Train with semantic supervision")
        print()
        sys.exit(0)
    else:
        print("\n  ✗✗✗ TESTS FAILED ✗✗✗")
        print("\n  Please check the error messages above.")
        print()
        sys.exit(1)

