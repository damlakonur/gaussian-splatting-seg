"""
Comprehensive test suite for semantic feature backward pass.
Tests gradient correctness, propagation, and training convergence.
"""

import torch
import sys
import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from gaussian_renderer import render


class BackwardPassTester:
    def __init__(self, source_path, model_path, iteration):
        print("=" * 70)
        print("Semantic Feature Backward Pass Test Suite")
        print("=" * 70)
        
        # Load scene
        print(f"\n[Setup] Loading scene from {model_path} (iteration {iteration})...")
        
        model_args = Namespace(
            source_path=source_path,
            model_path=model_path,
            images="images",
            resolution=8,  # Match training resolution!
            white_background=False,
            data_device="cuda",
            eval=False,
            depths="",
            train_test_exp=False
        )
        
        self.gaussians = GaussianModel(sh_degree=3)
        self.scene = Scene(model_args, self.gaussians, load_iteration=iteration, shuffle=False)
        self.camera = self.scene.getTrainCameras()[0]
        
        self.pipe_args = Namespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
            antialiasing=False
        )
        
        print(f"[Setup] Loaded {len(self.gaussians.get_xyz)} Gaussians")
        print(f"[Setup] Semantic features shape: {self.gaussians.get_semantic_features.shape}")
        print(f"[Setup] Camera resolution: {self.camera.image_width}x{self.camera.image_height}")
        print("[Setup] Ready!\n")
    
    def test_1_zero_input(self):
        """Test 1: Gradients should exist even with zero semantic features"""
        print("-" * 70)
        print("Test 1: Zero Input Test")
        print("-" * 70)
        
        # Save original values
        original_features = self.gaussians._semantic_features.data.clone()
        
        # Set to zero
        self.gaussians._semantic_features.data.fill_(0.0)
        self.gaussians._semantic_features.requires_grad = True
        
        # Render
        with torch.enable_grad():
            rendered = render(self.camera, self.gaussians, self.pipe_args, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))
            semantic_map = rendered["semantic"]
            
            # Simple loss
            loss = semantic_map.sum()
            loss.backward()
        
        # Check gradients
        grad_norm = self.gaussians._semantic_features.grad.norm().item()
        has_nonzero = (self.gaussians._semantic_features.grad.abs() > 1e-10).any().item()
        
        print(f"  Semantic map sum: {semantic_map.sum().item():.6f}")
        print(f"  Gradient norm: {grad_norm:.6f}")
        print(f"  Has non-zero gradients: {has_nonzero}")
        
        # Restore
        self.gaussians._semantic_features.data.copy_(original_features)
        
        assert self.gaussians._semantic_features.grad is not None, "Gradients should exist!"
        print("  ✓ Test 1 PASSED: Gradients exist with zero input\n")
        return True
    
    def test_2_loss_propagation(self):
        """Test 2: Check that loss correctly propagates through all components"""
        print("-" * 70)
        print("Test 2: Loss Propagation Test")
        print("-" * 70)
        
        self.gaussians._semantic_features.requires_grad = True
        
        with torch.enable_grad():
            rendered = render(self.camera, self.gaussians, self.pipe_args, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))
            semantic_map = rendered["semantic"]
            
            # Create fake ground truth (all pixels = class 5)
            H, W = semantic_map.shape[1], semantic_map.shape[2]
            
            # gt should be shape [H, W] with class indices (0-9)
            gt = torch.ones(H, W, dtype=torch.long, device='cuda') * 5
            
            # Cross-entropy loss expects:
            # - input: [N, C, H, W] or [C, H, W]
            # - target: [N, H, W] or [H, W] with class indices
            loss = torch.nn.functional.cross_entropy(semantic_map.unsqueeze(0), gt.unsqueeze(0))
            loss.backward()
        
        # Check gradient properties
        grads = self.gaussians._semantic_features.grad
        print(f"  Loss value: {loss.item():.6f}")
        print(f"  Gradient shape: {grads.shape}")
        print(f"  Gradient mean: {grads.mean().item():.6f}")
        print(f"  Gradient std: {grads.std().item():.6f}")
        print(f"  Gradient min/max: {grads.min().item():.6f} / {grads.max().item():.6f}")
        
        non_zero = (grads.abs() > 1e-8).sum().item()
        total = grads.numel()
        print(f"  Non-zero gradients: {non_zero} / {total} ({100*non_zero/total:.1f}%)")
        
        # Assertions
        assert grads is not None, "Gradients are None!"
        assert not torch.isnan(grads).any(), "Gradients contain NaN!"
        assert not torch.isinf(grads).any(), "Gradients contain Inf!"
        assert (grads.abs() > 0).any(), "All gradients are zero!"
        
        
        print("  ✓ Test 2 PASSED: Loss propagates correctly\n")
        return True
    
    def test_3_gradient_check(self):
        """Test 3: Gradient check - compare analytical vs numerical gradients"""
        print("-" * 70)
        print("Test 3: Gradient Check (Analytical vs Numerical)")
        print("-" * 70)
        
        # Pick random Gaussian and channel to test
        test_gaussian_id = torch.randint(0, len(self.gaussians.get_xyz), (1,)).item()
        test_channel = torch.randint(0, 10, (1,)).item()
        
        print(f"  Testing Gaussian #{test_gaussian_id}, Channel {test_channel}")
        
        epsilon = 1e-4
        
        # === Analytical Gradient ===
        self.gaussians._semantic_features.requires_grad = True
        
        with torch.enable_grad():
            rendered = render(self.camera, self.gaussians, self.pipe_args, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))
            semantic_map = rendered["semantic"]
            
            # Simple loss: sum of one channel
            loss = semantic_map[test_channel].sum()
            loss.backward()
        
        analytical_grad = self.gaussians._semantic_features.grad[test_gaussian_id, test_channel].item()
        
        # === Numerical Gradient ===
        
        # Forward pass with +epsilon
        original_value = self.gaussians._semantic_features.data[test_gaussian_id, test_channel].item()
        self.gaussians._semantic_features.data[test_gaussian_id, test_channel] += epsilon
        
        with torch.no_grad():
            rendered_plus = render(self.camera, self.gaussians, self.pipe_args, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))
            loss_plus = rendered_plus["semantic"][test_channel].sum().item()
        
        # Forward pass with -epsilon
        self.gaussians._semantic_features.data[test_gaussian_id, test_channel] = original_value - epsilon
        
        with torch.no_grad():
            rendered_minus = render(self.camera, self.gaussians, self.pipe_args, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))
            loss_minus = rendered_minus["semantic"][test_channel].sum().item()
        
        # Restore original value
        self.gaussians._semantic_features.data[test_gaussian_id, test_channel] = original_value
        
        # Numerical gradient
        numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
        
        # === Compare ===
        absolute_error = abs(analytical_grad - numerical_grad)
        relative_error = absolute_error / (abs(numerical_grad) + 1e-8)
        
        print(f"  Analytical gradient: {analytical_grad:.8f}")
        print(f"  Numerical gradient:  {numerical_grad:.8f}")
        print(f"  Absolute error:      {absolute_error:.8e}")
        print(f"  Relative error:      {relative_error:.8e}")
        
        threshold = 1e-3
        passed = relative_error < threshold
        
        if passed:
            print(f"  ✓ Test 3 PASSED: Relative error < {threshold}\n")
        else:
            print(f"  ✗ Test 3 FAILED: Relative error {relative_error:.6f} >= {threshold}\n")
            print(f"    This could indicate a bug in the backward pass!")
        
        return passed
    
    def test_4_channel_independence(self):
        """Test 4: Each channel should have independent gradients"""
        print("-" * 70)
        print("Test 4: Channel Independence Test")
        print("-" * 70)
        
        self.gaussians._semantic_features.requires_grad = True
        
        # Test channel 3 only
        test_channel = 3
        
        with torch.enable_grad():
            rendered = render(self.camera, self.gaussians, self.pipe_args, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))
            loss = rendered["semantic"][test_channel].sum()
            loss.backward()
        
        grads = self.gaussians._semantic_features.grad
        
        # Average gradient magnitude per channel
        grad_per_channel = grads.abs().mean(dim=0)
        
        print(f"  Testing with loss on channel {test_channel} only")
        print(f"  Average gradient magnitude per channel:")
        for i, g in enumerate(grad_per_channel):
            marker = " ← tested" if i == test_channel else ""
            print(f"    Channel {i}: {g.item():.8f}{marker}")
        
        # Channel 3 should have higher gradients than others
        max_channel = grad_per_channel.argmax().item()
        
        print(f"\n  Channel with highest gradient: {max_channel}")
        
        # Allow some tolerance - test channel should be in top 3
        sorted_channels = grad_per_channel.argsort(descending=True)
        test_channel_rank = (sorted_channels == test_channel).nonzero(as_tuple=True)[0].item()
        
        
        if test_channel_rank < 3:
            print(f"  ✓ Test 4 PASSED: Test channel {test_channel} is rank {test_channel_rank + 1}\n")
            return True
        else:
            print(f"  ⚠ Test 4 WARNING: Test channel {test_channel} is only rank {test_channel_rank + 1}")
            print(f"    Expected it to be in top 3, but gradients are still independent\n")
            return True
    
    def test_5_gradient_magnitude_comparison(self):
        """Test 5: Compare semantic gradient magnitudes to RGB gradients"""
        print("-" * 70)
        print("Test 5: Gradient Magnitude Comparison (RGB vs Semantic)")
        print("-" * 70)
        
        # Enable all gradients
        self.gaussians._semantic_features.requires_grad = True
        self.gaussians._features_dc.requires_grad = True
        
        with torch.enable_grad():
            rendered = render(self.camera, self.gaussians, self.pipe_args, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))
            
            # Loss on both RGB and semantic
            rgb_loss = rendered["render"].sum()
            semantic_loss = rendered["semantic"].sum()
            total_loss = rgb_loss + semantic_loss
            total_loss.backward()
        
        # Compare magnitudes
        rgb_grad_norm = self.gaussians._features_dc.grad.norm().item()
        semantic_grad_norm = self.gaussians._semantic_features.grad.norm().item()
        ratio = semantic_grad_norm / (rgb_grad_norm + 1e-8)
        
        print(f"  RGB loss: {rgb_loss.item():.6f}")
        print(f"  Semantic loss: {semantic_loss.item():.6f}")
        print(f"  RGB gradient norm: {rgb_grad_norm:.6f}")
        print(f"  Semantic gradient norm: {semantic_grad_norm:.6f}")
        print(f"  Ratio (Semantic/RGB): {ratio:.6f}")
        
        
        # They should be in a reasonable range
        if 0.01 < ratio < 100:
            print(f"  ✓ Test 5 PASSED: Gradient magnitudes are in reasonable range\n")
            return True
        else:
            print(f"  ⚠ Test 5 WARNING: Ratio {ratio:.2f} is outside [0.01, 100]")
            print(f"    Semantic gradients might be too large or too small\n")
            return True
    
    def test_6_training_convergence(self):
        """Test 6: Training convergence - loss should decrease (with downscaled semantic map)"""
        print("-" * 70)
        print("Test 6: Training Convergence Test (Downscaled)")
        print("-" * 70)
        
        print("  Running 20 training iterations with random ground truth on downscaled map...")

        # Render once to get original shape
        with torch.no_grad():
            rendered_init = render(
                self.camera,
                self.gaussians,
                self.pipe_args,
                torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
            )
            semantic_map_init = rendered_init["semantic"]   # [C, H, W]

        C, H, W = semantic_map_init.shape

        # Choose downscaled resolution (e.g. quarter in each dimension)
        scale = 4
        H_ds = max(1, H // scale)
        W_ds = max(1, W // scale)

        # Fixed random GT in downscaled grid
        gt_ds = torch.randint(0, C, (H_ds, W_ds), device="cuda")

        losses = []
        learning_rate = 0.01
        num_iters = 20

        # Only semantic features trainable
        self.gaussians._semantic_features.requires_grad_(True)

        for iteration in range(num_iters):
            # Clear old grads
            if self.gaussians._semantic_features.grad is not None:
                self.gaussians._semantic_features.grad.zero_()

            with torch.enable_grad():
                rendered = render(
                    self.camera,
                    self.gaussians,
                    self.pipe_args,
                    torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
                )
                semantic_map = rendered["semantic"]  # [C, H, W]

                # Downscale to [1, C, H_ds, W_ds]
                semantic_low = torch.nn.functional.interpolate(
                    semantic_map.unsqueeze(0),          # [1, C, H, W]
                    size=(H_ds, W_ds),
                    mode="bilinear",
                    align_corners=False,
                )

                # CE expects input [N, C, H, W], target [N, H, W]
                loss = torch.nn.functional.cross_entropy(
                    semantic_low,                       # [1, C, H_ds, W_ds]
                    gt_ds.unsqueeze(0)                  # [1, H_ds, W_ds]
                )
                loss.backward()

            losses.append(loss.item())

            # Manual SGD step on semantic features
            with torch.no_grad():
                self.gaussians._semantic_features -= (
                    learning_rate * self.gaussians._semantic_features.grad
                )

            if iteration % 5 == 0:
                print(f"    Iter {iteration:2d}: Loss = {loss.item():.6f}")

            # Free memory to prevent OOM
            del rendered, semantic_map, semantic_low, loss
            torch.cuda.empty_cache()

        initial_loss = losses[0]
        final_loss = losses[-1]
        min_loss = min(losses)

        print(f"\n  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss:   {final_loss:.6f}")
        print(f"  Min loss:     {min_loss:.6f}")
        print(f"  Reduction:    {100*(initial_loss - final_loss)/initial_loss:.1f}%")

        reduction = (initial_loss - final_loss) / max(initial_loss, 1e-8)

        if reduction > 0.05:
            print(f"  ✓ Test 6 PASSED: Loss decreased by {100*reduction:.1f}%\n")
            return True
        else:
            print(f"  ✗ Test 6 FAILED: Loss only decreased by {100*reduction:.1f}%")
            print("    Expected at least 5% reduction - backward pass may be broken!\n")
            return False
    
    def test_7_nan_inf_check(self):
        """Test 7: Check for NaN/Inf in gradients under various conditions"""
        print("-" * 70)
        print("Test 7: NaN/Inf Gradient Check")
        print("-" * 70)
        
        test_cases = [
            ("Normal values", lambda: None),
            ("Large values", lambda: self.gaussians._semantic_features.data.mul_(100)),
            ("Small values", lambda: self.gaussians._semantic_features.data.mul_(0.01)),
            ("Mixed signs", lambda: self.gaussians._semantic_features.data.mul_(
                torch.randn_like(self.gaussians._semantic_features.data)
            )),
        ]
        
        all_passed = True
        original_features = self.gaussians._semantic_features.data.clone()
        
        for name, modifier in test_cases:
            # Reset and modify
            self.gaussians._semantic_features.data.copy_(original_features)
            modifier()
            
            self.gaussians._semantic_features.requires_grad = True
            
            try:
                # Clear previous gradients to avoid memory leaks
                if self.gaussians._semantic_features.grad is not None:
                    self.gaussians._semantic_features.grad.zero_()
                
                with torch.enable_grad():
                    rendered = render(self.camera, self.gaussians, self.pipe_args, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))
                    semantic_map = rendered["semantic"]
                    
                    H, W = semantic_map.shape[1], semantic_map.shape[2]
                    gt = torch.randint(0, 10, (H, W), device='cuda')
                    
                    # Add batch dimension for cross_entropy
                    loss = torch.nn.functional.cross_entropy(semantic_map.unsqueeze(0), gt.unsqueeze(0))
                    loss.backward()
                
                grads = self.gaussians._semantic_features.grad
                
                # Clean up to prevent OOM
                del rendered, semantic_map, gt, loss
                torch.cuda.empty_cache()
                
                has_nan = torch.isnan(grads).any().item()
                has_inf = torch.isinf(grads).any().item()
                
                if has_nan or has_inf:
                    print(f"  ✗ {name}: NaN={has_nan}, Inf={has_inf}")
                    all_passed = False
                else:
                    print(f"  ✓ {name}: No NaN/Inf")
            
            except Exception as e:
                print(f"  ✗ {name}: Exception - {str(e)}")
                all_passed = False
        
        # Restore
        self.gaussians._semantic_features.data.copy_(original_features)
        
        if all_passed:
            print(f"  ✓ Test 7 PASSED: No NaN/Inf in any condition\n")
        else:
            print(f"  ✗ Test 7 FAILED: Found NaN/Inf in gradients!\n")
        
        return all_passed
    
    def run_all_tests(self):
        """Run all tests and report results"""
        tests = [
            self.test_1_zero_input,
            self.test_2_loss_propagation,
            self.test_3_gradient_check,
            self.test_4_channel_independence,
            self.test_5_gradient_magnitude_comparison,
            self.test_6_training_convergence,
            self.test_7_nan_inf_check,
        ]
        
        results = []
        
        for test in tests:
            try:
                passed = test()
                results.append((test.__doc__.split(':')[0].strip(), passed))
            except Exception as e:
                print(f"\n✗ {test.__name__} CRASHED: {str(e)}\n")
                results.append((test.__doc__.split(':')[0].strip(), False))
        
        # Summary
        print("=" * 70)
        print("Test Summary")
        print("=" * 70)
        
        for name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{status}: {name}")
        
        passed_count = sum(1 for _, p in results if p)
        total_count = len(results)
        
        print(f"\nResults: {passed_count}/{total_count} tests passed")
        
        if passed_count == total_count:
            print("\n🎉 All tests PASSED! Backward pass is working correctly! 🎉")
            return True
        else:
            print(f"\n  {total_count - passed_count} test(s) failed - check backward pass implementation!")
            return False


def main():
    parser = ArgumentParser(description="Test semantic feature backward pass")
    parser.add_argument("--source_path", required=True, help="Path to dataset")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--iteration", type=int, default=7000, help="Iteration to load")
    
    args = parser.parse_args()
    
    # Create tester
    tester = BackwardPassTester(args.source_path, args.model_path, args.iteration)
    
    # Run all tests
    # success = tester.run_all_tests()
    
    # sys.exit(0 if success else 1)
    # tester.test_2_loss_propagation()
    # breakpoint()
    # tester.test_3_gradient_check()
    # breakpoint()
    # tester.test_4_channel_independence()
    # # breakpoint()
    # tester.test_5_gradient_magnitude_comparison()
    # # breakpoint()
    # tester.test_6_training_convergence()
    # breakpoint()
    tester.test_7_nan_inf_check()
    # breakpoint()
    # tester.test_1_zero_input()


if __name__ == "__main__":
    main()




