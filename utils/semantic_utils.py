"""
Semantic segmentation utilities for ScanNet++ dataset.

The masks contain GLOBAL class indices from semantic_classes.txt (sparse, e.g., 0, 5, 32, 207).
This utility creates a remap to CONSECUTIVE local indices (0, 1, 2, ..., N-1).
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import os


def load_global_classes(classes_path: str = "/home/dkonur/scannetpp/metadata/semantic_classes.txt") -> List[str]:
    """Load global semantic class names from semantic_classes.txt."""
    with open(classes_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def scan_scene_classes(mask_dir: str, max_files: int = -1) -> Tuple[set, Dict[int, int]]:
    """
    Scan all masks in a scene to find unique global class indices.
    
    Args:
        mask_dir: Directory containing semantic masks
        max_files: Maximum files to scan (-1 for all)
    
    Returns:
        global_indices: Set of unique global class indices found
        pixel_counts: Dict of global_index -> total pixel count
    """
    mask_path = Path(mask_dir)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    
    mask_files = [f for f in mask_path.iterdir() 
                  if f.suffix == '.png' and not f.name.endswith('_viz.png')]
    
    if max_files > 0:
        mask_files = mask_files[:max_files]
    
    global_indices = set()
    pixel_counts = {}
    
    for mf in mask_files:
        mask = np.array(Image.open(mf))
        unique, counts = np.unique(mask, return_counts=True)
        global_indices.update(unique)
        for idx, cnt in zip(unique, counts):
            pixel_counts[idx] = pixel_counts.get(idx, 0) + cnt
    
    return global_indices, pixel_counts


def create_class_remap(
    mask_dir: str,
    classes_path: str = "/home/dkonur/scannetpp/metadata/semantic_classes.txt",
    output_path: str = None,
    background_class: int = None,
) -> Dict[int, int]:
    """
    Create a mapping from global sparse class indices to consecutive local indices.
    
    Args:
        mask_dir: Directory containing semantic masks
        classes_path: Path to semantic_classes.txt
        output_path: Optional path to save the remap as JSON
        background_class: Optional global class index to force as local index 0
    
    Returns:
        remap: Dict mapping global_index -> local_index
    """
    global_classes = load_global_classes(classes_path)
    global_indices, pixel_counts = scan_scene_classes(mask_dir)
    
    # Sort indices for consistent ordering
    sorted_indices = sorted(global_indices)
    
    # If background_class specified, put it first
    if background_class is not None and background_class in sorted_indices:
        sorted_indices.remove(background_class)
        sorted_indices = [background_class] + sorted_indices
    
    # Create remap: global_index -> local_index
    remap = {int(global_idx): int(local_idx) for local_idx, global_idx in enumerate(sorted_indices)}
    
    # Build output data
    output_data = {
        "num_classes": len(sorted_indices),
        "global_to_local": {str(k): int(v) for k, v in remap.items()},
        "local_to_global": {str(v): int(k) for k, v in remap.items()},
        "class_info": []
    }
    
    for local_idx, global_idx in enumerate(sorted_indices):
        class_name = global_classes[global_idx] if global_idx < len(global_classes) else "UNKNOWN"
        output_data["class_info"].append({
            "local_index": int(local_idx),
            "global_index": int(global_idx),
            "name": class_name,
            "pixel_count": int(pixel_counts.get(global_idx, 0))
        })
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved remap to {output_path}")
    
    return remap


def print_scene_info(mask_dir: str, classes_path: str = "/home/dkonur/scannetpp/metadata/semantic_classes.txt"):
    """Print information about classes in a scene."""
    global_classes = load_global_classes(classes_path)
    global_indices, pixel_counts = scan_scene_classes(mask_dir)
    
    print(f"\n{'='*60}")
    print(f"Scene: {mask_dir}")
    print(f"{'='*60}")
    print(f"Unique classes: {len(global_indices)}")
    print(f"\n{'Local':<6} {'Global':<8} {'Name':<30} {'Pixels':<12}")
    print("-" * 60)
    
    for local_idx, global_idx in enumerate(sorted(global_indices)):
        class_name = global_classes[global_idx] if global_idx < len(global_classes) else "UNKNOWN"
        pixels = pixel_counts.get(global_idx, 0)
        print(f"{local_idx:<6} {global_idx:<8} {class_name:<30} {pixels:<12}")
    
    print(f"\n⚠ Set NUM_SEMANTIC_CHANNELS = {len(global_indices)} in config.h")
    print(f"  Then recompile: cd submodules/diff-gaussian-rasterization && pip install .")
    
    return global_indices


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python semantic_utils.py <scene_id> [--save]")
        print("Example: python semantic_utils.py f36e3e1e53 --save")
        sys.exit(1)
    
    scene_id = sys.argv[1]
    save_remap = "--save" in sys.argv
    
    mask_dir = f"/home/dkonur/scannetpp/semantic_2d_output/{scene_id}/{scene_id}_4x"
    
    # Print info
    print_scene_info(mask_dir)
    
    # Save remap if requested
    if save_remap:
        output_path = f"/home/dkonur/scannetpp/metadata/class_remap_{scene_id}.json"
        create_class_remap(mask_dir, output_path=output_path)

