#!/usr/bin/env python3
"""
Preprocess segmentation masks: reduce to 3 classes
Usage: python preprocess_seg_masks_3class.py --input /path/to/seg_masks --output /path/to/output
"""

import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

def reduce_to_3_classes(mask):
    """
    Reduce to 3 semantic classes:
    - Class 0: Background
    - Class 1: Everything else (foreground)
    - Class 2: Rare/small objects
    
    Strategy: Group by frequency
    """
    unique_classes = np.unique(mask)
    
    # Count pixels per class
    class_counts = {}
    for cls in unique_classes:
        class_counts[cls] = (mask == cls).sum()
    
    # Sort by frequency
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create new mask
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    
    # Class 0: Keep background (usually most common or class 0)
    if sorted_classes[0][0] == 0:
        new_mask[mask == 0] = 0
        sorted_classes = sorted_classes[1:]
    else:
        new_mask[mask == sorted_classes[0][0]] = 0
        sorted_classes = sorted_classes[1:]
    
    # Class 1: Most common foreground classes (top 60%)
    split_point = len(sorted_classes) * 6 // 10
    for orig_class, _ in sorted_classes[:split_point]:
        new_mask[mask == orig_class] = 1
    
    # Class 2: Rare/small objects (remaining 40%)
    for orig_class, _ in sorted_classes[split_point:]:
        new_mask[mask == orig_class] = 2
    
    return new_mask

def process_directory(input_dir, output_dir):
    """Process all .npy files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .npy files
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy') or f.endswith('_seg.npy')]
    
    if len(npy_files) == 0:
        print(f"No .npy files found in {input_dir}")
        return
    
    print(f"Found {len(npy_files)} masks")
    print(f"Reducing to 3 classes...")
    
    for npy_file in tqdm(npy_files, desc="Processing"):
        # Load
        mask_path = os.path.join(input_dir, npy_file)
        mask = np.load(mask_path)
        
        # Reduce to 3 classes
        new_mask = reduce_to_3_classes(mask)
        
        # Save
        output_npy = os.path.join(output_dir, npy_file)
        np.save(output_npy, new_mask)
        
        # Save visualization
        vis_mask = (new_mask.astype(float) / 2.0 * 255).astype(np.uint8)
        png_file = npy_file.replace('.npy', '.png')
        Image.fromarray(vis_mask).save(os.path.join(output_dir, png_file))
    
    print(f"\n✓ Processed {len(npy_files)} masks → 3 classes")
    print(f"✓ Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    process_directory(args.input, args.output)



