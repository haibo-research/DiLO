#!/usr/bin/env python3
"""
Convert inverse scattering LMDB data to PNG images
"""
import os
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


def unnormalize(data, mean=0.5, std=0.25):
    """
    Unnormalize the data from training format to original range
    Based on the normalize function in LMDBData: (data - mean) / (2 * std)
    """
    return data * 2 * std + mean


def lmdb_to_png(lmdb_path, output_dir, resolution=128, num_channels=1, mean=0.5, std=0.25):
    """
    Convert LMDB database to PNG images
    
    Args:
        lmdb_path: Path to LMDB database
        output_dir: Output directory for PNG images
        resolution: Image resolution (default 128)
        num_channels: Number of channels (default 1)
        mean: Mean value for unnormalization
        std: Std value for unnormalization
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open LMDB database
    env = lmdb.open(lmdb_path, readonly=True, lock=False, create=False)
    txn = env.begin(write=False)
    
    # Get number of entries
    num_entries = txn.stat()['entries']
    print(f"Found {num_entries} entries in {lmdb_path}")
    
    # Process each entry
    for idx in tqdm(range(num_entries), desc=f"Converting {os.path.basename(lmdb_path)}"):
        # Read data from LMDB
        key = f'{idx}'.encode('utf-8')
        img_bytes = txn.get(key)
        
        if img_bytes is None:
            print(f"Warning: Key {idx} not found")
            continue
        
        # Convert bytes to numpy array
        img = np.frombuffer(img_bytes, dtype=np.float32).reshape(num_channels, resolution, resolution)
        
        # DO NOT unnormalize - keep original [0, 1] range from LMDB
        # This ensures background pixels (value 0.0) map to black (0) in PNG
        # instead of gray (127) which would result from unnormalizing
        
        # Convert to uint8 format (0-255)
        # Directly scale [0, 1] to [0, 255]
        img_normalized = np.clip(img, 0, 1)
        img_uint8 = (img_normalized * 255).astype(np.uint8)
        
        # For single channel, squeeze and save as grayscale
        if num_channels == 1:
            img_uint8 = img_uint8.squeeze(0)  # Remove channel dimension
            pil_img = Image.fromarray(img_uint8, mode='L')
        else:
            # For multi-channel, transpose to HWC format
            img_uint8 = img_uint8.transpose(1, 2, 0)
            pil_img = Image.fromarray(img_uint8)
        
        # Save as PNG
        output_path = os.path.join(output_dir, f'{str(idx).zfill(5)}.png')
        pil_img.save(output_path)
    
    print(f"Saved {num_entries} images to {output_dir}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Convert inverse scattering LMDB data to PNG images')
    parser.add_argument('--data_root', type=str, default='/amax/haibo/Diffusion4IP/InverseBench/data',
                      help='Root directory containing LMDB databases')
    parser.add_argument('--output_root', type=str, default='/amax/haibo/Diffusion4IP/data/inverse_scatter_png',
                      help='Root directory for output PNG images')
    parser.add_argument('--resolution', type=int, default=128,
                      help='Image resolution')
    parser.add_argument('--mean', type=float, default=0.5,
                      help='Mean value for unnormalization')
    parser.add_argument('--std', type=float, default=0.25,
                      help='Std value for unnormalization')
    parser.add_argument('--datasets', nargs='+', default=['inv-scatter-test', 'inv-scatter-val'],
                      help='List of dataset names to convert')
    
    args = parser.parse_args()
    
    # Convert each dataset
    for dataset_name in args.datasets:
        lmdb_path = os.path.join(args.data_root, dataset_name)
        
        if not os.path.exists(lmdb_path):
            print(f"Warning: {lmdb_path} does not exist, skipping...")
            continue
        
        output_dir = os.path.join(args.output_root, dataset_name)
        
        print(f"\n{'='*60}")
        print(f"Converting {dataset_name}")
        print(f"Input:  {lmdb_path}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")
        
        lmdb_to_png(
            lmdb_path=lmdb_path,
            output_dir=output_dir,
            resolution=args.resolution,
            num_channels=1,
            mean=args.mean,
            std=args.std
        )
    
    print(f"\n{'='*60}")
    print("All conversions completed!")
    print(f"Images saved to: {args.output_root}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

