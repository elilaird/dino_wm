#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
from pathlib import Path

def download_file(gdown_id, output_path):
    """Download a file using gdown"""
    cmd = ["gdown", gdown_id, "-O", str(output_path)]
    print(f"Downloading {output_path.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error downloading {output_path.name}: {result.stderr}")
        return False
    print(f"Successfully downloaded {output_path.name}")
    return True

def unzip_file(zip_path, extract_dir):
    """Unzip a file to the specified directory"""
    cmd = ["unzip", "-q", str(zip_path), "-d", str(extract_dir)]
    print(f"Extracting {zip_path.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error extracting {zip_path.name}: {result.stderr}")
        return False
    print(f"Successfully extracted {zip_path.name}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Download and extract Memory Maze dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for downloads")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # File IDs mapping
    files = {
        # "train-part0.zip": "19s80X0wUtQznkmYEzIwpJHevVQBbLCST",
        # "train-part1.zip": "1yq-59zawM7N0XNhONmqR3Zs_xPCXBI8m", 
        # "train-part2.zip": "1r19Lpy3iklyfGsl-m1bAZZVhRl5uRWLT",
        # "train-part3.zip": "1FZICWZkwOhG4rIIXk1P-BIWnJ2ZPj4Eu",
        # "train-part4.zip": "1HlbLM0kqeBHf1-_kd7shcJ9R-70g1jMo",
        "train-part5.zip": "1HtjmGrMqkOfGpztfTteFSlGTXYPPfyQX",
        "train-part6.zip": "11Dk2NWKcmMRh6sNcJeLqU_AQELawpBtP",
        "train-part7.zip": "10yjJ_kHZ-LpTkceXD_vkoxxep9L32Qoy",
        "train-part8.zip": "1tVxYcy9wBvihMi61q8fTG22XAJqB2f0H",
        "train-part9.zip": "1cZDRsXYJQ3fUgDT0f-AhM3pmibtObHM5",
        "eval.zip": "1usfAYW48V0L0flQMB_yI9XVk5xSKh-Hm"
    }
    
    print(f"Downloading Memory Maze dataset to {output_dir}")
    
    # Download all files
    downloaded_files = []
    for filename, gdown_id in files.items():
        output_path = output_dir / filename
        if download_file(gdown_id, output_path):
            downloaded_files.append(output_path)
        else:
            print(f"Failed to download {filename}")
            sys.exit(1)
    
    print(f"\nAll {len(downloaded_files)} files downloaded successfully!")
    
    # Extract all files
    print("\nExtracting all files...")
    for zip_file in downloaded_files:
        if not unzip_file(zip_file, output_dir):
            print(f"Failed to extract {zip_file.name}")
            sys.exit(1)
    
    print("\nAll files extracted successfully!")
    print(f"Dataset available at: {output_dir}")

if __name__ == "__main__":
    main()
