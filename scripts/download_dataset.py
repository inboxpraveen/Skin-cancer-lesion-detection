"""
Helper script to download HAM10000 dataset using Kaggle API
Note: Requires kaggle.json to be configured
"""

import os
import sys
import zipfile
import argparse

def download_dataset(output_dir='data'):
    """
    Download HAM10000 dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
    """
    try:
        import kaggle
    except ImportError:
        print("Error: kaggle package not installed.")
        print("Install with: pip install kaggle")
        sys.exit(1)
    
    # Check if kaggle.json exists
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_json):
        print("Error: Kaggle API credentials not found.")
        print("\nPlease follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section and click 'Create New API Token'")
        print("3. This downloads kaggle.json")
        print(f"4. Move it to {kaggle_dir}/")
        print(f"5. On Unix: chmod 600 {kaggle_json}")
        sys.exit(1)
    
    print("Downloading HAM10000 dataset from Kaggle...")
    print("This may take several minutes (2.6 GB download)...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download dataset
    os.system(f'kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p {output_dir}')
    
    print("\nExtracting files...")
    
    # Extract main zip
    main_zip = os.path.join(output_dir, 'skin-cancer-mnist-ham10000.zip')
    if os.path.exists(main_zip):
        with zipfile.ZipFile(main_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(main_zip)
        print("  Extracted main archive")
    
    # Extract image parts
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    for part in [1, 2]:
        part_zip = os.path.join(output_dir, f'HAM10000_images_part_{part}.zip')
        if os.path.exists(part_zip):
            with zipfile.ZipFile(part_zip, 'r') as zip_ref:
                zip_ref.extractall(images_dir)
            os.remove(part_zip)
            print(f"  Extracted image part {part}")
    
    # Remove CSV files we don't need
    csv_files = [
        'hmnist_8_8_RGB.csv',
        'hmnist_8_8_L.csv',
        'hmnist_28_28_RGB.csv',
        'hmnist_28_28_L.csv'
    ]
    
    for csv_file in csv_files:
        csv_path = os.path.join(output_dir, csv_file)
        if os.path.exists(csv_path):
            os.remove(csv_path)
    
    print("\nDataset downloaded and extracted successfully!")
    print(f"Location: {os.path.abspath(output_dir)}")
    print("\nYou can now start training:")
    print("  python src/train.py --model sequential")

def main():
    parser = argparse.ArgumentParser(
        description='Download HAM10000 dataset from Kaggle'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='Output directory for dataset'
    )
    
    args = parser.parse_args()
    download_dataset(args.output)

if __name__ == '__main__':
    main()

