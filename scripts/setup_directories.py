"""
Setup script to create necessary directory structure
"""

import os

# Define directory structure
directories = [
    'data',
    'data/images',
    'models',
    'models/checkpoints',
    'logs',
    'results',
    'screenshots',
]

def setup_directories():
    """Create all necessary directories."""
    print("Setting up directory structure...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create .gitkeep file to preserve empty directories in git
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, 'w') as f:
                f.write('')
        
        print(f"  Created: {directory}/")
    
    print("\nDirectory structure created successfully!")
    print("\nNext steps:")
    print("1. Download HAM10000 dataset from Kaggle")
    print("2. Extract HAM10000_metadata.csv to data/")
    print("3. Extract all images to data/images/")
    print("4. Run training: python src/train.py --model sequential")

if __name__ == '__main__':
    setup_directories()

