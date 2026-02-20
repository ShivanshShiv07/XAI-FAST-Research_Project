import os
import numpy as np
from PIL import Image
import json

def create_mock_data(base_dir, experiment_dir, category='hazelnut'):
    # Dataset structure
    dataset_root = os.path.join(base_dir, category)
    os.makedirs(os.path.join(dataset_root, 'train', 'good'), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, 'test', 'good'), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, 'test', 'defect'), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, 'ground_truth', 'defect'), exist_ok=True)
    
    # Anomaly maps structure
    exp_root = os.path.join(experiment_dir, category, 'test')
    os.makedirs(os.path.join(exp_root, 'good'), exist_ok=True)
    os.makedirs(os.path.join(exp_root, 'defect'), exist_ok=True)
    
    # Generate images
    size = (256, 256)
    
    # Train
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, size + (3,), dtype=np.uint8))
        img.save(os.path.join(dataset_root, 'train', 'good', f'{i:03d}.png'))
        
    # Test Good
    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, size + (3,), dtype=np.uint8))
        img.save(os.path.join(dataset_root, 'test', 'good', f'{i:03d}.png'))
        
        # Anomaly map (low scores)
        am = Image.fromarray(np.random.randint(0, 50, size, dtype=np.uint8))
        am.save(os.path.join(exp_root, 'good', f'{i:03d}.tiff')) # tiff as per readme

    # Test Defect
    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, size + (3,), dtype=np.uint8))
        img.save(os.path.join(dataset_root, 'test', 'defect', f'{i:03d}.png'))
        
        # Ground truth (circle in middle)
        gt = np.zeros(size, dtype=np.uint8)
        gt[100:150, 100:150] = 255
        Image.fromarray(gt).save(os.path.join(dataset_root, 'ground_truth', 'defect', f'{i:03d}_mask.png'))
        
        # Anomaly map (high scores in middle)
        am_arr = np.random.randint(0, 50, size, dtype=np.uint8)
        am_arr[100:150, 100:150] = 200 # match defect
        am = Image.fromarray(am_arr)
        am.save(os.path.join(exp_root, 'defect', f'{i:03d}.tiff'))

    print(f"Created mock data in {base_dir} and {experiment_dir}")

if __name__ == "__main__":
    create_mock_data('mock_data', 'mock_results/experiment_1/anomaly_maps')
