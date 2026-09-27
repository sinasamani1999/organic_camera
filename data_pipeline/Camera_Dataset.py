import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CameraOrganicDataset(Dataset):
    def __init__(self, num_samples=100, seq_length=10):
        """
        In this phase, we generate simulated data with a realistic structure to test the pipeline.
        In the future, this section will scan the actual files on your hard drive.
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        
        # Simulating game frame data: num_samples frames with 224x224 resolution
        self.visual_data = torch.randn(num_samples, 3, 224, 224)
        
        # Simulating mouse motion data: for each frame, a 10-frame sequence of [Delta_X, Delta_Y]
        self.motion_data = torch.randn(num_samples, seq_length, 2)
        
        # Simulating final labels (The target values the model should predict - next frame's organic displacement)
        self.target_data = torch.randn(num_samples, 2)

    def __len__(self):
        # Total number of frames/samples available for training
        return self.num_samples

    def __getitem__(self, idx):
        """
        When the dataset loader is called, this function returns the corresponding frame, 
        hand motion history, and the target label.
        """
        frame = self.visual_data[idx]
        motion_seq = self.motion_data[idx]
        target = self.target_data[idx]
        
        return frame, motion_seq, target

# ====== Test code to validate the data pipeline ======
if __name__ == "__main__":
    # 1. Create a dataset object with 100 dummy frame samples
    dataset = CameraOrganicDataset(num_samples=100, seq_length=10)
    
    # 2. Define DataLoader to split the dataset into batches of 4 (Batch Size = 4)
    # shuffle=True shuffles the data order in each epoch to prevent the model from memorizing
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 3. Test retrieving a batch from the data loader
    first_batch = next(iter(train_loader))
    frames, motions, targets = first_batch
    
    print("--- Data Pipeline layer test completed successfully ---")
    print(f"Total number of samples in dataset: {len(dataset)}")
    print(f"Loaded game frames batch shape: {frames.shape}")       # Expected: [4, 3, 224, 224]
    print(f"Mouse motion sequence batch shape: {motions.shape}")   # Expected: [4, 10, 2]
    print(f"Final target batch shape (Targets): {targets.shape}")   # Expected: [4, 2]