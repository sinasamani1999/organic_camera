import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class OrganicCameraDataset(Dataset):
    def __init__(self, video_dir, motion_dir, seq_length=10):
        self.video_dir = video_dir
        self.motion_dir = motion_dir
        self.seq_length = seq_length
        
        # Find all common names between videos and CSV files
        self.file_names = [os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith('.mp4')]
        
        # Define standard transforms for visual frames (ResNet-18)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Build a global index of all trainable frames across all videos
        self.samples = []
        self._prepare_dataset_indices()

    def _prepare_dataset_indices(self):
        """
        This function checks the total number of frames in each video and registers
        samples that have a sufficient motion history (at least equal to seq_length).
        """
        for name in self.file_names:
            csv_path = os.path.join(self.motion_dir, f"{name}.csv")
            if os.path.exists(csv_path):
                # Read motion data file
                df = pd.read_csv(csv_path)
                total_frames = len(df)
                
                # We need the past 10 frames to predict each frame
                for frame_idx in range(self.seq_length, total_frames):
                    self.samples.append({
                        'video_name': name,
                        'target_frame_idx': frame_idx
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_name = sample['video_name']
        target_idx = sample['target_frame_idx']
        
        # 1. Extract the target visual frame using OpenCV
        video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # In case of a read error, return an empty frame or zero tensor
            frame_tensor = torch.zeros(3, 224, 224)
        else:
            # Convert format from BGR to RGB and apply PyTorch transforms
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frame_tensor = self.transform(pil_img)
            
        # 2. Extract past motion sequence from the CSV file
        csv_path = os.path.join(self.motion_dir, f"{video_name}.csv")
        df = pd.read_csv(csv_path)
        
        # Slice the past 10 rows (from target_idx - seq_length to target_idx)
        motion_seq = df.iloc[target_idx - self.seq_length : target_idx][['Delta_X', 'Delta_Y']].values
        motion_tensor = torch.tensor(motion_seq, dtype=torch.float32)
        
        # 3. Target label (The amount of organic movement that occurred in this specific frame)
        target_motion = df.iloc[target_idx][['Delta_X', 'Delta_Y']].values
        target_tensor = torch.tensor(target_motion, dtype=torch.float32)
        
        return frame_tensor, motion_tensor, target_tensor

# ====== Data Pipeline Testing Section ======
if __name__ == "__main__":
    print("Dataset class compiled successfully.")
    print("This module is ready to connect to the actual data on your hard drive.")