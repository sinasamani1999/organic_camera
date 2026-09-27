import torch
import torch.nn as nn

# Import classes created separately in previous steps
import sys
import os

# اضافه کردن مسیر ریشه پروژه به پایتون برای حل مشکل Import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # حالت اول: وقتی از ریشه پروژه اجرا می‌شود
    from models.CNN_Base import VisualBackbone
    from models.Temporal_Base import TemporalBackbone
    from models.Attention_Base import CrossAttention
except ModuleNotFoundError:
    # حالت دوم: وقتی فایل به صورت مستقیم و جداگانه اجرا می‌شود
    from CNN_Base import VisualBackbone
    from Temporal_Base import TemporalBackbone
    from Attention_Base import CrossAttention
class OrganicCameraModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super(OrganicCameraModel, self).__init__()
        
        # 1. Initialize visual feature extraction layer (Phase 2.1)
        self.visual_backbone = VisualBackbone(hidden_dim=hidden_dim)
        
        # 2. Initialize temporal hand motion processing layer (Phase 2.2)
        self.temporal_backbone = TemporalBackbone(input_dim=2, hidden_size=64, hidden_dim=hidden_dim)
        
        # 3. Initialize Cross-Attention fusion layer (Phase 2.3)
        self.cross_attention = CrossAttention(feature_dim=hidden_dim)
        
        # 4. Final linear regression layer
        # Maps the 128-dimensional fused vector to 2 final values: [Predicted_Cam_X, Predicted_Cam_Y]
        self.fc_out = nn.Linear(hidden_dim, 2)

    def forward(self, video_frame, motion_sequence):
        """
        video_frame: [Batch_Size, 3, 224, 224] -> Game video frame
        motion_sequence: [Batch_Size, Sequence_Length, 2] -> Hand movement sequence from past frames
        """
        # A) Extract visual features from the game scene
        vis_feats = self.visual_backbone(video_frame) # Output: [Batch_Size, 128]
        
        # B) Extract dynamic features and acceleration from hand motion
        motion_feats = self.temporal_backbone(motion_sequence) # Output: [Batch_Size, 128]
        
        # C) Intelligently fuse vision and motion using the Cross-Attention mechanism
        fused_feats = self.cross_attention(vis_feats, motion_feats) # Output: [Batch_Size, 128]
        
        # D) Compute the final prediction for camera movement
        camera_prediction = self.fc_out(fused_feats) # Output: [Batch_Size, 2]
        
        return camera_prediction

# ====== Integrated AI Network Test Code ======
if __name__ == "__main__":
    # Simulate simultaneous input data (4 fully parallel samples)
    batch_size = 4
    seq_len = 10 # 10 historical frames
    
    dummy_frames = torch.randn(batch_size, 3, 224, 224) # Game frames
    dummy_motions = torch.randn(batch_size, seq_len, 2)  # Player's mouse data
    
    # Instantiate the entire hybrid proposed model
    full_model = OrganicCameraModel(hidden_dim=128)
    
    # Process simultaneously and get camera controller outputs
    predicted_camera_move = full_model(dummy_frames, dummy_motions)
    
    print("========================================================")
    print(" The entire CNN-GRU + Cross-Attention hybrid network architecture was built successfully.")
    print("========================================================")
    print(f"Input game frames shape: {dummy_frames.shape}")
    print(f"Time-series mouse data shape: {dummy_motions.shape}")
    print(f"Final output tensor shape (Predicted organic camera motion): {predicted_camera_move.shape}")
    print("--- Sample numerical model output for a single frame (CamX, CamY) ---")
    print(predicted_camera_move[0].detach().numpy())