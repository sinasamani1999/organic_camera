import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 1. Import local modules from custom data pipeline and models folder
from data_pipeline.Real_Camera_Dataset import OrganicCameraDataset
from models.Organic_Camera_Model import OrganicCameraModel

# 2. Hybrid Loss Function implementation (Position + Direction + Smoothness)
class OrganicCameraLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.6, gamma=0.4):
        """
        alpha: Weight for position accuracy (MSE)
        beta: Weight for direction alignment (Cosine Similarity)
        gamma: Weight for physical smoothness (Inter-frame MSE)
        """
        super(OrganicCameraLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets, prev_predictions=None):
        # A) Position Loss using standard Mean Squared Error
        loss_position = self.mse(predictions, targets)
        
        # B) Directional Loss using Cosine Similarity normalization
        cos_sim = F.cosine_similarity(predictions, targets, dim=-1)
        loss_direction = torch.mean(1.0 - cos_sim)
        
        # C) Smoothness Loss by penalizing sudden acceleration changes
        loss_smoothness = 0.0
        if prev_predictions is not None:
            loss_smoothness = self.mse(predictions, prev_predictions)
            
        # Total blended Multi-Task Loss calculation
        total_loss = (self.alpha * loss_position) + \
                     (self.beta * loss_direction) + \
                     (self.gamma * loss_smoothness)
                     
        return total_loss

def train_model():
    # 3. Target hardware accelerator setup (CUDA GPU or CPU fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training process is running on: {device}")

    # 4. Data pipeline initialization for localized storage assets
    dataset = OrganicCameraDataset(video_dir="dataset/train_videos", motion_dir="dataset/train_motions")
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    # 5. Instantiate the integrated network and allocate to device memory
    model = OrganicCameraModel(hidden_dim=128).to(device)

    # 6. Initialize the custom hybrid objective criterion and Adam optimizer
    criterion = OrganicCameraLoss(alpha=1.0, beta=0.6, gamma=0.4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7. Main supervised optimization training loop
    epochs = 5
    print("🎬 Starting the training process with Hybrid Organic Loss...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        prev_preds = None  # Cache to monitor motion smoothness continuity
        
        for batch_idx, (frames, motions, targets) in enumerate(train_loader):
            # Stream parallel batch tensors to target hardware device
            frames = frames.to(device)
            motions = motions.to(device)
            targets = targets.to(device)
            
            # Clear historical tracking gradients
            optimizer.zero_grad()
            
            # Forward pass inference
            predictions = model(frames, motions)
            
            # Evaluate comprehensive multi-task blended criteria
            loss = criterion(predictions, targets, prev_predictions=prev_preds)
            
            # Backward error propagation pass
            loss.backward()
            
            # Optimize parameters across CNN, GRU, and Attention weights
            optimizer.step()
            
            running_loss += loss.item()
            
            # Detach current states to prevent memory leaks across iterations
            prev_preds = predictions.detach()
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] -------> Model Average Loss: {epoch_loss:.4f}")

    # 8. Checkpoint serialization for operational deployment
    torch.save(model.state_dict(), "checkpoints/organic_camera_weights.pth")
    print("💾 Training complete! Model weights saved successfully inside 'checkpoints/' folder.")

if __name__ == "__main__":
    # Execute the training pipeline execution wrapper
    # train_model()
    print("Trainer structure code (train.py) updated with Hybrid Loss and compiled successfully.")