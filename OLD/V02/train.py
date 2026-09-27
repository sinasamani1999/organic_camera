import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import custom modules created in previous steps
from data_pipeline.Real_Camera_Dataset import OrganicCameraDataset
from models.Organic_Camera_Model import OrganicCameraModel

def train_model():
    # 1. Hardware setup for processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Training process is running on: {device}")

    # 2. Initialize the data pipeline for real data on disk
    # Replace these with your actual video and motion directory paths
    dataset = OrganicCameraDataset(video_dir="dataset/train_videos", motion_dir="dataset/train_motions")
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    # 3. Instantiate the integrated model and move it to the device (GPU/CPU)
    model = OrganicCameraModel().to(device)

    # 4. Define the loss function (Regression) and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Main training loop (starting with 5 experimental epochs)
    epochs = 5
    print("🎬 Starting the training process...")
    
    for epoch in range(epochs):
        model.train() # Set the model to training mode
        running_loss = 0.0
        
        for batch_idx, (frames, motions, targets) in enumerate(train_loader):
            # Move data batches to the selected device
            frames = frames.to(device)
            motions = motions.to(device)
            targets = targets.to(device)
            
            # Clear gradients from the previous step
            optimizer.zero_grad()
            
            # A) Forward pass -> Get model predictions
            predictions = model(frames, motions)
            
            # B) Calculate the loss for the current step
            loss = criterion(predictions, targets)
            
            # C) Backward pass -> Calculate gradients
            loss.backward()
            
            # D) Update network weights based on Attention and GRU calculations
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] -------> Model Average Loss: {epoch_loss:.4f}")

    # 6. Save the trained model weights after training finishes
    torch.save(model.state_dict(), "organic_camera_weights.pth")
    print("💾 Training complete! Model weights saved successfully: organic_camera_weights.pth")

if __name__ == "__main__":
    # Execute the training loop (Requires dataset folders to contain at least one video and CSV file)
    # train_model()
    print("Trainer structure code (train.py) compiled successfully and is ready to connect to data.")