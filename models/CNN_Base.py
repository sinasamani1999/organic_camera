import torch
import torch.nn as nn
import torchvision.models as models
import os

class VisualBackbone(nn.Module):
    def __init__(self, hidden_dim=128):
        """
        Completely offline loading of pre-trained ResNet-18 weights to extract game visual features.
        """
        super(VisualBackbone, self).__init__()
        
        # 1. Instantiate a raw ResNet model without pre-trained weights
        resnet = models.resnet18(weights=None)
        
        # 2. Resolve the weights file path dynamically relative to this script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "resnet18-f37072fd.pth")
        
        if os.path.exists(weights_path):
            # Load weights offline from system storage to memory safely
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            resnet.load_state_dict(state_dict)
            print("🚀 Official ResNet-18 weights successfully loaded 100% offline from models directory.")
        else:
            raise FileNotFoundError(
                f"❌ Error: Weights file not found! Checked path: '{weights_path}'. Ensure the file is inside the models folder."
            )
        
        # 3. Remove the final classification layers of ResNet (final AvgPool and FC layers)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # 4. Automatically reduce spatial dimensions to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 5. Linear Alignment Layer (Linear Embedding)
        self.fc_alignment = nn.Linear(512, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """
        x: Input game frames with dimensions [Batch_Size, 3, 224, 224]
        """
        # Pass the image through ResNet convolutional layers -> Output: [Batch_Size, 512, 7, 7]
        x = self.features(x)
        
        # Condense spatial dimensions -> Output: [Batch_Size, 512, 1, 1]
        x = self.global_pool(x)
        
        # Flatten the matrix into a 1D vector -> Output: [Batch_Size, 512]
        x = torch.flatten(x, 1)
        
        # Align dimensions via the linear layer -> Output: [Batch_Size, hidden_dim]
        x = self.fc_alignment(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

# ====== Test code to validate visual layer output dimensions ======
if __name__ == "__main__":
    dummy_frames = torch.randn(4, 3, 224, 224)
    
    try:
        visual_model = VisualBackbone(hidden_dim=128)
        output_features = visual_model(dummy_frames)
        
        print("\n--- Execution Successful ---")
        print(f"Input tensor size: {dummy_frames.shape}")
        print(f"Exported tensor size (ready for attention layer): {output_features.shape}")
        
    except Exception as e:
        print(f"Execution Failed: {e}")