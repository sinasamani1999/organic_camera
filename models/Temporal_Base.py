import torch
import torch.nn as nn

class TemporalBackbone(nn.Module):
    def __init__(self, input_dim=2, hidden_size=64, hidden_dim=128):
        """
        input_dim: Input dimension per frame containing hand movement [Delta_X, Delta_Y] (value is 2).
        hidden_size: Number of internal features/neurons in the GRU cell to retain motion memory.
        hidden_dim: Dimension of the final output vector, which must be aligned to 128 to match the visual component.
        """
        super(TemporalBackbone, self).__init__()
        
        # 1. Define the Recurrent Neural Network (GRU)
        # batch_first=True implies input dimensions will be [Batch_Size, Sequence_Length, Input_Dim]
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        
        # 2. Linear Alignment Layer
        # Transforms the GRU memory output (e.g., 64) to 128 dimensions for compatibility with the attention layer
        self.fc_alignment = nn.Linear(hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """
        x: Hand movement sequence from past frames with dimensions [Batch_Size, Sequence_Length, 2]
        Sequence_Length represents the number of historical frames the model looks back at as memory (e.g., past 10 frames).
        """
        # Pass data through the recurrent GRU cell
        # gru_out contains outputs from all time steps
        gru_out, _ = self.gru(x)
        
        # We only need the output of the last frame (the final time step in the sequence) to make the decision
        last_step_out = gru_out[:, -1, :] # Dimension transforms to [Batch_Size, hidden_size]
        
        # Align the dimensions of the motion vector using the linear layer -> Output: [Batch_Size, hidden_dim]
        out = self.fc_alignment(last_step_out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

# ====== Test code to validate temporal layer output dimensions ======
if __name__ == "__main__":
    # Simulate motion data:
    # Assume we have 4 samples (Batch) where each contains the player's mouse movement sequence over the past 10 frames (Sequence Length)
    # and for each frame, the mouse displacement [Mouse_X, Mouse_Y] is recorded.
    batch_size = 4
    sequence_length = 10
    input_features = 2 # Delta X, Delta Y
    
    dummy_motion = torch.randn(batch_size, sequence_length, input_features)
    
    # Instantiate the temporal model with an output dimension of 128
    temporal_model = TemporalBackbone(input_dim=2, hidden_size=64, hidden_dim=128)
    
    # Process the motion sequence through the model
    output_motion_features = temporal_model(dummy_motion)
    
    print("--- Temporal GRU layer test completed successfully ---")
    print(f"Input hand motion tensor shape (past 10 frames): {dummy_motion.shape}")
    print(f"Output temporal feature vector shape (ready for attention layer): {output_motion_features.shape}")
    # The output should be [4, 128], matching exactly with the output size of the ResNet visual layer.