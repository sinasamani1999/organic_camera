import torch
import torch.nn as nn
import math

class CrossAttention(nn.Module):
    def __init__(self, feature_dim=128):
        """
        feature_dim: Dimensionality of the input vectors (CNN and GRU), 
                     which we set to 128 in previous steps.
        """
        super(CrossAttention, self).__init__()
        
        # Define linear layers to construct Query, Key, and Value mechanisms
        self.q_linear = nn.Linear(feature_dim, feature_dim)
        self.k_linear = nn.Linear(feature_dim, feature_dim)
        self.v_linear = nn.Linear(feature_dim, feature_dim)
        
        # Softmax normalization layer to convert scores into probabilities (between 0 and 1)
        self.softmax = nn.Softmax(dim=-1)
        
        # Final output linear layer for fusion
        self.out_linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, visual_features, motion_features):
        """
        visual_features (CNN): [Batch_Size, feature_dim] -> Acts as Key and Value
        motion_features (GRU): [Batch_Size, feature_dim] -> Acts as Query
        """
        
        # 1. Generate Q, K, and V vectors
        # For cross-attention matrix multiplication, we need to unsqueeze to add a dummy sequence dimension
        Q = self.q_linear(motion_features).unsqueeze(1)  # Shape: [Batch_Size, 1, feature_dim]
        K = self.k_linear(visual_features).unsqueeze(1)  # Shape: [Batch_Size, 1, feature_dim]
        V = self.v_linear(visual_features).unsqueeze(1)  # Shape: [Batch_Size, 1, feature_dim]
        
        # 2. Calculate attention scores (Matrix multiplication of Query and Transpose of Key)
        # Divided by the square root of dimensions (Scaled Dot-Product) for gradient stability during training
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        
        # 3. Apply Softmax to obtain attention weights
        attention_weights = self.softmax(scores)
        
        # 4. Multiply attention weights by the Value vector to extract synchronized context
        context = torch.matmul(attention_weights, V)  # Shape: [Batch_Size, 1, feature_dim]
        
        # 5. Squeeze the dummy dimension and pass through the final linear layer
        context = context.squeeze(1)  # Shape: [Batch_Size, feature_dim]
        output = self.out_linear(context)
        
        return output

# ====== Test code to validate Cross-Attention layer output dimensions ======
if __name__ == "__main__":
    # Simulating outputs from step 2.1 and 2.2 (4 samples with 128-dimensional features)
    batch_size = 4
    dim = 128
    
    dummy_visual_out = torch.randn(batch_size, dim)  # Dummy CNN output
    dummy_motion_out = torch.randn(batch_size, dim)  # Dummy GRU output
    
    # Instantiate the Cross-Attention layer
    attention_layer = CrossAttention(feature_dim=128)
    
    # Intelligently fuse visual and temporal layers
    fused_features = attention_layer(dummy_visual_out, dummy_motion_out)
    
    print("--- Cross-Attention bridge layer test completed successfully ---")
    print(f"Input visual feature vector shape: {dummy_visual_out.shape}")
    print(f"Input temporal feature vector shape: {dummy_motion_out.shape}")
    print(f"Output fused feature vector shape (Fused Features): {fused_features.shape}")
    # The output should be [4, 128], indicating that vision and motion features have been intelligently fused.