import torch
import torch.nn as nn
from .gin_encoder import GraphEncoder
from .text_encoder import TextEncoder

class HybridFusionNetwork(nn.Module):
    """
    The unified multimodal network.
    Combines 3D topological features from a GNN with 1D global sequence features 
    from a Transformer to predict multiple continuous molecular properties.
    """
    def __init__(self, 
                 node_in_dim: int = 6, 
                 edge_in_dim: int = 3, 
                 gin_hidden_dim: int = 128, 
                 output_dim: int = 12, 
                 dropout: float = 0.2,
                 transformer_model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
        super(HybridFusionNetwork, self).__init__()
        
        # 1. Initialize both feature extraction engines
        self.graph_encoder = GraphEncoder(
            node_in_dim=node_in_dim, 
            edge_in_dim=edge_in_dim, 
            hidden_dim=gin_hidden_dim,
            dropout=dropout
        )
        self.text_encoder = TextEncoder(model_name=transformer_model_name)
        
        # 2. Calculate the combined size (128 + 768 = 896)
        combined_dim = gin_hidden_dim + self.text_encoder.out_dim
        
        # 3. The Master Prediction Head
        # Takes the 896-dim fused vector and maps it to the 12 target properties
        self.prediction_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward(self, data) -> torch.Tensor:
        """
        Executes the hybrid forward pass.
        
        Args:
            data: A PyTorch Geometric Data batch object that HAS BEEN INJECTED 
                  with `input_ids` and `attention_mask` attributes during preprocessing.
                  
        Returns:
            torch.Tensor: 12 property predictions of shape [batch_size, 12]
        """
        # Step A: Get 3D structural math (Output: [batch_size, 128])
        graph_embedding = self.graph_encoder(data)
        
        # Step B: Get global text sequence context (Output: [batch_size, 768])
        # We access the injected text attributes directly from the PyG data object
        text_embedding = self.text_encoder(input_ids=data.input_ids, 
                                           attention_mask=data.attention_mask)
        
        # Step C: Late Fusion (Concatenation on the feature dimension)
        # Resulting Shape: [batch_size, 896]
        fused_embedding = torch.cat([graph_embedding, text_embedding], dim=1)
        
        # Step D: Final regression output
        predictions = self.prediction_head(fused_embedding)
        
        return predictions