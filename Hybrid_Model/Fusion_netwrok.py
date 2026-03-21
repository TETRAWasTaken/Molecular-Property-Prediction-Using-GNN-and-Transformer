import torch
import torch.nn as nn
from GIN_2.Utils.GIN import GIN
from Transformers_2.Utils.Transformer import StandaloneChemBERTa

class HybridFusionModel(nn.Module):
    def __init__(self, 
                 gin_hidden_dim: int = 256, 
                 transformer_model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
                 mlp_hidden_dim: int = 512,
                 output_dim: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        
        #Initialize Encoders
        self.graph_encoder = GIN(hidden_dim=gin_hidden_dim, output_dim=output_dim)
        self.text_encoder = StandaloneChemBERTa(model_name=transformer_model_name, num_targets=output_dim)
        
        #Convert Predictors to Encoders by replacing the final layers with Identity
        self.graph_encoder.prediction_head[-1] = nn.Identity() 
        self.text_encoder.prediction_head = nn.Identity()
        
        #Calculate fused dimension (256 + 768 = 1024)
        concat_dim = gin_hidden_dim + self.text_encoder.hidden_size
        
        #Unified Fusion Head
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.BatchNorm1d(mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(mlp_hidden_dim // 2, output_dim)
        )

    def forward(self, graph_data, input_ids, attention_mask):
        #Extract features
        graph_embedding = self.graph_encoder(graph_data)
        text_embedding = self.text_encoder(input_ids, attention_mask)
        
        #Concatenate and Predict
        fused_embedding = torch.cat([graph_embedding, text_embedding], dim=1)
        return self.fusion_mlp(fused_embedding)