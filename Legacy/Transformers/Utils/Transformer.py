import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch.optim import AdamW
from typing import Dict, Any

class AttentionPooling(nn.Module):
    """
    Learns to dynamically weight the importance of each token in the sequence.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_scorer = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Calculate raw importance scores: [batch, seq_len]
        scores = self.attention_scorer(hidden_states).squeeze(-1)
        
        # Mask out padding tokens so Softmax pushes their weight to 0.0
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Convert scores to probabilities
        attn_weights = F.softmax(scores, dim=-1) 
        
        # Multiply each hidden state by its weight and sum them up
        pooled_output = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return pooled_output

class StandaloneChemBERTa(nn.Module):
    """
    A Transformer model for predicting multiple continuous targets from SMILES strings,
    upgraded with Attention Pooling.
    """
    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1", num_targets: int = 12):
        super().__init__()
        
        # 1. THE BODY
        self.transformer = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.transformer.config.hidden_size # Typically 768
        
        # 2. THE NECK (NEW: Attention Pooling)
        self.attention_pool = AttentionPooling(self.hidden_size)
        
        # 3. THE HEAD
        self.prediction_head = nn.Linear(self.hidden_size, num_targets)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer, Attention Pooling, and prediction head.
        """
        # Pass tokens through the 6 Transformer layers
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # outputs.last_hidden_state shape: [batch_size, seq_len, hidden_size]
        sequence_hidden_states = outputs.last_hidden_state 
        
        # NEW: Apply Attention Pooling instead of just grabbing the [CLS] token
        pooled_embedding = self.attention_pool(sequence_hidden_states, attention_mask)
        
        # Generate final predictions
        predictions = self.prediction_head(pooled_embedding)
        
        return predictions