import torch
import torch.nn as nn
from transformers import AutoModel
from torch.optim import AdamW
from typing import Dict, Any

class StandaloneChemBERTa(nn.Module):
    """
    A Transformer model for predicting multiple continuous targets from SMILES strings.
    
    This class is designed in two parts:
    1. The Body: A pre-trained ChemBERTa Transformer that generates a 768-dim embedding.
    2. The Head: A 12-neuron linear layer that translates the embedding into predictions.
    
    In the future hybrid architecture, the `prediction_head` will be removed, and 
    the 768-dim `[CLS]` embedding will be concatenated with a graph neural network output.
    
    Args:
        model_name (str): The Hugging Face model path.
        num_targets (int): The number of output properties to predict.
    """
    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1", num_targets: int = 12):
        super().__init__()
        
        #THE BODY
        #Loads the raw Transformer engine (without a classification head)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.transformer.config.hidden_size # Typically 768
        
        #THE HEAD
        #A linear layer mapping the 768-dim (CLS) token to the 12 target values
        self.prediction_head = nn.Linear(self.hidden_size, num_targets)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer and prediction head.
        
        Args:
            input_ids (torch.Tensor): Padded token integers of shape [batch_size, seq_len]
            attention_mask (torch.Tensor): Padding mask of shape [batch_size, seq_len]
            
        Returns:
            torch.Tensor: The predicted properties of shape [batch_size, num_targets]
        """
        #Pass tokens through the 6 Transformer layers
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        #Extract the [CLS] token representation (index 0 of the sequence)
        #outputs.last_hidden_state shape: [batch_size, seq_len, hidden_size]
        cls_embedding = outputs.last_hidden_state[:, 0, :] 
        
        #Generate final predictions
        predictions = self.prediction_head(cls_embedding)
        
        return predictions