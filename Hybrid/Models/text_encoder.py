import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    """
    Sequence feature extractor based on pre-trained ChemBERTa.
    Outputs a highly contextualized dense vector representing the 1D SMILES string.
    """
    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
        super(TextEncoder, self).__init__()
        
        # Load the raw Transformer engine (No classification head attached)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Expose the hidden dimension size (usually 768) so the Fusion module can see it
        self.out_dim = self.transformer.config.hidden_size 

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Passes tokenized SMILES through Transformer layers.
        Returns:
            torch.Tensor: The [CLS] token embedding of shape [batch_size, 768]
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Slice out the [CLS] token (Index 0) to get the global summary sequence
        cls_embedding = outputs.last_hidden_state[:, 0, :] 
        return cls_embedding