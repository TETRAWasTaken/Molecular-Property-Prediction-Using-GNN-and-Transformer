import torch
import torch.nn as nn
from functorch.dim import Tensor
from transformers import AutoModel, PreTrainedModel
from torch import tensor

class Transformer(nn.Module):
    """
    The Transformer class implements a neural network module inspired by the 
    Transformer architecture. This is primarily used in natural language 
    processing and other sequence modeling tasks. It consists of various 
    layers and sub-layers that allow it to model complex relationships in 
    sequential data.

    :ivar layer_stack: A stack of neural network layers used within the 
        Transformer model to process input data.
    :type layer_stack: nn.ModuleList
    :ivar embedding_size: The size of the embedding vector for each token in 
        the input sequence.
    :type embedding_size: int
    """
    
    def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MTR",
                 freeze_weights: bool = True):
        super(Transformer, self).__init__()

        self.transformer: PreTrainedModel = AutoModel.from_pretrained(model_name)
        self.out_dim = self.transformer.config.hidden_size
        
        if freeze_weights:
            self.freeze()
    
    def freeze(self):
        """
        Freezes the weights of the Transformer model.
        :return: 
        """
        for param in self.transformer.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """
        Unfreezes the weights of the Transformer model.
        :return: 
        """
        for param in self.transformer.parameters():
            param.requires_grad = True
            
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Forward pass through the Transformer model.

        :param input_ids: Tokenised SMILES sequences [batch_size, seq_len]
        :param attention_mask: Padding masks [batch_size, seq_len]
        :return Tensor: The [CLS] token embedding of shape [batch_size, out_dim]
        """
        outputs = self.transformer(input_ids=input_ids, 
                                   attention_mask=attention_mask)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding