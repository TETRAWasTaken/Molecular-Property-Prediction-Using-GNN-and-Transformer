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


def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, nan_mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates Mean Squared Error, ignoring missing target values (NaNs).
    
    Args:
        predictions (torch.Tensor): The model's predictions [batch_size, num_targets]
        targets (torch.Tensor): The true targets, with NaNs replaced by 0.0 [batch_size, num_targets]
        nan_mask (torch.Tensor): Binary mask (1 for valid, 0 for missing) [batch_size, num_targets]
        
    Returns:
        torch.Tensor: The scalar loss value, safely averaged over valid entries only.
    """
    #Calculate raw unreduced MSE
    loss_fn = nn.MSELoss(reduction='none')
    raw_loss = loss_fn(predictions, targets)
    
    #Zero out errors for missing data
    masked_loss = raw_loss * nan_mask
    
    #Safely average over the number of actual, valid data points in the batch
    valid_entries = nan_mask.sum()
    if valid_entries > 0:
        return masked_loss.sum() / valid_entries
    else:
        #For empty batches(if any)
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)


def train_transformer(model: nn.Module, train_loader: Any, val_loader: Any, 
                      epochs: int = 10, lr: float = 5e-5, save_path: str = "best_chemberta.pth"):
    """
    Main training engine with validation tracking and best-model saving.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    #AdamW is the standard optimizer for Transformer architectures
    optimizer = AdamW(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        #TRAINING PHASE
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            #Move batch arrays to GPU/CPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            nan_mask = batch['nan_mask'].to(device)
            
            optimizer.zero_grad()
            
            #Forward pass & Loss
            predictions = model(input_ids, attention_mask)
            loss = masked_mse_loss(predictions, labels, nan_mask)
            
            #Backpropagation
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        #VALIDATION PHASE
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad(): #Disable gradient tracking for speed and memory efficiency.
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                nan_mask = batch['nan_mask'].to(device)
                
                predictions = model(input_ids, attention_mask)
                loss = masked_mse_loss(predictions, labels, nan_mask)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        #Save the model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path}")
