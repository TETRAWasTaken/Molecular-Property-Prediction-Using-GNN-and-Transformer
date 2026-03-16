import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Any

from .Tokeniser import Tokeniser
from .Transformer import StandaloneChemBERTa
from .paths import Paths

class FineTuning(StandaloneChemBERTa):
    """
    Fine-tuning wrapper for ChemBERTa with custom training loop and validation tracking.
    Args:
        model_name (str): Name of the pre-trained ChemBERTa model to load.
        num_labels (int): Number of output labels for the regression task.
    """

    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1", num_labels: int = 12):
        super().__init__(model_name=model_name, num_targets=num_labels)
        self.paths = Paths()
        self.tokeniser = Tokeniser(
            qm8_path=self.paths.get_qm8_path(),
            qm9_path=self.paths.get_qm9_path(),
            model_name=model_name,
        )

    def train_transformer(self, train_loader: Any, val_loader: Any,
                          epochs: int = 10, lr: float = 5e-5, save_path: str = "best_chemberta.pth"):
        """
        Main training engine with validation tracking and best-model saving.
        """
        if len(train_loader) == 0:
            raise ValueError("train_loader is empty. Please provide at least one training batch.")
        if len(val_loader) == 0:
            raise ValueError("val_loader is empty. Please provide at least one validation batch.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        #AdamW is the standard optimizer for Transformer architectures
        optimizer = AdamW(self.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        
        print(f"Starting training on {device}...")
        
        for epoch in range(epochs):
            #TRAINING PHASE
            self.train()
            total_train_loss = 0.0
            
            for batch in train_loader:
                #Move batch arrays to GPU/CPU
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                nan_mask = batch['nan_mask'].to(device)
                
                optimizer.zero_grad()
                
                #Forward pass & Loss
                predictions = self(input_ids, attention_mask)
                loss = self.masked_mse_loss(predictions, labels, nan_mask)
                
                #Backpropagation
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
            avg_train_loss = total_train_loss / len(train_loader)
            
            #VALIDATION PHASE
            self.eval()
            total_val_loss = 0.0
            
            with torch.no_grad(): #Disable gradient tracking for speed and memory efficiency.w
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    nan_mask = batch['nan_mask'].to(device)
                    
                    predictions = self(input_ids, attention_mask)
                    loss = self.masked_mse_loss(predictions, labels, nan_mask)
                    total_val_loss += loss.item()
                    
            avg_val_loss = total_val_loss / len(val_loader)
            
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            #Save the model if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.state_dict(), save_path)
                print(f"--> Saved new best model to {save_path}")


    
    @staticmethod
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
        