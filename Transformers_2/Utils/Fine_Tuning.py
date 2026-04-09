from matplotlib.pylab import beta
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Any, Optional
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from .Transformer import StandaloneChemBERTa

class FineTuning(StandaloneChemBERTa):
    """
    Fine-tuning wrapper for ChemBERTa with custom training loop and per-property validation tracking.
    """

    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1", num_labels: int = 12):
        super().__init__(model_name=model_name, num_targets=num_labels)

    def train_transformer(self, train_loader: Any, val_loader: Any,
                          epochs: int = 15, lr: float = 5e-5, save_path: str = "best_chemberta.pth",
                          device: Optional[torch.device] = None, 
                          target_cols: list = None, scalers: dict = None):
        """
        Main training engine with per-property validation tracking and best-model saving.
        """
        if len(train_loader) == 0:
            raise ValueError("train_loader is empty. Please provide at least one training batch.")
        if len(val_loader) == 0:
            raise ValueError("val_loader is empty. Please provide at least one validation batch.")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        optimizer = AdamW(self.parameters(), lr=lr)
        best_val_loss = float('inf')
        
        print(f"Starting training on {device}...")
        
        for epoch in range(epochs):
            # --- TRAINING PHASE ---
            self.train()
            total_train_loss = 0.0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                nan_mask = batch['nan_mask'].to(device)
                
                optimizer.zero_grad()
                
                predictions = self(input_ids, attention_mask)
                loss = self.masked_mse_loss(predictions, labels, nan_mask)
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
            avg_train_loss = total_train_loss / len(train_loader)
            
            # --- VALIDATION PHASE ---
            self.eval()
            total_val_loss = 0.0
            
            all_preds = []
            all_targets = []
            all_masks = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    nan_mask = batch['nan_mask'].to(device)
                    
                    predictions = self(input_ids, attention_mask)
                    loss = self.masked_mse_loss(predictions, labels, nan_mask)
                    total_val_loss += loss.item()
                    
                    # Store arrays for metric calculation
                    all_preds.append(predictions.cpu())
                    all_targets.append(labels.cpu())
                    all_masks.append(nan_mask.cpu())
                    
            avg_val_loss = total_val_loss / len(val_loader)
            
            # --- METRIC CALCULATION ---
            y_pred = torch.cat(all_preds, dim=0).numpy()
            y_true = torch.cat(all_targets, dim=0).numpy()
            y_mask = torch.cat(all_masks, dim=0).numpy()

            # Denormalize predictions and targets for accurate physical MAE
            if scalers and target_cols:
                for i, col in enumerate(target_cols):
                    if col in scalers:
                        y_pred[:, i] = scalers[col].inverse_transform(y_pred[:, i].reshape(-1, 1)).flatten()
                        y_true[:, i] = scalers[col].inverse_transform(y_true[:, i].reshape(-1, 1)).flatten()

            mae_per_prop = []
            r2_per_prop = []
            num_targets = y_true.shape[1]

            for i in range(num_targets):
                valid_idx = y_mask[:, i] == 1
                if valid_idx.sum() > 0:
                    y_p = y_pred[valid_idx, i]
                    y_t = y_true[valid_idx, i]
                    mae_per_prop.append(mean_absolute_error(y_t, y_p))
                    r2_per_prop.append(r2_score(y_t, y_p))
                else:
                    mae_per_prop.append(0.0)
                    r2_per_prop.append(0.0)

            overall_mae = sum(mae_per_prop) / num_targets
            overall_r2 = sum(r2_per_prop) / num_targets
            
            print(f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {overall_mae:.4f} | Val R²: {overall_r2:.4f}")
            
            # Print full breakdown on the first epoch, every 5 epochs, or if we hit a new best model
            if epoch == 0 or (epoch + 1) % 5 == 0 or avg_val_loss < best_val_loss:
                print("\n  --- Per-Property Validation Breakdown ---")
                for i in range(num_targets):
                    prop_name = target_cols[i] if target_cols else f"Target_{i}"
                    print(f"  {prop_name.ljust(10)} | MAE: {mae_per_prop[i]:.4f} | R²: {r2_per_prop[i]:.4f}")
                print("  -----------------------------------------\n")
            
            # Save the model if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.state_dict(), save_path)
                print(f"--> Saved new best model to {save_path}\n")

    @staticmethod
    def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, nan_mask: torch.Tensor, beta=1.0) -> torch.Tensor:
    # Use SmoothL1Loss instead of MSELoss
    # 'beta' controls the threshold where it switches from squared to absolute loss
        loss_fn = nn.SmoothL1Loss(reduction='none', beta=beta)
    
        raw_loss = loss_fn(predictions, targets)
        masked_loss = raw_loss * nan_mask
    
        valid_entries = nan_mask.sum()
        if valid_entries > 0:
            return masked_loss.sum() / valid_entries
        else:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)