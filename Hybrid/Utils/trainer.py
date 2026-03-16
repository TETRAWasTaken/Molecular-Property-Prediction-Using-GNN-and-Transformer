import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from typing import Dict
from sklearn.metrics import mean_absolute_error, r2_score

from Hybrid.Models.fusion_network import HybridFusionNetwork

class HybridTrainer:
    """
    Manages the training, evaluation, and early stopping of the HybridFusionNetwork.
    Handles target normalization and custom masked loss for missing properties.
    """
    def __init__(self, 
                 model: HybridFusionNetwork,
                 learning_rate: float = 1e-4, 
                 weight_decay: float = 1e-4,
                 device: str = None):
        
        # 1. Setup Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                       "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        
        # 2. Normalizer State (To standardize the 12 target properties)
        self.mean_y = torch.zeros(12).to(self.device)
        self.std_y = torch.ones(12).to(self.device)
        self.is_fitted = False
        
        # 3. Optimization (AdamW is required here because of the ChemBERTa Transformer)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss(reduction='none') 
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)

    def fit_normalizer(self, train_loader):
        """Calculates Mean and Std of targets from training data, ignoring NaNs."""
        print("Computing dataset statistics for target normalization...")
        all_y = []
        all_mask = []
        
        for batch in train_loader:
            all_y.append(batch.y)
            all_mask.append(batch.mask)
            
        all_y = torch.cat(all_y, dim=0).to(self.device)
        all_mask = torch.cat(all_mask, dim=0).to(self.device)
        
        # Calculate mean and std using ONLY the valid entries (mask == 1)
        for i in range(all_y.shape[1]):
            valid_vals = all_y[:, i][all_mask[:, i] == 1]
            if len(valid_vals) > 0:
                self.mean_y[i] = valid_vals.mean()
                self.std_y[i] = valid_vals.std()
                if self.std_y[i] == 0:
                    self.std_y[i] = 1.0 # Prevent division by zero
                    
        self.is_fitted = True

    def masked_loss(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculates MSE safely by zeroing out errors where targets are missing."""
        raw_loss = self.criterion(predictions, targets)
        masked_loss = raw_loss * mask
        
        valid_entries = mask.sum()
        if valid_entries > 0:
            return masked_loss.sum() / valid_entries
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def train_epoch(self, loader) -> float:
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0
        
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward Pass: The model automatically unpacks the graph and tokens
            pred = self.model(batch)
            
            # Normalize Targets 
            y_real = batch.y
            y_norm = (y_real - self.mean_y) / self.std_y
            
            # Masked Loss Calculation
            loss = self.masked_loss(pred, y_norm, batch.mask)
            
            # Backprop
            loss.backward()
            
            # Gradient Clipping (Protects the Transformer weights from exploding)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def evaluate(self, loader) -> Dict[str, float]:
        """Evaluates model performance returning MAE and R2."""
        self.model.eval()
        preds_list = []
        true_list = []
        mask_list = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                
                # Predict
                pred_norm = self.model(batch)
                
                # Denormalize to get predictions back into real-world units
                pred_real = pred_norm * self.std_y + self.mean_y
                
                preds_list.append(pred_real.cpu())
                true_list.append(batch.y.cpu())
                mask_list.append(batch.mask.cpu())
                
        # Stack everything
        y_pred = torch.cat(preds_list, dim=0).numpy()
        y_true = torch.cat(true_list, dim=0).numpy()
        mask = torch.cat(mask_list, dim=0).numpy()
        
        # Filter arrays using the mask to calculate accurate metrics
        valid_y_true = y_true[mask == 1]
        valid_y_pred = y_pred[mask == 1]
        
        return {
            "mae": mean_absolute_error(valid_y_true, valid_y_pred),
            "r2": r2_score(valid_y_true, valid_y_pred)
        }

    def run_training(self, train_loader, val_loader, epochs: int = 50, patience: int = 15, save_path: str = "hybrid_model.pth"):
        """Full training loop with Early Stopping."""
        if not self.is_fitted:
            self.fit_normalizer(train_loader)
            
        best_val_mae = float('inf')
        early_stop_counter = 0
        best_weights = None
        
        print(f"\nStarting Hybrid Training on {self.device} for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            val_mae = metrics["mae"]
            
            self.scheduler.step(val_mae)
            
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss:.4f} | Val MAE: {val_mae:.4f}")
            
            # Early Stopping Check
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_weights = copy.deepcopy(self.model.state_dict())
                early_stop_counter = 0
                torch.save(best_weights, save_path)
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
        # Restore best model to the active instance
        if best_weights:
            self.model.load_state_dict(best_weights)
            print(f"Training complete. Best model weights restored and saved to {save_path}.")