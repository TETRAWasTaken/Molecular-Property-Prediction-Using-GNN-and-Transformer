import torch
import copy
from typing import Dict
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
import sys
import os
from typing import Dict, Any

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from GIN_2.Utils.GIN import GIN

class TrainingTesting(GIN):
    def __init__(self, 
                 node_in_dim: int = 26,   # Updated to match the new 26-dim RDKit features
                 edge_in_dim: int = 6,    # Updated to match 5 bond features + 1 3D distance
                 hidden_dim: int = 256,   # Increased for higher capacity
                 output_dim: int = 12, 
                 dropout: float = 0.0,    # Set to 0.0 initially to prevent underfitting
                 num_layers: int = 7,     # Updated to 7 (Assuming you implemented Jumping Knowledge)
                 learning_rate: float = 3e-4,
                 weight_decay: float = 1e-5,
                 device: str = "mps",
                 target_mean: torch.Tensor = None,
                 target_std: torch.Tensor = None):
        
        # Pass the updated hyperparams to the parent GIN class
        super().__init__(node_in_dim, edge_in_dim, hidden_dim, output_dim, num_layer=num_layers, dropout=dropout)
        
        self.device_name = device
        self.to(self.device_name)

        # Keep normalization stats on the same device for dynamic normalization.
        if target_mean is not None and target_std is not None:
            self.target_mean = target_mean.detach().clone().float().to(self.device_name)
            self.target_std = target_std.detach().clone().float().to(self.device_name)
            # Prevent division by zero if a target has 0 variance
            self.target_std[self.target_std == 0] = 1.0 
        else:
            self.target_mean = None
            self.target_std = None

        if learning_rate > 1e-3:
            raise ValueError(
                f"Learning rate {learning_rate} is too high for stable GIN training. "
                "Use a value <= 1e-3 (recommended 3e-4)."
            )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = torch.nn.SmoothL1Loss(reduction='none') 
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

    def _get_normalized_targets_and_mask(self, batch):
        """Helper to dynamically normalize targets and handle missing masks."""
        y_true = batch.y.float()
        
        # 1. Dynamically normalize targets if stats are provided
        if self.target_mean is not None and self.target_std is not None:
            y_true_norm = (y_true - self.target_mean) / self.target_std
        else:
            y_true_norm = y_true
            
        # 2. Fallback for mask (if batch.mask doesn't exist, assume all 1s)
        mask = getattr(batch, 'mask', torch.ones_like(y_true_norm))
        
        return y_true_norm, mask

    def inverse_transform(self, scaled_tensor: torch.Tensor) -> torch.Tensor:
        """
        De-normalizes a tensor of predictions or targets using the stored
        mean and standard deviation.
        """
        if self.target_mean is None or self.target_std is None:
            return scaled_tensor
        
        # Ensure stats are on the same device as the input tensor
        mean = self.target_mean.to(scaled_tensor.device)
        std = self.target_std.to(scaled_tensor.device)
        
        return scaled_tensor * std + mean

    def train_epoch(self, loader: DataLoader) -> float:
        self.train() 
        total_loss = 0.0
        
        for batch in loader:
            batch = batch.to(self.device_name)
            self.optimizer.zero_grad()
            
            pred = self(batch)
            y_true_norm, mask = self._get_normalized_targets_and_mask(batch)
            
            # Calculate masked loss
            loss = (self.criterion(pred, y_true_norm) * mask).sum() / mask.sum()

            if not torch.isfinite(loss):
                raise RuntimeError("Encountered non-finite training loss. Lower learning rate or inspect data.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        self.eval() 
        preds_list = []
        true_list = []
        mask_list = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device_name)
                pred_norm = self(batch)
                y_true_norm, mask = self._get_normalized_targets_and_mask(batch)
                
                preds_list.append(pred_norm.cpu())
                true_list.append(y_true_norm.cpu())
                mask_list.append(mask.cpu())
                
        y_pred_norm = torch.cat(preds_list, dim=0)
        y_true_norm = torch.cat(true_list, dim=0)
        y_mask = torch.cat(mask_list, dim=0)

        # Validation optimization metric (Normalized Loss)
        val_loss = ((torch.abs(y_pred_norm - y_true_norm) * y_mask).sum() / y_mask.sum()).item()

        # Denormalize for physical metrics (MAE and R2)
        y_pred_denorm = self.inverse_transform(y_pred_norm.to(self.device_name)).cpu()
        y_true_denorm = self.inverse_transform(y_true_norm.to(self.device_name)).cpu()
            
        mae_per_prop = []
        r2_per_prop = []
        
        num_targets = y_true_denorm.shape[1]
        for i in range(num_targets):
            # Isolate the predictions and targets for this specific column
            valid_idx = y_mask[:, i] == 1
            if valid_idx.sum() > 0:
                y_p = y_pred_denorm[valid_idx, i].numpy()
                y_t = y_true_denorm[valid_idx, i].numpy()
                mae_per_prop.append(mean_absolute_error(y_t, y_p))
                r2_per_prop.append(r2_score(y_t, y_p))
            else:
                mae_per_prop.append(0.0)
                r2_per_prop.append(0.0)
                
        return {
            "val_loss": val_loss,
            "mae": sum(mae_per_prop) / num_targets, # Overall average MAE
            "r2": sum(r2_per_prop) / num_targets,   # Overall average R2
            "mae_per_prop": mae_per_prop,           # List of 12 MAEs
            "r2_per_prop": r2_per_prop              # List of 12 R2s
        }

    def run_training(self, train_loader: DataLoader, val_loader: DataLoader, 
                     epochs: int = 50, patience: int = 20, target_cols: list = None):
        best_val_mae = float('inf')
        early_stop_counter = 0
        best_weights = None
        best_epoch = 0
        
        print(f"Starting Training on {self.device_name} for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            val_mae = metrics["mae"]
            val_loss = metrics["val_loss"]
            
            self.scheduler.step(val_loss)
            
            print(
                f"Epoch {epoch:03d} | Train Loss (norm): {loss:.4f} "
                f"| Val Loss (norm): {val_loss:.4f} | Val MAE (denorm): {val_mae:.4f} | Val R²: {metrics['r2']:.4f}"
            )
            
            # --- NEW: Print Per-Property Breakdown cleanly ---
            # Triggers on the 1st epoch, every 10th epoch, or if it finds a new "Best Model"
            if epoch == 1 or epoch % 10 == 0 or val_mae < best_val_mae:
                print("\n  --- Per-Property Validation Breakdown ---")
                for i in range(len(metrics["mae_per_prop"])):
                    prop_name = target_cols[i] if target_cols else f"Target_{i}"
                    print(f"  {prop_name.ljust(10)} | MAE: {metrics['mae_per_prop'][i]:.4f} | R²: {metrics['r2_per_prop'][i]:.4f}")
                print("  -----------------------------------------\n")
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                best_weights = copy.deepcopy(self.state_dict())
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
        if best_weights:
            self.load_state_dict(best_weights)
            print(f"Restored best model weights from Epoch {best_epoch}.")