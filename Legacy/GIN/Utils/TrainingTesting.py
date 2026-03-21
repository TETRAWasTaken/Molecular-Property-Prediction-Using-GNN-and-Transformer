import torch
import copy
from typing import Dict
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
import sys
import os

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from Legacy.GIN.Utils.GIN import GIN

class TrainingTesting(GIN):
    def __init__(self, 
                 node_in_dim: int, 
                 edge_in_dim: int, 
                 hidden_dim: int = 128, 
                 output_dim: int = 12, 
                 dropout: float = 0.2,
                 learning_rate: float = 3e-4,
                 weight_decay: float = 5e-4,
                 device: str = "mps",
                 target_mean: torch.Tensor = None,
                 target_std: torch.Tensor = None):
        
        super().__init__(node_in_dim, edge_in_dim, hidden_dim, output_dim, num_layer=3, dropout=dropout)
        
        self.device_name = device
        self.to(self.device_name)

        # Keep normalization stats on the same device for denormalized metrics.
        if target_mean is not None and target_std is not None:
            self.target_mean = target_mean.detach().clone().float().to(self.device_name)
            self.target_std = target_std.detach().clone().float().to(self.device_name)
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

    def train_epoch(self, loader: DataLoader) -> float:
        self.train() 
        total_loss = 0.0
        
        for batch in loader:
            batch = batch.to(self.device_name)
            self.optimizer.zero_grad()
            
            pred = self(batch)
            
            # batch.y is already normalized in preprocessing; optimize in normalized space.
            mask = batch.mask
            loss = (self.criterion(pred, batch.y) * mask).sum() / mask.sum()

            if not torch.isfinite(loss):
                raise RuntimeError("Encountered non-finite training loss. Lower learning rate or inspect data.")

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.eval() 
        preds_list = []
        true_list = []
        mask_list = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device_name)
                pred_norm = self(batch)
                
                preds_list.append(pred_norm.cpu())
                true_list.append(batch.y.cpu())
                mask_list.append(batch.mask.cpu())
                
        y_pred_norm = torch.cat(preds_list, dim=0)
        y_true_norm = torch.cat(true_list, dim=0)
        y_mask = torch.cat(mask_list, dim=0)

        # Validation optimization metric should stay in normalized space.
        val_loss = ((torch.abs(y_pred_norm - y_true_norm) * y_mask).sum() / y_mask.sum()).item()

        if self.target_mean is None or self.target_std is None:
            # Fallback to normalized metrics if denorm stats are unavailable.
            valid_idx = y_mask == 1
            y_pred_eval = y_pred_norm[valid_idx].numpy()
            y_true_eval = y_true_norm[valid_idx].numpy()
        else:
            mean_cpu = self.target_mean.detach().cpu().view(1, -1)
            std_cpu = self.target_std.detach().cpu().view(1, -1)
            y_pred_denorm = y_pred_norm * std_cpu + mean_cpu
            y_true_denorm = y_true_norm * std_cpu + mean_cpu
            valid_idx = y_mask == 1
            y_pred_eval = y_pred_denorm[valid_idx].numpy()
            y_true_eval = y_true_denorm[valid_idx].numpy()
        
        return {
            "val_loss": val_loss,
            "mae": mean_absolute_error(y_true_eval, y_pred_eval),
            "r2": r2_score(y_true_eval, y_pred_eval)
        }

    def run_training(self, 
                     train_loader: DataLoader, 
                     val_loader: DataLoader, 
                     epochs: int = 50, 
                     patience: int = 20):
        """Full training loop with Early Stopping."""
        
            
        best_val_mae = float('inf')
        early_stop_counter = 0
        best_weights = None
        
        print(f"Starting Training on {self.device_name} for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            val_mae = metrics["mae"]
            val_loss = metrics["val_loss"]
            
            self.scheduler.step(val_loss)
            
            print(
                f"Epoch {epoch:03d} | Train Loss (norm): {loss:.4f} "
                f"| Val Loss (norm): {val_loss:.4f} | Val MAE (denorm): {val_mae:.4f}"
            )
            
            # Early Stopping Check
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_weights = copy.deepcopy(self.state_dict())
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        # Restore best model
        if best_weights:
            self.load_state_dict(best_weights)
            print("Restored best model weights.")