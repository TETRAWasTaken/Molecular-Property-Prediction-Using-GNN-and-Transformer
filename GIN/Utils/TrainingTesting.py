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

from GIN.Utils.GIN import GIN

class TrainingTesting(GIN):
    def __init__(self, 
                 node_in_dim: int, 
                 edge_in_dim: int, 
                 hidden_dim: int = 128, 
                 output_dim: int = 12, 
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 weight_decay: float = 5e-4,
                 device: str = "mps"):
        
        super().__init__(node_in_dim, edge_in_dim, hidden_dim, output_dim, num_layer=3, dropout=dropout)
        
        self.device_name = device
        self.to(self.device_name)
        
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
            
            # Data is ALREADY normalized by preprocessing.py. Just calculate masked loss!
            mask = batch.mask
            loss = (self.criterion(pred, batch.y) * mask).sum() / mask.sum()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.eval() 
        preds_list = []
        true_list = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device_name)
                # For validation loss, we just compare the raw Z-scores to see if it's learning
                pred_norm = self(batch)
                
                preds_list.append(pred_norm.cpu())
                true_list.append(batch.y.cpu())
                
        y_pred = torch.cat(preds_list, dim=0).numpy()
        y_true = torch.cat(true_list, dim=0).numpy()
        
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }

    def run_training(self, 
                     train_loader: DataLoader, 
                     val_loader: DataLoader, 
                     epochs: int = 50, 
                     patience: int = 20):
        """Full training loop with Early Stopping."""
        
        if not self.is_fitted:
            self.fit_normalizer(train_loader)
            
        best_val_mae = float('inf')
        early_stop_counter = 0
        best_weights = None
        
        print(f"Starting Training on {self.device_name} for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            val_mae = metrics["mae"]
            
            self.scheduler.step(val_mae)
            
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val MAE: {val_mae:.4f}")
            
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