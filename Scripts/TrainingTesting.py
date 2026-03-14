import torch
import copy
from typing import Dict
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from GIN import GIN 

class TrainingTesting(GIN):
    """
    A self-training GNN class. 
    Inherits from GIN, meaning it IS a model, but has methods to train itself.
    """
    def __init__(self, 
                 node_in_dim: int, 
                 edge_in_dim: int, 
                 hidden_dim: int = 128, 
                 output_dim: int = 12, 
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 weight_decay: float = 5e-4,
                 device: str = "cuda"):
        
        # 1. Initialize the GIN Architecture (Superclass)
        super().__init__(node_in_dim, edge_in_dim, hidden_dim, output_dim, num_layer=3, dropout=dropout)
        
        # 2. Setup Device
        self.device_name = device if torch.cuda.is_available() else "cpu"
        self.to(self.device_name)
        
        # 3. Optimization Components
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = torch.nn.SmoothL1Loss(reduction='none') # Per-element loss for masking
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # 4. Normalizer State (Will be fitted later)
        self.register_buffer("mean_y", torch.zeros(output_dim))
        self.register_buffer("std_y", torch.ones(output_dim))
        self.is_fitted = False

    def fit_normalizer(self, loader: DataLoader):
        """Calculates Mean/Std of targets from training data for Z-score normalization."""
        all_y = []
        print("Computing dataset statistics for normalization...")
        for batch in loader:
            all_y.append(batch.y)
        all_y = torch.cat(all_y, dim=0)
        
        self.mean_y = torch.mean(all_y, dim=0).to(self.device_name)
        self.std_y = torch.std(all_y, dim=0).to(self.device_name)
        self.std_y[self.std_y == 0] = 1.0 # Handle constants
        self.is_fitted = True

    def train_epoch(self, loader: DataLoader) -> float:
        """Runs one epoch of training."""
        self.train() # Set GIN to training mode (enable dropout)
        total_loss = 0.0
        
        for batch in loader:
            batch = batch.to(self.device_name)
            self.optimizer.zero_grad()
            
            # Forward Pass (calls GIN.forward)
            pred = self(batch)
            
            # Normalize Targets (Crucial for convergence)
            y_real = batch.y
            y_norm = (y_real - self.mean_y) / self.std_y
            
            # Masked Loss Calculation (only on available targets)
            mask = batch.mask
            loss = (self.criterion(pred, y_norm) * mask).sum() / mask.sum()
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluates model performance (MAE, R2)."""
        self.eval() # Set GIN to eval mode
        preds_list = []
        true_list = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device_name)
                
                # Predict
                pred_norm = self(batch)
                
                # Denormalize to get real units
                pred_real = pred_norm * self.std_y + self.mean_y
                
                preds_list.append(pred_real.cpu())
                true_list.append(batch.y.cpu())
                
        # Metrics
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