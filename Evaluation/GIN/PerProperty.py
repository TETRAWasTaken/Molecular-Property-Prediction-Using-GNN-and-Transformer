import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, List

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GIN_2.Utils.preprocessing import RelationalGeometryPipeline
from GIN_2.Utils.TrainingTesting import TrainingTesting
from GIN_2.manual_run import get_dataset_stats

def evaluate_all_properties(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    target_cols: List[str],
    device: str
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates the GIN model on all properties.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch)
            
            # Inverse scale the predictions and targets
            scaled_preds = model.inverse_transform(predictions)
            scaled_targets = model.inverse_transform(batch.y)
            
            all_preds.append(scaled_preds.cpu())
            all_targets.append(scaled_targets.cpu())

    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()

    results = {}
    for i, col in enumerate(target_cols):
        p = y_pred[:, i]
        t = y_true[:, i]

        if len(t) < 2:
            rmse, mae, r2 = 0.0, 0.0, 0.0
        else:
            rmse = np.sqrt(mean_squared_error(t, p))
            mae = mean_absolute_error(t, p)
            r2 = r2_score(t, p)
            
        results[col] = {'RMSE': rmse, 'MAE': mae, 'R^2': r2}
            
    return results

def plot_results(results: Dict[str, float], property_name: str, save_dir: str):
    """
    Plots the evaluation results and saves them to a file.
    """
    metrics = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(8, 5))
    sns.barplot(x=metrics, y=values)
    plt.title(f'GIN Evaluation Metrics: {property_name}')
    plt.ylabel('Score')
    if values:
        plt.ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    plt.tight_layout()
    
    filename = f'{property_name}_evaluation.png'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
        
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # --- Configuration ---
    TARGET_COLS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
    MODEL_PATH = project_root / "models/GIN_model.pth"
    MOLECULE_CSV_PATH = project_root / "Dataset/New_QM9/molecule_properties.csv"
    ATOM_CSV_PATH = project_root / "Dataset/New_QM9/atom_properties.csv"
    SAVE_DIR = project_root / "evaluation_plots_gin"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data Loading ---
    test_loader = None
    node_in_dim, edge_in_dim = 0, 0
    
    if MOLECULE_CSV_PATH.exists() and ATOM_CSV_PATH.exists():
        print("Found dataset files. Preparing test dataloader...")
        
        target_mean, target_std = get_dataset_stats(str(MOLECULE_CSV_PATH), TARGET_COLS)
        
        pipeline = RelationalGeometryPipeline(
            root=str(project_root / 'GIN_2/data'), 
            mol_csv_path=str(MOLECULE_CSV_PATH),
            atom_csv_path=str(ATOM_CSV_PATH),
            target_cols=TARGET_COLS
        )
        
        _, _, test_loader = pipeline.get_loaders(batch_size=64)
        
        sample_batch = next(iter(test_loader))
        node_in_dim = sample_batch.num_node_features
        edge_in_dim = sample_batch.edge_attr.shape[1]
        
        print(f"Loaded test loader with {len(test_loader.dataset)} samples.")
    else:
        print("Dataset files not found. Cannot run evaluation.")

    # --- Model Initialization and Evaluation ---
    if test_loader and MODEL_PATH.exists():
        model = TrainingTesting(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=256,
            output_dim=len(TARGET_COLS),
            dropout=0.0,
            num_layers=7,
            device=DEVICE,
            target_mean=target_mean,
            target_std=target_std,
        )
        
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            print(f"Loaded GIN model state from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            exit()

        print("Starting evaluation for all properties...")
        all_results = evaluate_all_properties(model, test_loader, TARGET_COLS, DEVICE)
        
        for name, res in all_results.items():
            print(f"Results for {name}: {res}")
            plot_results(res, name, str(SAVE_DIR))
        
        print(f"\nEvaluation complete. Plots saved to: {SAVE_DIR}")
    elif not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Skipping evaluation.")
    else:
        print("No data loaded. Skipping evaluation.")
