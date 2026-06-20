import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, List

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GIN_2.Utils.preprocessing import RelationalGeometryPipeline
from GIN_2.Utils.TrainingTesting import TrainingTesting
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from Scripts.qm9_delta import apply_qm9_delta_learning

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
            
            # Inverse scale the predictions to get them back to the delta-corrected space
            scaled_preds = model.inverse_transform(predictions)
            
            all_preds.append(scaled_preds.cpu())
            # The 'y' from the loader is already the delta-corrected target
            all_targets.append(batch.y.cpu())

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

if __name__ == "__main__":
    
    # --- Configuration ---
    TARGET_COLS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    MODEL_PATH = project_root / "models/GIN_model.pth"
    MOLECULE_CSV_PATH = project_root / "Dataset/New_QM9/molecule_properties.csv"
    
    # The pipeline will use its own cache path, so we just need the root
    PIPELINE_ROOT = str(project_root / "GIN_2/data")

    # --- Data Loading ---
    if not MOLECULE_CSV_PATH.exists():
        print(f"FATAL: Molecule CSV not found at {MOLECULE_CSV_PATH}")
        sys.exit(1)

    print("Initializing data pipeline to get test loader...")
    pipeline = RelationalGeometryPipeline(
        root=PIPELINE_ROOT, 
        mol_csv_path=str(MOLECULE_CSV_PATH),
        # Assuming atom_properties.csv is in the same directory
        atom_csv_path=str(MOLECULE_CSV_PATH).replace("molecule_properties.csv", "atom_properties.csv"),
        target_cols=TARGET_COLS
    )
    
    # The get_loaders method provides train/val/test splits
    _, _, test_loader = pipeline.get_loaders(batch_size=64)
    
    print(f"Loaded test loader with {len(test_loader.dataset)} samples.")

    # --- Model Initialization and Evaluation ---
    if test_loader and MODEL_PATH.exists():
        # Calculate stats from the original CSV for model initialization
        df_mol = pd.read_csv(MOLECULE_CSV_PATH)
        # Apply the same delta learning to the stats calculation
        df_mol_delta = apply_qm9_delta_learning(df_mol.copy(), smiles_col='smiles', target_cols=TARGET_COLS)
        
        mean_vals = df_mol_delta[TARGET_COLS].mean().values
        std_vals = df_mol_delta[TARGET_COLS].std().values
        target_mean = torch.tensor(mean_vals, dtype=torch.float32)
        target_std = torch.tensor(std_vals, dtype=torch.float32)

        # Dynamically get model dimensions from the loaded data
        sample_batch = next(iter(test_loader))
        node_in_dim = sample_batch.num_node_features
        edge_in_dim = sample_batch.edge_attr.shape[1]

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
            sys.exit(1)

        print("\nStarting evaluation for all properties...")
        all_results = evaluate_all_properties(model, test_loader, TARGET_COLS, DEVICE)
        
        # Format results into a single pandas DataFrame
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df.index.name = 'Property'
        
        print("\n--- GIN Model Evaluation Results ---")
        print(results_df.to_string(float_format="%.4f"))
        print("------------------------------------\n")
        
    elif not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Skipping evaluation.")
    else:
        print("No data loaded. Skipping evaluation.")