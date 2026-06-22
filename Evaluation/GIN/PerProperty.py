import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, List, Tuple

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GIN_2.Utils.preprocessing import RelationalGeometryPipeline
from GIN_2.Utils.TrainingTesting import TrainingTesting
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from Scripts.qm9_delta import (
    apply_qm9_delta_learning,
    HARTREE_TO_EV,
    QM9_DELTA_TARGET_COLUMNS,
)

EV_TO_KCAL_MOL = 23.060548

def evaluate_all_properties(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    target_cols: List[str],
    device: str,
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch)
            scaled_preds = model.inverse_transform(predictions)
            all_preds.append(scaled_preds.cpu())
            all_targets.append(batch.y.cpu())

    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()

    results = {}
    for i, col in enumerate(target_cols):
        p, t = y_pred[:, i], y_true[:, i]
        metrics = {'RMSE': 0.0, 'MAE': 0.0, 'R^2': 0.0}
        if len(t) >= 2:
            metrics['RMSE'] = np.sqrt(mean_squared_error(t, p))
            metrics['MAE'] = mean_absolute_error(t, p)
            metrics['R^2'] = r2_score(t, p)
            if col in QM9_DELTA_TARGET_COLUMNS:
                error_kcal_mol = np.abs(t - p) * EV_TO_KCAL_MOL
                metrics['Chem. Acc. (%)'] = np.mean(error_kcal_mol <= 1.0) * 100.0
        results[col] = metrics
            
    return results, y_pred, y_true

if __name__ == "__main__":
    TARGET_COLS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = project_root / "models/GIN_model.pth"
    MOLECULE_CSV_PATH = project_root / "Dataset/New_QM9/molecule_properties.csv"
    PIPELINE_ROOT = str(project_root / "GIN_2/data")

    if not MOLECULE_CSV_PATH.exists():
        print(f"FATAL: Molecule CSV not found at {MOLECULE_CSV_PATH}")
        sys.exit(1)

    print("Initializing data pipeline...")
    pipeline = RelationalGeometryPipeline(
        root=PIPELINE_ROOT, mol_csv_path=str(MOLECULE_CSV_PATH),
        atom_csv_path=str(MOLECULE_CSV_PATH).replace("molecule_properties.csv", "atom_properties.csv"),
        target_cols=TARGET_COLS
    )
    _, _, test_loader = pipeline.get_loaders(batch_size=64)
    print(f"Loaded test loader with {len(test_loader.dataset)} samples.")

    if test_loader and MODEL_PATH.exists():
        df_mol = pd.read_csv(MOLECULE_CSV_PATH)
        df_mol_delta = apply_qm9_delta_learning(df_mol.copy(), smiles_col='smiles', target_cols=TARGET_COLS)
        target_mean = torch.tensor(df_mol_delta[TARGET_COLS].mean().values, dtype=torch.float32)
        target_std = torch.tensor(df_mol_delta[TARGET_COLS].std().values, dtype=torch.float32)

        sample_batch = next(iter(test_loader))
        model = TrainingTesting(
            node_in_dim=sample_batch.num_node_features, edge_in_dim=sample_batch.edge_attr.shape[1],
            hidden_dim=256, output_dim=len(TARGET_COLS), dropout=0.0, num_layers=7,
            device=DEVICE, target_mean=target_mean, target_std=target_std,
        )
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            print(f"Loaded GIN model state from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            sys.exit(1)

        print("\nStarting evaluation...")
        all_results, y_pred, y_true = evaluate_all_properties(model, test_loader, TARGET_COLS, DEVICE)
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df.index.name = 'Property'
        
        print("\n--- GIN Model Evaluation Results ---")
        print(results_df.to_string(float_format="%.4f"))
        print("------------------------------------\n")
        
        output_dir = Path(__file__).resolve().parent
        results_df.to_csv(output_dir / "evaluation_results.csv", float_format="%.4f")
        np.savez(output_dir / "predictions.npz", y_pred=y_pred, y_true=y_true)
        print(f"Results and predictions saved to {output_dir}")
        
    elif not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Skipping evaluation.")
    else:
        print("No data loaded. Skipping evaluation.")