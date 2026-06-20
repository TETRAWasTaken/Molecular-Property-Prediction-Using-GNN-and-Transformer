import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to sys.path to allow execution from any folder
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from the ONNX inference engine
from GUI.core.inference import (
    init_hybrid_engine,
    run_hybrid_regression_with_confidence,
    _PROPERTY_NAMES as TARGET_COLS,
)

def evaluate_all_properties_with_inference(
    smiles_list: List[str],
    true_targets: np.ndarray,
    target_cols: List[str],
    model_path: str = None,
    n_conformers: int = 3
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates the model on all properties using the ONNX inference engine.
    """
    try:
        init_hybrid_engine(model_path=model_path)
        print("Hybrid inference engine initialized.")
    except Exception as e:
        print(f"Failed to initialize hybrid engine: {e}")
        return {col: {'RMSE': 0.0, 'MAE': 0.0, 'R^2': 0.0} for col in target_cols}

    all_preds = []
    failures = []

    for i, smiles in enumerate(smiles_list):
        try:
            # apply_descaling=True ensures predictions are in their physical units
            result = run_hybrid_regression_with_confidence(
                smiles,
                model_path=model_path,
                n_conformers=n_conformers,
                apply_descaling=True,
            )
            all_preds.append(result['prediction'])
        except Exception as exc:
            failures.append((smiles, str(exc)))
            all_preds.append(np.full(len(target_cols), np.nan))
        
        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(smiles_list)} molecules...")

    if failures:
        print(f"\nEncountered {len(failures)} failures during inference.")

    y_pred = np.array(all_preds)
    y_true = true_targets

    results = {}
    for i, col in enumerate(target_cols):
        # Filter out NaNs from failed predictions and true values
        valid_idx = ~np.isnan(y_pred[:, i]) & ~np.isnan(y_true[:, i])
        
        p = y_pred[valid_idx, i]
        t = y_true[valid_idx, i]

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
    TOKENIZED_CACHE_PATH = project_root / "Transformers_2/outputs/cache/tokenized_dataset.pt"
    MOLECULE_CSV_PATH = project_root / "Dataset/New_QM9/molecule_properties.csv"
    
    # Let the inference engine resolve the model path automatically
    model_path = None

    # --- Data Loading ---
    test_smiles = []
    test_targets = None
    
    if TOKENIZED_CACHE_PATH.exists() and MOLECULE_CSV_PATH.exists():
        print("Found dataset files. Preparing test data...")
        
        df_mol = pd.read_csv(MOLECULE_CSV_PATH)
        df_mol['molecule_id'] = df_mol['molecule_id'].astype(str)
        smiles_map = df_mol.set_index('molecule_id')['smiles'].to_dict()

        transformer_data = torch.load(TOKENIZED_CACHE_PATH, weights_only=False)
        
        t_mol_ids = [str(m).strip() for m in transformer_data['mol_ids']]
        
        # Align original, unscaled targets from the CSV with the tokenized data
        df_mol_aligned = df_mol[df_mol['molecule_id'].isin(t_mol_ids)].set_index('molecule_id').loc[t_mol_ids]
        original_targets_aligned = df_mol_aligned[TARGET_COLS].values

        # Use an 80/10/10 train/validation/test split
        n_samples = len(t_mol_ids)
        train_size = int(0.8 * n_samples)
        val_size = int(0.1 * n_samples)
        test_size = n_samples - train_size - val_size

        generator = torch.Generator().manual_seed(42)
        # We only need the test set indices for this script
        _, _, test_indices_subset = torch.utils.data.random_split(range(n_samples), [train_size, val_size, test_size], generator=generator)
        test_indices = test_indices_subset.indices

        test_mol_ids = [t_mol_ids[i] for i in test_indices]
        test_smiles = [smiles_map[mol_id] for mol_id in test_mol_ids if mol_id in smiles_map]
        
        # Select the correct ground truth targets for the test set
        test_targets = original_targets_aligned[test_indices]
        
        print(f"Loaded test split with {len(test_smiles)} samples.")
    else:
        print("Dataset files or tokenized cache not found. Cannot run evaluation.")
        sys.exit(1)

    # --- Model Evaluation ---
    if test_smiles and test_targets is not None:
        print("\nStarting evaluation using ONNX inference engine...")
        all_results = evaluate_all_properties_with_inference(
            test_smiles, 
            test_targets,
            TARGET_COLS, 
            model_path=model_path,
            n_conformers=3 # Use 3 conformers for confidence, can be set to 1 for speed
        )
        
        # Format results into a single pandas DataFrame
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df.index.name = 'Property'
        
        print("\n--- Hybrid Model Evaluation Results (ONNX) ---")
        print(results_df.to_string(float_format="%.4f"))
        print("----------------------------------------------\n")

        # Save results to a CSV file
        output_path = Path(__file__).resolve().parent / "evaluation_results.csv"
        results_df.to_csv(output_path, float_format="%.4f")
        print(f"Results saved to {output_path}")

    else:
        print("No data loaded. Skipping evaluation.")