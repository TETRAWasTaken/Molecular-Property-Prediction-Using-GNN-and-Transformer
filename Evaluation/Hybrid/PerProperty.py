import os
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GUI.core.inference import (
    init_hybrid_engine,
    run_hybrid_regression_with_confidence,
    _PROPERTY_NAMES as TARGET_COLS,
)
from Scripts.qm9_delta import (
    HARTREE_TO_EV,
    QM9_DELTA_TARGET_COLUMNS,
)

EV_TO_KCAL_MOL = 23.060548

def init_worker(model_path_for_worker: str):
    init_hybrid_engine(model_path=model_path_for_worker)

def run_inference_for_smiles(smiles: str, n_conformers: int) -> tuple:
    try:
        result = run_hybrid_regression_with_confidence(
            smiles, model_path=None, n_conformers=n_conformers, apply_descaling=True
        )
        return smiles, result['prediction'], None
    except Exception as exc:
        return smiles, None, str(exc)

def evaluate_all_properties_with_inference(
    smiles_list: List[str],
    true_targets: np.ndarray,
    target_cols: List[str],
    model_path: str = None,
    n_conformers: int = 3,
    n_workers: int = -1,
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, np.ndarray, List[str]]:
    if n_workers < 1:
        n_workers = cpu_count()
    print(f"Using {n_workers} worker processes for inference.")

    all_preds, failures = [], []
    processed_smiles = []
    with Pool(processes=n_workers, initializer=init_worker, initargs=(model_path,)) as pool:
        task = partial(run_inference_for_smiles, n_conformers=n_conformers)
        results_iterator = pool.imap(task, smiles_list)
        for result in tqdm(results_iterator, total=len(smiles_list), desc="Running parallel inference"):
            smiles, prediction, error = result
            processed_smiles.append(smiles)
            if error:
                failures.append((smiles, error))
                all_preds.append(np.full(len(target_cols), np.nan))
            else:
                all_preds.append(prediction)

    if failures:
        print(f"\nEncountered {len(failures)} failures during inference.")

    y_pred, y_true = np.array(all_preds), true_targets
    results = {}
    for i, col in enumerate(target_cols):
        valid_idx = ~np.isnan(y_pred[:, i]) & ~np.isnan(y_true[:, i])
        p, t = y_pred[valid_idx, i], y_true[valid_idx, i]
        metrics = {'RMSE': 0.0, 'MAE': 0.0, 'R^2': 0.0}
        if len(t) >= 2:
            metrics['RMSE'] = np.sqrt(mean_squared_error(t, p))
            metrics['MAE'] = mean_absolute_error(t, p)
            metrics['R^2'] = r2_score(t, p)
            if col in QM9_DELTA_TARGET_COLUMNS:
                error_kcal_mol = np.abs(t - p) * EV_TO_KCAL_MOL
                metrics['Chem. Acc. (%)'] = np.mean(error_kcal_mol <= 1.0) * 100.0
        results[col] = metrics

    return results, y_pred, y_true, processed_smiles

if __name__ == "__main__":
    TOKENIZED_CACHE_PATH = project_root / "Transformers_2/outputs/cache/tokenized_dataset.pt"
    MOLECULE_CSV_PATH = project_root / "Dataset/New_QM9/molecule_properties.csv"
    model_path, N_WORKERS, N_CONFORMERS = None, -1, 1

    if not (TOKENIZED_CACHE_PATH.exists() and MOLECULE_CSV_PATH.exists()):
        print("Dataset files not found. Cannot run evaluation.")
        sys.exit(1)

    print("Preparing test data...")
    df_mol = pd.read_csv(MOLECULE_CSV_PATH)
    df_mol['molecule_id'] = df_mol['molecule_id'].astype(str)
    smiles_map = df_mol.set_index('molecule_id')['smiles'].to_dict()
    transformer_data = torch.load(TOKENIZED_CACHE_PATH, weights_only=False)
    t_mol_ids = [str(m).strip() for m in transformer_data['mol_ids']]
    df_mol_aligned = df_mol[df_mol['molecule_id'].isin(t_mol_ids)].set_index('molecule_id').loc[t_mol_ids]
    original_targets_aligned = df_mol_aligned[TARGET_COLS].values

    for i, col in enumerate(TARGET_COLS):
        if col in ['u0', 'u298', 'h298', 'g298', 'zpve', 'gap', 'homo', 'lumo']:
            original_targets_aligned[:, i] *= HARTREE_TO_EV

    n_samples = len(t_mol_ids)
    train_size, val_size = int(0.8 * n_samples), int(0.1 * n_samples)
    test_size = n_samples - train_size - val_size
    generator = torch.Generator().manual_seed(42)
    _, _, test_indices_subset = torch.utils.data.random_split(range(n_samples), [train_size, val_size, test_size], generator=generator)
    test_indices = test_indices_subset.indices
    test_smiles = [smiles_map[mol_id] for mol_id in [t_mol_ids[i] for i in test_indices] if mol_id in smiles_map]
    test_targets = original_targets_aligned[test_indices]
    print(f"Loaded test split with {len(test_smiles)} samples.")

    if test_smiles and test_targets is not None:
        print("\nStarting evaluation...")
        all_results, y_pred, y_true, processed_smiles = evaluate_all_properties_with_inference(
            test_smiles, test_targets, TARGET_COLS, model_path, N_CONFORMERS, N_WORKERS
        )
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df.index.name = 'Property'
        print("\n--- Hybrid Model Evaluation Results (ONNX) ---")
        print(results_df.to_string(float_format="%.4f"))
        print("----------------------------------------------\n")

        output_dir = Path(__file__).resolve().parent
        
        # --- Save Aggregate and Per-Molecule Results ---
        
        # 1. Save aggregate metrics
        results_df.to_csv(output_dir / "evaluation_results.csv", float_format="%.4f")
        
        # 2. Save raw predictions for plotting and analysis
        np.savez(output_dir / "predictions.npz", y_pred=y_pred, y_true=y_true, smiles=np.array(processed_smiles))
        
        # 3. Create and save per-molecule evaluation results
        df_true = pd.DataFrame(y_true, columns=[f"{col}_true" for col in TARGET_COLS])
        df_pred = pd.DataFrame(y_pred, columns=[f"{col}_pred" for col in TARGET_COLS])
        df_error = pd.DataFrame(np.abs(y_true - y_pred), columns=[f"{col}_error" for col in TARGET_COLS])
        
        df_per_molecule = pd.concat([pd.DataFrame({'smiles': processed_smiles}), df_true, df_pred, df_error], axis=1)
        
        per_molecule_output_path = output_dir / "per_molecule_evaluation_results.csv"
        df_per_molecule.to_csv(per_molecule_output_path, index=False, float_format="%.6f")

        print(f"Aggregate results saved to {output_dir / 'evaluation_results.csv'}")
        print(f"Per-molecule results saved to {per_molecule_output_path}")
        print(f"Raw predictions for analysis saved to {output_dir / 'predictions.npz'}")
    else:
        print("No data loaded. Skipping evaluation.")