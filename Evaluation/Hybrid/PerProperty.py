import os
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

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


# --- Worker process functions for multiprocessing ---

def init_worker(model_path_for_worker: str):
    """
    Initializer for each worker process in the pool.
    Initializes the ONNX inference engine for the process.
    """
    init_hybrid_engine(model_path=model_path_for_worker)


def run_inference_for_smiles(smiles: str, n_conformers: int) -> tuple:
    """
    The task executed by each worker process.
    Runs inference for a single SMILES string.
    """
    try:
        # The engine is initialized by init_worker.
        # Pass model_path=None because the engine is already initialized in this process.
        result = run_hybrid_regression_with_confidence(
            smiles,
            model_path=None,
            n_conformers=n_conformers,
            apply_descaling=True,
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
    n_workers: int = -1
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates the model on all properties using the ONNX inference engine,
    parallelized across multiple CPU cores.
    """
    if n_workers < 1:
        # Default to using all available CPU cores
        n_workers = cpu_count()
    print(f"Using {n_workers} worker processes for inference.")

    all_preds = []
    failures = []

    # Use a multiprocessing pool to parallelize inference.
    # The `initializer` function `init_worker` is called for each new worker process.
    with Pool(processes=n_workers, initializer=init_worker, initargs=(model_path,)) as pool:
        # Create a partial function for the worker task with the n_conformers argument fixed.
        task = partial(run_inference_for_smiles, n_conformers=n_conformers)

        # `pool.imap` applies the task to each item in smiles_list and returns an iterator.
        # The results will be in the same order as the input.
        # `tqdm` is used to display a progress bar.
        results_iterator = pool.imap(task, smiles_list)

        for result in tqdm(results_iterator, total=len(smiles_list), desc="Running parallel inference"):
            smiles, prediction, error = result
            if error:
                failures.append((smiles, error))
                all_preds.append(np.full(len(target_cols), np.nan))
            else:
                all_preds.append(prediction)

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

    # Number of worker processes to use. -1 means use all available CPU cores.
    N_WORKERS = 2

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
        _, _, test_indices_subset = torch.utils.data.random_split(range(n_samples),
                                                                    [train_size, val_size, test_size],
                                                                    generator=generator)
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
            n_conformers=1,  # Use 3 conformers for confidence, can be set to 1 for speed
            n_workers=N_WORKERS
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