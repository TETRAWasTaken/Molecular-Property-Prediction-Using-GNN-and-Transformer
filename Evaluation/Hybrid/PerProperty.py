import os
# Enable online download fallback for Hugging Face if the model is not cached
os.environ["HF_HUB_OFFLINE"] = "0"

import sys
from pathlib import Path
from typing import Dict, Any, List

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread-safe/headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to sys.path to allow execution from any folder
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from inference.py
from GUI.core.inference import (
    init_hybrid_engine,
    run_hybrid_regression,
    run_hybrid_regression_with_confidence,
    _descale_prediction_values,
    _PROPERTY_NAMES as TARGET_COLS, # Use property names from the authoritative source
)

def evaluate_all_properties_with_inference(
    smiles_list: List[str],
    true_targets: np.ndarray,
    target_cols: List[str],
    model_path: str = None,
    n_conformers: int = 3
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates the model on all properties using the inference engine.
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
            # Using confidence version to get more stable predictions, can be set to 1 conformer for speed
            if n_conformers > 1:
                result = run_hybrid_regression_with_confidence(
                    smiles,
                    model_path=model_path,
                    n_conformers=n_conformers,
                    apply_descaling=True,
                )
                prediction = result['prediction']
            else:
                # Fallback to single run if n_conformers is 1
                prediction = run_hybrid_regression(smiles, model_path=model_path)
                prediction = _descale_prediction_values(prediction, smiles=smiles)

            all_preds.append(prediction)
        except Exception as exc:
            failures.append((smiles, str(exc)))
            # Add a placeholder for failed predictions to keep alignment with true_targets
            all_preds.append(np.full(len(target_cols), np.nan))
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(smiles_list)} molecules...")

    if failures:
        print(f"\nEncountered {len(failures)} failures during inference.")
        for smiles, reason in failures[:5]:
            print(f"  - {smiles}: {reason}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more.")

    y_pred = np.array(all_preds)
    y_true = true_targets

    results = {}
    for i, col in enumerate(target_cols):
        # Filter out NaNs from failed predictions
        valid_idx = ~np.isnan(y_pred[:, i]) & ~np.isnan(y_true[:, i])
        
        p = y_pred[valid_idx, i]
        t = y_true[valid_idx, i]

        if len(t) < 2:
            if len(t) == 1:
                rmse = float(np.abs(t[0] - p[0]))
                mae = rmse
                r2 = 0.0
            else:
                rmse, mae, r2 = 0.0, 0.0, 0.0
        else:
            rmse = np.sqrt(mean_squared_error(t, p))
            mae = mean_absolute_error(t, p)
            r2 = r2_score(t, p)
            
        results[col] = {'RMSE': rmse, 'MAE': mae, 'R^2': r2}
            
    return results

def plot_results(results: Dict[str, float], property_name: str, save_dir: str = None):
    """
    Plots the evaluation results.

    Args:
        results: A dictionary containing evaluation metrics.
        property_name: The name of the property evaluated.
        save_dir: Optional directory to save the generated plots.
    """
    metrics = list(results.keys())
    values = [results[metric] for metric in metrics]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=metrics, y=values)
    plt.title(f'Evaluation Metrics: {property_name}')
    plt.ylabel('Score')
    if values:
        plt.ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    plt.tight_layout()
    
    filename = f'{property_name}_evaluation.png'
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
    else:
        save_path = filename
        
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Construct absolute paths from the project root
    TOKENIZED_CACHE_PATH = project_root / "Transformers_2/outputs/cache/tokenized_dataset.pt"
    MOLECULE_CSV_PATH = project_root / "Dataset/New_QM9/molecule_properties.csv"
    ATOM_CSV_PATH = project_root / "Dataset/New_QM9/atom_properties.csv"
    
    # Try finding the ONNX model at various paths
    model_path = project_root / "GUI/assets/hybrid_model.onnx"
    if not model_path.exists():
        print(f"ONNX model not found at {model_path}. The inference engine will try to find it.")
        model_path = None # Let the engine resolve it
    
    # Load data to get SMILES and targets for the validation set
    val_smiles = []
    val_targets = None
    
    try:
        from main import HybridDataset
        from GIN_2.Utils.preprocessing import RelationalGeometryPipeline
        
        if TOKENIZED_CACHE_PATH.exists() and MOLECULE_CSV_PATH.exists() and ATOM_CSV_PATH.exists():
            print("Found dataset files. Preparing validation data...")
            
            # Load the original molecule properties to get SMILES strings
            df_mol = pd.read_csv(MOLECULE_CSV_PATH)
            df_mol['molecule_id'] = df_mol['molecule_id'].astype(str)
            smiles_map = df_mol.set_index('molecule_id')['smiles'].to_dict()

            pyg_dataset = RelationalGeometryPipeline(
                root=str(project_root / 'GIN_2/data'), 
                mol_csv_path=str(MOLECULE_CSV_PATH),
                atom_csv_path=str(ATOM_CSV_PATH),
                target_cols=TARGET_COLS
            )
            pyg_graph_list = [g for g in pyg_dataset]
            
            transformer_data = torch.load(TOKENIZED_CACHE_PATH, weights_only=False)
            
            graph_dict = {}
            for g in pyg_graph_list:
                if torch.is_tensor(g.mol_id):
                     clean_id = str(g.mol_id.item())
                else:
                     clean_id = str(g.mol_id)
                graph_dict[clean_id] = g
                
            t_input_ids = transformer_data['input_ids']
            t_attention_masks = transformer_data['attention_mask']
            t_targets = transformer_data['labels']
            t_nan_mask = transformer_data['nan_mask']
            t_mol_ids = [str(m).strip() for m in transformer_data['mol_ids']]    

            aligned_graphs = []
            aligned_input_ids = []
            aligned_attention_masks = []
            aligned_targets = []
            aligned_nan_masks = []
            aligned_mol_ids = []
            
            for i, mol_id in enumerate(t_mol_ids):
                if mol_id in graph_dict:
                    aligned_graphs.append(graph_dict[mol_id])
                    aligned_input_ids.append(t_input_ids[i])
                    aligned_attention_masks.append(t_attention_masks[i])
                    aligned_targets.append(t_targets[i])
                    aligned_nan_masks.append(t_nan_mask[i])
                    aligned_mol_ids.append(mol_id)
                    
            if len(aligned_graphs) > 0:
                aligned_input_ids = torch.stack(aligned_input_ids)
                aligned_attention_masks = torch.stack(aligned_attention_masks)
                aligned_targets = torch.stack(aligned_targets)
                aligned_nan_masks = torch.stack(aligned_nan_masks)

                full_dataset = HybridDataset(
                    aligned_graphs, 
                    aligned_input_ids, 
                    aligned_attention_masks, 
                    aligned_targets, 
                    aligned_nan_masks
                )

                # Keep split consistent with main.py
                train_size = int(0.8 * len(full_dataset))
                val_size = len(full_dataset) - train_size
                
                generator = torch.Generator().manual_seed(42)
                _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)

                val_indices = val_dataset.indices
                
                # Correctly get SMILES strings for the validation set
                val_mol_ids = [aligned_mol_ids[i] for i in val_indices]
                val_smiles = [smiles_map[mol_id] for mol_id in val_mol_ids if mol_id in smiles_map]

                # Get original, unscaled targets for the validation set
                scalers = transformer_data.get('scalers', {})
                original_targets = torch.clone(aligned_targets)
                if scalers:
                    for i, col in enumerate(TARGET_COLS):
                        if col in scalers:
                            # Ensure the tensor is on the CPU before converting to numpy
                            target_numpy = original_targets[:, i].cpu().numpy().reshape(-1, 1)
                            inversed_numpy = scalers[col].inverse_transform(target_numpy)
                            original_targets[:, i] = torch.from_numpy(inversed_numpy).squeeze()
                
                val_targets = original_targets[val_indices].numpy()
                
                print(f"Loaded validation split with {len(val_smiles)} samples.")
            else:
                print("Warning: Could not align any molecules between Graph and Transformer datasets.")
        else:
            print("Dataset files or tokenized cache not found. Cannot run evaluation.")
    except Exception as e:
        print(f"Could not initialize and load data: {e}")
        print("Cannot run evaluation.")

    if val_smiles and val_targets is not None:
        print("Starting evaluation using inference engine...")
        all_results = evaluate_all_properties_with_inference(
            val_smiles, 
            val_targets, 
            TARGET_COLS, 
            model_path=str(model_path) if model_path else None,
            n_conformers=3 # Use 3 conformers for confidence, set to 1 for faster eval
        )
        for name in TARGET_COLS:
            if name in all_results:
                res = all_results[name]
                print(f"Results for {name}: {res}")
                plot_results(res, name, save_dir='evaluation_plots_inference')
    else:
        print("No data loaded. Skipping evaluation.")
