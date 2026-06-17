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

def evaluate_property(model: torch.nn.Module, loader: torch.utils.data.DataLoader, property_index: int, property_name: str, scalers: Dict[str, Any] = None) -> Dict[str, float]:
    """
    Evaluates the model on a dataset for a specific property index.
    
    Args:
        model: The trained HybridFusionModel.
        loader: DataLoader for the dataset (HybridDataset).
        property_index: The index of the property in the output tensor.
        property_name: Name of the property for scaling and logging.
        scalers: Dictionary of scalers to inverse transform the data.

    Returns:
        A dictionary containing RMSE, MAE, and R^2 scores.
    """
    model.eval()
    device = next(model.parameters()).device
    
    true_values = []
    predicted_values = []

    with torch.no_grad():
        for batch in loader:
            graph_data = batch['graph'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target']
            nan_mask = batch['nan_mask']

            outputs = model(graph_data, input_ids, attention_mask).cpu()
            
            # Filter by nan_mask for the specific property
            mask = nan_mask[:, property_index] == 1
            if mask.any():
                p = outputs[mask, property_index].detach().numpy()
                t = targets[mask, property_index].detach().numpy()
                
                if scalers and property_name in scalers:
                    p = scalers[property_name].inverse_transform(p.reshape(-1, 1)).flatten()
                    t = scalers[property_name].inverse_transform(t.reshape(-1, 1)).flatten()
                
                predicted_values.extend(p)
                true_values.extend(t)

    if len(true_values) < 2:
        if len(true_values) == 1:
            rmse = float(np.abs(true_values[0] - predicted_values[0]))
            mae = rmse
            return {'RMSE': rmse, 'MAE': mae, 'R^2': 0.0}
        return {'RMSE': 0.0, 'MAE': 0.0, 'R^2': 0.0}

    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R^2': r2
    }

def evaluate_all_properties(model: torch.nn.Module, loader: torch.utils.data.DataLoader, target_cols: List[str], scalers: Dict[str, Any] = None) -> Dict[str, Dict[str, float]]:
    """
    Evaluates the model on all properties using a single forward pass over the data loader.
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_preds = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for batch in loader:
            graph_data = batch['graph'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target']
            nan_mask = batch['nan_mask']
            
            outputs = model(graph_data, input_ids, attention_mask).cpu()
            
            all_preds.append(outputs.detach())
            all_targets.append(targets.detach())
            all_masks.append(nan_mask.detach())
            
    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()
    y_mask = torch.cat(all_masks, dim=0).numpy()
    
    results = {}
    for i, col in enumerate(target_cols):
        valid_idx = y_mask[:, i] == 1
        if valid_idx.any():
            p = y_pred[valid_idx, i]
            t = y_true[valid_idx, i]
            
            if scalers and col in scalers:
                p = scalers[col].inverse_transform(p.reshape(-1, 1)).flatten()
                t = scalers[col].inverse_transform(t.reshape(-1, 1)).flatten()
                
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
        else:
            results[col] = {'RMSE': 0.0, 'MAE': 0.0, 'R^2': 0.0}
            
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
        plt.ylim(0, max(values) * 1.2)
    plt.tight_layout()
    
    filename = f'{property_name}_evaluation.png'
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
    else:
        save_path = filename
        
    plt.savefig(save_path)
    plt.close()

def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, property_index: int, property_name: str, scalers: Dict[str, Any] = None, save_dir: str = None):
    """
    Helper to run evaluation and plotting for a specific property.
    """
    print(f"Evaluating {property_name}...")
    results = evaluate_property(model, loader, property_index, property_name, scalers)
    print(f"Results for {property_name}: {results}")
    plot_results(results, property_name, save_dir=save_dir)

if __name__ == "__main__":
    from main import HybridFusionModel, TOKENIZED_CACHE_PATH
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TARGET_COLS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
    
    # Initialize model
    model = HybridFusionModel(output_dim=len(TARGET_COLS)).to(device)
    
    # Try finding the pre-trained model at various paths
    model_path = os.environ.get("HYBRID_MODEL_OUTPUT_PATH", "best_hybrid_model.pth")
    if not os.path.exists(model_path) and os.path.exists("models/best_hybrid_model.pth"):
        model_path = "models/best_hybrid_model.pth"
        
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model state from {model_path}")
        except Exception as e:
            print(f"Error loading model weights from {model_path}: {e}")
    else:
        print(f"No pre-trained model found at {model_path}; using random weights.")
    
    # Load scalers from the cached tokenized dataset
    scalers = {}
    if os.path.exists(TOKENIZED_CACHE_PATH):
        try:
            transformer_data = torch.load(TOKENIZED_CACHE_PATH, weights_only=False)
            scalers = transformer_data.get('scalers', {})
            print(f"Loaded scalers from {TOKENIZED_CACHE_PATH}")
        except Exception as e:
            print(f"Could not load scalers from cached dataset: {e}")
    else:
        print(f"No cache found at {TOKENIZED_CACHE_PATH}")

    # Set up the dataloader consistent with main.py
    test_loader = None
    try:
        from main import MOLECULE_CSV_PATH, ATOM_CSV_PATH, HybridDataset
        from GIN_2.Utils.preprocessing import RelationalGeometryPipeline
        from torch_geometric.loader import DataLoader

        if os.path.exists(TOKENIZED_CACHE_PATH) and os.path.exists(MOLECULE_CSV_PATH) and os.path.exists(ATOM_CSV_PATH):
            print("Found dataset files. Preparing test/validation dataloader...")
            pyg_dataset = RelationalGeometryPipeline(
                root='GIN_2/data', 
                mol_csv_path=MOLECULE_CSV_PATH,
                atom_csv_path=ATOM_CSV_PATH,
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
            
            for i, mol_id in enumerate(t_mol_ids):
                if mol_id in graph_dict:
                    aligned_graphs.append(graph_dict[mol_id])
                    aligned_input_ids.append(t_input_ids[i])
                    aligned_attention_masks.append(t_attention_masks[i])
                    aligned_targets.append(t_targets[i])
                    aligned_nan_masks.append(t_nan_mask[i])
                    
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

                test_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
                print(f"Loaded validation split as test loader with {len(val_dataset)} samples.")
            else:
                print("Warning: Could not align any molecules between Graph and Transformer datasets.")
        else:
            print("Dataset files or tokenized cache not found. Running with test_loader = None.")
    except Exception as e:
        print(f"Could not initialize real test loader: {e}")
        print("Continuing with test_loader = None.")

    if test_loader is not None:
        print("Starting sequential evaluation for all properties...")
        # Evaluate all properties in a single pass (very efficient!)
        all_results = evaluate_all_properties(model, test_loader, TARGET_COLS, scalers)
        for name in TARGET_COLS:
            res = all_results[name]
            print(f"Results for {name}: {res}")
            plot_results(res, name)
    else:
        print("No test loader initialized. Skipping execution.")

