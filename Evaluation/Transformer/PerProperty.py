import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, List

# Add project root to sys.path to allow execution from any folder
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Transformers_2.Utils.Fine_Tuning import FineTuning
from Transformers_2.Utils.Tokeniser import Tokeniser

def evaluate_all_properties(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    target_cols: List[str],
    scalers: Dict[str, Any],
    device: str
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates the fine-tuned Transformer model on all properties.
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'] # Keep targets on CPU for scaling
            
            predictions = model(input_ids, attention_mask).cpu()
            
            all_preds.append(predictions)
            all_targets.append(targets)

    y_pred_scaled = torch.cat(all_preds, dim=0).numpy()
    y_true_scaled = torch.cat(all_targets, dim=0).numpy()

    results = {}
    for i, col in enumerate(target_cols):
        p_scaled = y_pred_scaled[:, i]
        t_scaled = y_true_scaled[:, i]

        # Inverse transform both predictions and targets to the delta-corrected space
        if scalers and col in scalers:
            p = scalers[col].inverse_transform(p_scaled.reshape(-1, 1)).flatten()
            t = scalers[col].inverse_transform(t_scaled.reshape(-1, 1)).flatten()
        else:
            p, t = p_scaled, t_scaled

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
    MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_PATH = project_root / "models/transformer_molecular_model.pth"
    MOLECULE_CSV_PATH = project_root / "Dataset/New_QM9/molecule_properties.csv"
    CACHE_PATH = project_root / "Transformers_2/outputs/cache/tokenized_dataset.pt"

    # --- Data Loading ---
    if not MOLECULE_CSV_PATH.exists():
        print(f"Molecule properties CSV not found at {MOLECULE_CSV_PATH}. Cannot run evaluation.")
        sys.exit(1)

    print("Preparing test dataloader from cache or by tokenizing...")
    tokeniser = Tokeniser(
        mol_path=str(MOLECULE_CSV_PATH),
        model_name=MODEL_NAME,
        target_cols=TARGET_COLS,
        cache_path=str(CACHE_PATH),
        force_rebuild=False,
        use_cache=True,
        verbose=True
    )
    
    artifacts = tokeniser.run_tokenizer(verbose=True)
    test_loader = artifacts["test_loader"]
    scalers = artifacts["scalers"]
    
    if not test_loader:
        print("Failed to create test loader. Exiting.")
        sys.exit(1)
        
    print(f"Loaded test loader with {len(test_loader.dataset)} samples.")

    # --- Model Initialization and Evaluation ---
    if test_loader and MODEL_PATH.exists():
        model = FineTuning(model_name=MODEL_NAME, num_labels=len(TARGET_COLS))
        
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"Loaded Transformer model state from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            sys.exit(1)

        print("\nStarting evaluation for all properties...")
        all_results = evaluate_all_properties(model, test_loader, TARGET_COLS, scalers, DEVICE)
        
        # Format results into a single pandas DataFrame
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df.index.name = 'Property'
        
        print("\n--- Transformer Model Evaluation Results ---")
        print(results_df.to_string(float_format="%.4f"))
        print("------------------------------------------\n")

    elif not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Please ensure the Transformer model has been trained.")
    else:
        print("No data loaded. Skipping evaluation.")