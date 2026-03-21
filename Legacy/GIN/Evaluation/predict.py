import os
import sys
import warnings
import torch

if __package__ in (None, ""):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from Legacy.GIN.Utils.TrainingTesting import TrainingTesting
from Legacy.GIN.Utils.preprocessing import MolecularPropertyPipeline
from Legacy.GIN.Utils.paths import Paths
from Legacy.GIN.Evaluation.preprocessing import bundle_dataset, all_files

def main():
    paths = Paths()
    model_path = paths.get_model_path()
    
    # 1. Load data using the Evaluation preprocessing script
    print("Loading molecules for prediction...")
    df = bundle_dataset(all_files)
    if df.empty:
        print("No molecules found for prediction.")
        return

    # 2. Setup Device and Pipeline
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # --- NEW: Get Normalization Stats from Cache ---
    pipeline = MolecularPropertyPipeline(paths.get_qm8_path(), paths.get_qm9_path())
    cache_path = pipeline._default_cache_path()
    
    if not os.path.exists(cache_path):
        print(f"Error: Cache not found at {cache_path}. Run training first.")
        return
        
    payload = torch.load(cache_path, map_location=device, weights_only=False)
    y_mean = payload.get("y_mean")
    y_std = payload.get("y_std")

    # Legacy cache files may not store normalization stats.
    # Fall back to identity denormalization to avoid crashing inference.
    target_cols = pipeline.target_cols
    target_dim = len(target_cols)
    if y_mean is None or y_std is None:
        warnings.warn(
            "Normalization stats (y_mean/y_std) were not found in cache. "
            "Using identity denormalization, so outputs are in normalized model space. "
            "Rebuild preprocessing cache to restore denormalized outputs.",
            RuntimeWarning,
        )
        y_mean = torch.zeros(target_dim, dtype=torch.float32, device=device)
        y_std = torch.ones(target_dim, dtype=torch.float32, device=device)
    else:
        y_mean = torch.as_tensor(y_mean, dtype=torch.float32, device=device).view(-1)
        y_std = torch.as_tensor(y_std, dtype=torch.float32, device=device).view(-1)

        if y_mean.numel() != target_dim or y_std.numel() != target_dim:
            warnings.warn(
                f"Normalization stats shape mismatch (mean={y_mean.numel()}, std={y_std.numel()}, expected={target_dim}). "
                "Using identity denormalization.",
                RuntimeWarning,
            )
            y_mean = torch.zeros(target_dim, dtype=torch.float32, device=device)
            y_std = torch.ones(target_dim, dtype=torch.float32, device=device)
    
    # 3. Initialize and Load Model
    # --- UPDATED DIMS: 7 and 4 ---
    model = TrainingTesting(
        node_in_dim=7,
        edge_in_dim=4,
        hidden_dim=128,
        output_dim=12,
        device=str(device)
    )

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please train the model first.")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 4. Perform Predictions
    sample_size = min(10, len(df))
    sample_df = df.sample(frac=1).reset_index(drop=True)
    print(f"\nPerforming predictions on {sample_size} random PC9 molecules...\n")
    
    dummy_y = torch.zeros(len(target_cols))
    dummy_mask = torch.zeros(len(target_cols))

    successful_predictions = 0
    for _, row in sample_df.iterrows():
        if successful_predictions >= sample_size:
            break

        smiles = row['smiles']
        print(f"Molecule: {smiles}")
        print(
            "Dataset actuals -> "
            f"homo: {row.get('homo', 'N/A')}, "
            f"lumo: {row.get('lumo', 'N/A')}, "
            f"gap: {row.get('gap', 'N/A')}, "
            f"e: {row.get('e', 'N/A')}"
        )
        
        # Convert SMILES to 3D graph
        graph = pipeline._smiles_to_graph((smiles, dummy_y, dummy_mask))
        
        if graph is None:
            print(f"  Failed 3D embedding for SMILES: {smiles}")
            continue

        # Prepare for inference
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        graph = graph.to(device)

        with torch.no_grad():
            pred_norm = model(graph)
            # --- NEW: DENORMALIZE ---
            pred_real = (pred_norm * y_std) + y_mean
        
        # Display results (properties defined in pipeline)
        print("-" * 60)
        print(f"{'Property':<15} | {'Prediction':<15} | {'Actual':<15}")
        print("-" * 60)
        for i, prop_name in enumerate(target_cols):
            pred_val = pred_real[0][i].item() # Use the denormalized value here!
            actual_val = row.get(prop_name, row.get(prop_name.lower(), "N/A"))
            try:
                actual_display = f"{float(actual_val):.4f}"
            except (TypeError, ValueError):
                actual_display = "N/A"
            print(f"{prop_name:<15} | {pred_val:<15.4f} | {actual_display:<15}")
        successful_predictions += 1
        print("=" * 60 + "\n")

    print(f"Completed predictions for {successful_predictions} molecules.")
        

if __name__ == "__main__":
    main()