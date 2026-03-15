import os
import sys
import torch
import random

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from GIN.Utils.TrainingTesting import TrainingTesting
from GIN.Utils.preprocessing import MolecularPropertyPipeline
from GIN.Utils.paths import Paths

# TODO: use the new dataset with unseen data to predict and get model metrics

def main():
    # ==================== Configuration ====================
    NUM_SAMPLES = 5
    BATCH_SIZE = 64

    paths = Paths()
    qm8_path = paths.get_qm8_path()
    qm9_path = paths.get_qm9_path()
    model_path = paths.get_model_path()

    # ==================== 1. Load Cached Preprocessed Data ====================
    print(f"Loading datasets from:\n - {qm8_path}\n - {qm9_path}")
    print("Running preprocessing pipeline with cache auto-detection...")

    # run_full_pipeline automatically loads GIN/outputs/cache/preprocessed_graphs.pt when valid.
    pipeline = MolecularPropertyPipeline(qm8_path, qm9_path)
    pipeline.run_full_pipeline(
        batch_size=BATCH_SIZE,
        use_cache=True,
        force_rebuild=False,
        verbose=True,
        show_progress=False,
    )

    total_graphs = len(pipeline.graphs) if pipeline.graphs is not None else 0
    if total_graphs == 0:
        print("No preprocessed graphs available for prediction.")
        return

    print(f"\nTotal cached molecules available: {total_graphs}")

    # ==================== 2. Load Model ====================
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Using device: {device}")

    # Hyperparameters must match training
    model = TrainingTesting(
        node_in_dim=6,
        edge_in_dim=3,
        hidden_dim=128,
        output_dim=12,
        device=str(device)
    )

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Explicitly move model to device again to ensure buffers are correct
    model.to(device)
    model.eval()

    # ==================== 3. Predict on Random Cached Test Samples ====================
    test_indices = (pipeline.split_indices or {}).get("test", list(range(total_graphs)))
    if not test_indices:
        test_indices = list(range(total_graphs))

    n_samples = min(NUM_SAMPLES, len(test_indices))
    print(f"\nSelecting {n_samples} random cached molecules for prediction...\n")

    sample_indices = random.sample(test_indices, n_samples)
    properties = pipeline.target_cols

    for graph_idx in sample_indices:
        graph = pipeline.graphs[graph_idx]
        smiles = (
            pipeline.smiles_list[graph_idx]
            if pipeline.smiles_list is not None and graph_idx < len(pipeline.smiles_list)
            else f"index_{graph_idx}"
        )
        print(f"Molecule: {smiles}")

        # Ensure a batch vector exists for single-graph inference.
        if not hasattr(graph, "batch") or graph.batch is None:
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)

        graph = graph.to(device)

        # Predict
        with torch.no_grad():
            pred_norm = model(graph)
            # Denormalize using stored stats in the model
            pred_real = pred_norm * model.std_y + model.mean_y

        # Display results
        print("-" * 50)
        print(f"{'Property':<10} | {'Prediction':<15} | {'Actual (if avail)':<15}")
        print("-" * 50)

        for i, prop_name in enumerate(properties):
            pred_val = pred_real[0][i].item()

            actual_val = graph.y[0][i].item()
            is_available = bool(graph.mask[0][i].item())
            actual_str = f"{actual_val:.4f}" if is_available else "N/A"

            print(f"{prop_name:<10} | {pred_val:<15.4f} | {actual_str:<15}")
        print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
