import argparse
import os
import sys
import pandas as pd
import torch
from art import *

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from Scripts.qm9_delta import apply_qm9_delta_learning

from GIN_2.Utils.paths import Paths
from GIN_2.Utils.preprocessing import RelationalGeometryPipeline
from GIN_2.Utils.TrainingTesting import TrainingTesting


def get_dataset_stats(csv_path: str, target_cols: list) -> tuple:
    """
    Calculates the global mean and standard deviation for dynamic normalization.
    """
    print(f"Calculating global statistics for {len(target_cols)} targets...")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = apply_qm9_delta_learning(df, smiles_col='smiles', target_cols=target_cols)
        df = df[target_cols]
        mean_vals = df[target_cols].mean().values
        std_vals = df[target_cols].std().values
        
        target_mean = torch.tensor(mean_vals, dtype=torch.float32)
        target_std = torch.tensor(std_vals, dtype=torch.float32)
        print("Global statistics calculated successfully from raw CSV.")
        return target_mean, target_std
    else:
        # Fallback to computing from processed cache if CSV path doesn't exist
        cache_file = 'GIN_2/data/processed/qm_merged_3d_graphs_delta.pt'
        if os.path.exists(cache_file):
            print(f"CSV file not found at {csv_path}. Extracting stats from preprocessed cache {cache_file}...")
            payload = torch.load(cache_file, map_location='cpu', weights_only=False)
            if isinstance(payload, tuple) and len(payload) >= 1:
                data = payload[0]
            else:
                data = payload
            if hasattr(data, 'y') and data.y is not None:
                import numpy as np
                y_np = data.y.numpy()
                mean_vals = np.nanmean(y_np, axis=0)
                std_vals = np.nanstd(y_np, axis=0)
                target_mean = torch.tensor(mean_vals, dtype=torch.float32)
                target_std = torch.tensor(std_vals, dtype=torch.float32)
                print("Global statistics extracted from cache successfully.")
                return target_mean, target_std
        raise FileNotFoundError(f"Could not calculate statistics: raw CSV {csv_path} and cache {cache_file} are both missing.")


class main:
    """
    Main class to manually run the molecular property prediction pipeline.
    """
    def __init__(
        self,
        mol_path: str,
        atom_path: str,
        force_rebuild: bool = False,
        save_path: str = None,
        verbose: bool = True,
        auto: bool = False,
    ):
        self.model = None
        self.mol_path = mol_path
        self.atom_path = atom_path
        self.force_rebuild = force_rebuild
        default_save_path = save_path or Paths().get_model_path()
        self.save_path = os.path.abspath(os.path.expanduser(default_save_path))
        self.verbose = verbose
        self.auto = auto

        # NEW: QM9 Target Columns
        self.TARGET_COLS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
        
        # NEW: Updated Architecture Hyperparameters
        self.BATCH_SIZE = 64
        self.HIDDEN_DIM = 256
        self.OUTPUT_DIM = len(self.TARGET_COLS)
        self.DROPOUT = 0.0     # 0.0 for initial training to prevent underfitting
        self.NUM_LAYERS = 7    # Deep GIN with Jumping Knowledge
        self.LEARNING_RATE = 3e-4
        self.WEIGHT_DECAY = 1e-5
        self.EPOCHS = 50      # Increased for proper convergence
        self.PATIENCE = 20
        
        # Smart device selector
        if torch.cuda.is_available():
            self.DEVICE = "cuda"
        elif torch.backends.mps.is_available():
            self.DEVICE = "mps"
        else:
            self.DEVICE = "cpu"
            
        self.target_mean = None
        self.target_std = None

    def _read_choice(self, prompt: str, valid_choices: tuple, default: str = None) -> str:
        """Small helper to keep interactive command handling consistent.

        In auto mode (or when stdin is unavailable), the *default* choice is
        returned immediately so the pipeline can run unattended.
        """
        # Auto-mode: skip interaction entirely.
        if self.auto:
            chosen = default if default in valid_choices else valid_choices[0]
            print(f"{prompt}  [auto-selected: '{chosen}']")
            return chosen

        while True:
            print(prompt)
            try:
                choice = input().strip().lower()
            except (EOFError, OSError):
                # stdin is closed (e.g. AzureML job, subprocess with no tty).
                chosen = default if default in valid_choices else valid_choices[0]
                print(
                    f"  [non-interactive environment detected — "
                    f"auto-selecting '{chosen}']"
                )
                return chosen
            if choice in valid_choices:
                return choice
            print("Invalid input")

    def preprocess(self):
        """Runs preprocessing with persistent cache and returns data loaders."""
        
        # 1. Calculate Global Statistics first
        self.target_mean, self.target_std = get_dataset_stats(self.mol_path, self.TARGET_COLS)

        # 2. Handle force rebuild (Manually delete the PyG cache file)
        cache_file = 'GIN_2/data/processed/qm_merged_3d_graphs_delta.pt'
        if self.force_rebuild and os.path.exists(cache_file):
            print("Force rebuild triggered. Deleting old cache...")
            os.remove(cache_file)

        # 3. Initialize PyG Pipeline (This automatically handles Dask processing or loading from disk)
        print("Initializing Relational Geometry Pipeline...")
        pipeline = RelationalGeometryPipeline(
            root='GIN_2/data', 
            mol_csv_path=self.mol_path, 
            atom_csv_path=self.atom_path, 
            target_cols=self.TARGET_COLS
        )
        
        # 4. Generate DataLoaders
        self.train_loader, self.val_loader, self.test_loader = pipeline.get_loaders(batch_size=self.BATCH_SIZE)

        # 5. Dynamically extract node/edge dimensions for the model
        sample_batch = next(iter(self.train_loader))
        self.node_in_dim = sample_batch.num_node_features
        self.edge_in_dim = sample_batch.edge_attr.shape[1]
        
        tprint("Preprocessing Completed")

    def build_model(self):
        """Initializes the TrainingTesting model."""
        self.model = TrainingTesting(
            node_in_dim=self.node_in_dim,      # Extracted dynamically (Should be 26)
            edge_in_dim=self.edge_in_dim,      # Extracted dynamically (Should be 6)
            hidden_dim=self.HIDDEN_DIM,
            output_dim=self.OUTPUT_DIM,
            dropout=self.DROPOUT,
            num_layers=self.NUM_LAYERS,
            learning_rate=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
            device=self.DEVICE,
            target_mean=self.target_mean,
            target_std=self.target_std,
        )
        print(f"\nModel initialized on {self.DEVICE} with {sum(p.numel() for p in self.model.parameters()):,} parameters.")

    def run(self):
        """Executes the full training and evaluation process."""
        tprint("Molecular Property Prediction", font='block-medium')
        tprint("GIN Training Pipeline", font='block-medium')

        cmd = self._read_choice("To perform data-preprocessing, enter 'S'", ("s",), default="s")
        if cmd == "s":
            self.preprocess()

        print("Building Model Configuration")
        self.build_model()

        while True:
            cmd = self._read_choice(
                "To perform training, enter 'S' | To check training device, enter 'D'",
                ("s", "d"),
            )
            if cmd == "d":
                print(f"Current Device: {self.DEVICE}")
                continue
            break

        # Training
        self.model.run_training(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=self.EPOCHS,
            patience=self.PATIENCE,
            target_cols=self.TARGET_COLS
        )

        # Evaluation
        test_metrics = self.model.evaluate(self.test_loader)
        print(f"\n{'=' * 60}")
        print(f"Test Results  —  MAE: {test_metrics['mae']:.4f}  |  R²: {test_metrics['r2']:.4f}")
        print(f"{'=' * 60}")

        # Save
        self.save_model()

    def save_model(self, save_path: str = None):
        """Saves the trained model state to a file."""
        if self.model is None:
            print("No model to save.")
            return

        target_path = os.path.abspath(os.path.expanduser(save_path or self.save_path))

        # If a directory was passed, save with a default filename inside it.
        if os.path.isdir(target_path):
            target_path = os.path.join(target_path, "GIN_model.pth")

        save_dir = os.path.dirname(target_path) or "."

        try:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.model.state_dict(), target_path)
            print(f"Model saved to {target_path}")
        except (PermissionError, OSError) as exc:
            fallback_path = os.path.abspath("GIN_2/outputs/GIN_model.pth")
            fallback_dir = os.path.dirname(fallback_path)
            os.makedirs(fallback_dir, exist_ok=True)
            torch.save(self.model.state_dict(), fallback_path)
            print(f"Warning: could not save to '{target_path}' ({exc}).")
            print(f"Model saved to fallback path: {fallback_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual GIN training runner")
    # Updated Arguments to match relational dataset structure
    parser.add_argument("--mol_csv", type=str, default="Dataset/New_QM9/molecule_properties.csv", help="Path to molecules CSV")
    parser.add_argument("--atom_csv", type=str, default="Dataset/New_QM9/atom_properties.csv", help="Path to atoms CSV")
    parser.add_argument("--save_path", type=str, default="GIN_2/outputs/GIN_model.pth", help="Optional custom model save path")
    parser.add_argument("--force_rebuild", action="store_true", help="Ignore cache and rebuild preprocessing")
    parser.add_argument("--quiet", action="store_true", help="Reduce preprocessing verbosity")
    parser.add_argument("--auto", action="store_true", help="Run non-interactively (skips all input() prompts)")
    args = parser.parse_args()

    # Paths fallback (Update these if your Paths class handles them differently)
    mol_path = args.mol_csv 
    atom_path = args.atom_csv 

    resolved_save_path = os.path.abspath(os.path.expanduser(args.save_path))
    if os.path.isdir(resolved_save_path):
        resolved_save_path = os.path.join(resolved_save_path, "GIN_model.pth")

    output_dir = os.path.dirname(resolved_save_path) or "."
    if not os.path.exists(output_dir):
        print(f"Output directory not found at '{output_dir}'. Creating it now...")
        os.makedirs(output_dir, exist_ok=True)

    probe_file = os.path.join(output_dir, ".dir_probe")
    with open(probe_file, "a", encoding="utf-8"):
        pass
    os.remove(probe_file)
    print(f"Output directory is writable: {output_dir}")

    runner = main(
        mol_path=mol_path,
        atom_path=atom_path,
        force_rebuild=args.force_rebuild,
        save_path=resolved_save_path,
        verbose=not args.quiet,
        auto=args.auto,
    )
    runner.run()