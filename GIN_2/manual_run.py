import argparse
import os
import sys
import uuid
import pandas as pd
import torch
from art import *

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from GIN_2.Utils.paths import Paths
from GIN_2.Utils.preprocessing import RelationalGeometryPipeline
from GIN_2.Utils.TrainingTesting import TrainingTesting


def get_dataset_stats(csv_path: str, target_cols: list) -> tuple:
    """
    Calculates the global mean and standard deviation for dynamic normalization.
    """
    print(f"Calculating global statistics for {len(target_cols)} targets...")
    df = pd.read_csv(csv_path, usecols=target_cols)
    mean_vals = df[target_cols].mean().values
    std_vals = df[target_cols].std().values
    
    target_mean = torch.tensor(mean_vals, dtype=torch.float32)
    target_std = torch.tensor(std_vals, dtype=torch.float32)
    print("Global statistics calculated successfully.")
    
    return target_mean, target_std


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
    ):
        self.model = None
        self.mol_path = mol_path
        self.atom_path = atom_path
        self.force_rebuild = force_rebuild
        self.save_path = save_path or Paths().get_model_path()
        self.verbose = verbose

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
        self.EPOCHS = 20      # Increased for proper convergence
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

    def _read_choice(self, prompt: str, valid_choices: tuple) -> str:
        """Small helper to keep interactive command handling consistent."""
        while True:
            print(prompt)
            choice = input().strip().lower()
            if choice in valid_choices:
                return choice
            print("Invalid input")

    def preprocess(self):
        """Runs preprocessing with persistent cache and returns data loaders."""
        
        # 1. Calculate Global Statistics first
        self.target_mean, self.target_std = get_dataset_stats(self.mol_path, self.TARGET_COLS)

        # 2. Handle force rebuild (Manually delete the PyG cache file)
        cache_file = './data/processed/qm_merged_3d_graphs.pt'
        if self.force_rebuild and os.path.exists(cache_file):
            print("Force rebuild triggered. Deleting old cache...")
            os.remove(cache_file)

        # 3. Initialize PyG Pipeline (This automatically handles Dask processing or loading from disk)
        print("Initializing Relational Geometry Pipeline...")
        pipeline = RelationalGeometryPipeline(
            root='./data', 
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

        cmd = self._read_choice("To perform data-preprocessing, enter 'S'", ("s",))
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

        target_path = save_path or self.save_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        torch.save(self.model.state_dict(), target_path)
        print(f"Model saved to {target_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual GIN training runner")
    # Updated Arguments to match relational dataset structure
    parser.add_argument("--mol_csv", type=str, default="Dataset/New_QM9/molecule_properties.csv", help="Path to molecules CSV")
    parser.add_argument("--atom_csv", type=str, default="Dataset/New_QM9/atom_properties.csv", help="Path to atoms CSV")
    parser.add_argument("--save_path", type=str, default="GIN_2/outputs/GIN_model.pth", help="Optional custom model save path")
    parser.add_argument("--force_rebuild", action="store_true", help="Ignore cache and rebuild preprocessing")
    parser.add_argument("--quiet", action="store_true", help="Reduce preprocessing verbosity")
    args = parser.parse_args()

    # Paths fallback (Update these if your Paths class handles them differently)
    mol_path = args.mol_csv 
    atom_path = args.atom_csv 

    output_dir = args.save_path
    if not os.path.exists(output_dir):
        print(f"Output directory not found at '{output_dir}'. Creating it now...")
        os.makedirs(output_dir, exist_ok=True)
    probe_file = os.path.join(output_dir, f".dir_probe_{uuid.uuid4().hex}")
    open(probe_file, "a").close()
    print(f"Directory write probe created: {probe_file}")

    runner = main(
        mol_path=mol_path,
        atom_path=atom_path,
        force_rebuild=args.force_rebuild,
        save_path=args.save_path,
        verbose=not args.quiet,
    )
    runner.run()