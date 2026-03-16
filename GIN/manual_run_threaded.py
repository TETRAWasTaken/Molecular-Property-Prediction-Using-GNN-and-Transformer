import argparse
import os
import sys

import torch
from art import *

# Support both `python GIN/manual_run.py` and `python -m GIN.manual_run`.
if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from GIN.Utils.paths import Paths
from GIN.Utils.preprocessing_multithreading import MolecularPropertyPipeline
from GIN.Utils.TrainingTesting import TrainingTesting


class main:
    """
    main class to manually run the molecular property prediction pipeline.
    """
    def __init__(
        self,
        qm8_path: str,
        qm9_path: str,
        use_cache: bool = True,
        force_rebuild: bool = False,
        cache_path: str = None,
        save_path: str = None,
        verbose: bool = True,
        show_progress: bool = True,
    ):
        self.model = None
        self.qm8_path = qm8_path
        self.qm9_path = qm9_path

        self.use_cache = use_cache
        self.force_rebuild = force_rebuild
        self.cache_path = cache_path
        self.save_path = save_path or Paths().get_model_path()
        self.verbose = verbose
        self.show_progress = show_progress

        self.BATCH_SIZE = 64
        self.HIDDEN_DIM = 128
        self.OUTPUT_DIM = 12
        self.DROPOUT = 0.2
        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 5e-4
        self.EPOCHS = 25
        self.PATIENCE = 20
        self.DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _read_choice(self, prompt: str, valid_choices: tuple[str, ...]) -> str:
        """Small helper to keep interactive command handling consistent."""
        while True:
            print(prompt)
            choice = input().strip().lower()
            if choice in valid_choices:
                return choice
            print("Invalid input")

    def preprocess(self):
        """Runs preprocessing with optional persistent cache and returns data loaders."""
        pipeline = MolecularPropertyPipeline(self.qm8_path, self.qm9_path)
        effective_cache_path = self.cache_path or pipeline._default_cache_path()

        if self.use_cache:
            cache_status = "found" if os.path.exists(effective_cache_path) else "not found"
            print(f"Cache {cache_status}: {effective_cache_path}")
        else:
            print("Cache disabled for this run.")

        self.train_loader, self.val_loader, self.test_loader = pipeline.run_full_pipeline(
            batch_size=self.BATCH_SIZE,
            use_cache=self.use_cache,
            force_rebuild=self.force_rebuild,
            cache_path=self.cache_path,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )

        sample_batch = next(iter(self.train_loader))
        self.node_in_dim = sample_batch.num_node_features
        self.edge_in_dim = sample_batch.edge_attr.shape[1]
        tprint("Preprocessing Completed")

    def build_model(self):
        """Initializes the TrainingTesting model."""
        self.model = TrainingTesting(
            node_in_dim=self.node_in_dim,
            edge_in_dim=self.edge_in_dim,
            hidden_dim=self.HIDDEN_DIM,
            output_dim=self.OUTPUT_DIM,
            dropout=self.DROPOUT,
            learning_rate=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
            device=self.DEVICE
        )
        print(f"\nModel initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters.")

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
            patience=self.PATIENCE
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
    parser.add_argument("--qm8_path", type=str, default=None, help="Path to qm8.csv")
    parser.add_argument("--qm9_path", type=str, default=None, help="Path to qm9.csv")
    parser.add_argument("--cache_path", type=str, default=None, help="Optional custom cache file path")
    parser.add_argument("--save_path", type=str, default=None, help="Optional custom model save path")
    parser.add_argument("--no_cache", action="store_true", help="Disable persistent preprocessing cache")
    parser.add_argument("--force_rebuild", action="store_true", help="Ignore cache and rebuild preprocessing")
    parser.add_argument("--quiet", action="store_true", help="Reduce preprocessing verbosity")
    parser.add_argument("--no_progress", action="store_true", help="Disable preprocessing progress bar")
    args = parser.parse_args()

    paths = Paths()
    qm8 = args.qm8_path or paths.get_qm8_path()
    qm9 = args.qm9_path or paths.get_qm9_path()

    runner = main(
        qm8,
        qm9,
        use_cache=not args.no_cache,
        force_rebuild=args.force_rebuild,
        cache_path=args.cache_path,
        save_path=args.save_path,
        verbose=not args.quiet,
        show_progress=not args.no_progress,
    )
    runner.run()