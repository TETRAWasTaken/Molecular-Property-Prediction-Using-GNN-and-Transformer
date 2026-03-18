import argparse
import os
import sys
import torch
from art import tprint

# Ensure the Project Root is in the system path so absolute imports work
if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from Hybrid.Utils.paths import Paths
from Hybrid.Utils.preprocessing import HybridMolecularPipeline
from Hybrid.Models.fusion_network import HybridFusionNetwork
from Hybrid.Utils.trainer import HybridTrainer

class HybridRunner:
    """
    Interactive class to manually run the Multimodal Hybrid pipeline.
    """
    def __init__(self, qm8_path: str, qm9_path: str, save_path: str = None):
        self.qm8_path = qm8_path
        self.qm9_path = qm9_path
        self.save_path = save_path or Paths().get_model_path()
        
        # Hyperparameters
        self.BATCH_SIZE = 64
        self.GIN_HIDDEN_DIM = 128
        self.OUTPUT_DIM = 12
        self.DROPOUT = 0.2
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 1e-4
        self.EPOCHS = 25
        self.PATIENCE = 15
        
        self.DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.pipeline = None
        self.model = None
        self.trainer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.node_in_dim = None
        self.edge_in_dim = None

    @staticmethod
    def _read_choice(prompt: str, valid_choices: tuple) -> str:
        while True:
            print(prompt)
            choice = input().strip().lower()
            if choice in valid_choices:
                return choice
            print("Invalid input")

    def preprocess(self):
        """Runs the Hybrid preprocessing to generate Graphs + Tokens."""
        self.pipeline = HybridMolecularPipeline(self.qm8_path, self.qm9_path)
        self.train_loader, self.val_loader, self.test_loader = self.pipeline.run_pipeline(batch_size=self.BATCH_SIZE)
        
        # Infer feature dimensions from the first batch
        sample_batch = next(iter(self.train_loader))
        self.node_in_dim = sample_batch.num_node_features
        self.edge_in_dim = sample_batch.edge_attr.shape[1]
        
        tprint("Preprocessing Completed")

    def build_model(self):
        """Initializes the Fusion Network and the Trainer."""
        self.model = HybridFusionNetwork(
            node_in_dim=self.node_in_dim,
            edge_in_dim=self.edge_in_dim,
            gin_hidden_dim=self.GIN_HIDDEN_DIM,
            output_dim=self.OUTPUT_DIM,
            dropout=self.DROPOUT
        )
        
        self.trainer = HybridTrainer(
            model=self.model,
            learning_rate=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
            device=self.DEVICE
        )
        print(f"\nHybrid Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters.")

    def run(self):
        """Main execution flow."""
        tprint("Hybrid Multimodal", font='block-medium')
        tprint("Training Pipeline", font='block-medium')

        cmd = self._read_choice("\nTo perform data-preprocessing, enter 'S':", ("s",))
        if cmd == "s":
            self.preprocess()

        print("\nBuilding Model Configuration...")
        self.build_model()

        while True:
            cmd = self._read_choice(
                "\nTo perform training, enter 'S' | To check training device, enter 'D':",
                ("s", "d")
            )
            if cmd == "d":
                print(f"Current Device: {self.DEVICE}")
                continue
            break

        # Training
        self.trainer.run_training(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=self.EPOCHS,
            patience=self.PATIENCE,
            save_path=self.save_path
        )

        # Evaluation
        train_metrics = self.trainer.evaluate(self.train_loader)
        val_metrics = self.trainer.evaluate(self.val_loader)
        test_metrics = self.trainer.evaluate(self.test_loader)
        
        print(f"\n{'=' * 60}")
        print(f"Hybrid Train Results —  MAE: {train_metrics['mae']:.4f}  |  R²: {train_metrics['r2']:.4f}")
        print(f"Hybrid Val Results   —  MAE: {val_metrics['mae']:.4f}  |  R²: {val_metrics['r2']:.4f}")
        print(f"Hybrid Test Results  —  MAE: {test_metrics['mae']:.4f}  |  R²: {test_metrics['r2']:.4f}")
        print(f"{'=' * 60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Hybrid training runner")
    parser.add_argument("--qm8_path", type=str, default=None)
    parser.add_argument("--qm9_path", type=str, default=None)
    args = parser.parse_args()

    paths = Paths()
    qm8 = args.qm8_path or paths.get_qm8_path()
    qm9 = args.qm9_path or paths.get_qm9_path()

    runner = HybridRunner(qm8, qm9)
    runner.run()