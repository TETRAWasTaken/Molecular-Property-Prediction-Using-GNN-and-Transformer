import os
import torch
from GIN.Utils.GIN import MolecularPropertyPipeline
from GIN.Utils.GIN import TrainingTesting
from art import *


class main:
    """
    main class to manually run the molecular property prediction pipeline.
    """
    def __init__(self, qm8_path: str, qm9_path: str):
        self.model = None
        self.qm8_path = qm8_path
        self.qm9_path = qm9_path
        
        self.BATCH_SIZE = 64
        self.HIDDEN_DIM = 128
        self.OUTPUT_DIM = 12
        self.DROPOUT = 0.2
        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 5e-4
        self.EPOCHS = 25
        self.PATIENCE = 20
        self.DEVICE = torch.device("cpu") if torch.backends.mps.is_available() else "cpu"

    def preprocess(self):
        """Runs the preprocessing pipeline and returns data loaders."""
        pipeline = MolecularPropertyPipeline(self.qm8_path, self.qm9_path)
        self.train_loader, self.val_loader, self.test_loader = pipeline.run_full_pipeline(batch_size=self.BATCH_SIZE)
        
        # Infer feature dimensions
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
        
        while True:
            print("To perform data-preprocessing, Enter 'S'")
            command = input()
        
            if command.lower() == 's':
                self.preprocess()
                break
        
            else:
                print("Inavlid Input")
                continue
        
        print("Building Model Configuration")
        self.build_model()
        
        while True:
            print("To perform Training, Enter 'S'")
            print("To check training device, enter 'D'")
            command = input().lower()
            
            if command == 's':
                break
            elif command == 'd':
                print(f"Current Device: {self.DEVICE}")
            else:
                print("Invalid Input")
        
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

    def save_model(self, save_path: str = "./outputs/gnn_molecular_model.pth"):
        """Saves the trained model state to a file."""
        if self.model is None:
            print("No model to save.")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # Example paths - adjusted to likely structure based on context
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qm8 = os.path.join(base_dir, "Dataset", "qm8.csv")
    qm9 = os.path.join(base_dir, "Dataset", "qm9.csv")
    
    runner = main(qm8, qm9)
    runner.run()
