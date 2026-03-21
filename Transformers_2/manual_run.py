import argparse
import os
import sys
import torch
from art import tprint

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from Transformers_2.Utils.Fine_Tuning import FineTuning
from Transformers_2.Utils.Tokeniser import Tokeniser
from Transformers_2.Utils.paths import Paths

class main:
    def __init__(
        self,
        mol_path: str,
        use_cache: bool = True,
        force_rebuild: bool = False,
        cache_path: str = None,
        save_path: str = None,
        verbose: bool = True,
        batch_size: int = 32,
        epochs: int = 20,
        learning_rate: float = 5e-5,
        max_length: int = 64,
        seed: int = 42,
    ):
        self.mol_path = mol_path
        self.use_cache = use_cache
        self.force_rebuild = force_rebuild
        self.cache_path = cache_path
        self.save_path = save_path or Paths().get_model_path()
        self.verbose = verbose

        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.LEARNING_RATE = learning_rate
        self.MAX_LENGTH = max_length
        self.SEED = seed
        self.MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
        
        self.TARGET_COLS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
        self.scalers = {}
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.tokeniser = None
        
        if torch.cuda.is_available(): self.DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available(): self.DEVICE = torch.device("mps")
        else: self.DEVICE = torch.device("cpu")

    def preprocess(self):
        self.tokeniser = Tokeniser(
            mol_path=self.mol_path,
            model_name=self.MODEL_NAME,
            max_length=self.MAX_LENGTH,
            batch_size=self.BATCH_SIZE,
            target_cols=self.TARGET_COLS,
            use_cache=self.use_cache,
            force_rebuild=self.force_rebuild,
            cache_path=self.cache_path,
            seed=self.SEED,
            verbose=self.verbose,
        )

        artifacts = self.tokeniser.run_tokenizer(verbose=self.verbose)
        self.train_loader = artifacts["train_loader"]
        self.val_loader = artifacts["val_loader"]
        self.test_loader = artifacts["test_loader"]
        self.scalers = artifacts["scalers"]
        self.TARGET_COLS = artifacts["target_cols"]

    def build_model(self):
        """Initialize the ChemBERTa fine-tuning model."""
        self.model = FineTuning(model_name=self.MODEL_NAME, num_labels=len(self.TARGET_COLS))
        print(f"Model initialized on target device {self.DEVICE}.")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self):
        """Run the fine-tuning loop and persist the best checkpoint."""
        if self.model is None:
            raise RuntimeError("Model has not been initialized. Call build_model() first.")

        save_dir = os.path.dirname(self.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        self.model.train_transformer(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=self.EPOCHS,
            lr=self.LEARNING_RATE,
            save_path=self.save_path,
            device=self.DEVICE,
            target_cols=self.TARGET_COLS,
            scalers=self.scalers
        )

    def evaluate(self) -> float:
        """Evaluate the best saved checkpoint on the held-out test split."""
        if self.model is None:
            raise RuntimeError("Model has not been initialized. Call build_model() first.")
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.save_path}")

        try:
            state_dict = torch.load(self.save_path, map_location=self.DEVICE, weights_only=True)
        except TypeError:
            state_dict = torch.load(self.save_path, map_location=self.DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.to(self.DEVICE)
        self.model.eval()

        total_loss = 0.0
        batch_count = 0
        
        # Trackers for column-wise loss
        num_targets = len(self.TARGET_COLS)
        col_squared_errors = torch.zeros(num_targets).to(self.DEVICE)
        col_counts = torch.zeros(num_targets).to(self.DEVICE)

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.DEVICE)
                attention_mask = batch["attention_mask"].to(self.DEVICE)
                labels = batch["labels"].to(self.DEVICE)
                nan_mask = batch["nan_mask"].to(self.DEVICE)

                predictions = self.model(input_ids, attention_mask)
                
                # Overall loss (your original code)
                loss = self.model.masked_mse_loss(predictions, labels, nan_mask)
                total_loss += loss.item()
                batch_count += 1
                
                # Column-wise loss calculation
                squared_errors = (predictions - labels) ** 2
                masked_errors = squared_errors * nan_mask
                
                # Sum errors and counts down the batch dimension (dim=0)
                col_squared_errors += masked_errors.sum(dim=0)
                col_counts += nan_mask.sum(dim=0)

        # Print the breakdown
        print("\n" + "="*35)
        print("  Per-Property Scaled Test MSE")
        print("="*35)
        
        col_mse = col_squared_errors / torch.clamp(col_counts, min=1.0)
        for i, col_name in enumerate(self.TARGET_COLS):
            print(f"{col_name.ljust(10)} : {col_mse[i].item():.4f}")
        
        print("="*35)

        test_loss = total_loss / max(batch_count, 1)
        print(f"Overall Test masked MSE: {test_loss:.4f}\n")
        
        return test_loss
    
    def run(self):
        """Execute preprocessing, training, and evaluation end to end."""
        torch.manual_seed(self.SEED)

        tprint("Molecular Property Prediction", font="block-medium")
        tprint("Transformer Training Pipeline", font="block-medium")
        print(f"Training device: {self.DEVICE}")

        self.preprocess()
        self.build_model()
        self.train()
        self.evaluate()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Transformer fine-tuning runner")
    parser.add_argument("--mol_csv", type=str, default="Dataset/New_QM9/molecule_properties.csv", help="Path to molecules CSV")
    parser.add_argument("--force_rebuild", action="store_true", help="Ignore cache and rebuild preprocessing")
    args = parser.parse_args()

    runner = main(
        mol_path=args.mol_csv,
        force_rebuild=args.force_rebuild,
    )
    runner.run()



