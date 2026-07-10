import argparse
import csv
import json
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
        epochs: int = 50,
        learning_rate: float = 5e-5,
        max_length: int = 64,
        seed: int = 42,
        preprocess_only: bool = False,
    ):
        self.mol_path = mol_path
        self.use_cache = use_cache
        self.force_rebuild = force_rebuild
        self.cache_path = cache_path
        self.paths = Paths()
        self.output_dir = self.paths.get_output_dir()
        self.artifacts_dir = self.paths.get_artifacts_dir()
        self.save_path = save_path or self.paths.get_model_path()
        self.verbose = verbose
        self.preprocess_only = preprocess_only

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
        self.tokenizer_outputs = None
        
        if torch.cuda.is_available(): self.DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available(): self.DEVICE = torch.device("mps")
        else: self.DEVICE = torch.device("cpu")

    def _prepare_runtime_paths(self):
        self.save_path = os.path.abspath(self.save_path)
        if self.cache_path:
            self.cache_path = os.path.abspath(self.cache_path)

        save_dir = os.path.dirname(self.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        if self.cache_path:
            cache_dir = os.path.dirname(self.cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

        if self.verbose:
            print(f"Model checkpoint path: {self.save_path}")
            print(f"Preprocessing cache path: {self.cache_path or self.paths.get_tokenized_dataset_path()}")

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
        self.tokenizer_outputs = getattr(self.tokeniser, "last_payload", None)

        os.makedirs(self.artifacts_dir, exist_ok=True)
        if self.tokenizer_outputs is not None:
            tokenizer_bundle_path = os.path.join(self.artifacts_dir, "tokenizer_outputs.pt")
            torch.save(self.tokenizer_outputs, tokenizer_bundle_path)
            if self.verbose:
                print(f"Saved tokenizer outputs to: {tokenizer_bundle_path}")

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
        saved_input_ids = []
        saved_attention_masks = []
        saved_labels = []
        saved_nan_masks = []
        saved_predictions = []
        
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
                saved_input_ids.append(input_ids.detach().cpu())
                saved_attention_masks.append(attention_mask.detach().cpu())
                saved_labels.append(labels.detach().cpu())
                saved_nan_masks.append(nan_mask.detach().cpu())
                saved_predictions.append(predictions.detach().cpu())
                
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

        self._save_evaluation_outputs(
            test_loss=test_loss,
            input_ids=saved_input_ids,
            attention_masks=saved_attention_masks,
            labels=saved_labels,
            nan_masks=saved_nan_masks,
            predictions=saved_predictions,
        )
        
        return test_loss

    def _save_evaluation_outputs(self, test_loss, input_ids, attention_masks, labels, nan_masks, predictions):
        os.makedirs(self.artifacts_dir, exist_ok=True)

        if input_ids:
            input_ids_tensor = torch.cat(input_ids, dim=0)
            attention_mask_tensor = torch.cat(attention_masks, dim=0)
            labels_tensor = torch.cat(labels, dim=0)
            nan_mask_tensor = torch.cat(nan_masks, dim=0)
            predictions_tensor = torch.cat(predictions, dim=0)
        else:
            input_ids_tensor = torch.empty((0, 0), dtype=torch.long)
            attention_mask_tensor = torch.empty((0, 0), dtype=torch.long)
            labels_tensor = torch.empty((0, len(self.TARGET_COLS)), dtype=torch.float32)
            nan_mask_tensor = torch.empty((0, len(self.TARGET_COLS)), dtype=torch.float32)
            predictions_tensor = torch.empty((0, len(self.TARGET_COLS)), dtype=torch.float32)

        bundle = {
            "test_loss": float(test_loss),
            "target_cols": self.TARGET_COLS,
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels_tensor,
            "nan_mask": nan_mask_tensor,
            "predictions": predictions_tensor,
        }

        bundle_path = os.path.join(self.artifacts_dir, "evaluation_outputs.pt")
        torch.save(bundle, bundle_path)

        csv_path = os.path.join(self.artifacts_dir, "evaluation_outputs.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            fieldnames = ["row_index", "split_index", "test_loss"]
            for col in self.TARGET_COLS:
                fieldnames.extend([f"pred_{col}", f"label_{col}", f"mask_{col}"])

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for row_index in range(predictions_tensor.shape[0]):
                row = {
                    "row_index": row_index,
                    "split_index": row_index,
                    "test_loss": float(test_loss),
                }
                for col_index, col_name in enumerate(self.TARGET_COLS):
                    row[f"pred_{col_name}"] = float(predictions_tensor[row_index, col_index].item())
                    row[f"label_{col_name}"] = float(labels_tensor[row_index, col_index].item())
                    row[f"mask_{col_name}"] = float(nan_mask_tensor[row_index, col_index].item())
                writer.writerow(row)

        summary_path = os.path.join(self.artifacts_dir, "evaluation_summary.json")
        summary = {
            "test_loss": float(test_loss),
            "num_samples": int(predictions_tensor.shape[0]),
            "num_targets": int(len(self.TARGET_COLS)),
            "artifacts": {
                "bundle": bundle_path,
                "csv": csv_path,
            },
        }
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)

        if self.verbose:
            print(f"Saved evaluation outputs to: {bundle_path}")
            print(f"Saved evaluation CSV to: {csv_path}")
            print(f"Saved evaluation summary to: {summary_path}")
    
    def run(self):
        """Execute preprocessing, training, and evaluation end to end."""
        torch.manual_seed(self.SEED)

        self._prepare_runtime_paths()

        tprint("Molecular Property Prediction", font="block-medium")
        tprint("Transformer Training Pipeline", font="block-medium")
        print(f"Training device: {self.DEVICE}")

        self.preprocess()

        if self.preprocess_only:
            print("\nPreprocessing complete. Tokenized dataset cache has been generated.")
            print("Skipping training and evaluation as requested.")
            return

        self.build_model()
        self.train()
        self.evaluate()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Transformer fine-tuning runner")
    parser.add_argument("--mol_csv", type=str, default="Dataset/New_QM9/molecule_properties.csv", help="Path to molecules CSV")
    parser.add_argument("--force_rebuild", action="store_true", help="Ignore cache and rebuild preprocessing")
    parser.add_argument("--preprocess_only", action="store_true", help="Only run the preprocessing step and create the cache")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the best model checkpoint")
    parser.add_argument("--cache_path", type=str, default=None, help="Path to save/load preprocessing cache")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=64, help="Tokenizer max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    runner = main(
        mol_path=args.mol_csv,
        force_rebuild=args.force_rebuild,
        save_path=args.save_path,
        cache_path=args.cache_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        preprocess_only=args.preprocess_only,
    )
    runner.run()