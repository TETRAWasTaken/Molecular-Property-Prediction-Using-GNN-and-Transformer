import argparse
import os
import sys
from typing import Dict
import pandas as pd
import torch
from art import tprint
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from Transformers_2.Utils.Fine_Tuning import FineTuning
from Transformers_2.Utils.Tokeniser import Tokeniser
from Transformers_2.Utils.paths import Paths

class TransformerDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, nan_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.nan_mask = nan_mask

    def __len__(self) -> int: return len(self.input_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
            "nan_mask": self.nan_mask[index],
        }

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
        self.cache_version = 2 # Incremented to force rebuild with new logic
        self.scalers = {}
        
        if torch.cuda.is_available(): self.DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available(): self.DEVICE = torch.device("mps")
        else: self.DEVICE = torch.device("cpu")

    def _default_cache_path(self) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, "Transformers_2", "outputs", "cache", "tokenized_qm9_data.pt")

    def _source_signature(self) -> dict:
        mol_abs = os.path.abspath(self.mol_path)
        return {
            "mol_path": mol_abs,
            "mol_mtime": os.path.getmtime(mol_abs),
            "max_length": self.MAX_LENGTH,
            "target_cols": self.TARGET_COLS,
            "cache_version": self.cache_version,
        }

    def _load_cache(self, cache_path: str) -> bool:
        if not os.path.exists(cache_path):
            return False

        try:
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(cache_path, map_location="cpu")

            if payload.get("signature") != self._source_signature():
                if self.verbose:
                    print(f"Cache found at {cache_path}, but inputs changed. Rebuilding preprocessing...")
                return False

            if self.verbose:
                print(f"Loaded tokenized dataset cache from: {cache_path}")

            self._build_dataloaders_from_payload(payload)
            return True
        except Exception as exc:
            if self.verbose:
                print(f"Failed to load cache ({exc}). Rebuilding preprocessing...")
            return False

    def _save_cache(self, cache_path: str, payload: dict):
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        torch.save(payload, cache_path)
        if self.verbose:
            print(f"Saved preprocessing cache to: {cache_path}")

    def _split_indices(self, dataset_size: int) -> Dict[str, list[int]]:
        if dataset_size < 3:
            raise ValueError("Need at least 3 valid molecules to create train/val/test splits.")

        generator = torch.Generator().manual_seed(self.SEED)
        indices = torch.randperm(dataset_size, generator=generator).tolist()

        train_end = max(1, int(dataset_size * 0.8))
        val_end = max(train_end + 1, int(dataset_size * 0.9))
        val_end = min(val_end, dataset_size - 1)

        split_indices = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }

        if not split_indices["val"] or not split_indices["test"]:
            raise ValueError(
                "Unable to create non-empty train/val/test splits from the preprocessed dataset."
            )

        return split_indices

    def _slice_tensor_dict(self, tensor_dict: dict, indices: list[int]) -> dict:
        index_tensor = torch.tensor(indices, dtype=torch.long)
        return {key: value.index_select(0, index_tensor) for key, value in tensor_dict.items()}

    def _make_loader(self, tensor_dict: dict, shuffle: bool) -> DataLoader:
        dataset = TransformerDataset(
            input_ids=tensor_dict["input_ids"],
            attention_mask=tensor_dict["attention_mask"],
            labels=tensor_dict["labels"],
            nan_mask=tensor_dict["nan_mask"],
        )
        return DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=shuffle)

    def _build_dataloaders_from_payload(self, payload: dict):
        tensor_dict = {
            "input_ids": payload["input_ids"],
            "attention_mask": payload["attention_mask"],
            "labels": payload["labels"],
            "nan_mask": payload["nan_mask"],
        }
        split_indices = payload["split_indices"]

        self.train_loader = self._make_loader(
            self._slice_tensor_dict(tensor_dict, split_indices["train"]),
            shuffle=True,
        )
        self.val_loader = self._make_loader(
            self._slice_tensor_dict(tensor_dict, split_indices["val"]),
            shuffle=False,
        )
        self.test_loader = self._make_loader(
            self._slice_tensor_dict(tensor_dict, split_indices["test"]),
            shuffle=False,
        )

        if self.verbose:
            print(
                f"Train samples: {len(split_indices['train'])} | "
                f"Val samples: {len(split_indices['val'])} | "
                f"Test samples: {len(split_indices['test'])}"
            )

    def preprocess(self):
        effective_cache_path = self.cache_path or self._default_cache_path()

        if self.use_cache and not self.force_rebuild and self._load_cache(effective_cache_path):
            tprint("Preprocessing Completed")
            return

        # Initialize and run Dask-powered Tokeniser
        self.tokeniser = Tokeniser(
            mol_path=self.mol_path,
            model_name=self.MODEL_NAME,
            max_length=self.MAX_LENGTH,
        )
        self.tokeniser.load_data(verbose=self.verbose)
        self.tokeniser.canonicalize_smiles(verbose=self.verbose)
        
        df_target = self.tokeniser.df

        if self.verbose: print("Scaling targets dynamically...")
        
        self.scalers = {} 
        raw_targets = df_target[self.TARGET_COLS].values
        scaled_targets = np.copy(raw_targets)

        for i, col_name in enumerate(self.TARGET_COLS):
            col_data = raw_targets[:, i]
            mask = ~np.isnan(col_data)
            if mask.any():
                scaler = StandardScaler()
                scaled_targets[mask, i] = scaler.fit_transform(col_data[mask].reshape(-1, 1)).flatten()
                self.scalers[col_name] = scaler
        
        scaled_targets = np.nan_to_num(scaled_targets, nan=0.0)
        labels = torch.tensor(scaled_targets, dtype=torch.float32)

        nan_mask = torch.tensor(
            df_target[self.TARGET_COLS].notna().astype(int).values, dtype=torch.float32
        )
        smiles_list = df_target["smiles"].astype(str).tolist()

        if self.verbose:
            print(f"Tokenizing {len(smiles_list)} molecules natively via Hugging Face...")

        # Let the Rust tokenizer handle the entire list at once
        token_dict = self.tokeniser.encode_smiles_list(smiles_list)
        
        payload = {
            "input_ids": token_dict["input_ids"],
            "attention_mask": token_dict["attention_mask"],
            "labels": labels,
            "nan_mask": nan_mask,
            "split_indices": self._split_indices(len(smiles_list)),
            "signature": self._source_signature(),
        }

        self._build_dataloaders_from_payload(payload)

        if self.use_cache:
            self._save_cache(effective_cache_path, payload)

        tprint("Preprocessing Completed")

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



