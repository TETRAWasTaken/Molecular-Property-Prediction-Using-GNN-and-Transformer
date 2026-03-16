import argparse
import os
import sys
from typing import Dict

import pandas as pd
import torch
from art import tprint
from torch.utils.data import DataLoader, Dataset

# Support both `python Transformers/manual_run.py` and `python -m Transformers.manual_run`.
if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from Transformers.Utils.Fine_Tuning import FineTuning
from Transformers.Utils.Tokeniser import Tokeniser
from Transformers.Utils.paths import Paths


def _looks_like_git_lfs_pointer(file_path: str) -> bool:
    """Detect Git LFS pointer files so preprocessing fails with a useful message."""
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        return first_line == "version https://git-lfs.github.com/spec/v1"
    except (OSError, UnicodeDecodeError):
        return False


class TransformerDataset(Dataset):
    """Tensor-backed dataset matching FineTuning.train_transformer batches."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        nan_mask: torch.Tensor,
    ):
        if not (len(input_ids) == len(attention_mask) == len(labels) == len(nan_mask)):
            raise ValueError("Dataset tensors must all have the same number of rows.")

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.nan_mask = nan_mask

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
            "nan_mask": self.nan_mask[index],
        }


class main:
    """Manual runner for the standalone transformer fine-tuning pipeline."""

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
        batch_size: int = 32,
        epochs: int = 10,
        learning_rate: float = 5e-5,
        max_length: int = 64,
        seed: int = 42,
    ):
        self.model = None
        self.tokeniser = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.df_merged = None

        self.qm8_path = qm8_path
        self.qm9_path = qm9_path
        self.use_cache = use_cache
        self.force_rebuild = force_rebuild
        self.cache_path = cache_path
        self.save_path = save_path or Paths().get_model_path()
        self.verbose = verbose
        self.show_progress = show_progress

        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.LEARNING_RATE = learning_rate
        self.MAX_LENGTH = max_length
        self.SEED = seed
        self.MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
        self.TARGET_COLS = [
            "E1-CC2",
            "E2-CC2",
            "f1-CC2",
            "f2-CC2",
            "mu",
            "alpha",
            "homo",
            "lumo",
            "gap",
            "r2",
            "zpve",
            "u0",
        ]
        self.cache_version = 1
        self.DEVICE = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def _default_cache_path(self) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, "Transformers", "outputs", "cache", "tokenized_transformer_data.pt")

    def _validate_source_files(self):
        for file_path in (self.qm8_path, self.qm9_path):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            if _looks_like_git_lfs_pointer(file_path):
                raise ValueError(
                    f"Dataset file at '{file_path}' is a Git LFS pointer, not the real CSV. "
                    "Run 'git lfs pull' and try again."
                )

    def _source_signature(self) -> dict:
        qm8_abs = os.path.abspath(self.qm8_path)
        qm9_abs = os.path.abspath(self.qm9_path)
        return {
            "qm8_path": qm8_abs,
            "qm9_path": qm9_abs,
            "qm8_mtime": os.path.getmtime(qm8_abs),
            "qm9_mtime": os.path.getmtime(qm9_abs),
            "qm8_size": os.path.getsize(qm8_abs),
            "qm9_size": os.path.getsize(qm9_abs),
            "max_length": self.MAX_LENGTH,
            "target_cols": self.TARGET_COLS,
            "cache_version": self.cache_version,
            "model_name": self.MODEL_NAME,
            "seed": self.SEED,
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
        """Load, canonicalize, merge, tokenize, split, and optionally cache the dataset."""
        self._validate_source_files()
        effective_cache_path = self.cache_path or self._default_cache_path()

        if self.use_cache and not self.force_rebuild and self._load_cache(effective_cache_path):
            tprint("Preprocessing Completed")
            return

        if self.use_cache and self.verbose:
            cache_status = "found" if os.path.exists(effective_cache_path) else "not found"
            print(f"Cache {cache_status}: {effective_cache_path}")
        elif self.verbose:
            print("Cache disabled for this run.")

        self.tokeniser = Tokeniser(
            qm8_path=self.qm8_path,
            qm9_path=self.qm9_path,
            model_name=self.MODEL_NAME,
            max_length=self.MAX_LENGTH,
        )
        self.tokeniser.qm8_path = self.qm8_path
        self.tokeniser.qm9_path = self.qm9_path
        self.tokeniser.load_data(verbose=self.verbose)
        self.tokeniser.validate_data(verbose=self.verbose)
        self.tokeniser.canonicalize_smiles(verbose=self.verbose)

        self.df_merged = pd.merge(self.tokeniser.df8, self.tokeniser.df9, on="smiles", how="outer")
        if self.verbose:
            print(f"Merged dataset shape: {self.df_merged.shape}")

        labels = torch.tensor(
            self.df_merged[self.TARGET_COLS].fillna(0).values,
            dtype=torch.float32,
        )
        nan_mask = torch.tensor(
            self.df_merged[self.TARGET_COLS].notna().astype(int).values,
            dtype=torch.float32,
        )
        smiles_list = self.df_merged["smiles"].astype(str).tolist()

        if self.verbose:
            print(f"Tokenizing {len(smiles_list)} molecules with max_length={self.MAX_LENGTH}...")

        token_dict = self.tokeniser.parallel_batch_encode(smiles_list, max_length=self.MAX_LENGTH)
        tensor_payload = {
            "input_ids": token_dict["input_ids"],
            "attention_mask": token_dict["attention_mask"],
            "labels": labels,
            "nan_mask": nan_mask,
        }
        split_indices = self._split_indices(len(smiles_list))
        payload = {
            **tensor_payload,
            "split_indices": split_indices,
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
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.DEVICE)
                attention_mask = batch["attention_mask"].to(self.DEVICE)
                labels = batch["labels"].to(self.DEVICE)
                nan_mask = batch["nan_mask"].to(self.DEVICE)

                predictions = self.model(input_ids, attention_mask)
                loss = self.model.masked_mse_loss(predictions, labels, nan_mask)
                total_loss += loss.item()
                batch_count += 1

        test_loss = total_loss / max(batch_count, 1)
        print(f"Test masked MSE: {test_loss:.4f}")
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
    parser.add_argument("--qm8_path", type=str, default=None, help="Path to qm8.csv")
    parser.add_argument("--qm9_path", type=str, default=None, help="Path to qm9.csv")
    parser.add_argument("--cache_path", type=str, default=None, help="Optional custom cache file path")
    parser.add_argument("--save_path", type=str, default=None, help="Optional custom model save path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for fine-tuning")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum tokenizer sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting")
    parser.add_argument("--no_cache", action="store_true", help="Disable persistent preprocessing cache")
    parser.add_argument("--force_rebuild", action="store_true", help="Ignore cache and rebuild preprocessing")
    parser.add_argument("--quiet", action="store_true", help="Reduce preprocessing verbosity")
    parser.add_argument("--no_progress", action="store_true", help="Reserved for compatibility with sibling runners")
    args = parser.parse_args()

    paths = Paths()
    qm8 = args.qm8_path or paths.get_qm8_path()
    qm9 = args.qm9_path or paths.get_qm9_path()

    runner = main(
        qm8_path=qm8,
        qm9_path=qm9,
        use_cache=not args.no_cache,
        force_rebuild=args.force_rebuild,
        cache_path=args.cache_path,
        save_path=args.save_path,
        verbose=not args.quiet,
        show_progress=not args.no_progress,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
    )
    runner.run()




