import os
from typing import Dict, List, Optional
from rdkit import Chem
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from art import tprint
import numpy as np
from sklearn.preprocessing import StandardScaler
from .paths import Paths

from Scripts.qm9_delta import apply_qm9_delta_learning


class Tokeniser:
    """
    End-to-end tokenizer and preprocessing pipeline for molecular property prediction.

    Minimal usage:
        tokeniser = Tokeniser(mol_path="Dataset/New_QM9/molecule_properties.csv")
        tokeniser.run_tokenizer()
    """

    DEFAULT_TARGET_COLS = [
        "mu", "alpha", "homo", "lumo", "gap", "r2",
        "zpve", "u0", "u298", "h298", "g298", "cv",
    ]

    class _TransformerDataset(Dataset):
        def __init__(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor,
            nan_mask: torch.Tensor,
        ):
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

    def __init__(
        self,
        mol_path: str,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        max_length: int = 64,
        batch_size: int = 32,
        target_cols: Optional[List[str]] = None,
        use_cache: bool = True,
        force_rebuild: bool = False,
        cache_path: Optional[str] = None,
        cache_version: int = 2,
        seed: int = 42,
        verbose: bool = True,
    ):
        # We leave Rust's parallelism ON because we are letting it handle the tokenization natively
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

        self.verbose = verbose
        self.model_name = model_name
        try:
            self.tokeniser = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except OSError:
            self.tokeniser = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        self.max_length = max_length
        self.batch_size = batch_size
        self.mol_path = mol_path
        self.target_cols = target_cols or self.DEFAULT_TARGET_COLS
        self.use_cache = use_cache
        self.force_rebuild = force_rebuild
        self.cache_path = cache_path
        self.cache_version = cache_version
        self.seed = seed

        self.ddf = None
        self.df = None  # Computed Pandas dataframe after Dask finishes
        self.scalers: Dict[str, StandardScaler] = {}
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.last_payload: Optional[dict] = None

    def _default_cache_path(self) -> str:
        return Paths().get_tokenized_dataset_path()

    def _source_signature(self) -> dict:
        mol_abs = os.path.abspath(self.mol_path)
        return {
            "mol_path": mol_abs,
            "mol_mtime": os.path.getmtime(mol_abs),
            "model_name": self.model_name,
            "max_length": self.max_length,
            "target_cols": self.target_cols,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "cache_version": self.cache_version,
        }

    def _save_cache(self, cache_path: str, payload: dict):
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        torch.save(payload, cache_path)
        if self.verbose:
            print(f"Saved preprocessing cache to: {cache_path}")

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

            self.scalers = payload.get("scalers", {})
            self._build_dataloaders_from_payload(payload)
            return True
        except Exception as exc:
            if self.verbose:
                print(f"Failed to load cache ({exc}). Rebuilding preprocessing...")
            return False

    def _split_indices(self, dataset_size: int) -> Dict[str, List[int]]:
        if dataset_size < 3:
            raise ValueError("Need at least 3 valid molecules to create train/val/test splits.")

        generator = torch.Generator().manual_seed(self.seed)
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
            raise ValueError("Unable to create non-empty train/val/test splits from the preprocessed dataset.")

        return split_indices

    @staticmethod
    def _slice_tensor_dict(tensor_dict: dict, indices: List[int]) -> dict:
        index_tensor = torch.tensor(indices, dtype=torch.long)
        return {key: value.index_select(0, index_tensor) for key, value in tensor_dict.items()}

    def _make_loader(self, tensor_dict: dict, shuffle: bool) -> DataLoader:
        dataset = self._TransformerDataset(
            input_ids=tensor_dict["input_ids"],
            attention_mask=tensor_dict["attention_mask"],
            labels=tensor_dict["labels"],
            nan_mask=tensor_dict["nan_mask"],
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

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

    def load_data(self, verbose: bool = True) -> None:
        """Loads the dataset lazily using Dask."""
        self.ddf = dd.read_csv(self.mol_path, blocksize="25MB")
        if verbose:
            print(f"Dataset lazily loaded. Partitions: {self.ddf.npartitions}")

    @staticmethod
    def _canonicalize(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol) if mol else None
        except Exception:
            return None

    def canonicalize_smiles(self, verbose: bool = True):
        """Uses Dask to parallelize RDKit canonicalization across all cores."""
        if verbose: print("\nCanonicalizing SMILES strings across CPU cores...")
        
        # Map the RDKit function across all Dask partitions
        def apply_canon(df):
            df['smiles'] = df['smiles'].apply(self._canonicalize)
            return df
            
        with ProgressBar():
            # We compute() here to finalize the Dask graph into a Pandas DataFrame
            self.df = self.ddf.map_partitions(apply_canon).compute(scheduler='processes')

        # Drop invalid SMILES strings
        self.df = self.df[self.df['smiles'].notna()]
        if verbose: print(f"Dataset after canonicalization: {self.df.shape}")

    def encode_smiles_list(self, smiles_list: List[str]) -> Dict[str, torch.Tensor]:
        """
        Relies on Hugging Face's native Rust multithreading for maximum speed.
        """
        if not smiles_list: raise ValueError("smiles_list is empty.")
        
        # The Hugging Face tokenizer is incredibly fast when fed a massive list natively
        encodings = self.tokeniser(
            smiles_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }
    
    def _validate_dataframe(self, df_target):
        if "smiles" not in df_target.columns:
            raise ValueError("Dataset must contain a 'smiles' column.")

        missing_targets = [col for col in self.target_cols if col not in df_target.columns]
        if missing_targets:
            raise ValueError(
                f"Dataset is missing required target columns: {missing_targets}"
            )

    def run_tokenizer(self, verbose: bool = True):
        self.verbose = verbose
        effective_cache_path = self.cache_path or self._default_cache_path()

        if self.use_cache and not self.force_rebuild and self._load_cache(effective_cache_path):
            tprint("Preprocessing Completed")
            return {
                "train_loader": self.train_loader,
                "val_loader": self.val_loader,
                "test_loader": self.test_loader,
                "scalers": self.scalers,
                "target_cols": self.target_cols,
            }

        self.load_data(verbose=self.verbose)
        self.canonicalize_smiles(verbose=self.verbose)

        df_target = self.df
        self._validate_dataframe(df_target)
        df_target = apply_qm9_delta_learning(df_target, smiles_col='smiles', target_cols=self.target_cols)

        if self.verbose:
            print("Scaling targets dynamically...")
        
        self.scalers = {}
        raw_targets = df_target[self.target_cols].values
        scaled_targets = np.copy(raw_targets)

        for i, col_name in enumerate(self.target_cols):
            col_data = raw_targets[:, i]
            mask = ~np.isnan(col_data)
            if mask.any():
                scaler = StandardScaler()
                scaled_targets[mask, i] = scaler.fit_transform(col_data[mask].reshape(-1, 1)).flatten()
                self.scalers[col_name] = scaler
        
        scaled_targets = np.nan_to_num(scaled_targets, nan=0.0)
        labels = torch.tensor(scaled_targets, dtype=torch.float32)

        nan_mask = torch.tensor(
            df_target[self.target_cols].notna().astype(int).values, dtype=torch.float32
        )
        smiles_list = df_target["smiles"].astype(str).tolist()

        if self.verbose:
            print(f"Tokenizing {len(smiles_list)} molecules natively via Hugging Face...")

        # Let the Rust tokenizer handle the entire list at once
        token_dict = self.encode_smiles_list(smiles_list)
        
        payload = {
            "input_ids": token_dict["input_ids"],
            "attention_mask": token_dict["attention_mask"],
            "labels": labels,
            "nan_mask": nan_mask,
            "mol_ids": df_target["molecule_id"].tolist(),
            "split_indices": self._split_indices(len(smiles_list)),
            "scalers": self.scalers,
            "signature": self._source_signature(),
        }
        self.last_payload = payload

        self._build_dataloaders_from_payload(payload)

        if self.use_cache:
            self._save_cache(effective_cache_path, payload)

        tprint("Preprocessing Completed")
        return {
            "train_loader": self.train_loader,
            "val_loader": self.val_loader,
            "test_loader": self.test_loader,
            "scalers": self.scalers,
            "target_cols": self.target_cols,
        }