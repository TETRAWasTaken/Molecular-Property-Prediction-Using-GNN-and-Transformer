import os
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from rdkit import Chem

import pandas as pd
import torch
from transformers import AutoTokenizer

def _tokenize_smiles_chunk(args: Tuple[List[str], str, int]) -> Dict[str, torch.Tensor]:
    """
    Worker function for process-based parallel tokenization.
    """
    smiles_chunk, model_name, max_length = args
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = tokenizer(
        smiles_chunk,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    }

class Tokeniser:
    """
    A class to handle tokenization of input text for machine learning models.

    This class is designed to preprocess input data into tokenized structures
    compatible with specific machine learning models. It uses a pre-defined
    tokenization model and processes the input text while enforcing constraints
    such as maximum length. It simplifies and standardizes the preparation of
    data for use in neural networks or other similar applications.

    :ivar model_name: The name of the model used for tokenization.
    :type model_name: str
    :ivar max_length: The maximum length of a tokenized sequence.
    :type max_length: int
    :ivar qm8_path: The file path to the QM8 dataset.
    :type qm8_path: str
    :ivar qm9_path: The file path to the QM9 dataset.
    :type qm9_path: str
    
    """
    def __init__(self, qm8_path: str, qm9_path: str,
                 model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
                 max_length: int = 64):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        self.model_name = model_name
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    @staticmethod
    def _validate_expected_columns(df: pd.DataFrame, dataset_name: str, dataset_path: str):
        """Fail fast with an actionable message when required dataset columns are missing."""
        required_columns = {"smiles"}
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            preview_columns = ", ".join(map(str, df.columns[:5])) or "<no columns>"
            raise ValueError(
                f"{dataset_name} dataset at '{dataset_path}' is missing required columns: {missing_columns}. "
                f"Found columns: {preview_columns}. "
                "If this file came from a Git clone, verify Git LFS assets were downloaded with 'git lfs pull'."
            )
        
    def validate_data(self, verbose: bool = True):
        """Check for missing values and data quality issues."""
        if not verbose:
            return
        print("\n--- QM8 Data Quality ---")
        print(f"Null values:\n{self.df8.isnull().sum()}")

        print("\n--- QM9 Data Quality ---")
        print(f"Null values:\n{self.df9.isnull().sum()}")

        # Check for empty SMILES
        empty_smiles_qm8 = (self.df8['smiles'].str.strip() == "").sum()
        empty_smiles_qm9 = (self.df9['smiles'].str.strip() == "").sum()
        print(f"\nEmpty SMILES in QM8: {empty_smiles_qm8}")
        print(f"Empty SMILES in QM9: {empty_smiles_qm9}")


    def load_data(self, verbose: bool = True) -> None:
        """
        Load QM8 and QM9 CSV files.
        """
        self.df8 = pd.read_csv(self.qm8_path)
        self.df9 = pd.read_csv(self.qm9_path)

        self._validate_expected_columns(self.df8, "QM8", self.qm8_path)
        self._validate_expected_columns(self.df9, "QM9", self.qm9_path)

        if verbose:
            print(f"QM8 dataset loaded with shape: {self.df8.shape}")
            print(f"QM9 dataset loaded with shape: {self.df9.shape}")

    @staticmethod
    def _chunk_smiles(smiles_list: List[str], chunk_size: int) -> List[List[str]]:
        """
        Splits SMILES into fixed-size chunks for parallel processing.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0.")
        return [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]

    @staticmethod
    def smiles_to_list(df: pd.DataFrame, smiles_col: str = "smiles_new") -> List[str]:
        """
        Converts a SMILES column to a clean Python list.
        """
        if smiles_col not in df.columns:
            raise KeyError(f"Column '{smiles_col}' not found in the provided dataframe.")

        smiles_series = df[smiles_col].dropna().astype(str)
        return smiles_series.tolist()

    @staticmethod
    def smiles_lengths(smiles_list: List[str]) -> List[int]:
        """
        Computes character lengths for each SMILES string.
        """
        return [len(s) for s in smiles_list]

    @staticmethod
    def max_smiles_length(smiles_list: List[str]) -> int:
        """
        Returns the maximum SMILES string length.
        """
        if not smiles_list:
            raise ValueError("smiles_list is empty. Cannot compute max length.")
        return max(len(s) for s in smiles_list)
    
    @staticmethod
    def _canonicalize(smiles):
        """Convert a SMILES string to its canonical form."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol) if mol else None
        except Exception:
            return None

    def canonicalize_smiles(
        self,
        verbose: bool = True,
        n_jobs: int = -1,
    ):
        """Canonicalize SMILES in both datasets."""
        if verbose:
            print("\nCanonicalizing SMILES strings...")

        smiles8 = self.df8['smiles'].tolist()
        smiles9 = self.df9['smiles'].tolist()

        num_cores = cpu_count() if n_jobs == -1 else n_jobs
        with Pool(num_cores) as pool:
            if verbose:
                print(f"Using {num_cores} cores for canonicalization.")

            self.df8['smiles'] = pool.map(self._canonicalize, smiles8)
            self.df9['smiles'] = pool.map(self._canonicalize, smiles9)

        # Remove rows with invalid SMILES
        self.df8 = self.df8[self.df8['smiles'].notna()]
        self.df9 = self.df9[self.df9['smiles'].notna()]

        if verbose:
            print(f"QM8 after canonicalization: {self.df8.shape}")
            print(f"QM9 after canonicalization: {self.df9.shape}")

    def encode(self, smiles: str) -> Dict[str, torch.Tensor]:
        """
        Encodes a given SMILES (Simplified Molecular Input Line Entry System) string into a
        dictionary containing PyTorch tensors. This method facilitates the transformation
        of chemical data into a machine-readable format suitable for deep learning models.

        :param smiles: The SMILES string representing a chemical compound.
        :type smiles: str
        :return: A dictionary where keys are strings and values are PyTorch tensors
                 representing the encoded form of the SMILES input.
        :rtype: Dict[str, torch.Tensor]
        """

        encoded = self.tokeniser(
            smiles,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ) 

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }

    def batch_encode(self, smiles_list: List[str], max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Batch tokenization with deterministic tensor width across all calls.
        """
        if not smiles_list:
            raise ValueError("smiles_list is empty. Provide at least one SMILES string.")

        effective_max_length = max_length if max_length is not None else self.max_length

        encodings = self.tokeniser(
            smiles_list,
            padding='max_length',
            truncation=True,
            max_length=effective_max_length,
            return_tensors='pt'
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }

    def parallel_batch_encode(
        self,
        smiles_list: List[str],
        max_length: Optional[int] = None,
        chunk_size: int = 4096,
        max_workers: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process-based parallel batch tokenization for very large SMILES lists.
        """
        if not smiles_list:
            raise ValueError("smiles_list is empty. Provide at least one SMILES string.")

        effective_max_length = max_length if max_length is not None else self.max_length
        chunks = self._chunk_smiles(smiles_list, chunk_size)

        if len(chunks) == 1:
            return self.batch_encode(smiles_list, max_length=effective_max_length)

        worker_count = max_workers if max_workers is not None else cpu_count()
        worker_count = max(1, worker_count if worker_count is not None else 1)
        worker_args = [(chunk, self.model_name, effective_max_length) for chunk in chunks]

        with Pool(worker_count) as pool:
            chunk_results = pool.map(_tokenize_smiles_chunk, worker_args)

        return {
            "input_ids": torch.cat([result["input_ids"] for result in chunk_results], dim=0),
            "attention_mask": torch.cat([result["attention_mask"] for result in chunk_results], dim=0),
        }
    
    def run(
        self,
        smiles_list: List[str],
        parallel: bool = False,
        chunk_size: int = 4096,
        max_length: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Backward-compatible entry point for batch tokenization.
        """
        if parallel:
            return self.parallel_batch_encode(
                smiles_list,
                max_length=max_length,
                chunk_size=chunk_size,
                max_workers=max_workers,
            )

        return self.batch_encode(smiles_list, max_length=max_length)
    
    def run_full_pipeline(self, verbose: bool = True):
        """Convenience method to run the entire tokenization pipeline."""
        self.load_data(verbose=verbose)
        self.validate_data(verbose=verbose)
        self.canonicalize_smiles(verbose=verbose)

        self.parallel_batch_encode(
            self.smiles_list,
            parallel=True,
            max_length=self.max_length,
            chunk_size=4096,
            max_workers=None,
        )

