import os
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count

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
        padding=True,
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
    
    """
    def __init__(self, qm8_path: str, qm9_path: str,
                 model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
                 max_length: int = 64):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        self.model_name = model_name
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.df8 = pd.read_csv(qm8_path)
        self.df9 = pd.read_csv(qm9_path)

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
        Notebook-style batch tokenization:
        tokenizer(smiles_list, padding=True, truncation=True, max_length=64, return_tensors='pt')
        """
        if not smiles_list:
            raise ValueError("smiles_list is empty. Provide at least one SMILES string.")

        effective_max_length = max_length if max_length is not None else self.max_length

        encodings = self.tokeniser(
            smiles_list,
            padding=True,
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

