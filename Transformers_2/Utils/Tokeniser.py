import os
from typing import Dict, List, Optional
from rdkit import Chem
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import pandas as pd
import torch
from transformers import AutoTokenizer

class Tokeniser:
    """
    Dask-optimized Tokeniser for single-dataset processing (QM9 molecules).
    """
    def __init__(self, mol_path: str, model_name: str = "seyonec/ChemBERTa-zinc-base-v1", max_length: int = 64):
        # We leave Rust's parallelism ON because we are letting it handle the tokenization natively
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true") 
        self.model_name = model_name
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.mol_path = mol_path
        self.df = None # Will hold the computed Pandas dataframe after Dask finishes

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