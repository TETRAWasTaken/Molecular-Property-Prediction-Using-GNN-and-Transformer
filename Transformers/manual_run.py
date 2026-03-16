from Utils import *
import argparse 
import os
import sys
from rdkit import Chem
from multiprocessing import Pool, cpu_count

import torch
from art import *

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from Transformers.Utils.Tokeniser import Tokeniser
from Transformers.Utils.Fine_Tuning import FineTuning
from Transformers.Utils.paths import Paths
from Transformers.Utils.Transformer import StandaloneChemBERTa

class main:
    """
    main class to manually run the molecular property prediction pipeline.
    """
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
    ):
        self.model = None
        self.qm8_path = qm8_path
        self.qm9_path = qm9_path

        self.use_cache = use_cache
        self.force_rebuild = force_rebuild
        self.cache_path = cache_path
        self.save_path = save_path or Paths().get_model_path()
        self.verbose = verbose
        self.show_progress = show_progress

    def data_loading(self):
        """Load and preprocess the QM8 and QM9 datasets."""
        if self.verbose:
            print("Loading datasets...")
        self.tokeniser = Tokeniser(
            qm8_path=self.qm8_path,
            qm9_path=self.qm9_path,
            model_name="seyonec/ChemBERTa-zinc-base-v1",
        )
        self.tokeniser.canonicalize_smiles(verbose=self.verbose)    

    


