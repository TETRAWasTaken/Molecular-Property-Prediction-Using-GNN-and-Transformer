from typing import Dict

import torch
from torch_geometric.data import InMemoryDataset
from transformers import AutoTokenizer

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
    :ivar tokeniser: The actual tokenization component or object. Handles
        the conversion of input text to tokenized outputs based on the
        specific model indicated by `model_name`.
    :type tokeniser: Any
    """
    def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MTR",
                 max_length: int = 128):
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

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