import pandas as pd
import numpy as np
import torch_geometric
from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx as nx
import matplotlib.pyplot as plt

from GIN import GIN
from preprocessing import MolecularPropertyPipeline


class TrainingTesting(GIN):
    """
    A comprehensive class for training and testing the GIN model on a given dataset.
    This class encapsulates the entire workflow, including data loading, model initialization, training loop, and evaluation metrics.
    It is designed to be flexible and can be easily adapted to different datasets and configurations.
    """
    def __init__(self, dataset: torch_geometric.data.Data):
        super().__init__(dataset.num_node_features, dataset.num_node_labels)
        self.dataset = dataset
        self.epochs = 50
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.001,
                                          weight_decay=0.0005)
        self.criterion = torch.nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.5,
                                                                    patience=10,
                                                                    verbose=True)
    def training(self, dataset: Any) -> None:
        """
        This method handles the training loop for the GIN model. It iterates over the training dataset, computes the loss, and updates the model parameters using backpropagation.
        The method also includes functionality for tracking training progress and can be extended to include features like early stopping or learning rate scheduling.

        :param dataset:
        :return:
        """
        self.train()
