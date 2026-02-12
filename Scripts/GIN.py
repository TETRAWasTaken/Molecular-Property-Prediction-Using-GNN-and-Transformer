import pandas as pd
import numpy as np
from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx as nx

from typing import Any
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d, Dropout
from torch_geometric.nn import GINConv, GINEConv
from torch_geometric.nn import global_add_pool, global_mean_pool
import torch_geometric
import torch.nn.functional as F

class GIN(torch.nn.Module):
    """
    This class is an implementation of Graph Isomorphism Network (GIN) for graph-level regression tasks.
    The GIN architecture is designed to capture the structural information of graphs effectively, making it suitable for tasks like molecular property prediction.

    :param node_in_dim: The dimensionality of the input node features.
    :param edge_in_dim: The dimensionality of the input edge features.
    :param hidden_dim: The dimensionality of the hidden layers in the GIN model.
    :param output_dim: The dimensionality of the output layer, which corresponds to the number of target properties to predict.
    :param dropout: The dropout rate for regularization during training.
    """
    def __init__(self, node_in_dim: int = 6,
                 edge_in_dim: int = 3,
                 hidden_dim: int = 128,
                 output_dim: int = 12,
                 dropout: float = 0.2):
        super(GIN, self).__init__()

        self.node_encoder = Linear(node_in_dim, hidden_dim)
        self.edge_encoder = Linear(edge_in_dim, hidden_dim)
        self.virtual_node_embedding = torch.nn.Parameter(torch.zeros(1, hidden_dim))

        self.vn_mlp1 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.vn_mlp2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.vn_mlp3 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))

        mlp1 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv1 = GINEConv(mlp1)
        self.bn1 = BatchNorm1d(hidden_dim)

        mlp2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv2 = GINEConv(mlp2)
        self.bn2 = BatchNorm1d(hidden_dim)

        mlp3 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv3 = GINEConv(mlp3)
        self.bn3 = BatchNorm1d(hidden_dim)

        self.dropout = Dropout(dropout)
        self.fc_out = Linear(hidden_dim, output_dim)

    def forward(self, data: torch_geometric.data.Data):
        """

        :param data:
        :return:
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        virtual_node_feat = self.virtual_node_embedding.expand(batch.max().item() + 1, -1)

        x = x + virtual_node_feat[batch]

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        virtual_node_feat = virtual_node_feat + global_add_pool(x, batch)
        virtual_node_feat = self.vn_mlp1(virtual_node_feat)

        x = x + virtual_node_feat[batch]
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)

        virtual_node_feat = virtual_node_feat + global_add_pool(x, batch)
        virtual_node_feat = self.vn_mlp2(virtual_node_feat)

        x = x + virtual_node_feat[batch]
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_add_pool(x, batch)

        x = self.dropout(x)
        return self.fc_out(x)
