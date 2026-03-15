import torch
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d, Dropout
from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_add_pool
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
    def __init__(self, 
                 node_in_dim: int = 6,
                 edge_in_dim: int = 3,
                 hidden_dim: int = 128,
                 output_dim: int = 12,
                 num_layer: int = 5,
                 dropout: float = 0.2):
        super(GIN, self).__init__()

        self.num_layer = num_layer
        self.dropout_rate = dropout

        self.node_encoder = Linear(node_in_dim, hidden_dim)
        self.edge_encoder = Linear(edge_in_dim, hidden_dim)
        
        self.virtual_node_embedding = torch.nn.Embedding(1, hidden_dim)  # Learnable virtual node embedding
        torch.nn.init.constant_(self.virtual_node_embedding.weight, 0)  # Initialize virtual node embedding to zero

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.vn_mlp = torch.nn.ModuleList()

        for _ in range(num_layer):
            mlp = Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Dropout(dropout)
            )
            self.convs.append(GINEConv(mlp))
            self.batch_norms.append(BatchNorm1d(hidden_dim))

            self.vn_mlp.append(Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Dropout(dropout)
            ))

        
        self.prediction_head = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, output_dim)
        )

    def forward(self, data: torch_geometric.data.Data):
        """

        :param data: torch_geometric.data.Data object containing the graph data, including node features, edge indices, edge attributes, and batch information.:
        :return:
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        h_list = [self.node_encoder(x)]
        edge_embeddings = self.edge_encoder(edge_attr)

        virtual_node_feat = self.virtual_node_embedding.weight.expand(batch.max().item() + 1, -1)

        for layer in range(self.num_layer):
            h = h_list[layer] + virtual_node_feat[batch]
            h = self.convs[layer](h, edge_index, edge_embeddings)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            h_list.append(h)

            if layer < self.num_layer - 1:
                virtual_node_feat = virtual_node_feat + global_add_pool(h, batch)
                virtual_node_feat = self.vn_mlp[layer](virtual_node_feat)
            
        h_graph = global_add_pool(h_list[-1], batch)

        return self.prediction_head(h_graph)