"""Graph Isomorphism Network (GIN) for molecular property regression.

Key improvements over the baseline:
- **Residual connections** after every GINEConv layer prevent feature washing
  in the 6-layer default stack.
- **ONNX-safe pooling**: replaces ``torch_geometric.nn.global_add_pool``
  (which depends on ``torch_scatter``) with a pure-PyTorch ``scatter_add``
  that ONNX Runtime can trace.
- **Learnable JK-Net weights**: a softmax-normalised ``nn.Parameter`` vector
  scales each layer's pooled output before concatenation, letting the model
  learn which message-passing depth is most informative.
- **``num_graphs`` parameter**: callers may pass the number of graphs
  explicitly so that ONNX tracing never needs ``batch.max().item()``.
"""

from typing import Optional

import torch
import torch.nn.functional as F
import torch_geometric
from torch.nn import (
    BatchNorm1d,
    Dropout,
    Linear,
    ReLU,
    Sequential,
)
from torch_geometric.nn import GINEConv


# ---------------------------------------------------------------------------
# ONNX-safe pooling helper
# ---------------------------------------------------------------------------

def _global_add_pool_safe(
    x: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """ONNX-traceable global-add pooling via :func:`scatter_add`.

    Drops the ``torch_scatter`` dependency so the operation survives
    ``torch.onnx.export``.  Functionally identical to
    ``torch_geometric.nn.global_add_pool``.

    Args:
        x: Node feature matrix, shape ``[total_nodes, feat_dim]``.
        batch: Graph assignment vector, shape ``[total_nodes]``.
        num_graphs: Total number of graphs in the batch.

    Returns:
        Graph-level feature matrix, shape ``[num_graphs, feat_dim]``.
    """
    out = torch.zeros(num_graphs, x.size(1), dtype=x.dtype, device=x.device)
    idx = batch.unsqueeze(1).expand(-1, x.size(1))
    return out.scatter_add(0, idx, x)


# ---------------------------------------------------------------------------
# GIN model
# ---------------------------------------------------------------------------

class GIN(torch.nn.Module):
    """Graph Isomorphism Network with residual connections and learnable JK-Net.

    Args:
        node_in_dim: Dimensionality of input node features (default 26).
        edge_in_dim: Dimensionality of input edge features (default 6).
        hidden_dim: Width of all internal layers.
        output_dim: Number of regression targets.
        num_layer: Number of GINEConv message-passing steps (default 6).
        dropout: Dropout probability used throughout the network.
    """

    def __init__(
        self,
        node_in_dim: int = 26,
        edge_in_dim: int = 6,
        hidden_dim: int = 256,
        output_dim: int = 12,
        num_layer: int = 6,
        dropout: float = 0.05,
    ) -> None:
        super(GIN, self).__init__()

        self.num_layer = num_layer
        self.dropout_rate = dropout

        # Input encoders
        self.node_encoder = Linear(node_in_dim, hidden_dim)
        self.edge_encoder = Linear(edge_in_dim, hidden_dim)

        # Virtual-node embedding (one shared vector per graph, initialised to 0)
        self.virtual_node_embedding = torch.nn.Embedding(1, hidden_dim)
        torch.nn.init.constant_(self.virtual_node_embedding.weight, 0)

        # Per-layer message-passing components
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.vn_mlp = torch.nn.ModuleList()

        for _ in range(num_layer):
            mlp = Sequential(
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
            self.vn_mlp.append(Sequential(
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Dropout(dropout),
            ))

        # Learnable JK-Net layer weights: softmax-normalised before concatenation
        # so the prediction head receives a weighted mix of all layer depths.
        self.jk_layer_weights = torch.nn.Parameter(torch.ones(num_layer + 1))

        # Final prediction head: input is the concatenation of (num_layer+1)
        # pooled graph embeddings, each scaled by its learned JK weight.
        self.prediction_head = Sequential(
            Linear(hidden_dim * (num_layer + 1), hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        data: torch_geometric.data.Data,
        num_graphs: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute graph-level regression predictions.

        Args:
            data: Batched PyG ``Data`` object with ``x``, ``edge_index``,
                ``edge_attr``, and ``batch`` attributes.
            num_graphs: Number of graphs in the batch.  When ``None`` the value
                is computed from ``data.batch`` using ``batch.max().item() + 1``
                (safe during training, **not** traceable by ONNX).  Pass
                explicitly when exporting to ONNX.

        Returns:
            Predicted properties, shape ``[num_graphs, output_dim]``.
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        if num_graphs is None:
            # Safe for training; avoided during ONNX export.
            num_graphs = int(batch.max()) + 1

        # --- Encode inputs ---
        h_list = [self.node_encoder(x)]
        edge_embeddings = self.edge_encoder(edge_attr)

        # Expand single virtual-node embedding to one vector per graph.
        # virtual_node_feat: [num_graphs, hidden_dim]
        virtual_node_feat = self.virtual_node_embedding.weight.expand(num_graphs, -1)

        # --- Message-passing with virtual node and residual connections ---
        for layer in range(self.num_layer):
            h_prev = h_list[layer]

            # Add broadcast virtual-node signal to every node in its graph.
            h = h_prev + virtual_node_feat[batch]
            h = self.convs[layer](h, edge_index, edge_embeddings)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)

            # Residual skip-connection: prevents feature washing in deep stacks.
            h = h + h_prev
            h_list.append(h)

            # Update virtual-node features (skip for the last layer).
            if layer < self.num_layer - 1:
                vn_agg = _global_add_pool_safe(h_list[-1], batch, num_graphs)
                vn_update = self.vn_mlp[layer](virtual_node_feat + vn_agg)
                virtual_node_feat = virtual_node_feat + vn_update

        # --- Jumping-Knowledge aggregation with learnable per-layer weights ---
        # Softmax ensures weights sum to 1, preserving the magnitude of each
        # layer's contribution while letting the model learn their relative
        # importance.
        jk_weights = torch.softmax(self.jk_layer_weights, dim=0)
        pooled_list = [
            _global_add_pool_safe(h, batch, num_graphs) for h in h_list
        ]
        scaled = [jk_weights[i] * pooled_list[i] for i in range(len(pooled_list))]
        h_graph = torch.cat(scaled, dim=1)  # [num_graphs, hidden_dim * (num_layer+1)]

        return self.prediction_head(h_graph)