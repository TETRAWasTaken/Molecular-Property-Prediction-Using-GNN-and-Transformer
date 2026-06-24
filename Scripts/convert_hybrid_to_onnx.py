"""Export HybridFusionModel to a batch-capable ONNX model.

Key improvements over the original exporter
--------------------------------------------
1. **``ExportFriendlyBilinear``** — uses ``torch.einsum`` instead of the
   previous ``unsqueeze`` chain, which produced an intermediate tensor of
   shape ``[B, out, in, in]`` (134 M elements for ``hidden_dim=512``).
   The einsum formulation is both correct and memory-efficient.

2. **``BatchHybridOnnxWrapper``** — the new wrapper accepts a full flattened
   graph batch (multiple molecules in one call) and derives ``num_graphs``
   from ``input_ids.shape[0]``.  This is an ONNX-traceable ``Shape`` op, so
   ONNX Runtime substitutes the real batch size at inference time.

3. **ONNX-safe GIN pooling** — the GIN now uses ``scatter_add`` internally
   (no ``torch_scatter``), so all ops survive ``torch.onnx.export``.

4. **Updated ``dynamic_axes``** — ``total_nodes`` and ``total_edges`` are
   separate dynamic dimensions from ``batch_size``, correctly reflecting
   that a batch of N molecules can have very different node/edge counts.

Usage
-----
::

    python Scripts/convert_hybrid_to_onnx.py \\
        --pth_path models/best_hybrid_model.pth \\
        --onnx_path GUI/assets/hybrid_model.onnx \\
        --batch_size 2

Then verify the export::

    python Scripts/verify_onnx_batch.py --onnx_path GUI/assets/hybrid_model.onnx
"""

import argparse
from pathlib import Path
import sys
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import HybridFusionModel


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _extract_state_dict(payload):
    """Unwrap a checkpoint dict to extract the bare state_dict."""
    if isinstance(payload, Mapping):
        if isinstance(payload.get("state_dict"), Mapping):
            return payload["state_dict"]
        if isinstance(payload.get("model_state_dict"), Mapping):
            return payload["model_state_dict"]
        if any(
            str(k).startswith("graph_encoder.") or str(k).startswith("fusion_mlp.")
            for k in payload.keys()
        ):
            return payload
    raise ValueError(
        "Could not find a valid HybridFusionModel state_dict in the checkpoint"
    )


def _infer_model_dimensions(state_dict):
    """Infer gin_hidden_dim, mlp_hidden_dim and output_dim from weight shapes."""
    required_keys = [
        "graph_encoder.node_encoder.weight",
        "fusion_mlp.0.weight",
        "fusion_mlp.8.weight",
    ]
    missing = [k for k in required_keys if k not in state_dict]
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")

    return {
        "gin_hidden_dim": int(state_dict["graph_encoder.node_encoder.weight"].shape[0]),
        "mlp_hidden_dim": int(state_dict["fusion_mlp.0.weight"].shape[0]),
        "output_dim":     int(state_dict["fusion_mlp.8.weight"].shape[0]),
    }


# ---------------------------------------------------------------------------
# ONNX-friendly bilinear
# ---------------------------------------------------------------------------

class ExportFriendlyBilinear(nn.Module):
    """``nn.Bilinear`` replacement that survives ONNX export.

    The standard ``nn.Bilinear`` and its naive manual expansion produce an
    ``[B, out_features, in_features, in_features]`` intermediate that hits
    ONNX shape-inference limits for large hidden dims.  This implementation
    uses ``torch.einsum`` which maps cleanly to the ``Einsum`` ONNX op
    (supported since opset 12).

    The bilinear form is:
        out[b, k] = Σ_{i,j} weight[k, i, j] · x1[b, i] · x2[b, j] + bias[k]
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, in_features))
        nn.init.xavier_uniform_(self.weight.view(out_features, -1)).view_as(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        # einsum: 'bi,kij,bj->bk'
        # B = batch, i/j = in_features, k = out_features
        out = torch.einsum("bi,kij,bj->bk", input1, self.weight, input2)
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# ONNX export wrapper — supports a batch of N molecules
# ---------------------------------------------------------------------------

class BatchHybridOnnxWrapper(nn.Module):
    """Flat-tensor wrapper that exposes the hybrid model to ONNX Runtime.

    Inputs (all are flat tensors — no PyG ``Data`` objects)
    --------------------------------------------------------
    x               : float32 [total_nodes, node_feat_dim]
    edge_index      : int64   [2, total_edges]
    edge_attr       : float32 [total_edges, edge_feat_dim]
    batch           : int64   [total_nodes]  — graph index per node (0…N-1)
    input_ids       : int64   [N, seq_len]
    attention_mask  : int64   [N, seq_len]

    Output
    ------
    predicted_properties : float32 [N, output_dim]

    ``num_graphs`` is derived from ``input_ids.shape[0]``, which is an
    ONNX-traceable ``Shape + Gather`` op.  ONNX Runtime substitutes the
    actual batch size at inference time when ``batch_size`` is declared as a
    dynamic axis.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Derive num_graphs from the transformer input: ONNX traces this as
        # Shape(input_ids)[0], which is fully dynamic.
        num_graphs = input_ids.shape[0]
        graph_data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch
        )
        return self.model(
            graph_data, input_ids, attention_mask, num_graphs=num_graphs
        )


# ---------------------------------------------------------------------------
# Dummy input builder
# ---------------------------------------------------------------------------

def build_dummy_inputs(
    batch_size: int,
    nodes_per_mol: int,
    edges_per_mol: int,
    seq_len: int,
):
    """Build a synthetic batch of ``batch_size`` molecules for ONNX tracing.

    Args:
        batch_size: Number of molecules in the dummy batch.
        nodes_per_mol: Average nodes per molecule (controls total_nodes).
        edges_per_mol: Average edges per molecule (controls total_edges).
        seq_len: Transformer sequence length.

    Returns:
        Tuple of ``(x, edge_index, edge_attr, batch, input_ids,
        attention_mask)``.
    """
    total_nodes = batch_size * nodes_per_mol
    total_edges = batch_size * edges_per_mol

    x            = torch.randn(total_nodes, 26, dtype=torch.float32)
    edge_attr    = torch.randn(total_edges, 6,  dtype=torch.float32)
    input_ids    = torch.ones((batch_size, seq_len), dtype=torch.long)
    attn_mask    = torch.ones((batch_size, seq_len), dtype=torch.long)

    # Build edge_index: connect nodes within each molecule's block.
    ei_src, ei_dst = [], []
    for mol in range(batch_size):
        offset = mol * nodes_per_mol
        for e in range(edges_per_mol):
            ei_src.append(offset + e % nodes_per_mol)
            ei_dst.append(offset + (e + 1) % nodes_per_mol)
    edge_index = torch.tensor([ei_src, ei_dst], dtype=torch.long)

    # batch vector: node → graph index
    batch_vec = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.long),
        torch.full((batch_size,), nodes_per_mol, dtype=torch.long),
    )

    return x, edge_index, edge_attr, batch_vec, input_ids, attn_mask


# ---------------------------------------------------------------------------
# Export entry point
# ---------------------------------------------------------------------------

def export_onnx(args) -> None:
    pth_path  = Path(args.pth_path).expanduser().resolve()
    onnx_path = Path(args.onnx_path).expanduser().resolve()

    if not pth_path.exists():
        raise FileNotFoundError(f"Model weights not found: {pth_path}")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load checkpoint ---
    state = torch.load(pth_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(state)
    inferred = _infer_model_dimensions(state)

    gin_hidden_dim = args.gin_hidden_dim or inferred["gin_hidden_dim"]
    mlp_hidden_dim = args.mlp_hidden_dim or inferred["mlp_hidden_dim"]
    output_dim     = args.output_dim     or inferred["output_dim"]

    # --- Build model ---
    model = HybridFusionModel(
        gin_hidden_dim=gin_hidden_dim,
        transformer_model=args.transformer_model,
        mlp_hidden_dim=mlp_hidden_dim,
        output_dim=output_dim,
        dropout=args.dropout,
    )

    # Swap in the ONNX-safe bilinear before loading weights.
    model.bilinear = ExportFriendlyBilinear(gin_hidden_dim, gin_hidden_dim, bias=True)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        print(
            f"[WARNING] strict load_state_dict failed ({exc}).\n"
            "This typically means the checkpoint was trained with the old "
            "architecture (e.g. AttentionPooling instead of DualPooling). "
            "Retrain the model and re-export.\n"
            "Attempting non-strict load for debugging only…"
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys:    {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

    model.eval()
    wrapper = BatchHybridOnnxWrapper(model).eval()

    # --- Dummy inputs ---
    dummy = build_dummy_inputs(
        batch_size=args.batch_size,
        nodes_per_mol=args.num_nodes,
        edges_per_mol=args.num_edges,
        seq_len=args.seq_len,
    )

    input_names  = ["x", "edge_index", "edge_attr", "batch", "input_ids", "attention_mask"]
    output_names = ["predicted_properties"]

    # total_nodes / total_edges are distinct dynamic dims from batch_size,
    # because different molecules have different graph sizes.
    dynamic_axes = {
        "x":                    {0: "total_nodes"},
        "edge_index":           {1: "total_edges"},
        "edge_attr":            {0: "total_edges"},
        "batch":                {0: "total_nodes"},
        "input_ids":            {0: "batch_size", 1: "seq_len"},
        "attention_mask":       {0: "batch_size", 1: "seq_len"},
        "predicted_properties": {0: "batch_size"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    print(f"[OK] Batch-capable ONNX export → {onnx_path}")
    print(f"     Dummy batch size used: {args.batch_size}")
    print(f"     Run Scripts/verify_onnx_batch.py to validate.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export HybridFusionModel .pth to a batch-capable ONNX model"
    )
    parser.add_argument(
        "--pth_path", type=str,
        default=str(PROJECT_ROOT / "models" / "best_hybrid_model.pth"),
    )
    parser.add_argument(
        "--onnx_path", type=str,
        default=str(PROJECT_ROOT / "GUI" / "assets" / "hybrid_model.onnx"),
    )
    parser.add_argument(
        "--transformer_model", type=str,
        default="seyonec/ChemBERTa-zinc-base-v1",
    )
    parser.add_argument("--gin_hidden_dim", type=int, default=None)
    parser.add_argument("--mlp_hidden_dim", type=int, default=None)
    parser.add_argument("--output_dim",     type=int, default=None)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--seq_len",        type=int, default=64)
    parser.add_argument(
        "--num_nodes", type=int, default=24,
        help="Average nodes per molecule in the dummy input.",
    )
    parser.add_argument(
        "--num_edges", type=int, default=56,
        help="Average edges per molecule in the dummy input.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Number of molecules in the dummy batch used for tracing.",
    )
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    export_onnx(parse_args())