import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data

from main import HybridFusionModel


class HybridOnnxWrapper(nn.Module):
    """Expose tensor-only inputs for ONNX export."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr, batch, input_ids, attention_mask):
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        return self.model(graph_data, input_ids, attention_mask)


def build_dummy_inputs(num_nodes: int, num_edges: int, seq_len: int):
    x = torch.randn(num_nodes, 26, dtype=torch.float32)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    edge_attr = torch.randn(num_edges, 6, dtype=torch.float32)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    input_ids = torch.ones((1, seq_len), dtype=torch.long)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long)
    return x, edge_index, edge_attr, batch, input_ids, attention_mask


def export_onnx(args):
    pth_path = Path(args.pth_path).expanduser().resolve()
    onnx_path = Path(args.onnx_path).expanduser().resolve()

    if not pth_path.exists():
        raise FileNotFoundError(f"Model weights not found: {pth_path}")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = HybridFusionModel(
        gin_hidden_dim=args.gin_hidden_dim,
        transformer_model=args.transformer_model,
        mlp_hidden_dim=args.mlp_hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
    )

    state = torch.load(pth_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    wrapper = HybridOnnxWrapper(model).eval()

    dummy_inputs = build_dummy_inputs(
        num_nodes=args.num_nodes,
        num_edges=args.num_edges,
        seq_len=args.seq_len,
    )

    input_names = ["x", "edge_index", "edge_attr", "batch", "input_ids", "attention_mask"]
    output_names = ["predicted_properties"]

    dynamic_axes = {
        "x": {0: "num_nodes"},
        "edge_index": {1: "num_edges"},
        "edge_attr": {0: "num_edges"},
        "batch": {0: "num_nodes"},
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "predicted_properties": {0: "batch_size"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(onnx_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    print(f"ONNX export successful: {onnx_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export HybridFusionModel .pth to ONNX")
    parser.add_argument("--pth_path", type=str, default="best_hybrid_model.pth")
    parser.add_argument("--onnx_path", type=str, default="GUI/assets/hybrid_model.onnx")
    parser.add_argument("--transformer_model", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    parser.add_argument("--gin_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--output_dim", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--num_nodes", type=int, default=24)
    parser.add_argument("--num_edges", type=int, default=56)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    export_onnx(parse_args())