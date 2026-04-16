#!/usr/bin/env python3
"""Predict molecular properties for a single SMILES string using a PyTorch GIN model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.qm9_delta import add_qm9_atom_reference_correction

from GIN_2.Utils.GIN import GIN


PROPERTY_NAMES = [
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u298",
    "h298",
    "g298",
    "cv",
]

DEFAULT_PROPERTY_STATS_PATH = PROJECT_ROOT / "models" / "qm9_property_stats.json"


def _load_property_stats(stats_path: Path | None) -> dict[str, dict[str, float]] | None:
    if stats_path is None or not stats_path.exists():
        return None
    try:
        with stats_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except Exception:
        return None

    properties = payload.get("properties") if isinstance(payload, dict) else None
    if not isinstance(properties, dict) or not properties:
        return None
    return payload


def _inverse_scale_predictions(values: list[float], payload: dict[str, dict[str, float]]) -> list[float]:
    de_scaled = list(values)
    properties = payload.get("properties", {}) if isinstance(payload, dict) else {}
    for idx, prop in enumerate(PROPERTY_NAMES):
        if idx >= len(de_scaled):
            break
        stats = properties.get(prop)
        if not isinstance(stats, dict):
            continue
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is None or std is None:
            continue
        try:
            mean_val = float(mean)
            std_val = float(std)
        except (TypeError, ValueError):
            continue
        de_scaled[idx] = (float(de_scaled[idx]) * std_val) + mean_val
    return de_scaled


def _apply_atom_reference_correction(values: list[float], smiles: str, payload: dict[str, dict[str, float]]) -> list[float]:
    atom_reference_payload = payload.get("atom_reference_energies") if isinstance(payload, dict) else None
    return add_qm9_atom_reference_correction(values, smiles, PROPERTY_NAMES, atom_reference_payload)


def one_hot_encoding(value, allowable_set):
    if value not in allowable_set:
        value = allowable_set[-1]
    return [value == allowed for allowed in allowable_set]


def get_node_features(atom):
    return (
        one_hot_encoding(atom.GetSymbol(), ["H", "C", "N", "O", "F", "Unknown"])
        + one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + one_hot_encoding(atom.GetFormalCharge(), [-1, 0, 1])
        + one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        + one_hot_encoding(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                "Unknown",
            ],
        )
        + [atom.GetIsAromatic(), atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
    )


def get_edge_features(bond):
    return one_hot_encoding(
        bond.GetBondType(),
        [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ],
    ) + [bond.GetIsConjugated()]


def build_graph_from_smiles(smiles: str, random_seed: int | None = None) -> Data:
    smiles = (smiles or "").strip()
    if not smiles:
        raise ValueError("SMILES is empty")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    params = AllChem.ETKDGv3()
    if random_seed is not None:
        params.randomSeed = int(random_seed)
    has_3d = AllChem.EmbedMolecule(mol, params) == 0
    if has_3d:
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
    else:
        conf = None

    x = torch.tensor([get_node_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)

    edge_indices: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        topo = get_edge_features(bond)

        if conf is not None:
            pi = conf.GetAtomPosition(i)
            pj = conf.GetAtomPosition(j)
            distance = float(((pi.x - pj.x) ** 2 + (pi.y - pj.y) ** 2 + (pi.z - pj.z) ** 2) ** 0.5)
        else:
            distance = 0.0

        full_edge_feature = topo + [distance]
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([full_edge_feature, full_edge_feature])

    if not edge_indices:
        edge_indices = [[0, 0]]
        edge_attrs = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    batch = torch.zeros(x.shape[0], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            return payload["model_state_dict"]
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        if any(str(k).startswith("node_encoder.") for k in payload.keys()):
            return payload

    raise ValueError("Could not find a valid GIN state_dict in the checkpoint")


def _infer_arch_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    try:
        node_encoder_w = state_dict["node_encoder.weight"]
        edge_encoder_w = state_dict["edge_encoder.weight"]
        pred_out_w = state_dict["prediction_head.4.weight"]
    except KeyError as exc:
        raise ValueError(f"Missing expected key in state_dict: {exc}") from exc

    num_layers = len({key.split(".")[1] for key in state_dict if key.startswith("convs.")})
    if num_layers < 1:
        raise ValueError("Could not infer num_layers from checkpoint")

    return {
        "node_in_dim": int(node_encoder_w.shape[1]),
        "edge_in_dim": int(edge_encoder_w.shape[1]),
        "hidden_dim": int(node_encoder_w.shape[0]),
        "output_dim": int(pred_out_w.shape[0]),
        "num_layer": int(num_layers),
    }


def load_gin_model(model_path: Path, device: torch.device, dropout: float = 0.0) -> GIN:
    payload = torch.load(str(model_path), map_location=device)
    state_dict = _extract_state_dict(payload)
    arch = _infer_arch_from_state_dict(state_dict)

    model = GIN(
        node_in_dim=arch["node_in_dim"],
        edge_in_dim=arch["edge_in_dim"],
        hidden_dim=arch["hidden_dim"],
        output_dim=arch["output_dim"],
        num_layer=arch["num_layer"],
        dropout=dropout,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict molecular properties from a SMILES string using a PyTorch GIN checkpoint."
    )
    parser.add_argument("smiles", help="Input SMILES string, e.g. CCO")
    parser.add_argument(
        "--model",
        help="Path to GIN checkpoint (.pth/.pt).",
        default="/Users/anshumaansoni/PycharmProjects/Molecular-Property-Prediction-Using-GNN-and-Transformer/models/GIN_model.pth"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device. Default: auto",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout used to instantiate architecture before loading weights.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RDKit conformer seed for deterministic edge-distance features.",
    )
    parser.add_argument(
        "--property-stats",
        default=str(DEFAULT_PROPERTY_STATS_PATH),
        help=(
            "Path to QM9 property stats JSON for inverse scaling. "
            "Use empty value to disable, e.g. --property-stats ''"
        ),
    )
    parser.add_argument(
        "--no-inverse-scale",
        action="store_true",
        help="Print raw scaled model outputs without attempting inverse scaling.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    if choice == "mps":
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    args = parse_args()
    smiles = args.smiles.strip()
    if not smiles:
        print("[ERROR] SMILES is empty.")
        return 2

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return 2

    try:
        device = resolve_device(args.device)
        model = load_gin_model(model_path=model_path, device=device, dropout=float(args.dropout))
        graph = build_graph_from_smiles(smiles, random_seed=args.seed).to(device)

        with torch.no_grad():
            pred = model(graph)

        values_raw = pred.detach().cpu().view(-1).tolist()

        values = values_raw
        scaling_mode = "raw_scaled"
        if not args.no_inverse_scale:
            stats_path = Path(args.property_stats).expanduser().resolve() if args.property_stats else None
            property_stats = _load_property_stats(stats_path)
            if property_stats:
                values = _inverse_scale_predictions(values_raw, property_stats)
                values = _apply_atom_reference_correction(values, smiles, property_stats)
                scaling_mode = f"inverse_scaled_and_delta_corrected_from:{stats_path}"
            else:
                scaling_mode = "raw_scaled (property stats unavailable)"

        print(f"Model: {model_path}")
        print(f"Device: {device}")
        print(f"Output mode: {scaling_mode}")
        print(f"SMILES: {smiles}")
        print("Predicted properties:")
        for index, value in enumerate(values):
            name = PROPERTY_NAMES[index] if index < len(PROPERTY_NAMES) else f"property_{index}"
            print(f"  {name:>6}: {float(value): .6f}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Prediction failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())