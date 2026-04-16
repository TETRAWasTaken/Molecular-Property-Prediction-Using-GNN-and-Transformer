#!/usr/bin/env python3
"""Predict molecular properties for a single SMILES string using a standalone Transformer model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.qm9_delta import add_qm9_atom_reference_correction

from Transformers_2.Utils.Transformer import StandaloneChemBERTa


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


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            return payload["model_state_dict"]
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        if any(str(key).startswith("transformer.") for key in payload.keys()):
            return payload

    raise ValueError("Could not find a valid Transformer state_dict in the checkpoint")


def _infer_num_targets(state_dict: dict[str, torch.Tensor]) -> int:
    key = "prediction_head.weight"
    if key not in state_dict:
        raise ValueError(f"Missing required key in checkpoint: {key}")
    return int(state_dict[key].shape[0])


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


def load_model(model_path: Path, model_name: str, device: torch.device) -> StandaloneChemBERTa:
    payload = torch.load(str(model_path), map_location=device)
    state_dict = _extract_state_dict(payload)
    num_targets = _infer_num_targets(state_dict)

    model = StandaloneChemBERTa(model_name=model_name, num_targets=num_targets)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict molecular properties from a SMILES string using a standalone Transformer checkpoint."
    )
    parser.add_argument("smiles", help="Input SMILES string, e.g. CCO")
    parser.add_argument(
        "--model",
        default="/Users/anshumaansoni/PycharmProjects/Molecular-Property-Prediction-Using-GNN-and-Transformer/models/transformer_molecular_model.pth",
        help="Path to Transformer checkpoint (.pth/.pt).",
    )
    parser.add_argument(
        "--model-name",
        default="seyonec/ChemBERTa-zinc-base-v1",
        help="Backbone model identifier used during training.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer identifier/path. Defaults to --model-name.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Max token length for SMILES tokenization.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device. Default: auto",
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


def main() -> int:
    args = parse_args()
    smiles = (args.smiles or "").strip()
    if not smiles:
        print("[ERROR] SMILES is empty.")
        return 2
    if Chem.MolFromSmiles(smiles) is None:
        print(f"[ERROR] Invalid SMILES: {smiles}")
        return 2

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return 2

    try:
        device = resolve_device(args.device)
        model = load_model(model_path=model_path, model_name=args.model_name, device=device)

        tokenizer_name = args.tokenizer or args.model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        encodings = tokenizer(
            [smiles],
            padding="max_length",
            truncation=True,
            max_length=int(args.max_length),
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            pred = model(input_ids=input_ids, attention_mask=attention_mask)

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
        print(f"Backbone: {args.model_name}")
        print(f"Tokenizer: {tokenizer_name}")
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