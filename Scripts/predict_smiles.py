#!/usr/bin/env python3
"""Predict molecular properties for a single SMILES string.

This script uses the existing C-backed hybrid inference engine through
GUI.core.inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GUI.core.inference import cleanup_hybrid_engine, init_hybrid_engine, run_hybrid_regression
from Scripts.qm9_delta import convert_delta_ev_to_kjmol, QM9_DELTA_TARGET_COLUMNS


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
    return properties


def _inverse_scale_predictions(values: list[float], property_stats: dict[str, dict[str, float]]) -> list[float]:
    de_scaled = list(values)
    for idx, prop in enumerate(PROPERTY_NAMES):
        if idx >= len(de_scaled):
            break
        stats = property_stats.get(prop)
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


def _convert_energies_to_delta_kjmol(
    values: list[float],
    smiles: str,
) -> list[float]:
    """Convert energy properties from eV (delta) to kJ/mol.
    
    Since the model uses delta learning, predictions are already delta energies in eV.
    We only need to convert the unit from eV to kJ/mol.
    
    For H298, G298, U298, U0: converts from eV (delta) to kJ/mol (delta)
    Other properties remain unchanged.
    
    Args:
        values: Array of property values (energy properties in eV after inverse scaling)
        smiles: Molecule SMILES string (not needed for delta-trained model)
        
    Returns:
        Updated array with energy properties in delta kJ/mol
    """
    converted = list(values)
    
    for idx, prop in enumerate(PROPERTY_NAMES):
        if idx >= len(converted):
            break
        if prop in QM9_DELTA_TARGET_COLUMNS:
            try:
                # values[idx] should be in eV (delta energy) at this point
                converted[idx] = convert_delta_ev_to_kjmol(float(converted[idx]))
            except Exception as e:
                print(f"[Warning] Failed to convert {prop} to delta kJ/mol: {e}")
                # Keep the original value if conversion fails
    
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict molecular properties from a single SMILES string."
    )
    parser.add_argument("smiles", 
                        help="Input SMILES string, e.g. CCO",
                        default="CCO")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional ONNX model path. If omitted, default model resolution is used.",
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
    smiles = args.smiles.strip()
    if not smiles:
        print("[ERROR] SMILES is empty.")
        return 2

    try:
        model_path = init_hybrid_engine(model_path=args.model)
        prediction = run_hybrid_regression(smiles, model_path=args.model)
        values_raw = np.asarray(prediction, dtype=np.float32).tolist()

        values = values_raw
        scaling_mode = "raw_scaled"
        if not args.no_inverse_scale:
            stats_path = Path(args.property_stats).expanduser().resolve() if args.property_stats else None
            property_stats = _load_property_stats(stats_path)
            if property_stats:
                values = _inverse_scale_predictions(values_raw, property_stats)
                # Convert energy properties from absolute Hartree to delta kJ/mol
                values = _convert_energies_to_delta_kjmol(values, smiles)
                scaling_mode = f"inverse_scaled_from:{stats_path} (energies in delta kJ/mol)"
            else:
                scaling_mode = "raw_scaled (property stats unavailable)"

        print(f"Model: {model_path}")
        print(f"Output mode: {scaling_mode}")
        print(f"SMILES: {smiles}")
        print("Predicted properties:")
        for index, value in enumerate(values):
            name = PROPERTY_NAMES[index] if index < len(PROPERTY_NAMES) else f"property_{index}"
            if name in QM9_DELTA_TARGET_COLUMNS:
                unit = "kJ/mol (delta)"
            elif name in ["mu"]:
                unit = "Debye"
            elif name in ["alpha"]:
                unit = "Ų"
            elif name in ["homo", "lumo", "gap"]:
                unit = "eV"
            elif name in ["r2"]:
                unit = "Ų"
            elif name in ["zpve"]:
                unit = "eV"
            elif name in ["cv"]:
                unit = "cal/(mol·K)"
            else:
                unit = "???"
            print(f"  {name:>6}: {float(value):>12.6f}  [{unit}]")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Prediction failed: {exc}")
        return 1
    finally:
        cleanup_hybrid_engine()


if __name__ == "__main__":
    raise SystemExit(main())