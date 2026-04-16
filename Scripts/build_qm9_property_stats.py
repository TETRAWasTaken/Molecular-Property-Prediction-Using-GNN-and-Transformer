#!/usr/bin/env python3
"""Build property scaling stats from a QM9 CSV file.

The output JSON is consumed by predict_smiles_transformer.py to convert
scaled model outputs back to physical units.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qm9_delta import (
    QM9_TARGET_COLUMNS,
    apply_qm9_delta_learning,
    build_qm9_atom_reference_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute QM9 property mean/std stats for inverse scaling at inference time."
    )
    parser.add_argument(
        "--qm9-csv",
        default="Dataset/New_QM9/molecule_properties.csv",
        help="Path to the QM9 molecule properties CSV.",
    )
    parser.add_argument(
        "--out",
        default="models/qm9_property_stats.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.qm9_csv).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not csv_path.exists():
        print(f"[ERROR] QM9 CSV not found: {csv_path}")
        return 2

    df = pd.read_csv(csv_path)
    df = apply_qm9_delta_learning(df, smiles_col="smiles", target_cols=QM9_TARGET_COLUMNS)

    missing = [name for name in QM9_TARGET_COLUMNS if name not in df.columns]
    if missing:
        print(f"[ERROR] Missing property columns in CSV: {missing}")
        return 2

    properties: dict[str, dict[str, float | int | None]] = {}
    for name in QM9_TARGET_COLUMNS:
        series = pd.to_numeric(df[name], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            properties[name] = {
                "mean": None,
                "std": None,
                "count": 0,
                "min": None,
                "max": None,
            }
            continue

        properties[name] = {
            "mean": float(valid.mean()),
            # Match sklearn StandardScaler behavior (population std, ddof=0)
            "std": float(valid.std(ddof=0)),
            "count": int(valid.shape[0]),
            "min": float(valid.min()),
            "max": float(valid.max()),
        }

    payload = {
        "source_csv": str(csv_path),
        "num_rows": int(df.shape[0]),
        "target_transform": "qm9_delta_learning",
        "delta_targets": ["u0", "u298", "h298", "g298"],
        "properties": properties,
        "atom_reference_energies": build_qm9_atom_reference_payload(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)

    print(f"[OK] Saved QM9 property stats to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())