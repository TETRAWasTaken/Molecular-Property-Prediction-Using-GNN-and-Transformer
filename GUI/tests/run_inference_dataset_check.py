#!/usr/bin/env python3
"""Run hybrid model inference over a SMILES CSV to validate end-to-end inference."""

from __future__ import annotations

import argparse
import csv
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Allow running this file directly: python GUI/core/run_inference_dataset_check.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GUI.core.inference import run_hybrid_regression, init_hybrid_engine, cleanup_hybrid_engine


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a SMILES dataset and save outputs.")
    parser.add_argument("--csv", default="GUI/assets/smiles_extract.csv", help="Path to input CSV file")
    parser.add_argument("--smiles-col", default="smiles", help="CSV column containing SMILES strings")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to run (0 = all)")
    parser.add_argument("--model", default=None, help="Optional ONNX model path")
    parser.add_argument("--out-dir", default="GUI/assets", help="Directory for output files")
    parser.add_argument("--prefix", default="inference_check", help="Output filename prefix")
    return parser.parse_args()


def _load_smiles(csv_path: Path, smiles_col: str, limit: int) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    smiles_list: List[str] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if smiles_col not in (reader.fieldnames or []):
            raise ValueError(
                f"Column '{smiles_col}' not found in CSV. Available: {reader.fieldnames}"
            )
        for row in reader:
            smiles = (row.get(smiles_col) or "").strip()
            if smiles:
                smiles_list.append(smiles)
                if limit > 0 and len(smiles_list) >= limit:
                    break

    if not smiles_list:
        raise ValueError("No valid SMILES rows found in CSV")

    return smiles_list


def _safe_stats(values: np.ndarray) -> Dict[str, float | None]:
    if values.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def main() -> int:
    args = _parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / f"{args.prefix}_results.csv"
    summary_json = out_dir / f"{args.prefix}_summary.json"

    smiles_list = _load_smiles(csv_path, args.smiles_col, args.limit)

    total = len(smiles_list)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    start = time.time()
    print(f"[INFO] Loaded {total} SMILES from {csv_path}")

    try:
        model_path = init_hybrid_engine(model_path=args.model)
        print(f"[INFO] Engine initialized with model: {model_path}")

        for idx, smiles in enumerate(smiles_list, start=1):
            row: Dict[str, Any] = {
                "index": idx,
                "smiles": smiles,
                "status": "ok",
                "error": "",
                "prediction": "",
            }
            try:
                pred = run_hybrid_regression(smiles, model_path=args.model)
                pred_list = [float(x) for x in np.asarray(pred, dtype=np.float32).tolist()]
                row["prediction"] = json.dumps(pred_list)
            except Exception as exc:  # noqa: BLE001
                row["status"] = "failed"
                row["error"] = str(exc)
                failures.append({"smiles": smiles, "error": str(exc)})
            rows.append(row)

            if idx % 50 == 0 or idx == total:
                print(f"[INFO] Processed {idx}/{total}")
    finally:
        cleanup_hybrid_engine()

    elapsed = time.time() - start

    success_count = sum(1 for r in rows if r["status"] == "ok")
    failed_count = total - success_count
    success_rate = (success_count / total) if total else 0.0

    prediction_values: List[float] = []
    for r in rows:
        if r["status"] != "ok":
            continue
        arr = np.asarray(json.loads(r["prediction"]), dtype=np.float32)
        finite_mask = np.isfinite(arr)
        prediction_values.extend(arr[finite_mask].tolist())

    pred_arr = np.asarray(prediction_values, dtype=np.float32)
    stats = _safe_stats(pred_arr)

    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "smiles", "status", "error", "prediction"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input_csv": str(csv_path),
        "smiles_column": args.smiles_col,
        "total_rows": total,
        "attempted": total,
        "succeeded": success_count,
        "failed": failed_count,
        "success_rate": round(success_rate * 100.0, 3),
        "elapsed_seconds": round(elapsed, 4),
        "avg_seconds_per_sample": round(elapsed / total, 6) if total else None,
        "prediction_stats": stats,
        "sample_failures": failures[:20],
        "outputs": {
            "results_csv": str(results_csv),
            "summary_json": str(summary_json),
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[RESULT] Inference run completed")
    print(f"[RESULT] Total: {total}")
    print(f"[RESULT] Success: {success_count}")
    print(f"[RESULT] Failed: {failed_count}")
    print(f"[RESULT] Success rate: {summary['success_rate']}%")
    print(f"[RESULT] Elapsed: {summary['elapsed_seconds']}s")
    print(f"[RESULT] Results CSV: {results_csv}")
    print(f"[RESULT] Summary JSON: {summary_json}")

    if failed_count > 0:
        print("[RESULT] Sample errors:")
        for item in failures[:5]:
            print(f"  - {item['smiles']}: {item['error']}")

    return 0 if success_count > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
