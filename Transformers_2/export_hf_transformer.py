import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from Transformers_2.Utils.Transformer import StandaloneChemBERTa
from Transformers_2.Utils.paths import Paths


def export_hf_transformer(
    checkpoint_path: str,
    export_dir: str,
    base_model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    num_targets: int = 12,
    tokenizer_source: str = None,
) -> Path:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    export_path = Path(export_dir).expanduser().resolve()
    export_path.mkdir(parents=True, exist_ok=True)

    model = StandaloneChemBERTa(model_name=base_model_name, num_targets=num_targets)
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")

    model.load_state_dict(state_dict, strict=True)

    tokenizer_name = tokenizer_source or base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model.transformer.save_pretrained(export_path)
    tokenizer.save_pretrained(export_path)

    return export_path


def _default_export_dir() -> str:
    return str(Path(Paths().base_dir) / "GUI" / "assets" / "transformer")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the trained ChemBERTa checkpoint into a Hugging Face-style export directory."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=Paths().get_model_path(),
        help="Path to transformer_molecular_model.pth",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default=_default_export_dir(),
        help="Directory to write the Hugging Face-style export",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="seyonec/ChemBERTa-zinc-base-v1",
        help="Base Hugging Face model name used to instantiate the backbone and tokenizer",
    )
    parser.add_argument(
        "--num_targets",
        type=int,
        default=12,
        help="Number of regression targets in the trained checkpoint",
    )
    parser.add_argument(
        "--tokenizer_source",
        type=str,
        default=None,
        help="Optional tokenizer source path or model name. Defaults to base_model_name.",
    )
    args = parser.parse_args()

    export_path = export_hf_transformer(
        checkpoint_path=args.checkpoint,
        export_dir=args.export_dir,
        base_model_name=args.base_model_name,
        num_targets=args.num_targets,
        tokenizer_source=args.tokenizer_source,
    )

    print(f"Exported Hugging Face-style transformer assets to: {export_path}")
    print("Point HYBRID_ATTENTION_MODEL_PATH at this directory to use it for explainability.")


if __name__ == "__main__":
    main()