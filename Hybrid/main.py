import os
import sys
import torch
import argparse

# Ensure the Project Root is in the system path
if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from Hybrid.Utils.preprocessing import HybridMolecularPipeline
from Hybrid.Models.fusion_network import HybridFusionNetwork
from Hybrid.Utils.trainer import HybridTrainer
from Hybrid.Utils.paths import Paths

IS_SAGEMAKER = "SM_MODEL_DIR" in os.environ

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qm8_path", type=str, default=None)
    parser.add_argument("--qm9_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()

    # Path Resolution (SageMaker vs Local)
    if IS_SAGEMAKER:
        data_dir = os.environ["SM_CHANNEL_TRAINING"]
        qm8_path = args.qm8_path or os.path.join(data_dir, "qm8.csv")
        qm9_path = args.qm9_path or os.path.join(data_dir, "qm9.csv")
        save_path = args.save_path or os.path.join(os.environ["SM_MODEL_DIR"], "hybrid_model.pth")
    else:
        paths = Paths()
        qm8_path = args.qm8_path or paths.get_qm8_path()
        qm9_path = args.qm9_path or paths.get_qm9_path()
        save_path = args.save_path or paths.get_model_path()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")

    # ==================== 1. Preprocessing ====================
    pipeline = HybridMolecularPipeline(qm8_path, qm9_path)
    train_loader, val_loader, test_loader = pipeline.run_pipeline(batch_size=args.batch_size)

    sample_batch = next(iter(train_loader))
    node_in_dim = sample_batch.num_node_features
    edge_in_dim = sample_batch.edge_attr.shape[1]

    # ==================== 2. Model Initialization ====================
    model = HybridFusionNetwork(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        gin_hidden_dim=128,
        output_dim=12
    )

    trainer = HybridTrainer(
        model=model,
        learning_rate=args.learning_rate,
        device=DEVICE
    )

    # ==================== 3. Training & Evaluation ====================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    trainer.run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=save_path
    )

    test_metrics = trainer.evaluate(test_loader)
    print(f"\n{'=' * 60}")
    print(f"Automated Run Complete — MAE: {test_metrics['mae']:.4f} | R²: {test_metrics['r2']:.4f}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()