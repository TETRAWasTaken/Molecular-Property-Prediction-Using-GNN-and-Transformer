import os
import sys
import torch

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from GIN.Utils.preprocessing import MolecularPropertyPipeline
from GIN.Utils.TrainingTesting import TrainingTesting
from GIN.Utils.paths import Paths
import argparse

IS_SAGEMAKER = "SM_MODEL_DIR" in os.environ


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qm8_path", type=str, default=None)
    parser.add_argument("--qm9_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--check_device", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    args = parser.parse_args()

    # Critical Environment Checks
    if args.check_device:
        print("GPU") if torch.cuda.is_available() else print("CPU")
        return

    # Resolve paths: SageMaker env vars take priority, then CLI args, then local defaults
    if IS_SAGEMAKER:
        data_dir = os.environ["SM_CHANNEL_TRAINING"]
        qm8_path = args.qm8_path or os.path.join(data_dir, "qm8.csv")
        qm9_path = args.qm9_path or os.path.join(data_dir, "qm9.csv")
        save_path = args.save_path or os.path.join(os.environ["SM_MODEL_DIR"], "gnn_molecular_model.pth")
    else:
        paths = Paths()
        qm8_path = args.qm8_path or paths.get_qm8_path()
        qm9_path = args.qm9_path or paths.get_qm9_path()
        save_path = args.save_path or paths.get_model_path()

    # ==================== Hyperparameters ====================
    BATCH_SIZE = args.batch_size
    HIDDEN_DIM = args.hidden_dim
    OUTPUT_DIM = 12
    DROPOUT = 0.2
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = 5e-4
    EPOCHS = args.epochs
    PATIENCE = 20
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ==================== 1. Preprocessing ====================
    pipeline = MolecularPropertyPipeline(qm8_path, qm9_path)
    train_loader, val_loader, test_loader = pipeline.run_full_pipeline(batch_size=BATCH_SIZE)

    # Infer feature dimensions from data
    sample_batch = next(iter(train_loader))
    node_in_dim = sample_batch.num_node_features  # 6
    edge_in_dim = sample_batch.edge_attr.shape[1]  # 3

    # ==================== 2. Model ====================
    model = TrainingTesting(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ==================== 3. Training ====================
    model.run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        patience=PATIENCE
    )

    # ==================== 4. Evaluation ====================
    test_metrics = model.evaluate(test_loader)
    print(f"\n{'=' * 60}")
    print(f"Test Results  —  MAE: {test_metrics['mae']:.4f}  |  R²: {test_metrics['r2']:.4f}")
    print(f"{'=' * 60}")

    # ==================== 5. Save Model ====================
    # Create outputs directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()