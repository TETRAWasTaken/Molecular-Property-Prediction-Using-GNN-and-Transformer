import torch
import pandas as pd
import os
import random
from rdkit import Chem
from torch_geometric.data import Data
from GIN.Utils.TrainingTesting import TrainingTesting
from GIN.Utils.preprocessing import MolecularPropertyPipeline

# TODO: use the new dataset with unseen data to predict and get model metrics

def smiles_to_graph(smiles: str):
    """
    Converts a SMILES string to a PyTorch Geometric Data object for inference.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    # --- Node Features (Atoms) ---
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append([
            atom.GetAtomicNum(),           # Atomic number
            atom.GetDegree(),              # Number of bonds
            atom.GetFormalCharge(),        # Charge
            atom.GetTotalNumHs(),          # Number of hydrogens
            int(atom.GetIsAromatic()),     # Is aromatic?
            int(atom.GetHybridization())   # Hybridization type
        ])
    x = torch.tensor(node_feats, dtype=torch.float)

    # --- Edge Features (Bonds) ---
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_feats = [
            float(bond.GetBondTypeAsDouble()),  # Bond type
            float(bond.GetIsConjugated()),      # Is conjugated?
            float(bond.GetIsAromatic())         # Is aromatic?
        ]

        # Add bidirectional edges
        edge_indices.append([i, j])
        edge_attrs.append(bond_feats)
        edge_indices.append([j, i])
        edge_attrs.append(bond_feats)

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # Create batch index (all zeros for a single graph)
    batch = torch.zeros(x.size(0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

def main():
    # ==================== Configuration ====================
    NUM_SAMPLES = 5

    # Define base directory to locate datasets
    # Assuming this script is in GIN/Utils/predict.py
    # We need to go up two levels to reach the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    qm8_path = os.path.join(base_dir, "Dataset", "qm8.csv")
    qm9_path = os.path.join(base_dir, "Dataset", "qm9.csv")
    MODEL_PATH = os.path.join(base_dir, "GIN", "outputs", "gnn_molecular_model.pth")

    # ==================== 1. Load Data ====================
    print(f"Loading datasets from:\n - {qm8_path}\n - {qm9_path}")

    # Use the pipeline class to handle loading and merging logic
    pipeline = MolecularPropertyPipeline(qm8_path, qm9_path)
    pipeline.load_data()
    pipeline.canonicalize_smiles()
    df_merged = pipeline.merge_datasets()

    print(f"\nTotal molecules available: {len(df_merged)}")

    # ==================== 2. Load Model ====================
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Using device: {device}")

    # Hyperparameters must match training
    model = TrainingTesting(
        node_in_dim=6,
        edge_in_dim=3,
        hidden_dim=128,
        output_dim=12,
        device=str(device)
    )

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please train the model first.")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Explicitly move model to device again to ensure buffers are correct
    model.to(device)
    model.eval()

    # ==================== 3. Predict on Random Samples ====================
    print(f"\nSelecting {NUM_SAMPLES} random molecules for prediction...\n")

    sample_indices = random.sample(range(len(df_merged)), NUM_SAMPLES)
    samples = df_merged.iloc[sample_indices]

    properties = ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2',
                  'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0']

    for idx, row in samples.iterrows():
        smiles = row['smiles_new']
        print(f"Molecule: {smiles}")

        # Convert to graph
        graph = smiles_to_graph(smiles)
        if not graph:
            print("  Invalid SMILES, skipping.")
            continue

        graph = graph.to(device)

        # Predict
        with torch.no_grad():
            pred_norm = model(graph)
            # Denormalize using stored stats in the model
            pred_real = pred_norm * model.std_y + model.mean_y

        # Display results
        print("-" * 50)
        print(f"{'Property':<10} | {'Prediction':<15} | {'Actual (if avail)':<15}")
        print("-" * 50)

        for i, prop_name in enumerate(properties):
            pred_val = pred_real[0][i].item()

            # Check if actual value exists in dataframe (it might be NaN)
            actual_val = row.get(prop_name, float('nan'))
            actual_str = f"{actual_val:.4f}" if pd.notna(actual_val) else "N/A"

            print(f"{prop_name:<10} | {pred_val:<15.4f} | {actual_str:<15}")
        print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
