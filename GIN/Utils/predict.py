import torch
import pandas as pd
import random
import os
import math
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from GIN.Utils.TrainingTesting import TrainingTesting
from GIN.Utils.preprocessing import MolecularPropertyPipeline
from GIN.Utils.paths import Paths

def smiles_to_graph(smiles: str):
    """
    Converts a SMILES string to a 3D PyTorch Geometric Data object for inference.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    mol = Chem.AddHs(mol) # Crucial for 3D shape
    
    # Generate 3D Conformer
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        AllChem.ComputeGasteigerCharges(mol)
    except ValueError:
        return None 

    conf = mol.GetConformer()
    
    # --- Node Features (Atoms) ---
    node_feats = []
    for i, atom in enumerate(mol.GetAtoms()):
        try:
            charge = float(atom.GetProp('_GasteigerCharge'))
            if math.isnan(charge) or math.isinf(charge):
                charge = 0.0
        except KeyError:
            charge = 0.0

        node_feats.append([
            atom.GetAtomicNum(),           
            atom.GetDegree(),              
            atom.GetFormalCharge(),        
            atom.GetTotalNumHs(),          
            int(atom.GetIsAromatic()),     
            int(atom.GetHybridization()),  
            charge                         # 7th Feature: Partial Charge
        ])
    x = torch.tensor(node_feats, dtype=torch.float)
    
    # --- Edge Features (Bonds) ---
    edge_indices = []
    edge_attrs = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        pos_i = conf.GetAtomPosition(i)
        pos_j = conf.GetAtomPosition(j)
        distance = math.sqrt((pos_i.x - pos_j.x)**2 + (pos_i.y - pos_j.y)**2 + (pos_i.z - pos_j.z)**2)

        bond_feats = [
            float(bond.GetBondTypeAsDouble()),  
            float(bond.GetIsConjugated()),      
            float(bond.GetIsAromatic()),        
            distance                            # 4th Feature: 3D Bond Length
        ]
        
        # Add bidirectional edges
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([bond_feats, bond_feats])
    
    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float) # Updated to 4
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    # Create batch index (all zeros for a single graph)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

def main():
    # ==================== Configuration ====================
    NUM_SAMPLES = 5
    
    paths = Paths()
    qm8_path = paths.get_qm8_path()
    qm9_path = paths.get_qm9_path()
    MODEL_PATH = paths.get_model_path()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    print(f"Using device: {device}")

    # ==================== 1. Load Denormalization Stats ====================
    pipeline = MolecularPropertyPipeline(qm8_path, qm9_path)
    cache_path = pipeline._default_cache_path()
    
    if not os.path.exists(cache_path):
        print(f"Error: Cache not found at {cache_path}. Run the training pipeline first to generate stats.")
        return
        
    print("Loading normalization stats from cache...")
    payload = torch.load(cache_path, map_location=device, weights_only=False)
    y_mean = payload.get("y_mean")
    y_std = payload.get("y_std")
    
    if y_mean is None or y_std is None:
        print("Error: y_mean or y_std not found in cache. Did you update preprocessing.py and run it?")
        return

    # ==================== 2. Load Raw Data for Sampling ====================
    # Quick load just to pick 5 random molecules
    pipeline.load_data(verbose=False)
    pipeline.canonicalize_smiles(verbose=False)
    df_merged = pipeline.merge_datasets(verbose=False)

    # ==================== 3. Load Model ====================
    # UPDATED DIMENSIONS: node=7, edge=4
    model = TrainingTesting(
        node_in_dim=7,
        edge_in_dim=4,
        hidden_dim=128,
        output_dim=12,
        device=str(device)
    )
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please train the model first.")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.to(device)
    model.eval()

    # ==================== 4. Predict on Random Samples ====================
    print(f"\nSelecting {NUM_SAMPLES} random molecules for prediction...\n")
    
    sample_indices = random.sample(range(len(df_merged)), NUM_SAMPLES)
    samples = df_merged.iloc[sample_indices]
    
    properties = ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 
                  'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0']

    for idx, row in samples.iterrows():
        # Make sure to use 'smiles' instead of 'smiles_new' unless you explicitly renamed it
        smiles = row['smiles'] 
        print(f"Molecule: {smiles}")
        
        # Convert to 3D graph
        graph = smiles_to_graph(smiles)
        if not graph:
            print("  Invalid SMILES or failed 3D embedding, skipping.")
            continue
            
        graph = graph.to(device)
        
        # Predict
        with torch.no_grad():
            pred_norm = model(graph)
            
            # STEP 2: The Denormalization Math
            pred_real = (pred_norm * y_std) + y_mean
            
        # Display results
        print("-" * 50)
        print(f"{'Property':<10} | {'Prediction':<15} | {'Actual (if avail)':<15}")
        print("-" * 50)
        
        for i, prop_name in enumerate(properties):
            pred_val = pred_real[0][i].item()
            
            actual_val = row.get(prop_name, float('nan'))
            actual_str = f"{actual_val:.4f}" if pd.notna(actual_val) else "N/A"
            
            print(f"{prop_name:<10} | {pred_val:<15.4f} | {actual_str:<15}")
        print("=" * 50 + "\n")

if __name__ == "__main__":
    main()