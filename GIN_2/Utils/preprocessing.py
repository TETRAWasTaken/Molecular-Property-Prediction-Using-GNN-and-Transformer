import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import multiprocessing
import ast

import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from typing import List, Tuple

from qm9_delta import apply_qm9_delta_learning

# ==========================================
# 1. Feature Engineering Helpers
# ==========================================
def one_hot_encoding(x, allowable_set):
    """Maps a feature to a one-hot vector. Maps to the last index if not found."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_node_features(atom):
    """Extracts 26-dimensional atomic features using RDKit topology."""
    return (
        one_hot_encoding(atom.GetSymbol(), ['H', 'C', 'N', 'O', 'F', 'Unknown']) +
        one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        one_hot_encoding(atom.GetFormalCharge(), [-1, 0, 1]) +
        one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
        one_hot_encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, 
            Chem.rdchem.HybridizationType.SP2, 
            Chem.rdchem.HybridizationType.SP3, 
            'Unknown'
        ]) +
        [atom.GetIsAromatic(), atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
    )

def get_edge_features(bond):
    """Extracts 5-dimensional topological bond features."""
    return (
        one_hot_encoding(bond.GetBondType(), [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]) +
        [bond.GetIsConjugated()]
    )

# ==========================================
# 2. Graph Generation Worker
# ==========================================
def process_merged_3d_graph(row: pd.Series, target_cols: List[str]) -> Data:
    """
    Worker function: Converts a merged row into a 3D PyG Data object.
    Uses True DFT coordinates and strictly validates atom ordering.
    """
    mol_id = row['molecule_id']
    smiles = row['smiles']
    targets = row[target_cols].values.astype(np.float32)
    
    # 1. Extract the grouped data
    atom_data = row['atom_data'] 
    if isinstance(atom_data, str):
        try:
            atom_data = ast.literal_eval(atom_data)
        except Exception as e:
            return None
    
    atom_data = np.array(atom_data)
    
    # Split the dataset symbols from the X, Y, Z coordinates
    dataset_symbols = atom_data[:, 0] 
    coords_matrix = atom_data[:, 1:].astype(np.float32) 
    pos = torch.tensor(coords_matrix, dtype=torch.float)

    # 2. Get Topology from RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    
    # 3. CRITICAL VALIDATION 1: Atom count match
    if pos.shape[0] != mol.GetNumAtoms():
        return None 

    # 4. CRITICAL VALIDATION 2: Atom ordering match
    for idx, atom in enumerate(mol.GetAtoms()):
        rdkit_symbol = atom.GetSymbol()
        dataset_symbol = dataset_symbols[idx]
        
        if rdkit_symbol != dataset_symbol:
            # The geometry is scrambled! Skip this molecule to protect the model.
            return None 

    # 5. Build Node Features (Shape: [num_atoms, 26])
    atom_features = [get_node_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    # 6. Build Edge Features with exact 3D Distances (Shape: [num_edges, 6])
    edge_indices = []
    edge_attrs = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        topological_features = get_edge_features(bond)
        
        # Calculate EXACT Euclidean distance using the DFT coordinates
        dist = torch.norm(pos[i] - pos[j], p=2).item()
        full_edge_feature = topological_features + [dist]
        
        # Undirected graph needs both directions
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [full_edge_feature, full_edge_feature]

    if not edge_indices: return None

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    y = torch.from_numpy(targets).view(1, -1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y, mol_id=mol_id)

# ==========================================
# 3. Dask Pipeline Manager
# ==========================================
class RelationalGeometryPipeline(InMemoryDataset):
    """
    Pipeline that merges two relational datasets (Molecules & Atoms) 
    and generates robust 3D PyTorch Geometric graphs.
    """
    def __init__(self, root: str, mol_csv_path: str, atom_csv_path: str, target_cols: List[str], transform=None):
        self.mol_csv = mol_csv_path
        self.atom_csv = atom_csv_path
        self.target_cols = target_cols
        super().__init__(root, transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [self.mol_csv, self.atom_csv]

    @property
    def processed_file_names(self):
        return ['qm_merged_3d_graphs_delta.pt']

    def download(self):
        pass 

    def process(self):
        print("1. Loading datasets into memory...")
        df_mol = pd.read_csv(self.mol_csv)
        df_atom = pd.read_csv(self.atom_csv)

        print("2. Grouping 3D atomic coordinates by molecule...")
        # CRITICAL: Sort by molecule_id AND atom_index to guarantee the coordinate sequence
        df_atom = df_atom.sort_values(['molecule_id', 'atom_index'])
        
        def extract_data(group):
            # Extract the symbol alongside the X, Y, Z coordinates for validation later
            return group[['atom', 'x', 'y', 'z']].values.tolist()
            
        coords_series = df_atom.groupby('molecule_id').apply(extract_data, include_groups = False)
        df_coords = coords_series.reset_index(name='atom_data')

        print("3. Merging molecule and atom data...")
        df_merged = df_mol.merge(df_coords, on='molecule_id', how='inner')
        df_merged = apply_qm9_delta_learning(df_merged, smiles_col='smiles', target_cols=self.target_cols)
        
        # Free up RAM
        del df_mol
        del df_atom
        del df_coords

        num_cores = multiprocessing.cpu_count()
        print(f"4. Initializing Dask across {num_cores} cores...")
        ddf = dd.from_pandas(df_merged, npartitions=num_cores * 2)
        
        def map_chunk(df):
            return df.apply(lambda row: process_merged_3d_graph(row, self.target_cols), axis=1)
        
        print("5. Generating graphs...")
        with ProgressBar():
            meta_blueprint = pd.Series(dtype=object, name='graphs')
            graph_series = ddf.map_partitions(map_chunk, meta=meta_blueprint).compute(scheduler='processes')
        
        data_list = [g for g in graph_series if g is not None]
        print(f"Successfully processed {len(data_list)} 3D graphs.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_loaders(self, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Splits the dataset and returns loaders."""
        dataset = self.shuffle()
        n = len(dataset)
        
        train_set = dataset[:int(0.8 * n)]
        val_set = dataset[int(0.8 * n):int(0.9 * n)]
        test_set = dataset[int(0.9 * n):]
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader