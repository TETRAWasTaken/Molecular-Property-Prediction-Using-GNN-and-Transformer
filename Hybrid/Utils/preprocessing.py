import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from multiprocessing import Pool, cpu_count

# Since Transformers is in the root directory alongside Hybrid, we can import 
# your existing Tokeniser directly without rewriting it!
from Transformers.tokeniser import Tokeniser 

class HybridMolecularPipeline:
    """
    Comprehensive pipeline for the Hybrid Model.
    Generates PyTorch Geometric 3D graphs, extracts 1D Transformer tokens, 
    and fuses them into a single unified Data object for the DataLoader.
    """
    def __init__(self, qm8_path: str, qm9_path: str, max_seq_len: int = 64):
        self.qm8_path = qm8_path
        self.qm9_path = qm9_path
        self.max_seq_len = max_seq_len
        
        self.df_merged = None
        self.Y_tensor = None
        self.mask_tensor = None
        self.hybrid_graphs = []
        self.split_indices = None
        
        # 12 Targets from your merged datasets
        self.target_cols = ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 
                            'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0']

    # ==========================================
    # 1. Standard Data Loading (Identical to GIN)
    # ==========================================
    def load_and_merge(self):
        """Loads QM8/QM9, canonicalizes, merges, and generates Target/Mask tensors."""
        print("Loading and merging datasets...")
        df8 = pd.read_csv(self.qm8_path).dropna(subset=['smiles'])
        df9 = pd.read_csv(self.qm9_path).dropna(subset=['smiles'])
        
        # Merge on SMILES
        self.df_merged = pd.merge(df8, df9, on='smiles', how='outer')
        
        # Create Targets and NaNs Mask
        Y = self.df_merged[self.target_cols].fillna(0).values
        mask = self.df_merged[self.target_cols].notna().astype(int).values
        
        self.Y_tensor = torch.tensor(Y, dtype=torch.float32)
        self.mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        print(f"Merged Dataset Shape: {self.df_merged.shape}")
        return self.df_merged

    # ==========================================
    # 2. Graph Generation
    # ==========================================
    @staticmethod
    def _create_base_graph(args):
        """Worker function to build the basic structural Graph."""
        smiles, y_values, mask_values = args
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None

        # Node Features
        node_feats = [
            [atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
             atom.GetTotalNumHs(), int(atom.GetIsAromatic()), int(atom.GetHybridization())]
            for atom in mol.GetAtoms()
        ]
        x = torch.tensor(node_feats, dtype=torch.float)

        # Edge Features
        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feats = [float(bond.GetBondTypeAsDouble()), float(bond.GetIsConjugated()), float(bond.GetIsAromatic())]
            
            # Bidirectional edges
            edge_indices.extend([[i, j], [j, i]])
            edge_attrs.extend([bond_feats, bond_feats])

        if len(edge_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        y = y_values.clone().detach().view(1, -1)
        mask = mask_values.clone().detach().view(1, -1)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, mask=mask)

    # ==========================================
    # 3. Hybrid Data Fusion
    # ==========================================
    def generate_hybrid_dataset(self):
        """Builds graphs, tokenizes text, and fuses them into unified PyG Data objects."""
        smiles_list = self.df_merged['smiles'].tolist()
        num_molecules = len(smiles_list)
        
        print(f"\nStep A: Generating 3D Graphs for {num_molecules} molecules...")
        args_list = [(smiles_list[i], self.Y_tensor[i], self.mask_tensor[i]) for i in range(num_molecules)]
        
        with Pool(cpu_count()) as pool:
            base_graphs = pool.map(self._create_base_graph, args_list)

        print("Step B: Generating 1D Transformer Tokens...")
        tokeniser = Tokeniser(self.qm8_path, self.qm9_path, max_length=self.max_seq_len)
        # Using your highly optimized parallel tokeniser
        token_dict = tokeniser.parallel_batch_encode(smiles_list)
        
        all_input_ids = token_dict['input_ids']
        all_attention_masks = token_dict['attention_mask']

        print("Step C: Fusing Graphs and Tokens...")
        self.hybrid_graphs = []
        
        for i, graph in enumerate(base_graphs):
            if graph is not None:
                # THE PYTORCH GEOMETRIC TRICK:
                # We unsqueeze(0) to force shape [1, seq_len]. 
                # This ensures the PyG DataLoader stacks them perfectly into [batch_size, seq_len]
                graph.input_ids = all_input_ids[i].unsqueeze(0)
                graph.attention_mask = all_attention_masks[i].unsqueeze(0)
                
                self.hybrid_graphs.append(graph)

        print(f"Successfully generated {len(self.hybrid_graphs)} Hybrid Data objects.")
        return self.hybrid_graphs

    # ==========================================
    # 4. DataLoader Creation
    # ==========================================
    def create_dataloaders(self, batch_size: int = 64):
        """Splits the unified data and returns train/val/test loaders."""
        print(f"\nCreating Hybrid DataLoaders (Batch Size: {batch_size})...")
        num_graphs = len(self.hybrid_graphs)
        
        # Shuffle indices for splitting
        indices = torch.randperm(num_graphs).tolist()
        train_end = int(num_graphs * 0.8)
        val_end = int(num_graphs * 0.9)
        
        # Save split indices for caching
        self.split_indices = {"train": indices[:train_end], "val": indices[train_end:val_end], "test": indices[val_end:]}

        train_data = [self.hybrid_graphs[i] for i in self.split_indices["train"]]
        val_data = [self.hybrid_graphs[i] for i in self.split_indices["val"]]
        test_data = [self.hybrid_graphs[i] for i in self.split_indices["test"]]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
        return train_loader, val_loader, test_loader

    # ==========================================
    # 5. Master Execution Wrapper
    # ==========================================
    def run_pipeline(self, batch_size: int = 64):
        self.load_and_merge()
        self.generate_hybrid_dataset()
        return self.create_dataloaders(batch_size=batch_size)