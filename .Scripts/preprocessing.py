import pandas as pd 
import numpy as np 
from rdkit import Chem
import torch
from torch_geometric.data import Data 
from torch_geometric.loader import DataLoader
import networkx as nx
import matplotlib.pyplot as plt

class MolecularPropertyPipeline:
    '''
    Comprehensive class for the entire molecular property prediction pipeline.
    Handles data loading, preprocessing, canonicalization, merging, and graph generation.
    Operates on the full dataset.
    '''
    
    def __init__(self, qm8_path: str, qm9_path: str):
        """Initialize the pipeline with paths to QM8 and QM9 datasets."""
        self.qm8_path = qm8_path
        self.qm9_path = qm9_path
        self.df8 = None
        self.df9 = None
        self.df_merged = None
        self.Y = None
        self.Y_tensor = None
        self.mask = None
        self.mask_tensor = None
        self.graphs = None
        self.data_loader = None
        self.target_cols = ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2',  # QM8
                           'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0']  # QM9
    
    # ==================== Data Loading ====================
    def load_data(self):
        """Load QM8 and QM9 CSV files."""
        print("Loading QM8 and QM9 datasets...")
        self.df8 = pd.read_csv(self.qm8_path)
        self.df9 = pd.read_csv(self.qm9_path)
        print(f"QM8 shape: {self.df8.shape}")
        print(f"QM9 shape: {self.df9.shape}")
        return self.df8, self.df9
    
    def validate_data(self):
        """Check for missing values and data quality issues."""
        print("\n--- QM8 Data Quality ---")
        print(f"Null values:\n{self.df8.isnull().sum()}")
        
        print("\n--- QM9 Data Quality ---")
        print(f"Null values:\n{self.df9.isnull().sum()}")
        
        # Check for empty SMILES
        empty_smiles_qm8 = (self.df8['smiles'].str.strip() == "").sum()
        empty_smiles_qm9 = (self.df9['smiles'].str.strip() == "").sum()
        print(f"\nEmpty SMILES in QM8: {empty_smiles_qm8}")
        print(f"Empty SMILES in QM9: {empty_smiles_qm9}")
    
    # ==================== SMILES Canonicalization ====================
    def _canonicalize(self, smiles):
        """Convert a SMILES string to its canonical form."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol) if mol else None
        except:
            return None
    
    def canonicalize_smiles(self):
        """Canonicalize SMILES in both datasets."""
        print("\nCanonicalizing SMILES strings...")
        self.df8['smiles_new'] = self.df8['smiles'].apply(self._canonicalize)
        self.df9['smiles_new'] = self.df9['smiles'].apply(self._canonicalize)
        
        # Remove rows with invalid SMILES
        self.df8 = self.df8[self.df8['smiles_new'].notna()]
        self.df9 = self.df9[self.df9['smiles_new'].notna()]
        
        print(f"QM8 after canonicalization: {self.df8.shape}")
        print(f"QM9 after canonicalization: {self.df9.shape}")
    
    # ==================== Data Merging ====================
    def merge_datasets(self):
        """Merge QM8 and QM9 datasets using outer join on canonicalized SMILES."""
        print("\nMerging QM8 and QM9 datasets...")
        self.df_merged = pd.merge(self.df8, self.df9, on='smiles_new', how='outer')
        print(f"Merged dataset shape: {self.df_merged.shape}")
        return self.df_merged
    
    # ==================== Targets and Masks ====================
    def create_targets_and_masks(self):
        """
        Create target matrices (Y) and masks.
        - Y: Contains property values (NaN replaced with 0)
        - mask: Indicates which values are real (1) vs. NaN replacements (0)
        """
        print("\nCreating target matrices and masks...")
        
        # Fill NaNs with 0 (for numerical stability)
        self.Y = self.df_merged[self.target_cols].fillna(0).values
        
        # Create mask: 1 where data exists, 0 where it was NaN
        self.mask = self.df_merged[self.target_cols].notna().astype(int).values
        
        print(f"Target (Y) shape: {self.Y.shape}")
        print(f"Mask shape: {self.mask.shape}")
        
        # Print sparsity information
        total_values = self.Y.size
        missing_values = (self.mask == 0).sum()
        sparsity = (missing_values / total_values) * 100
        print(f"Data sparsity: {sparsity:.2f}%")
        
        return self.Y, self.mask
    
    def get_tensors(self):
        """Convert Y and mask to PyTorch tensors."""
        print("\nConverting to PyTorch tensors...")
        self.Y_tensor = torch.tensor(self.Y, dtype=torch.float32)
        self.mask_tensor = torch.tensor(self.mask, dtype=torch.float32)
        
        print(f"Y tensor shape: {self.Y_tensor.shape}")
        print(f"Mask tensor shape: {self.mask_tensor.shape}")
        
        return self.Y_tensor, self.mask_tensor
    
    # ==================== Graph Generation ====================
    def _smiles_to_graph(self, smiles: str, y_values, mask_values):
        """
        Convert a SMILES string to a PyTorch Geometric Data object (graph).
        
        Args:
            smiles: Canonical SMILES string
            y_values: Target property values for this molecule
            mask_values: Mask values for this molecule
            
        Returns:
            Data object with node features, edge indices/attributes, targets, and masks
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
                float(bond.GetBondTypeAsDouble()),  # Bond type (1.0=single, 2.0=double, 1.5=aromatic)
                float(bond.GetIsConjugated()),      # Is conjugated?
                float(bond.GetIsAromatic())         # Is aromatic?
            ]
            
            # Add bidirectional edges
            edge_indices.append([i, j])
            edge_attrs.append(bond_feats)
            edge_indices.append([j, i])
            edge_attrs.append(bond_feats)
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # --- Targets and Masks ---
        y = y_values.clone().detach().view(1, -1)
        mask = mask_values.clone().detach().view(1, -1)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, mask=mask)
    
    def generate_graphs(self):
        """Generate graph representations for all molecules."""
        print("\nGenerating molecular graphs...")
        self.graphs = []
        
        num_molecules = len(self.df_merged)
        failed_count = 0
        
        for i in range(num_molecules):
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{num_molecules} molecules ({100*(i+1)/num_molecules:.1f}%)")
            
            smiles = self.df_merged['smiles_new'].iloc[i]
            y_val = self.Y_tensor[i]
            mask_val = self.mask_tensor[i]
            
            graph = self._smiles_to_graph(smiles, y_val, mask_val)
            
            if graph:
                self.graphs.append(graph)
            else:
                failed_count += 1
        
        print(f"\nGraph generation complete!")
        print(f"Successfully generated {len(self.graphs)} graphs")
        print(f"Failed to convert: {failed_count} molecules")
        
        return self.graphs
    
    # ==================== DataLoader ====================
    def create_dataloader(self, batch_size: int = 32, shuffle: bool = True):
        """
        Create a PyTorch Geometric DataLoader for batch processing.
        
        Args:
            batch_size: Number of graphs per batch
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader object
        """
        print(f"\nCreating DataLoader (batch_size={batch_size})...")
        self.data_loader = DataLoader(self.graphs, batch_size=batch_size, shuffle=shuffle)
        
        # Get info from first batch
        first_batch = next(iter(self.data_loader))
        print(f"Number of batches: {len(self.data_loader)}")
        print(f"Molecules per batch: {first_batch.num_graphs}")
        print(f"Atoms per batch: {first_batch.x.shape[0]}")
        print(f"Node features: {first_batch.num_node_features}")
        print(f"Edge features: {first_batch.edge_attr.shape[1]}")
        print(f"Target shape per batch: {first_batch.y.shape}")
        
        return self.data_loader
    
    # ==================== Full Pipeline ====================
    def run_full_pipeline(self, batch_size: int = 32):
        """
        Execute the complete pipeline from raw data to DataLoader.
        
        Args:
            batch_size: Batch size for the DataLoader
            
        Returns:
            DataLoader object for model training
        """
        print("=" * 60)
        print("Starting Full Molecular Property Prediction Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_data()
        self.validate_data()
        
        # Step 2: Canonicalize SMILES
        self.canonicalize_smiles()
        
        # Step 3: Merge datasets
        self.merge_datasets()
        
        # Step 4: Create targets and masks
        self.create_targets_and_masks()
        
        # Step 5: Convert to tensors
        self.get_tensors()
        
        # Step 6: Generate graphs
        self.generate_graphs()
        
        # Step 7: Create DataLoader
        self.create_dataloader(batch_size=batch_size)
        
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        
        return self.data_loader
    
    # ==================== Utility Methods ====================
    def get_summary(self):
        """Print summary statistics of the pipeline."""
        print("\n--- Pipeline Summary ---")
        print(f"Total molecules: {len(self.df_merged)}")
        print(f"Valid graphs created: {len(self.graphs)}")
        print(f"Target features: {len(self.target_cols)}")
        print(f"Data sparsity: {((self.mask == 0).sum() / self.mask.size) * 100:.2f}%")
        print(f"DataLoader batches: {len(self.data_loader) if self.data_loader else 'Not created'}")
    
    def visualize_molecule(self, index: int):
        """Visualize a molecule structure from the graphs."""
        if index >= len(self.graphs):
            print(f"Index {index} out of range. Available graphs: {len(self.graphs)}")
            return
        
        data = self.graphs[index]
        smiles = self.df_merged['smiles_new'].iloc[index]
        
        G = nx.Graph()
        
        # Create node labels with atomic symbols
        node_labels = {}
        for i in range(data.num_nodes):
            atomic_num = int(data.x[i][0])
            symbol = Chem.GetPeriodicTable().GetElementSymbol(atomic_num)
            node_labels[i] = f"{i}:{symbol}"
            G.add_node(i)
        
        # Add edges
        edge_list = data.edge_index.t().tolist()
        G.add_edges_from(edge_list)
        
        # Plot
        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(G, seed=42)
        
        nx.draw(G, pos, with_labels=True, labels=node_labels,
                node_color='skyblue', node_size=1000,
                font_color='black', font_weight='bold', font_size=9,
                edge_color='gray', width=2)
        
        plt.title(f"Molecule {index}: {smiles}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

