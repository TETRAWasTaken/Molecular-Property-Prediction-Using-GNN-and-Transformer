import os
import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import math
from rdkit.Chem import AllChem


def _looks_like_git_lfs_pointer(file_path: str) -> bool:
    """Detect whether a file contains a Git LFS pointer instead of real data."""
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        return first_line == "version https://git-lfs.github.com/spec/v1"
    except (OSError, UnicodeDecodeError):
        return False


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
        self.cache_version = 1
        self.split_indices = None
        self.smiles_list = None

    def _print_progress(self, current: int, total: int, width: int = 32):
        """Render a lightweight terminal progress bar without extra dependencies."""
        if total <= 0:
            return
        filled = int(width * current / total)
        bar = "#" * filled + "-" * (width - filled)
        pct = (100.0 * current) / total
        print(f"\r[{bar}] {current}/{total} ({pct:5.1f}%)", end="", flush=True)
        if current >= total:
            print()

    # ==================== Data Loading ====================
    def load_data(self, verbose: bool = True):
        """Load QM8 and QM9 CSV files."""
        if verbose:
            print("Loading QM8 and QM9 datasets...")

        self.df8 = pd.read_csv(self.qm8_path)
        self.df9 = pd.read_csv(self.qm9_path)

        self._validate_expected_columns(self.df8, "QM8", self.qm8_path)
        self._validate_expected_columns(self.df9, "QM9", self.qm9_path)

        if verbose:
            print(f"QM8 shape: {self.df8.shape}")
            print(f"QM9 shape: {self.df9.shape}")

    @staticmethod
    def _validate_expected_columns(df: pd.DataFrame, dataset_name: str, dataset_path: str):
        """Fail fast with an actionable message when required dataset columns are missing."""
        required_columns = {"smiles"}
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            preview_columns = ", ".join(map(str, df.columns[:5])) or "<no columns>"
            raise ValueError(
                f"{dataset_name} dataset at '{dataset_path}' is missing required columns: {missing_columns}. "
                f"Found columns: {preview_columns}. "
                "If this file came from a Git clone, verify Git LFS assets were downloaded with 'git lfs pull'."
            )

    def validate_data(self, verbose: bool = True):
        """Check for missing values and data quality issues."""
        if not verbose:
            return
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
    @staticmethod
    def _canonicalize(smiles):
        """Convert a SMILES string to its canonical form."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol) if mol else None
        except Exception:
            return None

    def canonicalize_smiles(
        self,
        verbose: bool = True,
        n_jobs: int = -1,
    ):
        """Canonicalize SMILES in both datasets."""
        if verbose:
            print("\nCanonicalizing SMILES strings...")

        smiles8 = self.df8['smiles'].tolist()
        smiles9 = self.df9['smiles'].tolist()

        num_cores = cpu_count() if n_jobs == -1 else n_jobs
        with Pool(num_cores) as pool:
            if verbose:
                print(f"Using {num_cores} cores for canonicalization.")

            self.df8['smiles'] = pool.map(self._canonicalize, smiles8)
            self.df9['smiles'] = pool.map(self._canonicalize, smiles9)

        # Remove rows with invalid SMILES
        self.df8 = self.df8[self.df8['smiles'].notna()]
        self.df9 = self.df9[self.df9['smiles'].notna()]

        if verbose:
            print(f"QM8 after canonicalization: {self.df8.shape}")
            print(f"QM9 after canonicalization: {self.df9.shape}")

    # ==================== Data Merging ====================
    def merge_datasets(self, verbose: bool = True):
        """Merge QM8 and QM9 datasets using outer join on canonicalized SMILES."""
        if verbose:
            print("\nMerging QM8 and QM9 datasets...")
        self.df_merged = pd.merge(self.df8, self.df9, on='smiles', how='outer')
        if verbose:
            print(f"Merged dataset shape: {self.df_merged.shape}")
        return self.df_merged

    # ==================== Targets and Masks ====================
    def create_targets_and_masks(self, verbose: bool = True):
        """
        Create target matrices (Y) and masks.
        - Y: Contains property values (NaN replaced with 0)
        - mask: Indicates which values are real (1) vs. NaN replacements (0)
        """
        if verbose:
            print("\nCreating target matrices and masks...")

        # Fill NaNs with 0 (for numerical stability)
        self.Y = self.df_merged[self.target_cols].fillna(0).values

        # Create mask: 1 where data exists, 0 where it was NaN
        self.mask = self.df_merged[self.target_cols].notna().astype(int).values

        if verbose:
            print(f"Target (Y) shape: {self.Y.shape}")
            print(f"Mask shape: {self.mask.shape}")

        # Print sparsity information
        total_values = self.Y.size
        missing_values = (self.mask == 0).sum()
        sparsity = (missing_values / total_values) * 100
        if verbose:
            print(f"Data sparsity: {sparsity:.2f}%")

        return self.Y, self.mask

    def get_tensors(self, verbose: bool = True):
        """Convert Y and mask to PyTorch tensors and perform Masked Z-Score Normalization."""
        if verbose:
            print("\nConverting to PyTorch tensors and Normalizing...")
            
        self.Y_tensor = torch.tensor(self.Y, dtype=torch.float32)
        self.mask_tensor = torch.tensor(self.mask, dtype=torch.float32)

        # --- MASKED NORMALIZATION ---
        # 1. Calculate sum and count of only the valid (masked) values
        sum_y = (self.Y_tensor * self.mask_tensor).sum(dim=0)
        count_y = self.mask_tensor.sum(dim=0)
        
        # 2. Calculate true mean
        self.y_mean = sum_y / count_y
        
        # 3. Calculate true standard deviation
        diff_sq = ((self.Y_tensor - self.y_mean) * self.mask_tensor) ** 2
        self.y_std = torch.sqrt(diff_sq.sum(dim=0) / count_y)
        self.y_std[self.y_std == 0] = 1.0 # Safety fallback to prevent division by zero

        # 4. Apply Z-score normalization ONLY to real values. Keep fake values as 0.0.
        self.Y_tensor = torch.where(
            self.mask_tensor == 1,
            (self.Y_tensor - self.y_mean) / self.y_std,
            0.0 
        )

        if verbose:
            print(f"Y tensor shape: {self.Y_tensor.shape}")
            print("Target data successfully normalized (Mean=0, Std=1).")
            
        return self.Y_tensor, self.mask_tensor

    # ==================== Graph Generation ====================
    @staticmethod
    def _smiles_to_graph(args):
        smiles, y_values, mask_values = args
        
        # 1. Load and prepare for 3D
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
            
        mol = Chem.AddHs(mol) # Hydrogens are required for correct 3D geometry
        
        # 2. Generate 3D Conformer & Calculate Charges
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            AllChem.ComputeGasteigerCharges(mol)
        except ValueError:
            # If 3D embedding fails (rare, but happens), drop the molecule
            return None 

        conf = mol.GetConformer()

        # --- Node Features (Atoms) ---
        node_feats = []
        for i, atom in enumerate(mol.GetAtoms()):
            # Safely extract partial charge
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
                charge                         # NEW: Partial Charge (Index 6)
            ])
        x = torch.tensor(node_feats, dtype=torch.float)

        # --- Edge Features (Bonds) ---
        edge_indices = []
        edge_attrs = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # NEW: Calculate exact 3D distance between atom i and j
            pos_i = conf.GetAtomPosition(i)
            pos_j = conf.GetAtomPosition(j)
            distance = math.sqrt((pos_i.x - pos_j.x)**2 + (pos_i.y - pos_j.y)**2 + (pos_i.z - pos_j.z)**2)

            bond_feats = [
                float(bond.GetBondTypeAsDouble()),  
                float(bond.GetIsConjugated()),      
                float(bond.GetIsAromatic()),        
                distance                            # NEW: 3D Bond Length in Å (Index 3)
            ]

            # Add bidirectional edges
            edge_indices.extend([[i, j], [j, i]])
            edge_attrs.extend([bond_feats, bond_feats])

        if len(edge_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float) # Note: dim is now 4
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # --- Targets and Masks ---
        y = y_values.clone().detach().view(1, -1)
        mask = mask_values.clone().detach().view(1, -1)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, mask=mask)
    def generate_graphs(
        self,
        verbose: bool = True,
        show_progress: bool = True,
        n_jobs: int = -1,
    ):
        """Generate graph representations for all molecules."""
        if verbose:
            print("\nGenerating molecular graphs...")

        num_molecules = len(self.df_merged)
        num_cores = cpu_count() if n_jobs == -1 else n_jobs

        smiles_list_full = self.df_merged['smiles'].tolist()
        args_list = [
            (smiles_list_full[i], self.Y_tensor[i], self.mask_tensor[i])
            for i in range(num_molecules)
        ]

        if verbose:
            print(f"Using {num_cores} cores for graph generation on {num_molecules} molecules")

        with Pool(num_cores) as pool:
            if show_progress:
                results = []
                for i, result in enumerate(pool.imap(MolecularPropertyPipeline._smiles_to_graph, args_list)):
                    results.append(result)
                    self._print_progress(i + 1, num_molecules)
            else:
                results = pool.map(MolecularPropertyPipeline._smiles_to_graph, args_list)

        self.graphs = []
        self.smiles_list = []
        failed_count = 0

        for i, graph in enumerate(results):
            if graph:
                self.graphs.append(graph)
                self.smiles_list.append(smiles_list_full[i])
            else:
                failed_count += 1

        if verbose:
            print(f"\nGraph generation complete!")
            print(f"Successfully generated {len(self.graphs)} graphs")
            print(f"Failed to convert: {failed_count} molecules")
        return self.graphs

    def _default_cache_path(self) -> str:
        """Store cache in GIN/outputs/cache by default."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(project_root, "GIN", "outputs", "cache", "preprocessed_graphs.pt")

    def _source_signature(self) -> dict:
        """Fingerprint raw datasets to invalidate stale cache automatically."""
        qm8_abs = os.path.abspath(self.qm8_path)
        qm9_abs = os.path.abspath(self.qm9_path)
        return {
            "qm8_path": qm8_abs,
            "qm9_path": qm9_abs,
            "qm8_mtime": os.path.getmtime(qm8_abs),
            "qm9_mtime": os.path.getmtime(qm9_abs),
            "qm8_size": os.path.getsize(qm8_abs),
            "qm9_size": os.path.getsize(qm9_abs),
            "target_cols": self.target_cols,
            "cache_version": self.cache_version,
        }

    def _load_cache(self, cache_path: str, verbose: bool = True) -> bool:
        """Load preprocessed graphs and saved split indices when cache is valid."""
        if not os.path.exists(cache_path):
            return False

        try:
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(cache_path, map_location="cpu")

            if payload.get("signature") != self._source_signature():
                if verbose:
                    print(f"Cache found at {cache_path}, but source data changed. Rebuilding...")
                return False

            self.graphs = payload["graphs"]
            self.split_indices = payload["split_indices"]
            self.smiles_list = payload.get("smiles_list")
            self.y_mean = payload.get("y_mean")
            self.y_std = payload.get("y_std")
            if verbose:
                print(f"Loaded {len(self.graphs)} preprocessed graphs from cache: {cache_path}")
            return True
        except Exception as exc:
            if verbose:
                print(f"Failed to load cache ({exc}). Rebuilding from raw data...")
            return False

    def _save_cache(self, cache_path: str, verbose: bool = True):
        """Persist expensive preprocessing outputs for future runs."""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        payload = {
            "graphs": self.graphs,
            "split_indices": self.split_indices,
            "smiles_list": self.smiles_list,
            "signature": self._source_signature(),
            "y_mean": self.y_mean, 
            "y_std": self.y_std
        }
        torch.save(payload, cache_path)
        if verbose:
            print(f"Saved preprocessed cache to: {cache_path}")

    def _build_split_indices(self, num_graphs: int, train_ratio: float, val_ratio: float):
        """Generate and store train/val/test split indices."""
        indices = torch.randperm(num_graphs).tolist()
        train_end = int(num_graphs * train_ratio)
        val_end = int(num_graphs * (train_ratio + val_ratio))
        self.split_indices = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }

    # ==================== DataLoader ====================
    def create_dataloader(
        self,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        split_indices: dict = None,
        verbose: bool = True,
    ):
        """
        Split graphs into train/val/test sets and create DataLoaders.

        Args:
            batch_size: Number of graphs per batch
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation (rest goes to test)
            split_indices: Optional precomputed split indices
            verbose: Print detailed information

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if verbose:
            print(f"\nCreating DataLoaders (batch_size={batch_size})...")

        num_graphs = len(self.graphs)
        if split_indices is None:
            self._build_split_indices(num_graphs, train_ratio, val_ratio)
            split_indices = self.split_indices
        else:
            self.split_indices = split_indices

        train_graphs = [self.graphs[i] for i in split_indices["train"]]
        val_graphs = [self.graphs[i] for i in split_indices["val"]]
        test_graphs = [self.graphs[i] for i in split_indices["test"]]

        self.train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

        if verbose:
            print(f"Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")

            # Get info from first batch
            first_batch = next(iter(self.train_loader))
            print(f"Number of batches: {len(self.train_loader)}")
            print(f"Molecules per batch: {first_batch.num_graphs}")
            print(f"Atoms per batch: {first_batch.x.shape[0]}")
            print(f"Node features: {first_batch.num_node_features}")
            print(f"Edge features: {first_batch.edge_attr.shape[1]}")
            print(f"Target shape per batch: {first_batch.y.shape}")

        return self.train_loader, self.val_loader, self.test_loader

    # ==================== Full Pipeline ====================
    def run_full_pipeline(
        self,
        batch_size: int = 32,
        use_cache: bool = True,
        force_rebuild: bool = False,
        cache_path: str = None,
        verbose: bool = True,
        show_progress: bool = True,
        n_jobs: int = -1,
    ):
        """
        Execute the complete pipeline from raw data to DataLoader.

        Args:
            batch_size: Batch size for the DataLoader
            use_cache: Load/save preprocessed graphs on disk
            force_rebuild: Ignore cache and regenerate from raw files
            cache_path: Optional custom .pt cache file path
            verbose: Print detailed information
            show_progress: Display progress bar during graph generation
            n_jobs: Number of CPU cores for graph generation

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        cache_path = cache_path or self._default_cache_path()

        if use_cache and not force_rebuild and self._load_cache(cache_path, verbose=verbose):
            self.create_dataloader(batch_size=batch_size, split_indices=self.split_indices, verbose=verbose)
            return self.train_loader, self.val_loader, self.test_loader

        if verbose:
            print("=" * 60)
            print("Starting Full Molecular Property Prediction Pipeline")
            print("=" * 60)

        self.load_data(verbose=verbose)
        self.validate_data(verbose=verbose)
        self.canonicalize_smiles(
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.merge_datasets(verbose=verbose)
        self.create_targets_and_masks(verbose=verbose)
        self.get_tensors(verbose=verbose)
        self.generate_graphs(
            verbose=verbose,
            show_progress=show_progress,
            n_jobs=n_jobs,
        )

        self._build_split_indices(len(self.graphs), train_ratio=0.8, val_ratio=0.1)
        if use_cache:
            self._save_cache(cache_path, verbose=verbose)

        self.create_dataloader(batch_size=batch_size, split_indices=self.split_indices, verbose=verbose)

        if verbose:
            print("\n" + "=" * 60)
            print("Pipeline Complete!")
            print("=" * 60)

        return self.train_loader, self.val_loader, self.test_loader

    # ==================== Utility Methods ====================
    def get_summary(self):
        """Print summary statistics of the pipeline."""
        print("\n--- Pipeline Summary ---")
        total_molecules = len(self.df_merged) if self.df_merged is not None else (len(self.graphs) if self.graphs else 0)
        print(f"Total molecules: {total_molecules}")
        print(f"Valid graphs created: {len(self.graphs) if self.graphs else 0}")
        print(f"Target features: {len(self.target_cols)}")
        if self.mask is not None:
            print(f"Data sparsity: {((self.mask == 0).sum() / self.mask.size) * 100:.2f}%")
        else:
            print("Data sparsity: N/A (loaded from cache)")
        print(f"DataLoader batches: {len(self.train_loader) if self.train_loader else 'Not created'}")

    