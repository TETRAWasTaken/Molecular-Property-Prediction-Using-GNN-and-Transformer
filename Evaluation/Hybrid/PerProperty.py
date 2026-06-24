import os
import sys
from functools import partial
from multiprocessing import cpu_count, freeze_support
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GUI.core.inference import (
    init_hybrid_engine,
    run_hybrid_regression_with_confidence,
    _PROPERTY_NAMES as TARGET_COLS,
)
from Scripts.qm9_delta import (
    HARTREE_TO_EV,
    QM9_DELTA_TARGET_COLUMNS,
    apply_qm9_delta_learning,
)

EV_TO_KCAL_MOL = 23.060548

import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention_scorer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention_scorer(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return self.dropout(attn_output)

class CompatibleStandaloneChemBERTa(nn.Module):
    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        num_targets: int = 12,
        pool_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from transformers import AutoModel
        try:
            self.transformer = AutoModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.transformer = AutoModel.from_pretrained(model_name, local_files_only=False)

        self.hidden_size: int = self.transformer.config.hidden_size
        self.pooled_hidden_size: int = self.hidden_size

        self.attention_pool = AttentionPooling(self.hidden_size, dropout=pool_dropout)
        self.prediction_head = nn.Linear(self.pooled_hidden_size, num_targets)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.attention_pool(outputs.last_hidden_state, attention_mask)
        return self.prediction_head(pooled)

from GIN_2.Utils.GIN import _global_add_pool_safe, GIN

class CompatibleGIN(GIN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if hasattr(self, 'jk_layer_weights'):
            delattr(self, 'jk_layer_weights')
            self.register_parameter('jk_layer_weights', None)

    def forward(self, data, num_graphs=None):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        if num_graphs is None:
            num_graphs = int(batch.max()) + 1

        h_list = [self.node_encoder(x)]
        edge_embeddings = self.edge_encoder(edge_attr)

        virtual_node_feat = self.virtual_node_embedding.weight.expand(num_graphs, -1)

        for layer in range(self.num_layer):
            h_prev = h_list[layer]
            h = h_prev + virtual_node_feat[batch]
            h = self.convs[layer](h, edge_index, edge_embeddings)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            h = h + h_prev
            h_list.append(h)

            if layer < self.num_layer - 1:
                vn_agg = _global_add_pool_safe(h_list[-1], batch, num_graphs)
                vn_update = self.vn_mlp[layer](virtual_node_feat + vn_agg)
                virtual_node_feat = virtual_node_feat + vn_update

        pooled_list = [
            _global_add_pool_safe(h, batch, num_graphs) for h in h_list
        ]
        h_graph = torch.cat(pooled_list, dim=1)

        return self.prediction_head(h_graph)

class CompatibleHybridFusionModel(nn.Module):
    def __init__(
        self,
        gin_hidden_dim: int = 512,
        transformer_model: str = "seyonec/ChemBERTa-zinc-base-v1",
        mlp_hidden_dim: int = 1024,
        output_dim: int = 12,
        dropout: float = 0.1,
        num_gin_layers: int = 5,
    ) -> None:
        super().__init__()
        self.graph_encoder = CompatibleGIN(hidden_dim=gin_hidden_dim, output_dim=output_dim, num_layer=num_gin_layers)
        self.text_encoder = CompatibleStandaloneChemBERTa(
            model_name=transformer_model, num_targets=output_dim
        )

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.pooled_hidden_size, gin_hidden_dim),
            nn.BatchNorm1d(gin_hidden_dim),
            nn.ReLU(),
        )

        self.graph_gate = nn.Sequential(
            nn.Linear(gin_hidden_dim, gin_hidden_dim), nn.Sigmoid()
        )
        self.text_gate = nn.Sequential(
            nn.Linear(gin_hidden_dim, gin_hidden_dim), nn.Sigmoid()
        )

        self.bilinear = nn.Bilinear(gin_hidden_dim, gin_hidden_dim, gin_hidden_dim)

        self.graph_encoder.prediction_head[-1] = nn.Identity()
        self.text_encoder.prediction_head = nn.Identity()

        concat_dim = gin_hidden_dim * 3

        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.BatchNorm1d(mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        graph_data,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_graphs=None,
    ) -> torch.Tensor:
        graph_embedding = self.graph_encoder(graph_data, num_graphs=num_graphs)
        raw_text_embedding = self.text_encoder(input_ids, attention_mask)

        text_embedding = self.text_projector(raw_text_embedding)

        g_weight = self.graph_gate(graph_embedding)
        t_weight = self.text_gate(text_embedding)

        weighted_graph = graph_embedding * g_weight
        weighted_text = text_embedding * t_weight

        interaction = torch.relu(self.bilinear(weighted_graph, weighted_text))
        fused_embedding = torch.cat(
            [weighted_graph, weighted_text, interaction], dim=1
        )
        return self.fusion_mlp(fused_embedding)

def load_model_from_checkpoint(model_path, device, target_cols):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    
    # Infer GIN layers and pooling strategy from checkpoint keys
    num_layers = 0
    has_dual_pool = True
    for key in state.keys():
        if key.startswith("graph_encoder.convs."):
            try:
                layer_idx = int(key.split(".")[2])
                num_layers = max(num_layers, layer_idx + 1)
            except ValueError:
                pass
        if "attention_pool" in key:
            has_dual_pool = False
            
    print(f"Detected checkpoint architecture: GIN layers = {num_layers}, DualPool = {has_dual_pool}")
    
    if has_dual_pool and num_layers == 6:
        from main import HybridFusionModel
        model = HybridFusionModel(output_dim=len(target_cols)).to(device)
    else:
        model = CompatibleHybridFusionModel(output_dim=len(target_cols), num_gin_layers=num_layers).to(device)
        
    model.load_state_dict(state)
    model.eval()
    return model

def evaluate_all_properties_with_inference(
    smiles_list: List[str],
    true_targets: np.ndarray,
    target_cols: List[str],
    model_path: str = None,
    n_conformers: int = 3,
    n_workers: int = -1,
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, np.ndarray, List[str]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using PyTorch on device: {device} for inference.")
    
    if not model_path:
        model_path = str(project_root / "models" / "best_hybrid_model.pth")
    
    print(f"Loading model checkpoint from {model_path}...")
    model = load_model_from_checkpoint(model_path, device, target_cols)

    # Load caches for fast SMILES lookup
    cache_lookup = {}
    scalers = None
    
    TOKENIZED_CACHE_PATH = project_root / "Transformers_2/outputs/cache/tokenized_dataset.pt"
    MOLECULE_CSV_PATH = project_root / "Dataset/New_QM9/molecule_properties.csv"
    ATOM_CSV_PATH = project_root / "Dataset/New_QM9/atom_properties.csv"
    
    print("Loading preprocessed cache to speed up evaluation...")
    try:
        from GIN_2.Utils.preprocessing import RelationalGeometryPipeline
        
        # Load GIN preprocessed graphs
        pyg_dataset = RelationalGeometryPipeline(
            root=str(project_root / 'GIN_2/data'), 
            mol_csv_path=str(MOLECULE_CSV_PATH),
            atom_csv_path=str(ATOM_CSV_PATH),
            target_cols=target_cols
        )
        
        graph_dict = {}
        for g in pyg_dataset:
            clean_id = str(g.mol_id.item()) if torch.is_tensor(g.mol_id) else str(g.mol_id)
            graph_dict[clean_id] = g
            
        # Load tokenized representations
        transformer_data = torch.load(TOKENIZED_CACHE_PATH, weights_only=False)
        t_input_ids = transformer_data['input_ids']
        t_attention_masks = transformer_data['attention_mask']
        t_mol_ids = [str(m).strip() for m in transformer_data['mol_ids']]
        scalers = transformer_data.get('scalers')
        
        token_dict = {}
        for i, mol_id in enumerate(t_mol_ids):
            token_dict[mol_id] = (t_input_ids[i], t_attention_masks[i])
            
        # Map smiles -> (graph, input_ids, attention_mask)
        df_mol = pd.read_csv(MOLECULE_CSV_PATH)
        df_mol['molecule_id'] = df_mol['molecule_id'].astype(str).str.strip()
        smiles_map = df_mol.set_index('molecule_id')['smiles'].to_dict()
        
        for mol_id, smiles in smiles_map.items():
            if mol_id in graph_dict and mol_id in token_dict:
                cache_lookup[smiles] = (graph_dict[mol_id], token_dict[mol_id][0], token_dict[mol_id][1])
                
        print(f"Loaded {len(cache_lookup)} molecules from GIN and ChemBERTa preprocessed caches.")
    except Exception as e:
        print(f"Could not load preprocessed caches ({e}). Running entirely on-the-fly.")

    # Helper function for generating features on the fly
    from transformers import AutoTokenizer
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    
    tokenizer = None
    def generate_features_on_the_fly(smiles: str, tok, seed=42):
        from GUI.core.inference import build_graph_tensors_from_smiles
        try:
            node_features, edge_indices, edge_attr, batch_index = build_graph_tensors_from_smiles(
                smiles, random_seed=seed
            )
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
            encoded = tok(smiles, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            return graph, input_ids, attention_mask
        except Exception as exc:
            return None

    # We will construct a list of final predictions
    final_preds = [None] * len(smiles_list)
    failures = []
    
    # We collect the indices of successful molecules for batched inference
    successful_indices = []
    batch_graphs = []
    batch_input_ids = []
    batch_attention_masks = []
    
    for idx, smiles in enumerate(smiles_list):
        smiles = smiles.strip()
        if smiles in cache_lookup:
            graph, ids, mask = cache_lookup[smiles]
            clean_graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
            batch_graphs.append(clean_graph)
            batch_input_ids.append(ids)
            batch_attention_masks.append(mask)
            successful_indices.append(idx)
        else:
            if tokenizer is None:
                try:
                    from GUI.core.inference import _resolve_explainability_model_path
                    model_path_hf = _resolve_explainability_model_path()
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path_hf), local_files_only=True)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
            
            # Generate features for multiple conformers
            conf_feats = []
            for c in range(n_conformers):
                feat = generate_features_on_the_fly(smiles, tokenizer, seed=17 + c)
                if feat is not None:
                    conf_feats.append(feat)
            
            if conf_feats:
                # Add each conformer to the batch, mapping them to the same global index
                for feat in conf_feats:
                    graph, ids, mask = feat
                    clean_graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
                    batch_graphs.append(clean_graph)
                    batch_input_ids.append(ids)
                    batch_attention_masks.append(mask)
                    successful_indices.append(idx)
            else:
                failures.append((smiles, "Feature generation failed"))
                final_preds[idx] = np.full(len(target_cols), np.nan)

    if failures:
        print(f"\nEncountered {len(failures)} failures during feature preparation.")

    if successful_indices:
        from torch.utils.data import Dataset
        
        class InferenceDataset(Dataset):
            def __init__(self, graphs, ids, masks):
                self.graphs = graphs
                self.ids = ids
                self.masks = masks
            def __len__(self):
                return len(self.graphs)
            def __getitem__(self, idx):
                return {
                    'graph': self.graphs[idx],
                    'input_ids': self.ids[idx],
                    'attention_mask': self.masks[idx]
                }
                
        inf_dataset = InferenceDataset(batch_graphs, batch_input_ids, batch_attention_masks)
        loader = DataLoader(inf_dataset, batch_size=256, shuffle=False, num_workers=0)
        
        inferred_preds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Running batched PyTorch inference"):
                b_graph = batch['graph'].to(device)
                b_ids = batch['input_ids'].to(device)
                b_mask = batch['attention_mask'].to(device)
                
                preds = model(b_graph, b_ids, b_mask)
                inferred_preds.append(preds.cpu())
                
        y_pred_scaled = torch.cat(inferred_preds, dim=0).numpy()
        
        # Descale predictions
        if scalers is not None:
            for i, col in enumerate(target_cols):
                if col in scalers:
                    y_pred_scaled[:, i] = scalers[col].inverse_transform(y_pred_scaled[:, i].reshape(-1, 1)).flatten()
        else:
            print("Warning: Training scalers not found. Outputting raw scaled predictions.")
            
        # Group and average predictions by global index
        pred_dict = {}
        for local_idx, global_idx in enumerate(successful_indices):
            if global_idx not in pred_dict:
                pred_dict[global_idx] = []
            pred_dict[global_idx].append(y_pred_scaled[local_idx])
            
        for global_idx, preds_list in pred_dict.items():
            final_preds[global_idx] = np.mean(preds_list, axis=0)

    y_pred = np.array(final_preds)
    y_true = true_targets
    
    # Calculate metrics
    results = {}
    for i, col in enumerate(target_cols):
        valid_idx = ~np.isnan(y_pred[:, i]) & ~np.isnan(y_true[:, i])
        p, t = y_pred[valid_idx, i], y_true[valid_idx, i]
        metrics = {'RMSE': 0.0, 'MAE': 0.0, 'R^2': 0.0}
        if len(t) >= 2:
            metrics['RMSE'] = np.sqrt(mean_squared_error(t, p))
            metrics['MAE'] = mean_absolute_error(t, p)
            metrics['R^2'] = r2_score(t, p)
            if col in QM9_DELTA_TARGET_COLUMNS:
                error_kcal_mol = np.abs(t - p) * EV_TO_KCAL_MOL
                metrics['Chem. Acc. (%)'] = np.mean(error_kcal_mol <= 1.0) * 100.0
        results[col] = metrics
        
    return results, y_pred, y_true, smiles_list

if __name__ == "__main__":
    freeze_support()
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    TOKENIZED_CACHE_PATH = project_root / "Transformers_2/outputs/cache/tokenized_dataset.pt"
    MOLECULE_CSV_PATH = project_root / "Dataset/New_QM9/molecule_properties.csv"
    # Checkpoint path — used to load saved split indices (Bug 3 fix)
    CHECKPOINT_PATH = project_root / "models" / "best_hybrid_model.pth"
    model_path, N_WORKERS, N_CONFORMERS = str(CHECKPOINT_PATH), 1, 1  # -1 uses cpu_count() for parallel processing

    if not (TOKENIZED_CACHE_PATH.exists() and MOLECULE_CSV_PATH.exists()):
        print("Dataset files not found. Cannot run evaluation.")
        sys.exit(1)

    print("Preparing test data...")
    df_mol = pd.read_csv(MOLECULE_CSV_PATH)
    df_mol['molecule_id'] = df_mol['molecule_id'].astype(str).str.strip()
    smiles_map = df_mol.set_index('molecule_id')['smiles'].to_dict()
    transformer_data = torch.load(TOKENIZED_CACHE_PATH, weights_only=False)
    t_mol_ids = [str(m).strip() for m in transformer_data['mol_ids']]

    # Ensure we strictly use mol_ids that exist in both the cache and the CSV
    valid_mol_ids = [m for m in t_mol_ids if m in smiles_map]
    if len(valid_mol_ids) != len(t_mol_ids):
        print(f"Warning: Found {len(t_mol_ids) - len(valid_mol_ids)} missing molecule IDs in CSV.")

    df_mol_aligned = df_mol.set_index('molecule_id').loc[valid_mol_ids]
    
    # Load targets directly from the tokenized cache to ensure perfect consistency with training
    t_targets = transformer_data['labels'].numpy()
    t_scalers = transformer_data.get('scalers')
    
    # Descale the true targets to physical units using the same scalers as the model predictions
    original_targets = np.copy(t_targets)
    if t_scalers is not None:
        for i, col in enumerate(TARGET_COLS):
            if col in t_scalers:
                original_targets[:, i] = t_scalers[col].inverse_transform(original_targets[:, i].reshape(-1, 1)).flatten()
                
    # Build a map from mol_id -> target row index in transformer_data
    mol_id_to_idx = {mol_id: idx for idx, mol_id in enumerate(t_mol_ids)}
    
    # Extract original_targets aligned with valid_mol_ids
    original_targets_aligned = np.array([original_targets[mol_id_to_idx[m]] for m in valid_mol_ids], dtype=np.float64)

    n_samples = len(valid_mol_ids)

    # Bug 3 fix: try to load the exact test indices saved by main.py so the
    # evaluation set matches training exactly.  Fall back to re-splitting only
    # if the checkpoint does not contain split_info (e.g. an old checkpoint).
    test_indices = None
    if CHECKPOINT_PATH.exists():
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'split_info' in ckpt:
                split_info = ckpt['split_info']
                if split_info.get('dataset_length') == n_samples:
                    test_indices = split_info['test_indices']
                    print(f"Loaded {len(test_indices)} test indices from checkpoint split_info.")
                else:
                    print(
                        f"Warning: checkpoint dataset_length ({split_info.get('dataset_length')}) "
                        f"!= current dataset length ({n_samples}). Falling back to re-split."
                    )
        except Exception as exc:
            print(f"Warning: could not read split_info from checkpoint ({exc}). Falling back to re-split.")

    if test_indices is None:
        # Fallback: reproduce the same 80/10/10 split used by main.py
        train_size = int(0.8 * n_samples)
        val_size = int(0.1 * n_samples)
        test_size = n_samples - train_size - val_size
        generator = torch.Generator().manual_seed(42)
        _, _, test_indices_subset = torch.utils.data.random_split(
            range(n_samples), [train_size, val_size, test_size], generator=generator
        )
        test_indices = list(test_indices_subset.indices)
        print(f"Reconstructed {len(test_indices)} test indices via fallback re-split.")

    test_smiles = [smiles_map[valid_mol_ids[i]] for i in test_indices]
    test_targets = original_targets_aligned[test_indices]
    print(f"Loaded test split with {len(test_smiles)} samples.")

    if test_smiles and test_targets is not None:
        print("\nStarting evaluation...")
        all_results, y_pred, y_true, processed_smiles = evaluate_all_properties_with_inference(
            test_smiles, test_targets, TARGET_COLS, model_path, N_CONFORMERS, N_WORKERS
        )
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df.index.name = 'Property'
        print("\n--- Hybrid Model Evaluation Results (PyTorch) ---")
        print(results_df.to_string(float_format="%.4f"))
        print("----------------------------------------------\n")

        output_dir = Path(__file__).resolve().parent
        
        # --- Save Aggregate and Per-Molecule Results ---
        
        # 1. Save aggregate metrics
        results_df.to_csv(output_dir / "evaluation_results.csv", float_format="%.4f")
        
        # 2. Save raw predictions for plotting and analysis
        np.savez(output_dir / "predictions.npz", y_pred=y_pred, y_true=y_true, smiles=np.array(processed_smiles))
        
        # 3. Create and save per-molecule evaluation results
        df_true = pd.DataFrame(y_true, columns=[f"{col}_true" for col in TARGET_COLS])
        df_pred = pd.DataFrame(y_pred, columns=[f"{col}_pred" for col in TARGET_COLS])
        df_error = pd.DataFrame(np.abs(y_true - y_pred), columns=[f"{col}_error" for col in TARGET_COLS])
        
        df_per_molecule = pd.concat([pd.DataFrame({'smiles': processed_smiles}), df_true, df_pred, df_error], axis=1)
        
        per_molecule_output_path = output_dir / "per_molecule_evaluation_results.csv"
        df_per_molecule.to_csv(per_molecule_output_path, index=False, float_format="%.6f")

        print(f"Aggregate results saved to {output_dir / 'evaluation_results.csv'}")
        print(f"Per-molecule results saved to {per_molecule_output_path}")
        print(f"Raw predictions for analysis saved to {output_dir / 'predictions.npz'}")
    else:
        print("No data loaded. Skipping evaluation.")