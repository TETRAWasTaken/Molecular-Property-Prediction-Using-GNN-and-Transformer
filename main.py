"""Train and evaluate a hybrid molecular property predictor that combines GIN graph
representations with ChemBERTa text embeddings for aligned multitask regression.

The script prepares cached graph and tokenizer artifacts, aligns them by molecule
identifier, trains a fusion model, and reports validation losses and per-property
metrics with early stopping.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"

import multiprocessing

import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader

from GIN_2.Utils.GIN import GIN
from Transformers_2.Utils.Transformer import StandaloneChemBERTa


class HybridDataset(Dataset):
    """Dataset wrapper that keeps graph, token, target, and mask tensors aligned.

    Each item exposes the PyG graph object together with the corresponding
    tokenized text inputs, regression targets, and NaN mask used to ignore
    missing labels during loss computation.

    Args:
        pyg_graph_list: List of PyG graph objects aligned to the tokenizer output.
        tokenized_input_ids: Token ID tensors for each molecule.
        tokenized_attention_masks: Attention mask tensors for each molecule.
        targets: Multitask regression targets.
        nan_mask: Mask with 1s for valid targets and 0s for missing targets.
    """

    def __init__(self, pyg_graph_list, tokenized_input_ids, tokenized_attention_masks, targets, nan_mask):
        self.graphs = pyg_graph_list
        self.input_ids = tokenized_input_ids
        self.attention_masks = tokenized_attention_masks
        self.targets = targets
        self.nan_mask = nan_mask

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return {
            'graph': self.graphs[idx],
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'target': self.targets[idx],
            'nan_mask': self.nan_mask[idx]
        }

class HybridFusionModel(nn.Module):
    """Fuse graph and text embeddings into a multitask molecular property head.

    The model encodes molecular graphs with a GIN backbone, encodes molecular
    text with ChemBERTa, projects the text embedding into the graph embedding
    space, applies learned gating to both modalities, constructs an explicit
    bilinear interaction, and predicts all target properties with a shared MLP.

    Args:
        gin_hidden_dim: Hidden dimension used by the graph encoder and fusion blocks.
        transformer_model: Hugging Face model name for the ChemBERTa encoder.
        mlp_hidden_dim: Hidden dimension used by the fusion MLP.
        output_dim: Number of regression targets to predict.
        dropout: Dropout probability applied inside the fusion MLP.
    """

    def __init__(self, gin_hidden_dim=512, transformer_model="seyonec/ChemBERTa-zinc-base-v1", mlp_hidden_dim=1024, output_dim=12, dropout=0.1):
        super().__init__()
        
        self.graph_encoder = GIN(hidden_dim=gin_hidden_dim, output_dim=output_dim)
        self.text_encoder = StandaloneChemBERTa(model_name=transformer_model, num_targets=output_dim)

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.hidden_size, gin_hidden_dim),
            nn.BatchNorm1d(gin_hidden_dim),
            nn.ReLU()
        )

        self.graph_gate = nn.Sequential(nn.Linear(gin_hidden_dim, gin_hidden_dim), nn.Sigmoid())
        self.text_gate = nn.Sequential(nn.Linear(gin_hidden_dim, gin_hidden_dim), nn.Sigmoid())

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
            
            nn.Linear(mlp_hidden_dim // 2, output_dim)
        )

    def forward(self, graph_data, input_ids, attention_mask):
        """Compute multitask predictions from graph and text inputs.

        Args:
            graph_data: Batched PyG graph data.
            input_ids: Token IDs for the text branch.
            attention_mask: Attention masks for the text branch.

        Returns:
            A tensor of multitask regression predictions.
        """

        graph_embedding = self.graph_encoder(graph_data)
        raw_text_embedding = self.text_encoder(input_ids, attention_mask)

        text_embedding = self.text_projector(raw_text_embedding)

        g_weight = self.graph_gate(graph_embedding)
        t_weight = self.text_gate(text_embedding)

        weighted_graph = graph_embedding * g_weight
        weighted_text = text_embedding * t_weight

        interaction = torch.relu(self.bilinear(weighted_graph, weighted_text))
        fused_embedding = torch.cat([weighted_graph, weighted_text, interaction], dim=1)
        
        return self.fusion_mlp(fused_embedding)

def run_gin_preprocessing():
    """Build or reuse the cached GIN graph preprocessing artifacts for QM9-like data.

    Args:
        None.
    """

    print("[Process 1] Starting GIN Preprocessing...")
    from GIN_2.Utils.preprocessing import RelationalGeometryPipeline
    
    pipeline = RelationalGeometryPipeline(
        root='GIN_2/data', 
        mol_csv_path='Dataset/New_QM9/molecule_properties.csv', 
        atom_csv_path='Dataset/New_QM9/atom_properties.csv', 
        target_cols=['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
    )
    print("[Process 1] GIN Preprocessing Complete.")


def run_transformer_preprocessing():
    """Build or reuse the cached ChemBERTa tokenization artifacts for the dataset.

    Args:
        None.
    """

    print("[Process 2] Starting Transformer Preprocessing...")
    from Transformers_2.Utils.Tokeniser import Tokeniser
    
    tokeniser = Tokeniser(
        mol_path='Dataset/New_QM9/molecule_properties.csv',
        model_name="seyonec/ChemBERTa-zinc-base-v1",
        max_length=64,
        use_cache=True 
    )
    tokeniser.run_tokenizer(verbose=False)
    print("[Process 2] Transformer Preprocessing Complete.")

def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, nan_mask: torch.Tensor, beta=1.0) -> torch.Tensor:
    """Compute a NaN-masked SmoothL1 regression loss over valid target entries.

    The mask is expected to contain 1s for valid targets and 0s for missing
    targets. The beta parameter controls the transition point for SmoothL1Loss.

    Args:
        predictions: Model outputs with shape matching targets.
        targets: Ground-truth multitask regression labels.
        nan_mask: Binary mask indicating valid target entries.
        beta: SmoothL1 transition point between L2 and L1 behavior.

    Returns:
        The average masked SmoothL1 loss across valid entries.
    """

    loss_fn = nn.SmoothL1Loss(reduction='none', beta=beta)
    
    raw_loss = loss_fn(predictions, targets)
    masked_loss = raw_loss * nan_mask
    
    valid_entries = nan_mask.sum()
    if valid_entries > 0:
        return masked_loss.sum() / valid_entries
    else:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

if __name__ == '__main__':
    print("Initiating parallel preprocessing for Graph and Text pipelines...\n")
    p1 = multiprocessing.Process(target=run_gin_preprocessing)
    p2 = multiprocessing.Process(target=run_transformer_preprocessing)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
    print("\nAll preprocessing finished. Proceeding to Data Loading...\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    TARGET_COLS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
    
    print("Loading cached datasets from disk...")
    
    from GIN_2.Utils.preprocessing import RelationalGeometryPipeline
    pyg_dataset = RelationalGeometryPipeline(
        root='GIN_2/data', 
        mol_csv_path='Dataset/New_QM9/molecule_properties.csv', 
        atom_csv_path='Dataset/New_QM9/atom_properties.csv', 
        target_cols=TARGET_COLS
    )
    pyg_graph_list = [g for g in pyg_dataset]

    transformer_data = torch.load('Transformers_2/outputs/cache/tokenized_dataset.pt', weights_only=False)
    
    graph_dict = {}
    for g in pyg_graph_list:
        if torch.is_tensor(g.mol_id):
             clean_id = str(g.mol_id.item())
        else:
            clean_id = str(g.mol_id)
    
        graph_dict[clean_id] = g
        
    t_input_ids = transformer_data['input_ids']
    t_attention_masks = transformer_data['attention_mask']
    t_targets = transformer_data['labels']
    t_nan_mask = transformer_data['nan_mask']
    t_mol_ids = [str(m).strip() for m in transformer_data['mol_ids']]    
    t_scalers = transformer_data['scalers'] 

    aligned_graphs = []
    aligned_input_ids = []
    aligned_attention_masks = []
    aligned_targets = []
    aligned_nan_masks = []
    
    print(f"Sample Transformer ID: '{t_mol_ids[0]}' (Type: {type(t_mol_ids[0])})")
    sample_graph_id = list(graph_dict.keys())[0]
    print(f"Sample Graph ID: '{sample_graph_id}' (Type: {type(sample_graph_id)})")
    
    print("Aligning multimodal data...")
    for i, mol_id in enumerate(t_mol_ids):
        if mol_id in graph_dict:
            aligned_graphs.append(graph_dict[mol_id])
            aligned_input_ids.append(t_input_ids[i])
            aligned_attention_masks.append(t_attention_masks[i])
            aligned_targets.append(t_targets[i])
            aligned_nan_masks.append(t_nan_mask[i])
            
    if len(aligned_input_ids) == 0:
        raise ValueError("Zero molecules were aligned! Check if 'mol_id' in your Graph objects matches the IDs in 'tokenized_dataset.pt'.")
    print(f"Alignment complete! Kept {len(aligned_graphs)} valid multimodal molecules.")

    aligned_input_ids = torch.stack(aligned_input_ids)
    aligned_attention_masks = torch.stack(aligned_attention_masks)
    aligned_targets = torch.stack(aligned_targets)
    aligned_nan_masks = torch.stack(aligned_nan_masks)

    full_dataset = HybridDataset(
        aligned_graphs, 
        aligned_input_ids, 
        aligned_attention_masks, 
        aligned_targets, 
        aligned_nan_masks
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = HybridFusionModel().to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.graph_encoder.parameters(), 'lr': 3e-4, 'weight_decay': 1e-3},
        {'params': model.fusion_mlp.parameters(), 'lr': 3e-4, 'weight_decay': 1e-2},
        {'params': model.text_encoder.parameters(), 'lr': 5e-5, 'weight_decay': 1e-5}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    epochs = 50
    freeze_transformer_epochs = 5
    patience = 10
    early_stop_counter = 0
    best_val_loss = float('inf')

    print(f"\nStarting Hybrid Training on {device}...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        if epoch < freeze_transformer_epochs:
            for param in model.text_encoder.transformer.parameters():
                param.requires_grad = False
        else:
            for param in model.text_encoder.transformer.parameters():
                param.requires_grad = True

        for batch in train_loader:
            graph_data = batch['graph'].to(device)
            b_input_ids = batch['input_ids'].to(device)
            b_attention_mask = batch['attention_mask'].to(device)
            b_targets = batch['target'].to(device)
            b_nan_mask = batch['nan_mask'].to(device)
            
            optimizer.zero_grad()
            
            predictions = model(graph_data, b_input_ids, b_attention_mask)
            
            loss = masked_mse_loss(predictions, b_targets, b_nan_mask)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        
        all_preds = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for batch in val_loader:
                graph_data = batch['graph'].to(device)
                b_input_ids = batch['input_ids'].to(device)
                b_attention_mask = batch['attention_mask'].to(device)
                b_targets = batch['target'].to(device)
                b_nan_mask = batch['nan_mask'].to(device)
                
                predictions = model(graph_data, b_input_ids, b_attention_mask)
                loss = masked_mse_loss(predictions, b_targets, b_nan_mask)
                total_val_loss += loss.item()

                all_preds.append(predictions.cpu())
                all_targets.append(b_targets.cpu())
                all_masks.append(b_nan_mask.cpu())
                
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        y_pred = torch.cat(all_preds, dim=0).numpy()
        y_true = torch.cat(all_targets, dim=0).numpy()
        y_mask = torch.cat(all_masks, dim=0).numpy()

        for i, col in enumerate(TARGET_COLS):
            if col in t_scalers:
                y_pred[:, i] = t_scalers[col].inverse_transform(y_pred[:, i].reshape(-1, 1)).flatten()
                y_true[:, i] = t_scalers[col].inverse_transform(y_true[:, i].reshape(-1, 1)).flatten()

        mae_per_prop = []
        r2_per_prop = []
        
        for i in range(len(TARGET_COLS)):
            valid_idx = y_mask[:, i] == 1
            if valid_idx.sum() > 0:
                y_p = y_pred[valid_idx, i]
                y_t = y_true[valid_idx, i]
                mae_per_prop.append(mean_absolute_error(y_t, y_p))
                r2_per_prop.append(r2_score(y_t, y_p))
            else:
                mae_per_prop.append(0.0)
                r2_per_prop.append(0.0)

        overall_mae = sum(mae_per_prop) / len(TARGET_COLS)
        overall_r2 = sum(r2_per_prop) / len(TARGET_COLS)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {overall_mae:.4f} | Val R²: {overall_r2:.4f}")
        
        if epoch == 0 or (epoch + 1) % 5 == 0 or avg_val_loss < best_val_loss:
            print("\n  --- Per-Property Validation Breakdown ---")
            for i in range(len(TARGET_COLS)):
                print(f"  {TARGET_COLS[i].ljust(10)} | MAE: {mae_per_prop[i]:.4f} | R²: {r2_per_prop[i]:.4f}")
            print("  -----------------------------------------\n")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_hybrid_model.pth")
            early_stop_counter = 0
            print("  --> Saved new best model.")
        else:
            early_stop_counter += 1
            print(f"  --> No improvement for {early_stop_counter} epochs.")
            
        if early_stop_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}. Restoring best weights.")
            model.load_state_dict(torch.load("best_hybrid_model.pth"))
            break