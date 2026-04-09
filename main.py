import os
import torch
import torch.nn as nn
import multiprocessing
import subprocess
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader

# 1. Modular Imports: Pull directly from your standalone folders
from GIN_2.Utils.GIN import GIN
from Transformers_2.Utils.Transformer import StandaloneChemBERTa

# ==========================================
# 2. DATASET & ALIGNMENT
# ==========================================
class HybridDataset(Dataset):
    """
    Ensures PyG 3D Graphs and Tokenized Text stay perfectly aligned during batching.
    Now includes nan_mask for robust loss calculation.
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

# ==========================================
# 3. HYBRID FUSION MODEL ARCHITECTURE
# ==========================================
class HybridFusionModel(nn.Module):
    def __init__(self, gin_hidden_dim=256, transformer_model="seyonec/ChemBERTa-zinc-base-v1", mlp_hidden_dim=512, output_dim=12, dropout=0.1):
        super().__init__()
        
        self.graph_encoder = GIN(hidden_dim=gin_hidden_dim, output_dim=output_dim)
        self.text_encoder = StandaloneChemBERTa(model_name=transformer_model, num_targets=output_dim)
        
        # Convert them to feature extractors (Encoders)
        self.graph_encoder.prediction_head[-1] = nn.Identity() 
        self.text_encoder.prediction_head = nn.Identity()
        
        concat_dim = gin_hidden_dim + self.text_encoder.hidden_size
        
        # Unified MLP
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
        graph_embedding = self.graph_encoder(graph_data)
        text_embedding = self.text_encoder(input_ids, attention_mask)
        fused_embedding = torch.cat([graph_embedding, text_embedding], dim=1)
        return self.fusion_mlp(fused_embedding)


# ==========================================
# 4. MULTIPROCESSING WRAPPERS
# ==========================================
def run_gin_preprocessing():
    print("[Process 1] Starting GIN Preprocessing...")
    # Import the pipeline directly
    from GIN_2.Utils.preprocessing import RelationalGeometryPipeline
    
    # Initialize it. If the cache doesn't exist, it will build it.
    pipeline = RelationalGeometryPipeline(
        root='GIN_2/data', 
        mol_csv_path='Dataset/New_QM9/molecule_properties.csv', 
        atom_csv_path='Dataset/New_QM9/atom_properties.csv', 
        target_cols=['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
    )
    print("[Process 1] GIN Preprocessing Complete.")


def run_transformer_preprocessing():
    print("[Process 2] Starting Transformer Preprocessing...")
    from Transformers_2.Utils.Tokeniser import Tokeniser
    
    tokeniser = Tokeniser(
        mol_path='Dataset/New_QM9/molecule_properties.csv',
        model_name="seyonec/ChemBERTa-zinc-base-v1",
        max_length=64,
        use_cache=True 
    )
    # Just run the tokenizer to build the cache, skip training
    tokeniser.run_tokenizer(verbose=False)
    print("[Process 2] Transformer Preprocessing Complete.")

def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, nan_mask: torch.Tensor) -> torch.Tensor:
    loss_fn = nn.MSELoss(reduction='none')
    raw_loss = loss_fn(predictions, targets)
    masked_loss = raw_loss * nan_mask
    valid_entries = nan_mask.sum()
    if valid_entries > 0:
        return masked_loss.sum() / valid_entries
    else:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

# ==========================================
# 5. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == '__main__':
    # --- A. Parallel Preprocessing Phase ---
    print("Initiating parallel preprocessing for Graph and Text pipelines...\n")
    p1 = multiprocessing.Process(target=run_gin_preprocessing)
    p2 = multiprocessing.Process(target=run_transformer_preprocessing)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
    print("\nAll preprocessing finished. Proceeding to Data Loading...\n")

   # --- B. Data Loading Phase ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    TARGET_COLS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
    
    print("Loading cached datasets from disk...")
    
    # 1. Load PyG Graphs natively using the Pipeline
    from GIN_2.Utils.preprocessing import RelationalGeometryPipeline
    pyg_dataset = RelationalGeometryPipeline(
        root='GIN_2/data', 
        mol_csv_path='Dataset/New_QM9/molecule_properties.csv', 
        atom_csv_path='Dataset/New_QM9/atom_properties.csv', 
        target_cols=TARGET_COLS
    )
    # Extract the individual graphs into a standard list
    pyg_graph_list = [g for g in pyg_dataset]

    # 2. Load Transformer Data
    transformer_data = torch.load('Transformers_2/outputs/cache/tokenized_dataset.pt', weights_only=False)
    # 3. Build a dictionary mapping mol_id -> Graph for instant lookup
    graph_dict = {}
    for g in pyg_graph_list:
    # If mol_id is a tensor, we use .item() to get the scalar value
        if torch.is_tensor(g.mol_id):
        # .item() gets the number, then we cast to string for the match
             clean_id = str(g.mol_id.item())
        else:
            clean_id = str(g.mol_id)
    
        graph_dict[clean_id] = g
    # 4. Extract Tokeniser data
    t_input_ids = transformer_data['input_ids']
    t_attention_masks = transformer_data['attention_mask']
    t_targets = transformer_data['labels']
    t_nan_mask = transformer_data['nan_mask']
    t_mol_ids = [str(m).strip() for m in transformer_data['mol_ids']]    
    t_scalers = transformer_data['scalers'] 

    # 5. DYNAMIC ALIGNMENT: Only keep data that exists in BOTH pipelines
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

    # 6. Stack the text lists back into tensors
    aligned_input_ids = torch.stack(aligned_input_ids)
    aligned_attention_masks = torch.stack(aligned_attention_masks)
    aligned_targets = torch.stack(aligned_targets)
    aligned_nan_masks = torch.stack(aligned_nan_masks)

    # 7. Create Dataset and Split
    full_dataset = HybridDataset(
        aligned_graphs, 
        aligned_input_ids, 
        aligned_attention_masks, 
        aligned_targets, 
        aligned_nan_masks
    )

    # Recalculate train/val split on the freshly aligned data
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Use a fixed seed so the train/val split is identical every time you run main.py
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # --- C. Model & Optimizer Setup ---
    model = HybridFusionModel().to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.graph_encoder.parameters(), 'lr': 3e-4},
        {'params': model.fusion_mlp.parameters(), 'lr': 3e-4},
        {'params': model.text_encoder.parameters(), 'lr': 5e-5}
    ])
    
    epochs = 15
    freeze_transformer_epochs = 5
    best_val_loss = float('inf')

    # --- D. Training Engine ---
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
            
            # CHANGED: Use the custom masked MSE loss
            loss = masked_mse_loss(predictions, b_targets, b_nan_mask)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        # --- E. Validation Engine ---
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
                
                # Accumulate for physical metrics
                all_preds.append(predictions.cpu())
                all_targets.append(b_targets.cpu())
                all_masks.append(b_nan_mask.cpu())
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # --- Physical Metric Calculation (Denormalization) ---
        y_pred = torch.cat(all_preds, dim=0).numpy()
        y_true = torch.cat(all_targets, dim=0).numpy()
        y_mask = torch.cat(all_masks, dim=0).numpy()

        # Reverse the scaling using the dictionary
        for i, col in enumerate(TARGET_COLS):
            if col in t_scalers:
                y_pred[:, i] = t_scalers[col].inverse_transform(y_pred[:, i].reshape(-1, 1)).flatten()
                y_true[:, i] = t_scalers[col].inverse_transform(y_true[:, i].reshape(-1, 1)).flatten()

        mae_per_prop = []
        r2_per_prop = []
        
        # Calculate metrics only on valid (non-NaN) entries
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
        
        # --- Print the clean summary ---
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {overall_mae:.4f} | Val R²: {overall_r2:.4f}")
        
        # Print the detailed breakdown every 5 epochs or on a new best model
        if epoch == 0 or (epoch + 1) % 5 == 0 or avg_val_loss < best_val_loss:
            print("\n  --- Per-Property Validation Breakdown ---")
            for i in range(len(TARGET_COLS)):
                print(f"  {TARGET_COLS[i].ljust(10)} | MAE: {mae_per_prop[i]:.4f} | R²: {r2_per_prop[i]:.4f}")
            print("  -----------------------------------------\n")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_hybrid_model.pth")