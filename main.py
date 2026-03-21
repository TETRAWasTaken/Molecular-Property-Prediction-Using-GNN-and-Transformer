import torch
import torch.nn as nn
from torch.utils.data import Dataset
# IMPORTANT: We use PyTorch Geometric's DataLoader, not standard PyTorch's!
# It knows how to batch standard tensors AND Graph objects in the same dictionary.
from torch_geometric.loader import DataLoader 

from hybrid_model.fusion_network import HybridFusionModel

# ==========================================
# 1. BATCH CONSISTENCY: Custom Dataset
# ==========================================
class HybridDataset(Dataset):
    """
    Ensures that Graph #N and Text #N are always paired together in the exact same batch.
    """
    def __init__(self, pyg_graph_list, tokenized_input_ids, tokenized_attention_masks, targets):
        self.graphs = pyg_graph_list
        self.input_ids = tokenized_input_ids
        self.attention_masks = tokenized_attention_masks
        self.targets = targets

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        # PyG's DataLoader will automatically collate this dictionary perfectly.
        return {
            'graph': self.graphs[idx],
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'target': self.targets[idx]
        }

# ==========================================
# 2. INITIALIZATION & DATA LOADING
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridFusionModel().to(device)

# --- MOCK DATA SETUP (Replace with your actual loaded data) ---
# dummy_graphs = [torch_geometric.data.Data(...) for _ in range(100)]
# dummy_input_ids = torch.randint(0, 1000, (100, 128)) 
# dummy_masks = torch.ones((100, 128))
# dummy_targets = torch.randn((100, 12))
# --------------------------------------------------------------

# Create Dataset and DataLoader
# dataset = HybridDataset(dummy_graphs, dummy_input_ids, dummy_masks, dummy_targets)
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ==========================================
# 3. OPTIMIZATION SETUP
# ==========================================
# We assign different learning rates to different parts of the architecture
optimizer = torch.optim.AdamW([
    {'params': model.graph_encoder.parameters(), 'lr': 3e-4},    
    {'params': model.fusion_mlp.parameters(), 'lr': 3e-4},       
    {'params': model.text_encoder.parameters(), 'lr': 5e-5}      
])

criterion = nn.MSELoss()

# ==========================================
# 4. TRAINING LOOP WITH GRADIENT MANAGEMENT
# ==========================================
epochs = 50
freeze_transformer_epochs = 5  # Freeze ChemBERTa for the first 5 epochs

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    # Check if we should freeze or unfreeze the text encoder
    if epoch < freeze_transformer_epochs:
        # Freeze ChemBERTa body to let GIN and MLP catch up
        for param in model.text_encoder.transformer.parameters():
            param.requires_grad = False
    else:
        # Unfreeze for fine-tuning
        for param in model.text_encoder.transformer.parameters():
            param.requires_grad = True

    # Iterate over the perfectly aligned batches
    # for batch in train_loader:
    #     # Move data to GPU/CPU
    #     graph_data = batch['graph'].to(device)
    #     input_ids = batch['input_ids'].to(device)
    #     attention_mask = batch['attention_mask'].to(device)
    #     targets = batch['target'].to(device)
    #     
    #     optimizer.zero_grad()
    #     
    #     # Forward pass
    #     predictions = model(graph_data, input_ids, attention_mask)
    #     
    #     # Calculate loss
    #     loss = criterion(predictions, targets)
    #     
    #     # Backward pass & Optimize
    #     loss.backward()
    #     optimizer.step()
    #     
    #     total_loss += loss.item()
        
    # print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
    pass # Remove this pass when you uncomment the loop above