import torch
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer
import sys
import os

# Ensure absolute imports work
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Hybrid.Models.fusion_network import HybridFusionNetwork

print("--- STARTING SANITY CHECK ---")

print("\n1. Initializing Hybrid Master Model...")
model = HybridFusionNetwork(node_in_dim=6, edge_in_dim=3, gin_hidden_dim=128, output_dim=12)

print("\n2. Generating Dummy Graphs and REAL Tokens...")
# Dummy Graph 1
x1 = torch.randn(5, 6) 
edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
edge_attr1 = torch.randn(4, 3) 
data1 = Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1)

# Dummy Graph 2 
x2 = torch.randn(4, 6) 
edge_index2 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
edge_attr2 = torch.randn(3, 3)
data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2)

# Batch the graphs together
batch = Batch.from_data_list([data1, data2])

# Generate REAL tokens using the actual HuggingFace tokenizer
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
dummy_smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO"] # Aspirin and Ethanol
tokens = tokenizer(dummy_smiles, padding='max_length', max_length=64, truncation=True, return_tensors='pt')

# Inject the real tokens into the PyG batch
batch.input_ids = tokens['input_ids']
batch.attention_mask = tokens['attention_mask']

print("\n3. Pushing Data Through the Network...")
try:
    model.eval() 
    with torch.no_grad():
        output = model(batch)
        
    print(f"\n✅ SUCCESS! Wiring is perfect.")
    print(f"Final Output Shape: {list(output.shape)} (Expected: [2, 12])")
    
except Exception as e:
    print(f"\n❌ FAILED! The network crashed.")
    print(f"Error Traceback: {e}")