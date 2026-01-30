# Hybrid Graph-Text Transformer for Molecular Property Prediction

A high-throughput Multi-Task Deep Learning pipeline that predicts 25+ quantum chemical properties simultaneously by integrating QM7, QM8, and QM9 datasets.

This project implements a **Hybrid "Chimera" Architecture** that fuses a Graph Neural Network (GIN) with a Transformer Encoder (BERT-style) to capture both 3D topological structures and long-range chemical dependencies.

---

## 🏗️ Architecture

The model uses a dual-stream encoder approach to maximize the information extracted from molecular representations:

- **Stream A (Graph Expert):** A Graph Isomorphism Network (GIN) processes the molecular graph (Atoms & Bonds) to capture local connectivity and 3D geometry.
- **Stream B (Sequence Expert):** A Transformer Encoder processes SMILES strings to capture global grammar, functional groups, and stereochemistry.
- **Fusion Layer:** Concatenates embeddings from both streams ($V_{gnn} \oplus V_{text}$) and feeds them into task-specific heads.

---

## 🚀 Key Features

- **Multi-Task Learning (MTL):** Simultaneously trains on a unified superset of QM7, QM8, and QM9.
- **Smart Loss Masking:** Implements a custom Masked MSE Loss to handle sparse labels, intelligently ignoring missing properties from incompatible datasets.
- **Hybrid Fusion:** Leverages the synergy between GNNs (topological connectivity) and Transformers (sequence contexts).
- **Z-Score Normalization:** Automatically scales targets to manage the wide range of property scales (e.g., from Atomization Energy in QM7 to Dipole Moment in QM9).
- **Optimized Engineering:**
  - **Lazy Loading:** Uses LMDB/Pickle for efficient data streaming and reduced memory footprint.
  - **Mixed Precision (AMP):** Optimized for NVIDIA Tensor Cores to accelerate training.

---

## 🛠️ Installation

### Prerequisites
- Linux or Windows (WSL2 recommended)
- NVIDIA GPU (CUDA 11.8+)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/hybrid-molecular-gnn.git
cd hybrid-molecular-gnn

# Create a conda environment
conda create -n molgnn python=3.9
conda activate molgnn

# Install Dependencies
# Note: Ensure PyTorch is compatible with your CUDA version!
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install rdkit pandas numpy wandb
```

---

## 📂 Data Preparation

Since feature extraction via RDKit is CPU-intensive, data is pre-processed before training to ensure maximum throughput.

1. **Download Datasets:** Place `qm7.csv`, `qm8.csv`, and `qm9.csv` in the `data/raw/` folder.
2. **Run the ETL Pipeline:** This script merges datasets, normalizes targets, generates graphs/tokens, and saves processed data.

```bash
python src/data/preprocess.py --workers 8
```

**Output:** 
- `data/processed/train_data.pt`
- `data/processed/meta_stats.json`

---

## 🧠 Training

Initiate the training loop with Mixed Precision and Weights & Biases (WandB) logging:

```bash
python train.py \
    --batch_size 64 \
    --epochs 100 \
    --lr 0.001 \
    --gnn_type "gin" \
    --fusion_type "concat"
```

### Monitoring
The training script logs separate loss curves to WandB for each dataset:
- `loss/qm7`
- `loss/qm8`
- `loss/qm9`

---

## 📂 Project Structure

```plaintext
hybrid-molecular-gnn/
├── data/
│   ├── raw/                 # Original CSV files
│   └── processed/           # Featurized PyG objects (ignored by git)
├── src/
│   ├── models/
│   │   ├── hybrid_model.py  # GNN + Transformer architecture
│   │   └── layers.py        # Custom Fusion & Masking layers
│   ├── data/
│   │   ├── dataset.py       # Custom PyTorch Dataset (Lazy Loader)
│   │   └── featurizer.py    # RDKit logic & Tokenizer
│   └── utils/
│       └── metrics.py       # Masked Loss implementation
├── train.py                 # Main training entry point
├── requirements.txt
└── README.md
```

---

## 🔬 Engineering Details

- **Loss Function:** `MaskedMSELoss` using vectorized `torch.isnan` masking.
- **Batching:** A custom `collate_fn` handles diagonal stacking for Graphs and pad-masking for Text simultaneously.
- **Transformer:** Utilizes `nn.TransformerEncoder` (Bidirectional) for superior regression performance compared to GPT-style decoders.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
