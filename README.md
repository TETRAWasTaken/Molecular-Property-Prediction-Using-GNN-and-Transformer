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

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
