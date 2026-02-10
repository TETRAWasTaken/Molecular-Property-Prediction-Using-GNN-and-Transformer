# Hybrid Graph-Text Transformer for Molecular Property Prediction

A high-throughput Multi-Task Deep Learning pipeline that predicts 25+ quantum chemical properties simultaneously by integrating QM7, QM8, and QM9 datasets.

This project implements a **Hybrid "Chimera" Architecture** that fuses a Graph Neural Network (GIN) with a Transformer Encoder (BERT-style) to capture both 3D topological structures and long-range chemical dependencies.

---

## Architecture

The model uses a dual-stream encoder approach to maximize the information extracted from molecular representations:

- **Stream A (Graph Expert):** A Graph Isomorphism Network (GIN) processes the molecular graph (Atoms & Bonds) to capture local connectivity and 3D geometry.
- **Stream B (Sequence Expert):** A Transformer Encoder processes SMILES strings to capture global grammar, functional groups, and stereochemistry.
- **Fusion Layer:** Concatenates embeddings from both streams ($V_{gnn} \oplus V_{text}$) and feeds them into task-specific heads.
