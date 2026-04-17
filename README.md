# Hybrid Graph-Text Transformer for Molecular Property Prediction

This repository contains a hybrid molecular property predictor that combines:

- a GIN graph encoder (structure features)
- a Transformer encoder (SMILES sequence features)
- a fused inference engine used by the desktop GUI

## GUI Setup (Quick)

Run all commands from the project root.

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install PySide6
```

3. Verify required GUI assets exist:

- `GUI/assets/hybrid_model.onnx`
- `GUI/core/libhybrid_engine.dylib` (macOS)

4. Launch the GUI:

```bash
python GUI/guimain.py
```

## Optional: Rebuild the Native Inference Library (macOS)

Only needed if `libhybrid_engine.dylib` is missing or outdated.

```bash
cd GUI/core
make clean && make
cd ../..
```

## Platform Notes (Linux / Windows)

- Linux shared library: `GUI/core/libhybrid_engine.so`
- Windows shared library: `GUI/core/hybrid_engine.dll`

Linux rebuild:

```bash
cd GUI/core
make so
cd ../..
```

Windows rebuild:

Use your Windows ONNX Runtime C SDK/include+lib paths and compile `GUI/core/inference.c` to `hybrid_engine.dll`.
This repo currently bundles ONNX Runtime binaries for macOS under `GUI/assets/onnxruntime-osx-arm64-1.24.4`.

Windows launch:

```powershell
python GUI/guimain.py
```

## Troubleshooting

- `ONNX model file was not found`:
	Place the model at `GUI/assets/hybrid_model.onnx` or set `HYBRID_ONNX_MODEL_PATH`.
- Qt WebEngine rendering issues on macOS:
	`GUI_FORCE_SOFTWARE_WEBENGINE=1 python GUI/guimain.py`
