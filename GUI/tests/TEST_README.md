# Inference Module Test Suite

## Overview

This test suite (`test_inference.py`) validates the core functionality of the neural network inference module in `inference.py`. It tests graph building from SMILES strings, feature extraction, tokenization, and error handling.

## Running Tests

### Run all tests:
```bash
python GUI/core/test_inference.py
```

### Run with verbose output:
```bash
python GUI/core/test_inference.py -v
```

### Run specific test class:
```bash
python -m unittest GUI.core.test_inference.TestGraphTensorBuilding
```

### Run specific test:
```bash
python -m unittest GUI.core.test_inference.TestGraphTensorBuilding.test_simple_methane
```

## Test Coverage

### TestOneHotEncoding (3 tests)
- Validates one-hot encoding utility for molecular features
- Tests valid elements, invalid elements, and degree encoding

### TestNodeFeatures (2 tests)
- Tests node feature extraction from atoms
- Covers hydrogen and carbon atoms

### TestEdgeFeatures (2 tests)
- Tests edge feature extraction from bonds
- Covers single and double bonds

### TestGraphTensorBuilding (10 tests) ⭐ Core functionality
- Tests graph tensor generation from SMILES
- Covers simple molecules (methane, ethane, water), benzene, and complex structures
- **Validates error handling:**
  - Empty/None SMILES
  - Whitespace-only SMILES
  - Invalid SMILES strings
- Verifies tensor shapes and data types

### TestSMILESTokenization (4 tests)
- Tests ChemBERTa tokenizer encoding
- Handles padding, truncation, and long sequences
- Tests empty SMILES and edge cases

### TestModelPathResolution (3 tests)
- Tests model file path resolution
- Validates error handling for missing models
- Tests tilde expansion in paths

### TestEdgeCases (4 tests)
- Single-atom molecules
- Large molecules (caffeine)
- Charged molecules
- Aromatic molecules

### TestCleanup (1 test)
- Validates cleanup function doesn't raise errors

## Test Statistics

- **Total Tests:** 29
- **Pass Rate:** 100% (when properly configured)
- **Execution Time:** ~1-2 seconds

## Requirements

The test suite requires the following dependencies from `requirements.txt`:
- `rdkit` - Molecular structure parsing and feature extraction
- `numpy` - Array operations
- `transformers` - SMILES tokenization via ChemBERTa
- `PySide6` (for inference module imports, not directly used in tests)

## Notes

- ✅ Tests for Python portion of inference (SMILES parsing, tokenization, graph building)
- ℹ️ Tests do NOT require the compiled C library (libhybrid_engine) - those require full inference setup
- ✅ Tests validate error handling and edge cases thoroughly
- ⚠️ Inference tests (full end-to-end with C engine) require the model file at `GUI/assets/hybrid_model.onnx`

## Common Test Failures and Troubleshooting

### Issue: Import errors
**Solution:** Ensure you run from the project root and have activated the virtual environment
```bash
source .venv/bin/activate
```

### Issue: rdkit not installed
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Missing tokenizer model
**Solution:** The tokenizer downloads `seyonec/ChemBERTa-zinc-base-v1` on first use. Ensure internet connection.

## Example Test Output

```
test_simple_methane (__main__.TestGraphTensorBuilding.test_simple_methane)
Test graph building for methane (CH4). ... ok
test_benzene (__main__.TestGraphTensorBuilding.test_benzene)
Test graph building for benzene. ... ok
test_invalid_smiles_raises_error (__main__.TestGraphTensorBuilding.test_invalid_smiles_raises_error)
Test that invalid SMILES raises ValueError. ... ok

Ran 29 tests in 1.345s
OK
```

## To Add More Tests

1. Create a new test class inheriting from `unittest.TestCase`
2. Add test methods starting with `test_`
3. Use assertions like `self.assertEqual()`, `self.assertTrue()`, etc.
4. Tests are auto-discovered by unittest

Example:
```python
class TestNewFeature(unittest.TestCase):
    def test_something(self):
        result = some_function()
        self.assertEqual(result, expected_value)
```
