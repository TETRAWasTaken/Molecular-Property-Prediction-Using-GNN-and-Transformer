import os
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path to allow execution from any folder
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from the ONNX inference engine
from GUI.core.inference import (
    init_hybrid_engine,
    run_hybrid_regression_with_confidence,
    _descale_prediction_values,
    _PROPERTY_NAMES as TARGET_COLS,
)

def test_single_molecule(smiles: str):
    """
    Runs a test inference on a single molecule and prints the result.
    """
    print(f"--- Testing Single Molecule: {smiles} ---")
    
    try:
        # Initialize the engine. The model path will be resolved automatically.
        print("Initializing hybrid inference engine...")
        init_hybrid_engine()
        print("Engine initialized successfully.")

        # Run inference with confidence calculation (1 conformer for speed)
        print("Running inference...")
        result = run_hybrid_regression_with_confidence(
            smiles,
            n_conformers=1,
            apply_descaling=True
        )
        
        prediction = result['prediction']
        
        print("\n--- Test Result ---")
        if np.isnan(prediction).any():
            print("ERROR: Prediction contains NaN values.")
        else:
            print("SUCCESS: Inference completed without NaN values.")
        
        print("\nPredicted Properties (descaled):")
        for name, value in zip(TARGET_COLS, prediction):
            print(f"  - {name}: {value:.6f}")
            
    except Exception as e:
        print("\n--- AN ERROR OCCURRED ---")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Using a simple, common molecule for the test
    test_smiles = "CCO"  # Ethanol
    test_single_molecule(test_smiles)