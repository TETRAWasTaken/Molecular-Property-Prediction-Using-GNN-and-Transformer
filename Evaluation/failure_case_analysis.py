import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from collections import Counter

def analyze_failure_cases(property_index: int, property_name: str, top_percent: float = 1.0):
    """
    Identifies the top percentage of worst-performing molecules for a given property
    and analyzes their chemical features.
    """
    base_dir = Path(__file__).resolve().parent
    predictions_path = base_dir / "Hybrid" / "predictions.npz"

    if not predictions_path.exists():
        print(f"Predictions for Hybrid model not found at {predictions_path}.")
        return

    data = np.load(predictions_path, allow_pickle=True)
    y_pred = data["y_pred"]
    y_true = data["y_true"]
    smiles_list = data["smiles"]

    if property_index >= y_pred.shape[1]:
        print(f"Property index {property_index} is out of bounds.")
        return

    # Calculate absolute error for the specified property
    absolute_error = np.abs(y_true[:, property_index] - y_pred[:, property_index])
    
    # Get the indices of the top N% of errors
    n_worst = int(len(absolute_error) * (top_percent / 100.0))
    worst_indices = np.argsort(absolute_error)[-n_worst:]

    print(f"--- Failure Case Analysis for '{property_name}' (Top {top_percent}%) ---")
    print(f"Identified {len(worst_indices)} molecules with the highest prediction error.\n")

    worst_smiles = []
    print("Top Failing Molecules (SMILES and Error):")
    for idx in reversed(worst_indices):  # Print from worst to least-worst
        smiles = smiles_list[idx]
        error = absolute_error[idx]
        worst_smiles.append(smiles)
        print(f"  - {smiles}: Error = {error:.4f} eV")

    # --- Chemical Feature Analysis ---
    ring_counts = Counter()
    functional_group_counts = Counter()

    # Define some simple functional group SMARTS patterns
    functional_groups = {
        "ester": "[#6&!$(C=O)]-O-C=O",
        "amide": "C(=O)-N",
        "ether": "[OD2]([#6;!$(C=O)])[#6;!$(C=O)]",
        "ketone": "[#6][C](=O)[#6]",
        "aldehyde": "[CX3H1](=O)[#6]",
        "carboxylic_acid": "C(=O)[O;H1]",
        "nitro": "N(=O)=O",
    }

    for smiles in worst_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue

        # Analyze ring systems
        sssr = Chem.GetSymmSSSR(mol)
        for ring in sssr:
            ring_size = len(ring)
            ring_counts[ring_size] += 1
            if ring_size <= 4: # Specifically note small rings
                ring_counts[f"strained_ring_{ring_size}"] += 1


        # Analyze functional groups
        for name, smarts in functional_groups.items():
            pattern = Chem.MolFromSmarts(smarts)
            if mol.HasSubstructMatch(pattern):
                functional_group_counts[name] += 1

    print("\n--- Chemical Feature Summary of Failure Cases ---")
    
    print("\nRing Systems Found:")
    if not ring_counts:
        print("  - No notable ring systems found.")
    else:
        for ring_size, count in ring_counts.most_common():
            print(f"  - {ring_size}-membered rings: {count} occurrences")

    print("\nFunctional Groups Found:")
    if not functional_group_counts:
        print("  - No common functional groups found.")
    else:
        for group, count in functional_group_counts.most_common():
            print(f"  - {group}: {count} occurrences")
            
    print("\n--- End of Analysis ---")


if __name__ == "__main__":
    # --- Configuration ---
    # You can change this to plot the error for a different property.
    # The index corresponds to the order in TARGET_COLS.
    # 0: mu, 1: alpha, 2: homo, 3: lumo, 4: gap, 5: r2, 6: zpve, 
    # 7: u0, 8: u298, 9: h298, 10: g298, 11: cv
    PROPERTY_NAMES = {
        0: "mu",
        1: "alpha",
        2: "homo",
        3: "lumo",
        4: "gap",
        5: "r2",
        6: "zpve",
        7: "u0",
        8: "u298",
        9: "h298",
        10: "g298",
        11: "cv"
    }

    for i, name in PROPERTY_NAMES.items():
        analyze_failure_cases(i, name)