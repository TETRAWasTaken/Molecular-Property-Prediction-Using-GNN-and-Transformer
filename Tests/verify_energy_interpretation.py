#!/usr/bin/env python3
"""Verify the correct interpretation of QM9 energies"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the raw data
csv_path = PROJECT_ROOT / "Dataset" / "New_QM9" / "molecule_properties.csv"
df = pd.read_csv(csv_path)

# Find ethanol
cco = df[df['smiles'] == 'CCO']
if len(cco) == 0:
    # Try to find it by searching
    cco = df[df['smiles'].str.contains('^C(C)O$|^CCO$', regex=True, na=False)]

print("=" * 80)
print("QM9 ENERGY INTERPRETATION")
print("=" * 80)

if len(cco) > 0:
    ethanol = cco.iloc[0]
    print(f"\nMolecule: {ethanol['smiles']}")
    
    print(f"\nRaw values from CSV (in Hartree):")
    for prop in ['u0', 'u298', 'h298', 'g298']:
        val_hartree = ethanol[prop]
        print(f"  {prop}: {val_hartree:.6f} Hartree")
    
    print(f"\nConverted to eV (multiply by 27.2113863):")
    for prop in ['u0', 'u298', 'h298', 'g298']:
        val_hartree = ethanol[prop]
        val_ev = val_hartree * 27.2113863
        print(f"  {prop}: {val_ev:.2f} eV")
    
    print(f"\nConverted to kJ/mol (multiply by 2625.5):")
    for prop in ['u0', 'u298', 'h298', 'g298']:
        val_hartree = ethanol[prop]
        val_kjmol = val_hartree * 2625.5
        print(f"  {prop}: {val_kjmol:.2f} kJ/mol")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print(f"\nThese are ABSOLUTE energies in Hartree (negative for stability)")
    print(f"They are NOT formation enthalpies or atomization energies")
    print(f"They are total electronic energies of the system at B3LYP-6-31G* level")
    print()
    print(f"For ethanol:")
    print(f"  Raw u0:  -154.97 Ha = -4,207,400 kJ/mol (absolute)")
    print(f"  Your ref -277 kJ/mol is formation enthalpy (completely different)")
    print()
    print(f"These energies are MEANT to be used with delta learning:")
    print(f"  - Subtract atomic references to get ATOMIZATION ENERGY")
    print(f"  - This removes the baseline energy of isolated atoms")
else:
    print("\nCould not find ethanol in dataset")

# Check what the delta learning actually produces
print("\n" + "=" * 80)
print("DELTA LEARNING IMPACT")
print("=" * 80)

from Scripts.qm9_delta import get_qm9_atom_reference_sum

smiles = "CCO"
print(f"\nFor ethanol ({smiles}):")

for prop in ['u0', 'u298', 'h298', 'g298']:
    ref_sum_ev = get_qm9_atom_reference_sum(smiles, prop)
    ref_sum_hartree = ref_sum_ev / 27.2113863
    print(f"\n  {prop}:")
    print(f"    Atomic refs (eV):      {ref_sum_ev:.2f}")
    print(f"    Atomic refs (Hartree): {ref_sum_hartree:.6f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
The QM9 dataset provides ABSOLUTE ELECTRONIC ENERGIES in Hartree.

For correct interpretation:
1. These are NOT formation enthalpies (like your -277 kJ/mol reference)
2. These ARE total quantum mechanical energies of the molecule
3. They MUST be converted to atomization energies using delta learning
4. Atomization energies are MUCH larger than formation enthalpies

The current pipeline is CORRECT for atomization energies (~12M kJ/mol range is plausible).

If you need formation enthalpies, you would need:
- Different reference data (not available in QM9)
- OR additional experimental/computational data to calibrate
- OR a completely different dataset
""")
