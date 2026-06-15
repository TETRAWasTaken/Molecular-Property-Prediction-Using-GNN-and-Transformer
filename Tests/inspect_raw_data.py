#!/usr/bin/env python3
"""Inspect the actual raw data in molecule_properties.csv"""

import pandas as pd
from pathlib import Path

csv_path = Path(__file__).parent.parent / "Dataset" / "New_QM9" / "molecule_properties.csv"

# Load the CSV
df = pd.read_csv(csv_path)

print("=" * 100)
print("CSV STRUCTURE")
print("=" * 100)
print(f"\nShape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

print("\n" + "=" * 100)
print("SEARCH FOR ETHANOL (C2H6O / CCO)")
print("=" * 100)

# Look for exact CCO
exact_cco = df[df['smiles'] == 'CCO']
if len(exact_cco) > 0:
    print(f"\nFound {len(exact_cco)} rows with exact SMILES='CCO'")
    print("\nFirst match:")
    for col in df.columns:
        val = exact_cco.iloc[0][col]
        print(f"  {col:20}: {val}")
else:
    print("\nNo exact 'CCO' found. Searching for molecules containing CCO...")
    cco_containing = df[df['smiles'].str.contains('CCO', na=False)]
    print(f"Found {len(cco_containing)} molecules containing 'CCO'")
    
    # Show a few examples
    if len(cco_containing) > 0:
        print("\nFirst few rows with CCO substructure:")
        for idx, (_, row) in enumerate(cco_containing.head(3).iterrows()):
            print(f"\n  Row {idx}:")
            print(f"    smiles: {row['smiles']}")
            print(f"    u0:     {row.get('u0', 'N/A')}")
            print(f"    u298:   {row.get('u298', 'N/A')}")
            print(f"    h298:   {row.get('h298', 'N/A')}")
            print(f"    g298:   {row.get('g298', 'N/A')}")

print("\n" + "=" * 100)
print("SAMPLE OF ENERGY VALUE RANGES")
print("=" * 100)

energy_cols = ['u0', 'u298', 'h298', 'g298']
for col in energy_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  mean:   {df[col].mean():.2f}")
        print(f"  std:    {df[col].std():.2f}")
        print(f"  min:    {df[col].min():.2f}")
        print(f"  max:    {df[col].max():.2f}")
        print(f"  sample: {df[col].iloc[0]:.6f}")

print("\n" + "=" * 100)
print("ENERGY UNIT ANALYSIS")
print("=" * 100)

sample_u0 = df['u0'].iloc[0]
sample_u298 = df['u298'].iloc[0]

print(f"\nFirst row u0 value: {sample_u0:.6f}")
print(f"First row u298 value: {sample_u298:.6f}")
print(f"\nIf these are in Hartree:")
print(f"  u0 in eV:      {sample_u0 * 27.2113863:.2f}")
print(f"  u0 in kJ/mol:  {sample_u0 * 27.2113863 * 96.485333:.2f}")
print(f"\nIf these are already in eV:")
print(f"  u0 in kJ/mol:  {sample_u0 * 96.485333:.2f}")
