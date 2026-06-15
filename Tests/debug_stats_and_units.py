#!/usr/bin/env python3
"""Debug script to understand the actual model output and stats."""

from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GUI.core.inference import run_hybrid_regression
from Scripts.qm9_delta import get_qm9_atom_reference_sum


def debug_model_output():
    """Debug what the model actually outputs before any conversions."""
    print("=" * 80)
    print("DEBUGGING MODEL OUTPUT FOR ETHANOL (CCO)")
    print("=" * 80)
    
    smiles = "CCO"
    
    # Get raw model output (scaled, between -1 and 1 typically)
    print("\n1. RAW MODEL OUTPUT (scaled/normalized):")
    raw_output = run_hybrid_regression(smiles)
    print(f"   Raw output shape: {raw_output.shape}")
    print(f"   Values (scaled): {raw_output.tolist()}")
    
    # Load property stats
    stats_path = PROJECT_ROOT / "models" / "qm9_property_stats.json"
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    print("\n2. PROPERTY STATS (mean, std):")
    property_names = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv"]
    
    for idx, prop in enumerate(property_names):
        if idx < len(raw_output):
            prop_stats = stats['properties'][prop]
            mean = prop_stats['mean']
            std = prop_stats['std']
            print(f"   {prop:>6}: mean={mean:>15.2f}, std={std:>10.2f}")
    
    print("\n3. INVERSE SCALED VALUES (raw × std + mean):")
    inverse_scaled = []
    for idx, prop in enumerate(property_names):
        if idx < len(raw_output):
            prop_stats = stats['properties'][prop]
            mean = prop_stats['mean']
            std = prop_stats['std']
            value = float(raw_output[idx]) * std + mean
            inverse_scaled.append(value)
            
            unit_hint = ""
            if prop in ["u0", "u298", "h298", "g298"]:
                unit_hint = " ← ENERGY PROPERTY"
            
            print(f"   {prop:>6}: {value:>15.2f}{unit_hint}")
    
    print("\n4. ATOMIC REFERENCES (eV):")
    for prop in ["u0", "u298", "h298", "g298"]:
        ref_sum = get_qm9_atom_reference_sum(smiles, prop)
        print(f"   {prop}: {ref_sum:>15.2f} eV (sum of atomic refs)")
    
    print("\n5. WHAT ARE THESE STATS REPRESENTING?")
    print(f"   Mean of u0: {stats['properties']['u0']['mean']:.2f}")
    print(f"   Min of u0:  {stats['properties']['u0']['min']:.2f}")
    print(f"   Max of u0:  {stats['properties']['u0']['max']:.2f}")
    print()
    print("   Analysis:")
    print(f"   - If these are Hartree: Range would be ~-1 to +0.5 Ha (reasonable)")
    print(f"   - If these are eV: Range would be ~-27 to +13 eV (too small for absolute energy)")
    print(f"   - If already converted to something else: Need to determine what")
    
    # Let's check if they could be in eV
    print("\n6. CHECKING UNIT HYPOTHESIS:")
    mean_u0 = stats['properties']['u0']['mean']
    print(f"   If mean={mean_u0} is in Hartree:")
    print(f"     = {mean_u0 * 27.2113863:.2f} eV")
    print(f"     = {mean_u0 * 27.2113863 * 96.485333:.2f} kJ/mol")
    
    print(f"\n   If mean={mean_u0} is in eV:")
    print(f"     = {mean_u0 * 96.485333:.2f} kJ/mol")
    
    print(f"\n   If mean={mean_u0} is in kcal/mol:")
    print(f"     = {mean_u0 * 4.184:.2f} kJ/mol")
    
    print("\n" + "=" * 80)
    print("CONCLUSION: What's the correct interpretation?")
    print("=" * 80)
    print(f"Expected ethanol H298: ~-277 kJ/mol (formation enthalpy)")
    print(f"We're getting: ~12,000,000 kJ/mol (way too large)")
    print()
    print("This suggests either:")
    print("1. Stats are NOT in Hartree (check what unit training data was)")
    print("2. These are atomization energies (vastly different from formation)")
    print("3. The model is predicting total energy, not delta energy")
    print("4. There's a fundamental misunderstanding of what the model outputs")


if __name__ == "__main__":
    try:
        debug_model_output()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
