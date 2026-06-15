#!/usr/bin/env python3
"""Check if the stats were computed on delta or absolute energies"""

import pandas as pd
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load raw data
csv_path = PROJECT_ROOT / "Dataset" / "New_QM9" / "molecule_properties.csv"
df = pd.read_csv(csv_path)

# Load stats
stats_path = PROJECT_ROOT / "models" / "qm9_property_stats.json"
with open(stats_path, 'r') as f:
    stats = json.load(f)

print("=" * 80)
print("STATS vs RAW DATA COMPARISON")
print("=" * 80)

# Check if this is delta-encoded
print("\nFrom qm9_property_stats.json:")
print(f"  'target_transform': '{stats.get('target_transform', 'N/A')}'")
print(f"  'delta_targets': {stats.get('delta_targets', [])}")

print("\nEnergy property stats from JSON:")
for prop in ['u0', 'u298', 'h298', 'g298']:
    prop_stats = stats['properties'][prop]
    print(f"\n  {prop}:")
    print(f"    mean: {prop_stats['mean']:.2f}")
    print(f"    std:  {prop_stats['std']:.2f}")
    print(f"    min:  {prop_stats['min']:.2f}")
    print(f"    max:  {prop_stats['max']:.2f}")

print("\n\nRaw data from CSV (should match if stats are on raw data):")
for prop in ['u0', 'u298', 'h298', 'g298']:
    raw_mean = df[prop].mean()
    raw_std = df[prop].std()
    raw_min = df[prop].min()
    raw_max = df[prop].max()
    
    print(f"\n  {prop}:")
    print(f"    mean: {raw_mean:.2f}")
    print(f"    std:  {raw_std:.2f}")
    print(f"    min:  {raw_min:.2f}")
    print(f"    max:  {raw_max:.2f}")
    
    # Check if they match (allowing for small rounding differences)
    if abs(raw_mean - stats['properties'][prop]['mean']) < 1:
        print(f"    ✓ Stats were computed on RAW data")
    else:
        print(f"    ✗ Stats were computed on TRANSFORMED data (delta)")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

target_transform = stats.get('target_transform')
delta_targets = stats.get('delta_targets', [])

if target_transform == 'qm9_delta_learning' and delta_targets:
    print(f"\nStats were computed on DELTA-ENCODED data for: {delta_targets}")
    print("\nThis means:")
    print("  1. Raw CSV values are absolute Hartree energies")
    print("  2. Stats mean/std are for DELTA energies (after atomic ref subtraction)")
    print("  3. Model is trained to predict DELTA energies directly")
    print("  4. No need to subtract atomic refs again - model already does delta learning")
else:
    print(f"\nStats were computed on RAW (non-delta) data")
    print("\nThis means:")
    print("  1. Raw CSV values are absolute Hartree energies")
    print("  2. Stats mean/std are for absolute energies")
    print("  3. Model predicts absolute energies")
    print("  4. Must subtract atomic refs to get delta energies")
