import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from typing import List
from joblib import Parallel, delayed

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.environ.get(
    "SM_CHANNEL_TRAINING",
    os.path.join(script_dir, "../Dataset/133660_curatedQM9_outof_133885"),
)
all_files = glob.glob(os.path.join(path, "*.xyz"))


# QM9 extended XYZ property order.
QM9_PROPERTY_COLUMNS = [
    "tag",
    "index",
    "A",
    "B",
    "C",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u298",
    "h298",
    "g298",
    "cv",
]


def _safe_to_number(value: str):
    normalized = value.replace("D", "E").replace("E^", "E").replace("e^", "e")
    try:
        if normalized.isdigit() or (normalized.startswith("-") and normalized[1:].isdigit()):
            return int(normalized)
        return float(normalized)
    except (TypeError, ValueError):
        return value


def _safe_to_float(value: str) -> float:
    normalized = value.replace("D", "E").replace("E^", "E").replace("e^", "e")
    return float(normalized)


def _property_columns_for(values: List[str]) -> List[str]:
    if len(values) <= len(QM9_PROPERTY_COLUMNS):
        return QM9_PROPERTY_COLUMNS[: len(values)]

    overflow = len(values) - len(QM9_PROPERTY_COLUMNS)
    return QM9_PROPERTY_COLUMNS + [f"property_{idx}" for idx in range(overflow)]


def read_extended_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_atoms = int(lines[0].strip())
    properties = lines[1].strip().split()
    property_columns = _property_columns_for(properties)
    property_data = {
        col: _safe_to_number(value)
        for col, value in zip(property_columns, properties)
    }

    atom_lines = lines[2: 2 + num_atoms]
    molecule_id = Path(file_path).stem
    atom_rows = []
    for atom_index, line in enumerate(atom_lines):
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        atom_rows.append(
            {
                "molecule_id": molecule_id,
                "atom_index": atom_index,
                "atom": parts[0],
                "x": _safe_to_float(parts[1]),
                "y": _safe_to_float(parts[2]),
                "z": _safe_to_float(parts[3]),
                "charge": _safe_to_float(parts[4]) if len(parts) > 4 else np.nan,
            }
        )
    
    current_line = 2 + num_atoms

    smiles = lines[current_line + 1].strip().split()[0]

    molecule_data = {
        "molecule_id": molecule_id,
        "source_file": os.path.basename(file_path),
        "num_atoms": num_atoms,
        "smiles": smiles,
        **property_data,
    }

    return molecule_data, atom_rows


def load_all_molecules(file_paths):
    results = Parallel(n_jobs=-1)(delayed(read_extended_xyz)(fp) for fp in file_paths)
    return results


def bundle_dataset(file_paths):
    parsed = load_all_molecules(file_paths)
    molecule_rows = [item[0] for item in parsed]
    atom_rows = [atom for _, atoms in parsed for atom in atoms]

    molecules_df = pd.DataFrame(molecule_rows)
    atoms_df = pd.DataFrame(atom_rows)
    return molecules_df, atoms_df


if __name__ == "__main__":
    molecules_df, atoms_df = bundle_dataset(all_files)

    merged_df = atoms_df.merge(molecules_df, on="molecule_id", how="left", validate="many_to_one")

    molecule_output = os.environ.get("QM9_MOLECULE_CSV", os.path.join(script_dir, "new_qm9_molecules.csv"))
    atom_output = os.environ.get("QM9_ATOM_CSV", os.path.join(script_dir, "new_qm9_atoms.csv"))
    merged_output = os.environ.get("QM9_MERGED_CSV", os.path.join(script_dir, "new_qm9.csv"))

    molecules_df.to_csv(molecule_output, index=False)
    atoms_df.to_csv(atom_output, index=False)
    merged_df.to_csv(merged_output, index=False)

    print(f"Saved molecule-level CSV: {molecule_output} ({len(molecules_df)} rows)")
    print(f"Saved atom-level CSV: {atom_output} ({len(atoms_df)} rows)")
    print(f"Saved merged CSV: {merged_output} ({len(merged_df)} rows)")