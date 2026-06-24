from __future__ import annotations

from typing import Iterable

import pandas as pd
from rdkit import Chem


# Unit conversion constants
HARTREE_TO_EV = 27.2113863  # 1 Hartree = 27.2113863 eV
EV_TO_KJMOL = 96.485333  # 1 eV = 96.485333 kJ/mol
HARTREE_TO_KJMOL = HARTREE_TO_EV * EV_TO_KJMOL  # 1 Hartree = 2625.5 kJ/mol

QM9_TARGET_COLUMNS = [
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

QM9_DELTA_TARGET_COLUMNS = ["u0", "u298", "h298", "g298"]

_QM9_ATOM_REFERENCE_VALUES = {
    "u0": {
        1: -13.61312172,
        6: -1029.86312267,
        7: -1485.30251237,
        8: -2042.61123593,
        9: -2713.48485589,
    },
    "u298": {
        1: -13.54887564,
        6: -1029.79887659,
        7: -1485.23829350,
        8: -2042.54701705,
        9: -2713.42063702,
    },
    "h298": {
        1: -13.54887564,
        6: -1029.79887659,
        7: -1485.23829350,
        8: -2042.54701705,
        9: -2713.42063702,
    },
    "g298": {
        1: -13.90303183,
        6: -1030.25891228,
        7: -1485.71166277,
        8: -2043.01812778,
        9: -2713.88796536,
    },
}

_QM9_ATOMIC_SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
}


def convert_hartree_to_kjmol(value_hartree: float) -> float:
    """Convert energy from Hartree to kJ/mol.
    
    Args:
        value_hartree: Energy value in Hartree atomic units
        
    Returns:
        Energy value in kJ/mol
    """
    return float(value_hartree) * HARTREE_TO_KJMOL


def convert_ev_to_kjmol(value_ev: float) -> float:
    """Convert energy from eV to kJ/mol.
    
    Args:
        value_ev: Energy value in electron volts
        
    Returns:
        Energy value in kJ/mol
    """
    return float(value_ev) * EV_TO_KJMOL


def convert_delta_ev_to_kjmol(
    delta_energy_ev: float,
) -> float:
    """Convert delta energy in eV to delta energy in kJ/mol.
    
    This is used when the model has already been trained with delta learning,
    so the predictions are already DELTA energies (relative to atomic references).
    We only need to convert the unit from eV to kJ/mol.
    
    Args:
        delta_energy_ev: Delta energy in eV (already adjusted for atomic references)
        
    Returns:
        Delta energy in kJ/mol
    """
    return float(delta_energy_ev) * EV_TO_KJMOL



def _parse_smiles(smiles: str):
    smiles = (smiles or "").strip()
    if not smiles:
        raise ValueError("SMILES is empty")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return Chem.AddHs(mol)


def get_qm9_atom_reference_sum(smiles: str, property_name: str) -> float:
    reference_values = _QM9_ATOM_REFERENCE_VALUES.get(property_name)
    if reference_values is None:
        raise KeyError(f"Unsupported QM9 delta target: {property_name}")

    return get_qm9_atom_reference_totals(smiles).get(property_name, 0.0)


def get_qm9_atom_reference_totals(smiles: str) -> dict[str, float]:
    mol = _parse_smiles(smiles)
    totals = {property_name: 0.0 for property_name in _QM9_ATOM_REFERENCE_VALUES}

    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        for property_name, reference_values in _QM9_ATOM_REFERENCE_VALUES.items():
            totals[property_name] += float(reference_values.get(atomic_number, 0.0))

    return totals


def apply_qm9_delta_learning(
    dataframe: pd.DataFrame,
    smiles_col: str = "smiles",
    target_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Subtract per-atom reference energies from delta-learning target columns.

    The QM9 CSV stores u0/u298/h298/g298 in Hartree.  The atomic reference
    energies (``_QM9_ATOM_REFERENCE_VALUES``) are in eV.  This function first
    converts those columns from Hartree to eV, then subtracts the reference
    sum so the network only needs to learn the residual (delta) energy.
    All other target columns are left unchanged.
    """
    target_cols = list(target_cols or QM9_TARGET_COLUMNS)
    delta_cols = [col for col in target_cols if col in QM9_DELTA_TARGET_COLUMNS]

    if not delta_cols:
        return dataframe.copy()

    if smiles_col not in dataframe.columns:
        raise ValueError(f"Dataset must contain a '{smiles_col}' column for delta learning.")

    result = dataframe.copy()

    # Step 1: Convert Hartree → eV for the energy columns so units match the
    # atomic reference values.
    for col in delta_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce") * HARTREE_TO_EV

    # Step 2: Subtract the per-molecule atomic reference sum (in eV).
    reference_totals = result[smiles_col].apply(get_qm9_atom_reference_totals)
    for col in delta_cols:
        reference_sum = reference_totals.apply(lambda totals: float(totals.get(col, 0.0)))
        result[col] = result[col] - reference_sum

    return result


def build_qm9_atom_reference_payload() -> dict:
    return {
        "source": "torch_geometric.datasets.qm9.atomrefs",
        "unit": "eV",
        "atomic_numbers": [1, 6, 7, 8, 9],
        "atomic_symbols": [_QM9_ATOMIC_SYMBOLS[z] for z in [1, 6, 7, 8, 9]],
        "by_property": {
            property_name: {
                _QM9_ATOMIC_SYMBOLS[atomic_number]: float(value)
                for atomic_number, value in values.items()
            }
            for property_name, values in _QM9_ATOM_REFERENCE_VALUES.items()
        },
    }


def add_qm9_atom_reference_correction(
    values: list[float],
    smiles: str,
    property_names: Iterable[str],
    atom_reference_payload: dict | None = None,
) -> list[float]:
    adjusted = list(values)
    property_names = list(property_names)
    payload = atom_reference_payload or build_qm9_atom_reference_payload()
    by_property = payload.get("by_property") if isinstance(payload, dict) else None

    if not isinstance(by_property, dict):
        by_property = build_qm9_atom_reference_payload()["by_property"]

    mol = _parse_smiles(smiles)
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    for index, property_name in enumerate(property_names):
        if index >= len(adjusted):
            break

        property_refs = by_property.get(property_name)
        if not isinstance(property_refs, dict):
            continue

        correction = 0.0
        for atomic_number in atomic_numbers:
            symbol = _QM9_ATOMIC_SYMBOLS.get(atomic_number)
            if symbol is None:
                continue
            try:
                correction += float(property_refs.get(symbol, 0.0))
            except (TypeError, ValueError):
                continue

        adjusted[index] = float(adjusted[index]) + correction

    return adjusted