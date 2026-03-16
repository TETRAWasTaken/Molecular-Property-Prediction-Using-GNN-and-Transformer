import pandas as pd 
import glob
import sys
import os
from typing import Dict, Tuple
from joblib import Parallel, delayed


script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "../../Dataset/PC9_data/XYZ")
all_files = glob.glob(os.path.join(path, "*.xyz"))

def read_extended_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    num_atoms = int(lines[0].strip())
    properties = lines[1].strip().split()
    
    current_line = 2 + num_atoms
    
    frequencies = lines[current_line].strip().split()
    smiles = lines[current_line + 1].strip().split()[0]
    
    molecule_data = {
        'properties': properties,
        'frequencies': [float(f) for f in frequencies],
        'smiles': smiles,
    }
    
    return molecule_data

def extract_data(path: str) -> Tuple[float, float, float, float, str]:
    data = read_extended_xyz(path)
    homo = data.get('properties')[7]
    lumo = data.get('properties')[8]
    gap = data.get('properties')[9]
    e = data.get('properties')[12]
    smiles = data.get('smiles')
    
    return homo, lumo, gap, e, smiles

def load_all_molecules(file_paths):
    results = Parallel(n_jobs=-1)(delayed(extract_data)(fp) for fp in file_paths)
    return results

def bundle_dataset(file_paths):
    molecules = load_all_molecules(file_paths)
    df = pd.DataFrame(molecules, columns=['homo', 'lumo', 'gap', 'e', 'smiles'])
    return df

if __name__ == "__main__":
    data = bundle_dataset(all_files)
    print(data)