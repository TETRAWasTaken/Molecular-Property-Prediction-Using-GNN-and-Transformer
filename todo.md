# TODO list

## Graph Neural Networks
- [ ] Research RDKIT, for SMILES to graph conversion

## Transformers
- [ ] Research BERT, for sentence embedding
- [ ] Research SMILES data conversion for BERT

## Links
- https://quantum-machine.org/datasets/
- https://www.biorxiv.org/content/10.1101/2024.10.31.621293v2.abstract
- https://moleculenet.org/datasets-1
- https://www.kaggle.com/code/mjmurphy28/predicting-atomization-energy-qm7

## Dataset EDA

### QM7 ( Anshumaan )
* Contains a X feature with **Coulomb Matrix** of each Molecule [ 7165 x 23 x 23 ]
* Contains a Z feature with **Charge** of each Atom in Molecule [ 7165 x 23 x 1 ]
* Contains a R feature with **Cartesian Coordinates** of each Atom in Molecule [ 7165 x 23 x 3 ]
* Contains a T feature with **Atomisation Energy** of each Molecule [ 7165 x 1 ]
* Contains a P subset which hold the validation set for the dataset. 


### QM8 ( Anurag )



### QM9 ( Wani and Diya )
#### TARGET FEATURE DESCRIPTIONS:
* QM8 (Electronic Properties / Excited States)
    * E1-CC2:   First electronic excitation energy (how much energy to move an electron to its first 'jump').
    * E2-CC2:   Second electronic excitation energy (energy for the second-level 'jump').
    * f1-CC2:   Oscillator strength of the first jump (how likely/bright the transition is).
    * f2-CC2:   Oscillator strength of the second jump (intensity of the second transition).
    
    *Note: 'CC2' refers to the high-level quantum chemistry method used for these calculations.*

* QM9 (Geometric & Thermodynamic Properties) 
    * A, B, C:  Rotational constants. They define how the molecule's mass is distributed and how it spins.
    * mu:       Dipole moment. Measures the 'magnet-like' polarity of the molecule.
    * alpha:    Isotropic polarizability. How easily the electron cloud can be 'squished' by an external field.
    * homo:     Highest Occupied Molecular Orbital. The energy of the outermost 'full' electron layer.
    * lumo:     Lowest Unoccupied Molecular Orbital. The energy of the first available 'empty' electron layer.
    * gap:      The Energy Gap (LUMO minus HOMO). Determines if the molecule is a conductor or insulator.

