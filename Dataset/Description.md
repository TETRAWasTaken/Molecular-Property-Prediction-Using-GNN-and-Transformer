# Dataset Description

The Project Uses 3 Differet Datasets; QM7, QM8, QM9. Each Datasets contains different molecules and properties needed for the model to be trained on. 

## QM7

QM7 Dataset is contains Coulomb Matrix, Atomic Charges, Atomization Energy, and Cartesian Coordinates of the Molecules. 

| Features | Dimensions | Additional Info |
| -------- | ---------- | --------------- |
| Coulomb Matrix | 7165 x 23 x 23 | Coulomb Matrix shows the electric replusion between an Atom and every other Atom in the Molecule |
| Nuclear Charges | 7165 x 23 x 1 | Electric charge of each atom in the molecule |
| Coordinates | 7165 x 23 x 3 | Cartesian Coordinates of each Atom inside the molecule |
| Atomization Energies | 7165 x 1 | Atomization energy of each Molecule |
| Smiles | 6834 | SMILES String of each molecule |

## QM8

QM8 Dataset contains 2D molecules with multiple electronic properties computed using different methods.

| Features | Dimensions | Additional Info |
| -------- | ---------- | --------------- |
| smiles | 21786 | SMILES String of each molecule |
| E1-CC2 | 21786 x 1 | Excitation energy (CC2) |
| E2-CC2 | 21786 x 1 | Excitation energy (CC2) |
| f1-CC2 | 21786 x 1 | Oscillator strength (CC2) |
| f2-CC2 | 21786 x 1 | Oscillator strength (CC2) |
| E1-PBE0 | 21786 x 1 | Excitation energy (PBE0) |
| E2-PBE0 | 21786 x 1 | Excitation energy (PBE0) |
| f1-PBE0 | 21786 x 1 | Oscillator strength (PBE0) |
| f2-PBE0 | 21786 x 1 | Oscillator strength (PBE0) |
| E1-PBE0.1 | 21786 x 1 | Excitation energy (PBE0.1) |
| E2-PBE0.1 | 21786 x 1 | Excitation energy (PBE0.1) |
| f1-PBE0.1 | 21786 x 1 | Oscillator strength (PBE0.1) |
| f2-PBE0.1 | 21786 x 1 | Oscillator strength (PBE0.1) |
| E1-CAM | 21786 x 1 | Excitation energy (CAM) |
| E2-CAM | 21786 x 1 | Excitation energy (CAM) |
| f1-CAM | 21786 x 1 | Oscillator strength (CAM) |
| f2-CAM | 21786 x 1 | Oscillator strength (CAM) |

## QM9

QM9 Dataset contains equilibrium geometries and multiple quantum chemical properties.

| Features | Dimensions | Additional Info |
| -------- | ---------- | --------------- |
| mol_id | 133885 | Molecule identifier |
| smiles | 133885 | SMILES String of each molecule |
| A | 133885 x 1 | Rotational constant A |
| B | 133885 x 1 | Rotational constant B |
| C | 133885 x 1 | Rotational constant C |
| mu | 133885 x 1 | Dipole moment |
| alpha | 133885 x 1 | Isotropic polarizability |
| homo | 133885 x 1 | HOMO energy |
| lumo | 133885 x 1 | LUMO energy |
| gap | 133885 x 1 | HOMO-LUMO gap |
| r2 | 133885 x 1 | Electronic spatial extent |
| zpve | 133885 x 1 | Zero-point vibrational energy |
| u0 | 133885 x 1 | Internal energy at 0 K |
| u298 | 133885 x 1 | Internal energy at 298 K |
| h298 | 133885 x 1 | Enthalpy at 298 K |
| g298 | 133885 x 1 | Free energy at 298 K |
| cv | 133885 x 1 | Heat capacity at 298 K |
| u0_atom | 133885 x 1 | Atomization energy at 0 K |
| u298_atom | 133885 x 1 | Atomization energy at 298 K |
| h298_atom | 133885 x 1 | Atomization enthalpy at 298 K |
| g298_atom | 133885 x 1 | Atomization free energy at 298 K |

