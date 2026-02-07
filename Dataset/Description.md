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

