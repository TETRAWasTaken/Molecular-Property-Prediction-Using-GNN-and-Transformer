import numpy as np
import pandas as pd
import scipy.io as sio

class QM7_Preprocessing:
    def __init__(self):
        self.data = sio.loadmat('../Dataset/qm7.mat')

    def Extract_Data(self):
        self.CM = self.data.get('X', [])  # Coulomb Matrix
        self.Q = pd.DataFrame(self.data['Z'])  # Nuclear Charge
        self.COOR = self.data.get('R', [])  # Coordinates
        self.AET = pd.DataFrame(self.data['T'])  # Atomization Energies
        print("Extracted Features from the Dataset")

    def Compile_Data(self):
        Data = pd.DataFrame({
            'Coulomb Matrix': list(self.CM),
            'Nuclear Charge': list(self.Q),
            'Coordinates': list(self.COOR),
            'Atomization Energies': list(self.AET)
        })
        return Data

    def Save_Data(self, path):
        self.Compile_Data().to_csv(path)

    def Load_Smiles(self):
        self.SMILES = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv')
        print("Smiles Loaded from URL")
