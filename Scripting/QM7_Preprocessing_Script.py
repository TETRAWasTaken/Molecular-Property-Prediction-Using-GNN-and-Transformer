"""
QM7 Dataset Preprocessing Script
=================================
This script loads the QM7 dataset from a .mat file, processes the molecular features,
adds SMILES representations, and saves the final dataset as a CSV file.

Features extracted:
- Coulomb Matrix: 3D representation of molecular structure (23x23 for each molecule)
- Nuclear Charges: Atomic charges for each atom in the molecule
- Coordinates: 3D Cartesian coordinates of each atom
- Atomization Energies: Target variable for prediction
- SMILES: Chemical structure representation strings
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import warnings
from pathlib import Path
from typing import Dict, Tuple
import logging

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QM7DataPreprocessor:
    """
    A class to preprocess the QM7 molecular property dataset.
    
    Attributes:
        mat_file_path (str): Path to the QM7 .mat file
        smiles_url (str): URL to download SMILES data
        output_path (str): Path where the processed CSV will be saved
        data (pd.DataFrame): The processed dataset
    """
    
    def __init__(
        self, 
        mat_file_path: str = "../Dataset/qm7.mat",
        smiles_url: str = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv",
        output_path: str = "../Dataset/qm7_processed.csv"
    ):
        """
        Initialize the QM7 preprocessor.
        
        Args:
            mat_file_path: Path to the QM7 .mat file
            smiles_url: URL to download SMILES representations
            output_path: Path to save the processed dataset
        """
        self.mat_file_path = Path(mat_file_path)
        self.smiles_url = smiles_url
        self.output_path = Path(output_path)
        self.qm7_raw = None
        self.data = None
        
        logger.info("QM7DataPreprocessor initialized")
    
    def load_mat_file(self) -> Dict:
        """
        Load the QM7 .mat file.
        
        Returns:
            Dictionary containing the raw QM7 data
            
        Raises:
            FileNotFoundError: If the .mat file doesn't exist
        """
        if not self.mat_file_path.exists():
            raise FileNotFoundError(f"QM7 .mat file not found at {self.mat_file_path}")
        
        logger.info(f"Loading QM7 data from {self.mat_file_path}")
        self.qm7_raw = sio.loadmat(str(self.mat_file_path))
        logger.info(f"Loaded data with keys: {list(self.qm7_raw.keys())}")
        
        return self.qm7_raw
    
    def extract_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract and reshape features from the raw QM7 data.
        
        Returns:
            Tuple containing (Coulomb Matrix, Nuclear Charges, Coordinates, Atomization Energies)
        """
        if self.qm7_raw is None:
            raise ValueError("QM7 data not loaded. Call load_mat_file() first.")
        
        logger.info("Extracting features from QM7 data")
        
        # Extract and reshape features
        # X: Coulomb Matrix [7165 x 23 x 23]
        coulomb_matrix = self.qm7_raw['X'].reshape(7165, 23, 23)
        
        # Z: Nuclear Charges [7165 x 23]
        nuclear_charges = self.qm7_raw['Z'].reshape(7165, 23)
        
        # R: Cartesian Coordinates [7165 x 23 x 3]
        coordinates = self.qm7_raw['R'].reshape(7165, 23, 3)
        
        # T: Atomization Energies [7165 x 1]
        atomization_energies = self.qm7_raw['T'].reshape(7165, 1)
        
        logger.info(f"Extracted features - CM: {coulomb_matrix.shape}, "
                   f"Q: {nuclear_charges.shape}, "
                   f"COOR: {coordinates.shape}, "
                   f"AE: {atomization_energies.shape}")
        
        return coulomb_matrix, nuclear_charges, coordinates, atomization_energies
    
    def create_dataframe(
        self, 
        coulomb_matrix: np.ndarray,
        nuclear_charges: np.ndarray,
        coordinates: np.ndarray,
        atomization_energies: np.ndarray
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the extracted features.
        
        Args:
            coulomb_matrix: Coulomb matrix arrays
            nuclear_charges: Nuclear charge arrays
            coordinates: Coordinate arrays
            atomization_energies: Atomization energy values
            
        Returns:
            DataFrame containing all features
        """
        logger.info("Creating DataFrame from features")
        
        # Convert arrays to lists to store as object columns in DataFrame
        self.data = pd.DataFrame({
            'Coulomb Matrix': list(coulomb_matrix),
            'Nuclear Charges': list(nuclear_charges),
            'Coordinates': list(coordinates),
            'Atomization Energies': list(atomization_energies)
        })
        
        logger.info(f"DataFrame created with shape: {self.data.shape}")
        return self.data
    
    def add_smiles(self) -> pd.DataFrame:
        """
        Download and add SMILES representations to the dataset.
        
        Returns:
            DataFrame with SMILES column added
            
        Raises:
            Exception: If SMILES data cannot be downloaded
        """
        logger.info(f"Downloading SMILES data from {self.smiles_url}")
        
        try:
            smiles_df = pd.read_csv(self.smiles_url)
            logger.info(f"SMILES data downloaded - Shape: {smiles_df.shape}, Columns: {list(smiles_df.columns)}")
            
            if 'smiles' not in smiles_df.columns:
                raise ValueError("SMILES column not found in downloaded data")
            
            # Add SMILES to the main dataframe
            self.data['smiles'] = smiles_df['smiles']
            logger.info("SMILES column added to dataset")
            
            # Check for null values
            null_counts = self.data.isnull().sum()
            if null_counts.sum() > 0:
                logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
            else:
                logger.info("No null values found in the dataset")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Failed to download SMILES data: {str(e)}")
            raise
    
    def save_to_csv(self) -> None:
        """
        Save the processed dataset to a CSV file.
        Note: Multi-dimensional arrays (Coulomb Matrix, Coordinates, etc.) 
        will be saved as string representations.
        """
        if self.data is None:
            raise ValueError("No data to save. Run the preprocessing pipeline first.")
        
        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving processed dataset to {self.output_path}")
        
        # Convert list columns to string representation for CSV compatibility
        data_to_save = self.data.copy()
        for col in ['Coulomb Matrix', 'Nuclear Charges', 'Coordinates']:
            data_to_save[col] = data_to_save[col].apply(lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x))
        
        # Flatten Atomization Energies if needed
        data_to_save['Atomization Energies'] = data_to_save['Atomization Energies'].apply(
            lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x
        )
        
        # Save to CSV
        data_to_save.to_csv(self.output_path, index=False)
        logger.info(f"Dataset successfully saved! Shape: {data_to_save.shape}")
        logger.info(f"File location: {self.output_path.absolute()}")
    
    def get_dataset_info(self) -> Dict:
        """
        Get summary information about the processed dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self.data is None:
            raise ValueError("No data available. Run the preprocessing pipeline first.")
        
        info = {
            'total_molecules': len(self.data),
            'features': list(self.data.columns),
            'null_values': self.data.isnull().sum().to_dict(),
            'atomization_energy_stats': {
                'min': float(np.min([x[0] if isinstance(x, (list, np.ndarray)) else x 
                                     for x in self.data['Atomization Energies']])),
                'max': float(np.max([x[0] if isinstance(x, (list, np.ndarray)) else x 
                                     for x in self.data['Atomization Energies']])),
                'mean': float(np.mean([x[0] if isinstance(x, (list, np.ndarray)) else x 
                                      for x in self.data['Atomization Energies']])),
            }
        }
        
        return info
    
    def preprocess(self) -> pd.DataFrame:
        """
        Execute the complete preprocessing pipeline.
        
        Returns:
            The fully processed DataFrame
        """
        logger.info("=" * 60)
        logger.info("Starting QM7 Dataset Preprocessing Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load the .mat file
        self.load_mat_file()
        
        # Step 2: Extract features
        cm, q, coor, ae = self.extract_features()
        
        # Step 3: Create DataFrame
        self.create_dataframe(cm, q, coor, ae)
        
        # Step 4: Add SMILES
        self.add_smiles()
        
        # Step 5: Save to CSV
        self.save_to_csv()
        
        # Step 6: Display summary
        info = self.get_dataset_info()
        logger.info("=" * 60)
        logger.info("Preprocessing Complete!")
        logger.info(f"Total Molecules: {info['total_molecules']}")
        logger.info(f"Features: {', '.join(info['features'])}")
        logger.info(f"Atomization Energy Range: {info['atomization_energy_stats']['min']:.2f} to {info['atomization_energy_stats']['max']:.2f}")
        logger.info("=" * 60)
        
        return self.data


def main():
    """Main execution function."""
    # Initialize preprocessor
    preprocessor = QM7DataPreprocessor(
        mat_file_path="../Dataset/qm7.mat",
        smiles_url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv",
        output_path="../Dataset/qm7_processed.csv"
    )
    
    # Run preprocessing pipeline
    processed_data = preprocessor.preprocess()
    
    print("\n QM7 dataset preprocessing completed successfully!")
    print(f" Dataset shape: {processed_data.shape}")
    print(f" Saved to: {preprocessor.output_path.absolute()}")


if __name__ == "__main__":
    main()
