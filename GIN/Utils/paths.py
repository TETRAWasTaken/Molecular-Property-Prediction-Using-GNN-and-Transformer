import os 

class Paths:
    """
    A configuration class containing all the Paths to all the required files in the Pipeline
    Making the pipeline more modular
    """
    
    def __init__(self):
        # The base_dir should point to the project root
        # GIN/Utils/paths.py -> GIN/Utils -> GIN -> Project Root
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.dataset_dir = os.path.join(self.base_dir, "Dataset")
        self.output_dir = os.path.join(self.base_dir, "GIN", "outputs")
        
        self.qm8_path = os.path.join(self.dataset_dir, "qm8.csv")
        self.qm9_path = os.path.join(self.dataset_dir, "qm9.csv")
        self.model_path = os.path.join(self.output_dir, "gnn_molecular_model.pth")

    def get_qm8_path(self):
        return self.qm8_path

    def get_qm9_path(self):
        return self.qm9_path

    def get_model_path(self):
        return self.model_path