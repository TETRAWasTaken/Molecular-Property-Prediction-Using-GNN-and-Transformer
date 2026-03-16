import os

class Paths:
    """
    Configuration class managing all directory and file paths for the Hybrid Pipeline.
    Ensures the Hybrid model artifacts are isolated from standalone GIN/Transformer outputs.
    """
    def __init__(self):
        # Hybrid/Utils/paths.py -> Hybrid/Utils -> Hybrid -> Project Root
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Shared Dataset Directory (same as before)
        self.dataset_dir = os.path.join(self.base_dir, "Dataset")
        self.qm8_path = os.path.join(self.dataset_dir, "qm8.csv")
        self.qm9_path = os.path.join(self.dataset_dir, "qm9.csv")
        
        # Hybrid-Specific Output Directories
        self.output_dir = os.path.join(self.base_dir, "Hybrid", "outputs")
        self.cache_dir = os.path.join(self.output_dir, "cache")
        self.model_path = os.path.join(self.output_dir, "hybrid_molecular_model.pth")
        
        # Ensure output directories exist
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_qm8_path(self) -> str:
        return self.qm8_path

    def get_qm9_path(self) -> str:
        return self.qm9_path

    def get_model_path(self) -> str:
        return self.model_path
        
    def get_cache_path(self, filename: str = "hybrid_graphs.pt") -> str:
        return os.path.join(self.cache_dir, filename)