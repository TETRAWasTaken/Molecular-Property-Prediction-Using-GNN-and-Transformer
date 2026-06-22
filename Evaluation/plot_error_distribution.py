import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_error_distribution(property_index: int, property_name: str):
    """
    Loads prediction results from GIN, Transformer, and Hybrid models,
    and plots the Kernel Density Estimate (KDE) of their error distributions.
    """
    base_dir = Path(__file__).resolve().parent
    model_dirs = {
        "GIN": base_dir / "GIN",
        "Transformer": base_dir / "Transformer",
        "Hybrid": base_dir / "Hybrid",
    }

    plt.figure(figsize=(12, 8))
    
    for model_name, model_dir in model_dirs.items():
        predictions_path = model_dir / "predictions.npz"
        if not predictions_path.exists():
            print(f"Predictions for {model_name} not found at {predictions_path}. Skipping.")
            continue

        data = np.load(predictions_path)
        y_pred = data["y_pred"]
        y_true = data["y_true"]

        # Ensure the property index is valid
        if property_index >= y_pred.shape[1]:
            print(f"Property index {property_index} is out of bounds for {model_name} model.")
            continue

        # Calculate the error for the specified property
        error = y_true[:, property_index] - y_pred[:, property_index]
        
        # Plot the KDE for the error
        sns.kdeplot(error, label=f"{model_name} Error", fill=True, alpha=0.2)

    plt.title(f'Error Distribution (KDE) for {property_name}')
    plt.xlabel(f'Error in {property_name} (eV)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Save the plot
    output_path = base_dir / "error_distribution_kde.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    # --- Configuration ---
    # You can change this to plot the error for a different property.
    # The index corresponds to the order in TARGET_COLS.
    # 0: mu, 1: alpha, 2: homo, 3: lumo, 4: gap, 5: r2, 6: zpve, 
    # 7: u0, 8: u298, 9: h298, 10: g298, 11: cv
    PROPERTY_TO_PLOT_INDEX = 7
    PROPERTY_TO_PLOT_NAME = "u0"

    plot_error_distribution(PROPERTY_TO_PLOT_INDEX, PROPERTY_TO_PLOT_NAME)