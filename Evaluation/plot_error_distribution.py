import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_all_properties_error_distribution():
    """
    Loads prediction results from GIN, Transformer, and Hybrid models,
    and plots the Kernel Density Estimate (KDE) of their error distributions
    for all 12 properties in a single image with subplots.
    """
    base_dir = Path(__file__).resolve().parent
    model_dirs = {
        "GIN": base_dir / "GIN",
        "Transformer": base_dir / "Transformer",
        "Hybrid": base_dir / "Hybrid",
    }

    # Load predictions for all available models
    model_predictions = {}
    for model_name, model_dir in model_dirs.items():
        predictions_path = model_dir / "predictions.npz"
        if not predictions_path.exists():
            print(f"Predictions for {model_name} not found at {predictions_path}. Skipping.")
            continue

        try:
            data = np.load(predictions_path)
            model_predictions[model_name] = {
                "y_pred": data["y_pred"],
                "y_true": data["y_true"]
            }
            print(f"Loaded predictions for {model_name} (shape: {data['y_pred'].shape})")
        except Exception as e:
            print(f"Error loading predictions for {model_name}: {e}")

    if not model_predictions:
        print("No predictions loaded. Exiting.")
        return

    # Property names and their index/units
    properties = {
        0: ("mu", "Debye"),
        1: ("alpha", "Bohr³"),
        2: ("homo", "eV"),
        3: ("lumo", "eV"),
        4: ("gap", "eV"),
        5: ("r2", "Bohr²"),
        6: ("zpve", "eV"),
        7: ("u0", "eV"),
        8: ("u298", "eV"),
        9: ("h298", "eV"),
        10: ("g298", "eV"),
        11: ("cv", "cal/(mol·K)")
    }

    # Create a 4x3 subplot grid
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    axes = axes.flatten()

    for idx, (name, unit) in properties.items():
        ax = axes[idx]
        
        # Plot KDE for each model's error for this property
        has_plots = False
        all_errors = []
        for model_name, data in model_predictions.items():
            y_pred = data["y_pred"]
            y_true = data["y_true"]

            if idx >= y_pred.shape[1]:
                continue

            error = y_true[:, idx] - y_pred[:, idx]
            sns.kdeplot(error, label=f"{model_name}", fill=True, alpha=0.15, ax=ax)
            all_errors.append(error)
            has_plots = True

        # Shorten x-axis range to 1st to 99th percentile to exclude extreme outliers
        if all_errors:
            combined_errors = np.concatenate(all_errors)
            xlim_min = np.percentile(combined_errors, 1)
            xlim_max = np.percentile(combined_errors, 99)
            padding = (xlim_max - xlim_min) * 0.05
            ax.set_xlim(xlim_min - padding, xlim_max + padding)

        ax.set_title(f'{name.upper()} Error Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Error ({unit})', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        if has_plots:
            ax.legend(fontsize=9)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = base_dir / "error_distribution_kde.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_all_properties_error_distribution()