import os
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importance(importances, feature_names, coin_name, threshold=0):
    # Determine the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the 'feature_importances' directory within 'results' in 'forecasting'
    feature_importances_dir = os.path.join(current_dir, "..", "results", "feature_importances")

    # Ensure the 'feature_importances' directory exists
    if not os.path.exists(feature_importances_dir):
        os.makedirs(feature_importances_dir)

    # Construct the filename with the coin name
    save_path = os.path.join(feature_importances_dir, f"feature_importance_{coin_name}.png")

    # Sort and select significant feature importances
    indices = np.argsort(importances)[::-1]
    significant_indices = [i for i in indices if importances[i] > threshold]
    
    # Plot the feature importances
    plt.figure(figsize=(10, 8))
    plt.title(f"Feature Importances for {coin_name}")
    plt.barh(range(len(significant_indices)), importances[significant_indices], align="center")
    plt.yticks(range(len(significant_indices)), [feature_names[i] for i in significant_indices])
    plt.gca().invert_yaxis()
    plt.xlabel('Feature Importance')
    
    # Save the figure
    plt.savefig(save_path, format='png', bbox_inches='tight')
    plt.show()