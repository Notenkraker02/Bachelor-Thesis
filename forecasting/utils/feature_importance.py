import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importance(importances, feature_names, threshold=0):
    indices = np.argsort(importances)[::-1]
    significant_indices = [i for i in indices if importances[i] > threshold]
    
    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances")
    plt.barh(range(len(significant_indices)), importances[significant_indices], align="center")
    plt.yticks(range(len(significant_indices)), [feature_names[i] for i in significant_indices])
    plt.gca().invert_yaxis()
    plt.xlabel('Feature Importance')
    plt.show() 