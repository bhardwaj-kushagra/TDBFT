"""
Plotting Utilities.
"""
import matplotlib.pyplot as plt
import os

def plot_trust_evolution(vehicles, save_path="results/trust_evolution.png"):
    """
    Plots the history of global trust scores for all vehicles.
    """
    plt.figure(figsize=(10, 6))
    
    for vid, v in vehicles.items():
        color = 'red' if v.is_malicious else 'green'
        alpha = 0.5 if v.is_malicious else 0.3
        # Start at 0.5 default
        plt.plot(v.trust_history, color=color, alpha=alpha, label='_nolegend_')
        
    # Legend hack
    plt.plot([], [], color='red', label='Malicious')
    plt.plot([], [], color='green', label='Honest')
    
    plt.title("Evolution of Global Trust Scores")
    plt.xlabel("Simulation Step")
    plt.ylabel("Global Trust Score")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_detection_metrics(vehicles, save_path="results/detection_metrics.png"):
    """
    Plots Detection Rate (TPR) and False Positive Rate (FPR) vs Trust Threshold.
    """
    import numpy as np
    
    thresholds = np.linspace(0, 1, 100)
    tprs = []
    fprs = []
    
    malicious_nodes = [v for v in vehicles.values() if v.is_malicious] # Includes SWING if they are flagged as malicious in vehicle.py
    honest_nodes = [v for v in vehicles.values() if not v.is_malicious]
    
    total_malicious = len(malicious_nodes)
    total_honest = len(honest_nodes)
    
    if total_malicious == 0 or total_honest == 0:
        print("Skipping detection plot: insufficient node types.")
        return

    for t in thresholds:
        # Detected as malicious if trust < t
        detected_malicious = [v for v in malicious_nodes if v.global_trust_score < t]
        false_positives = [v for v in honest_nodes if v.global_trust_score < t]
        
        tpr = len(detected_malicious) / total_malicious if total_malicious else 0
        fpr = len(false_positives) / total_honest if total_honest else 0
        
        tprs.append(tpr)
        fprs.append(fpr)
        
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, tprs, label='Detection Rate (TPR)', color='blue')
    plt.plot(thresholds, fprs, label='False Positive Rate (FPR)', color='red', linestyle='--')
    
    plt.title("Detection Performance vs Trust Threshold")
    plt.xlabel("Trust Threshold (Nodes < T considered Malicious)")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()
