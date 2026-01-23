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

def plot_comparative_trust(vehicles, save_path="results/comparative_trust_evolution.png"):
    """
    Plots the average trust evolution for:
    1. Swing Attacker
    2. Normal Vehicle
    3. Malicious Vehicle
    Matches the style of Fig 8.
    """
    import numpy as np

    honest_hist = []
    malicious_hist = []
    swing_hist = []

    for v in vehicles.values():
        if v.behavior_type == 'HONEST':
            honest_hist.append(v.trust_history)
        elif v.behavior_type == 'MALICIOUS':
            malicious_hist.append(v.trust_history)
        elif v.behavior_type == 'SWING':
            swing_hist.append(v.trust_history)

    # Calculate averages
    # Assuming all histories are same length
    t = range(len(list(vehicles.values())[0].trust_history))
    
    plt.figure(figsize=(10, 6))
    
    if swing_hist:
        avg_swing = np.mean(swing_hist, axis=0)
        plt.plot(t, avg_swing, 'o-', label='Swing attack', color='blue', markevery=5)
        
    if honest_hist:
        avg_honest = np.mean(honest_hist, axis=0)
        plt.plot(t, avg_honest, '+-', label='A normal vehicle', color='lime', markevery=5)
        
    if malicious_hist:
        avg_malicious = np.mean(malicious_hist, axis=0)
        plt.plot(t, avg_malicious, '*-', label='A malicious vehicle', color='red', markevery=5)
        
    # Threshold line
    plt.axhline(y=0.45, color='magenta', linestyle='--', label='Trust threshold')

    plt.title("Change of vehicle global trust values in different situations")
    plt.xlabel("Time (steps)")
    plt.ylabel("Global trust value")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_final_trust_distribution(vehicles, save_path="results/final_trust_malicious.png"):
    """
    Plots the final global trust values of up to 10 malicious vehicles.
    Matches the style of Fig 9 (but only for our Proposed model).
    """
    import numpy as np
    
    malicious_nodes = [v for v in vehicles.values() if v.behavior_type == 'MALICIOUS']
    
    # Take first 10
    subset = malicious_nodes[:10]
    
    if not subset:
        return

    ids = [i+1 for i in range(len(subset))]
    scores = [v.global_trust_score for v in subset]
    
    plt.figure(figsize=(10, 6))
    
    # Bar width
    width = 0.4
    
    plt.bar(ids, scores, width, label='Our proposed', color='deepskyblue', edgecolor='black')
    
    # Reference line average
    avg_val = np.mean(scores)
    plt.axhline(y=avg_val, color='magenta', linestyle='--', label=f'Average value ({avg_val:.2f})')
    
    # Threshold line
    plt.axhline(y=0.45, color='black', linestyle='--', label='Trust threshold')
    
    plt.xlabel("Malicious vehicles")
    plt.ylabel("Global trust value")
    plt.title("Trust values of malicious vehicles")
    plt.xticks(ids)
    plt.legend()
    plt.grid(True, axis='y')
    plt.ylim(0, 1.0)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_swing_analysis(global_history, local_history, save_path="results/swing_analysis.png"):
    """
    Plots the specific trust evolution of a Swing Attacker.
    Compares:
    1. Global Trust (Aggregated by RSU with forgetting factor)
    2. Local Trust (Perceived by a single observer)
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(global_history, label='Global Trust (RSU)', color='purple', linewidth=2)
    plt.plot(local_history, label='Local Trust (Observer)', color='orange', linestyle='--', alpha=0.8)
    
    plt.title("Swing Attacker Analysis: Local vs Global Trust")
    plt.xlabel("Simulation Step")
    plt.ylabel("Trust Score")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()
