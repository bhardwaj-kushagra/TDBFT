"""
Plotting Utilities. (Refactored for Statistical Rigor)
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_trust_evolution(vehicles, save_path="results/trust_evolution.png"):
    """
    Plots the history of global trust scores.
    Graph 1 Change: Uses Percentile-based thresholds instead of absolute ones.
    """
    plt.figure(figsize=(10, 6))
    
    final_scores = []
    
    for vid, v in vehicles.items():
        color = 'red' if v.is_malicious else 'green'
        alpha = 0.5 if v.is_malicious else 0.3
        plt.plot(v.trust_history, color=color, alpha=alpha, label='_nolegend_')
        final_scores.append(v.global_trust_score)
        
    # Legend hack
    plt.plot([], [], color='red', label='Malicious')
    plt.plot([], [], color='green', label='Honest')
    
    # Calculate Percentiles from Final Step
    if final_scores:
        p30 = np.percentile(final_scores, 30) # Bottom 30% might be malicious cutoff?
        p_top30 = np.percentile(final_scores, 70) # Top 30% Trusted
        
        # Plot Dynamic Thresholds
        plt.axhline(y=p_top30, color='blue', linestyle='--', label=f'Top 30% (>{p_top30:.4f})')
        plt.axhline(y=p30, color='orange', linestyle='--', label=f'Bottom 30% (<{p30:.4f})')
    
    plt.title("Evolution of Global Trust Scores (Percentile Thresholds)")
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
    Graph 5 Change: Sweeps thresholds over observed range, not [0,1].
    """
    import numpy as np
    
    malicious_nodes = [v for v in vehicles.values() if v.is_malicious]
    honest_nodes = [v for v in vehicles.values() if not v.is_malicious]
    
    all_scores = [v.global_trust_score for v in vehicles.values()]
    min_trust = min(all_scores) if all_scores else 0
    max_trust = max(all_scores) if all_scores else 1
    
    # Graph 5 Fix: Adaptive Sweep
    thresholds = np.linspace(min_trust, max_trust + 0.0001, 50)
    
    tprs = []
    fprs = []
    
    total_malicious = len(malicious_nodes)
    total_honest = len(honest_nodes)
    
    if total_malicious == 0 or total_honest == 0:
        return

    for t in thresholds:
        # Detected if trust < t
        detected_malicious = [v for v in malicious_nodes if v.global_trust_score < t]
        false_positives = [v for v in honest_nodes if v.global_trust_score < t]
        
        tpr = len(detected_malicious) / total_malicious
        fpr = len(false_positives) / total_honest
        
        tprs.append(tpr)
        fprs.append(fpr)
        
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, tprs, label='Detection Rate (TPR)', color='blue')
    plt.plot(thresholds, fprs, label='False Positive Rate (FPR)', color='red', linestyle='--')
    
    plt.title(f"Detection Performance (Range: {min_trust:.4f} - {max_trust:.4f})")
    plt.xlabel("Trust Threshold")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_comparative_trust(vehicles, save_path="results/comparative_trust_normalized.png"):
    """
    Graph 2 Change: Plot Normalized Difference (Z-Score) or Rank Percentile.
    """
    import numpy as np

    # Group histories
    histories = {
        'HONEST': [],
        'MALICIOUS': [],
        'SWING': []
    }
    
    for v in vehicles.values():
        if v.behavior_type in histories:
            histories[v.behavior_type].append(v.trust_history)

    # 1. Calculate Aggregates per Step
    # Assume all equal length
    if not vehicles: return
    n_steps = len(list(vehicles.values())[0].trust_history)
    steps = range(n_steps)
    
    mean_trust_per_step = []
    std_trust_per_step = []
    
    for t_idx in range(n_steps):
        # Gather all scores at this step
        step_scores = [v.trust_history[t_idx] for v in vehicles.values()]
        mean_trust_per_step.append(np.mean(step_scores))
        std_trust_per_step.append(np.std(step_scores) + 1e-9) # Avoid div0
        
    plt.figure(figsize=(10, 6))
    
    # Plot Normalized Z-Scores per group
    for cat, lists in histories.items():
        if not lists: continue
        
        avg_raw = np.mean(lists, axis=0)
        # Check lengths match
        if len(avg_raw) != len(mean_trust_per_step): continue
        
        # Normalized: (Average_of_Group - Global_Mean) / Global_Std
        normalized = (avg_raw - mean_trust_per_step) / std_trust_per_step
        
        plt.plot(steps, normalized, label=f'{cat} (Normalized)', markevery=10)

    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Normalized Trust Difference (Z-Score) vs Time")
    plt.xlabel("Time (steps)")
    plt.ylabel("Z-Score (Std Devs from Mean)")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_final_trust_distribution(vehicles, save_path="results/final_rank_distribution.png"):
    """
    Graph 4 Change: Plot Rank Index / Distance from Committee Cutoff.
    """
    import numpy as np
    
    # Sort all vehicles by trust
    sorted_vehicles = sorted(vehicles.values(), key=lambda x: x.global_trust_score, reverse=True)
    
    ranks = range(1, len(sorted_vehicles) + 1)
    scores = [v.global_trust_score for v in sorted_vehicles]
    colors = ['red' if v.is_malicious else 'green' for v in sorted_vehicles]
    ids = [v.id for v in sorted_vehicles]
    
    committee_size = 5 # As per experiment
    
    plt.figure(figsize=(12, 6))
    
    # Scatter plot of Ranks
    plt.scatter(ranks, scores, c=colors, s=100, alpha=0.7)
    
    # Draw Committee Cutoff
    if len(sorted_vehicles) >= committee_size:
        cutoff_score = sorted_vehicles[committee_size-1].global_trust_score
        plt.axvline(x=committee_size + 0.5, color='blue', linestyle='--', label='Committee Cutoff')
        plt.axhline(y=cutoff_score, color='gray', linestyle=':', alpha=0.5)

    # Label points
    for i, txt in enumerate(ids):
        # Annotate sparsely to avoid clutter
        if i < 10 or i % 5 == 0:
            plt.annotate(txt, (ranks[i], scores[i]), fontsize=8, alpha=0.7)

    plt.title("Vehicle Ranking & Committee Selection")
    plt.xlabel("Rank (1 = Highest Trust)")
    plt.ylabel("Global Trust Score")
    
    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='green', marker='o', linestyle=''),
                    Line2D([0], [0], color='red', marker='o', linestyle='')]
    plt.legend(custom_lines, ['Honest', 'Malicious'])
    
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_swing_analysis(global_history, local_history, save_path="results/swing_analysis.png"):
    """
    Graph 3 Change: Plots Sliding Window Local Trust vs Global Trust.
    User Note: "Plot local trust over a sliding window... This will show Oscillation"
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(global_history, label='Global Trust (Smoothed/Decaying)', color='purple', linewidth=2)
    plt.plot(local_history, label='Local Trust (Sliding Window=20)', color='orange', linestyle='-', alpha=0.8)
    
    plt.title("Swing Attacker Dynamics: Global Stability vs Local Oscillation")
    plt.xlabel("Simulation Step")
    plt.ylabel("Trust Score")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()
