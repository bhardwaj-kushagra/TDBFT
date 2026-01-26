"""
Plotting Utilities. (Refactored for Statistical Rigor)
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def normalize_histories(vehicles):
    """
    Normalizes trust scores per time step using Min-Max normalization.
    Returns:
        dict: {vid: np.array(normalized_history)}
    """
    if not vehicles:
        return {}
    
    # Check length
    sample_v = next(iter(vehicles.values()))
    n_steps = len(sample_v.trust_history)
    
    # 1. Gather Matrix: (Steps, Vehicles)
    # maintain consistent order
    vids = list(vehicles.keys())
    raw_matrix = np.zeros((n_steps, len(vids)))
    
    for i, vid in enumerate(vids):
        hist = vehicles[vid].trust_history
        # Truncate or pad if mismatch (shouldn't happen in sync sim)
        length = min(len(hist), n_steps) 
        raw_matrix[:length, i] = hist[:length]
        
    # 2. Normalize per Step (Row-wise)
    norm_matrix = np.zeros_like(raw_matrix)
    
    for t in range(n_steps):
        row = raw_matrix[t, :]
        rmin, rmax = np.min(row), np.max(row)
        if rmax > rmin:
            norm_matrix[t, :] = (row - rmin) / (rmax - rmin + 1e-9)
        else:
            norm_matrix[t, :] = 0.5 # Default if all equal
            
    # 3. Distribute back
    normalized_data = {}
    for i, vid in enumerate(vids):
        normalized_data[vid] = norm_matrix[:, i]
        
    return normalized_data

def plot_trust_evolution(vehicles, save_path="results/trust_evolution.png"):
    """
    Plots the history of global trust scores (Normalized).
    Graph 1 Fix: Plot t_norm. Honest vs Malicious. Percentile lines.
    "Since VehicleRank produces relative trust scores... global trust values are min-max normalized..."
    """
    norm_data = normalize_histories(vehicles)
    
    plt.figure(figsize=(10, 6))
    
    # Burn-in: Skip first 5 steps
    burn_in = 5
    
    all_final_scores = []
    
    for vid, v in vehicles.items():
        if vid not in norm_data: continue
        
        full_hist = norm_data[vid]
        if len(full_hist) <= burn_in: continue
        
        history = full_hist[burn_in:]
        steps = range(burn_in, burn_in + len(history))
        
        color = 'red' if v.is_malicious else 'green'
        alpha = 0.15 if v.is_malicious else 0.1 # Lighter for individual traces
        
        plt.plot(steps, history, color=color, alpha=alpha, label='_nolegend_')
        all_final_scores.append(history[-1])

    # Add Percentile Trends (Median of Honest vs Median of Malicious?)
    # Or just population percentiles. 
    # Let's plot the MEDIAN trend line for Honest vs Malicious to show separation clearly.
    
    vids = list(vehicles.keys())
    if norm_data:
        n_steps = len(next(iter(norm_data.values())))
        steps = range(burn_in, n_steps)
        
        honest_ids = [v.id for v in vehicles.values() if not v.is_malicious]
        mal_ids = [v.id for v in vehicles.values() if v.is_malicious]
        
        honest_matrix = np.array([norm_data[uid] for uid in honest_ids])
        mal_matrix = np.array([norm_data[uid] for uid in mal_ids])
        
        if honest_matrix.shape[0] > 0:
            h_median = np.median(honest_matrix, axis=0)[burn_in:]
            plt.plot(steps, h_median, color='darkgreen', linewidth=2, label='Honest Median')
            
        if mal_matrix.shape[0] > 0:
            m_median = np.median(mal_matrix, axis=0)[burn_in:]
            plt.plot(steps, m_median, color='darkred', linewidth=2, label='Malicious Median')
            
        # Top 30% cutoff line (approximate from last step)
        # Actually, let's plot the Top 30% threshold over time
        all_matrix = np.array([norm_data[uid] for uid in vids])
        p70_trend = np.percentile(all_matrix, 70, axis=0)[burn_in:]
        plt.plot(steps, p70_trend, color='blue', linestyle='--', label='Top 30% (Committee Cutoff)')

    plt.ylim(0, 1.05)
    plt.title("Evolution of Global Trust Scores (Normalized)")
    plt.xlabel("Simulation Step")
    plt.ylabel("Normalized Trust Score")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_detection_metrics(vehicles, save_path="results/detection_metrics.png"):
    """
    Plots Detection Performance across Trust Percentiles.
    Graph 3 Fix: X-axis = Trust Percentile (0-100).
    "Detection performance is evaluated across trust percentiles..."
    """
    # Use normalized scores from final step
    norm_data = normalize_histories(vehicles)
    
    # Gather (score, is_maliciuos)
    data_points = []
    for vid, v in vehicles.items():
        if vid in norm_data:
            final_score = norm_data[vid][-1]
            data_points.append({'score': final_score, 'is_mal': v.is_malicious})
            
    if not data_points: return
    
    # Sort by score ascending
    data_points.sort(key=lambda x: x['score'])
    scores = [x['score'] for x in data_points]
    
    total_malicious = sum(1 for x in data_points if x['is_mal'])
    total_honest = len(data_points) - total_malicious
    
    tprs = []
    fprs = []
    percentiles = np.linspace(0, 100, 100)
    
    for p in percentiles:
        # Threshold at p-th percentile
        thresh = np.percentile(scores, p)
        
        # Predicted Malicious if score < thresh
        # (Low trust = Malicious)
        
        # Count stats
        tp = sum(1 for x in data_points if x['score'] < thresh and x['is_mal'])
        fp = sum(1 for x in data_points if x['score'] < thresh and not x['is_mal'])
        
        tpr = tp / total_malicious if total_malicious > 0 else 0
        fpr = fp / total_honest if total_honest > 0 else 0
        
        tprs.append(tpr)
        fprs.append(fpr)
        
    plt.figure(figsize=(10, 6))
    plt.plot(percentiles, tprs, label='Detection Rate (TPR)', color='blue')
    plt.plot(percentiles, fprs, label='False Positive Rate (FPR)', color='red', linestyle='--')
    
    # Mark "Median of Bottom 30%"
    # 30th percentile is x=30
    idx_30 = 30 # roughly index 30 in linspace(0,100,100)
    if idx_30 < len(tprs):
        plt.axvline(x=30, color='green', linestyle=':', label='Bottom 30% Threshold')
        plt.annotate(f'TPR={tprs[idx_30]:.2f}', 
                     xy=(30, tprs[idx_30]), 
                     xytext=(35, tprs[idx_30]-0.15),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title("Detection Performance vs Trust Percentile")
    plt.xlabel("Trust Threshold (Percentile)")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_comparative_trust(vehicles, save_path="results/comparative_trust_normalized.png"):
    """
    Graph 2: Normalized Trust Difference (Z-Score)
    "Z-score is computed on t_norm... Mention cold start explicitly."
    """
    norm_data = normalize_histories(vehicles)
    if not norm_data: return

    # Group histories
    histories = {'HONEST': [], 'MALICIOUS': [], 'SWING': []}
    
    for vid, v in vehicles.items():
        if vid in norm_data:
            histories[v.behavior_type].append(norm_data[vid])

    # Calc global stats per step
    all_hists = np.array(list(norm_data.values())) # (N_vehicles, N_steps) - wait, dict values order?
    # Better:
    all_hists = np.array([norm_data[uid] for uid in vehicles.keys()])
    
    # Transpose to (Steps, Vehicles) for step-wise stats
    step_data = all_hists.T 
    
    mean_per_step = np.mean(step_data, axis=1)
    std_per_step = np.std(step_data, axis=1) + 1e-9
    
    plt.figure(figsize=(10, 6))
    burn_in = 5
    steps = np.arange(len(mean_per_step))
    
    # Plot Z-scores
    for cat, lists in histories.items():
        if not lists: continue
        
        # Average trust of this group at each step
        group_matrix = np.array(lists).T # (Steps, GroupSize)
        group_avg = np.mean(group_matrix, axis=1)
        
        # Z-score of the GROUP AVERAGE relative to GLOBAL dist
        # z = (GroupMean - GlobalMean) / GlobalStd
        z_scores = (group_avg - mean_per_step) / std_per_step
        
        # Apply burn-in for plotting
        plt.plot(steps[burn_in:], z_scores[burn_in:], label=f'{cat} (Z-Score)', markevery=10)

    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.fill_between(steps[burn_in:], -1, 1, color='gray', alpha=0.2, label='±1 Std Dev')

    plt.title("Normalized Trust Difference (Z-Score)\n(First 5 steps omitted for cold-start)")
    plt.xlabel("Simulation Step")
    plt.ylabel("Z-Score (Std Devs from Global Mean)")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_final_trust_distribution(vehicles, save_path="results/final_rank_distribution.png"):
    """
    Graph 4 Fix: Label Y-axis as Normalized Global Trust.
    Committee is Top-k ranks.
    """
    norm_data = normalize_histories(vehicles)
    
    # Build list of (vid, score, is_mal)
    data = []
    for vid, v in vehicles.items():
        if vid in norm_data:
            score = norm_data[vid][-1] # Final step
            data.append((v, score))
            
    # Sort by score desc (Rank 1 is highest score)
    data.sort(key=lambda x: x[1], reverse=True)
    
    ranks = range(1, len(data) + 1)
    scores = [x[1] for x in data]
    colors = ['red' if x[0].is_malicious else 'green' for x in data]
    ids = [x[0].id for x in data]
    
    committee_size = 5
    
    plt.figure(figsize=(12, 6))
    plt.scatter(ranks, scores, c=colors, s=100, alpha=0.7)
    
    if len(data) >= committee_size:
        plt.axvline(x=committee_size + 0.5, color='blue', linestyle='--', label='Committee Cutoff')
        # Threshold at cutoff
        cutoff = scores[committee_size-1]
        plt.axhline(y=cutoff, color='gray', linestyle=':', alpha=0.5)

    # Label top-k and bottom-k
    top_k, bottom_k = 5, 5
    n_vehicles = len(data)
    
    for i, txt in enumerate(ids):
        rank = i + 1
        if rank <= top_k or rank > (n_vehicles - bottom_k):
             plt.annotate(txt, (ranks[i], scores[i]), fontsize=8, alpha=0.7)

    plt.title("Vehicle Ranking & Committee Selection")
    plt.xlabel("Rank (1 = Highest Trust)")
    plt.ylabel("Normalized Global Trust (VehicleRank)")
    
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
    Graph 6: Swing Attacker Dynamics.
    "Label local trust as Bayesian... Label global trust as smoothed VehicleRank"
    """
    plt.figure(figsize=(10, 6))
    
    # Global history should already be normalized by caller (or here if we had context)
    # Assuming caller passes normalized trace
    plt.plot(global_history, label='Global Trust (Smoothed VehicleRank)', color='purple', linewidth=2)
    plt.plot(local_history, label='Local Trust (Bayesian Window=10)', color='orange', linestyle='-', alpha=0.8)
    
    plt.title("Swing Attacker Dynamics")
    plt.xlabel("Simulation Step")
    plt.ylabel("Trust Score")
    plt.legend()
    plt.grid(True)
    
    # Add text box
    textstr = "While swing attackers manipulate local trust,\ntheir global trust remains suppressed due to\ntemporal smoothing and network-wide propagation."
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=9,
        verticalalignment='bottom', bbox=props)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_trust_convergence(vehicles, save_path="results/trust_convergence.png"):
    """
    Plots the fraction of nodes with stable rank over time.
    Shows how fast the system settles.
    Metric: Fraction of nodes whose rank changed by <= 5% of total nodes since last step.
    """
    import numpy as np
    
    # 1. Build Score Matrix (Steps x Vehicles)
    if not vehicles: return
    
    vids = sorted(list(vehicles.keys()))
    sample_v = vehicles[vids[0]]
    n_steps = len(sample_v.trust_history)
    n_vehicles = len(vids)
    
    score_matrix = np.zeros((n_steps, n_vehicles))
    for i, vid in enumerate(vids):
        hist = vehicles[vid].trust_history
        length = min(len(hist), n_steps)
        score_matrix[:length, i] = hist[:length]
        
    # 2. Compute Ranks at each step
    # Rank 0 = Highest Trust
    rank_matrix = np.zeros((n_steps, n_vehicles), dtype=int)
    
    for t in range(n_steps):
        scores = score_matrix[t, :]
        # efficient rank: argsort of argsort of negated scores
        temp = np.argsort(-scores)
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(n_vehicles)
        rank_matrix[t, :] = ranks
        
    # 3. Compute Stability
    # Threshold for "stable" = e.g. 5% of N
    threshold = max(1, int(0.05 * n_vehicles))
    
    stability_fraction = []
    # Skip t=0
    plot_steps = range(1, n_steps)
    
    for t in plot_steps:
        diffs = np.abs(rank_matrix[t] - rank_matrix[t-1])
        stable_count = np.sum(diffs <= threshold)
        frac = stable_count / n_vehicles
        stability_fraction.append(frac)
        
    # 4. Plot
    plt.figure(figsize=(10, 6))
    
    burn_in = 5
    # Adjust plot steps if burn_in applies
    valid_indices = [i for i, s in enumerate(plot_steps) if s > burn_in]
    
    if valid_indices:
        plt.plot([plot_steps[i] for i in valid_indices], [stability_fraction[i] for i in valid_indices], color='navy', linewidth=2)
    else:
        plt.plot(plot_steps, stability_fraction, color='navy', linewidth=2)
        
    plt.title(f"Trust Convergence Time\n(Stable Rank = change <= {threshold} positions)")
    plt.xlabel("Simulation Step")
    plt.ylabel("Fraction of Nodes with Stable Rank")
    plt.ylim(0, 1.05)
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_dag_structure(dag, save_path="results/dag_structure.png"):
    """
    Visualizes the DAG structure.
    Uses a layered layout based on parent-child relationships.
    """
    blocks = dag.blocks
    if not blocks:
        print("DAG is empty, skipping plot.")
        return

    # 1. Compute Depths (Layering)
    depths = {} # block_id -> int
    
    # Initialize all depths to 0
    for bid in blocks:
        depths[bid] = 0
        
    # Relax edges: depth(child) >= depth(parent) + 1
    # Repeat until convergence (Longest Path)
    
    changed = True
    max_depth = 0
    loop_count = 0
    
    while changed and loop_count < len(blocks) + 2:
        changed = False
        loop_count += 1
        for bid, block in blocks.items():
            current_depth = depths[bid]
            max_p_depth = -1
            
            # Find max parent depth
            has_known_parents = False
            for pid in block.parents:
                if pid in depths:
                    has_known_parents = True
                    max_p_depth = max(max_p_depth, depths[pid])
            
            new_depth = 0
            if has_known_parents:
                new_depth = max_p_depth + 1
            
            if new_depth > current_depth:
                depths[bid] = new_depth
                changed = True
                max_depth = max(max_depth, new_depth)
    
    # 2. Assign Y-coordinates (Layout)
    layers = {}
    for bid, d in depths.items():
        if d not in layers: layers[d] = []
        layers[d].append(bid)
        
    y_coords = {}
    for d, bids in layers.items():
        n = len(bids)
        for i, bid in enumerate(bids):
            y_coords[bid] = i - (n - 1) / 2.0
            
    # 3. Plot
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Draw Edges first
    for bid, block in blocks.items():
        x1, y1 = depths[bid], y_coords[bid]
        for pid in block.parents:
            if pid in depths:
                x2, y2 = depths[pid], y_coords[pid]
                ax.plot([x2, x1], [y2, y1], color='gray', alpha=0.5, zorder=1)
                
    # Draw Nodes
    x_vals = [depths[bid] for bid in blocks]
    y_vals = [y_coords[bid] for bid in blocks]
    
    # Color by Validator
    validators = sorted(list(set(b.validator_id for b in blocks.values())))
    val_map = {v: i for i, v in enumerate(validators)}
    colors = [val_map[blocks[bid].validator_id] for bid in blocks]
    
    sc = ax.scatter(x_vals, y_vals, c=colors, cmap='tab10', s=300, zorder=2, edgecolors='black')
    
    # Labels
    for bid in blocks:
        ax.text(depths[bid], y_coords[bid], bid[:4], fontsize=8, ha='center', va='center', color='white', fontweight='bold', zorder=3)
        
    plt.title(f"DAG Structure (Height: {max_depth+1}, Blocks: {len(blocks)})")
    plt.xlabel("Layer (Temporal Depth)")
    plt.yticks([]) # Hide Y axis
    plt.grid(False)
    
    # Legend for validators
    import matplotlib.patches as mpatches
    cmap = plt.get_cmap('tab10')
    if len(validators) <= 10:
        patches = [mpatches.Patch(color=cmap(val_map[v]), label=f'Val {v}') for v in validators]
        plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1))
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()


