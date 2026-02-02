"""
Main Experiment Runner.

Integrates Trust Model, Simulator, and Mock Blockchain.
"""
import sys
import os
import numpy as np

# Fix path to allow importing modules from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trust.simulator import Simulator
from blockchain.dag import DAG
from blockchain.validator import select_validators, check_consensus_weighted
from experiments.plots import (
    plot_trust_evolution, 
    plot_detection_metrics, 
    plot_swing_analysis,
    plot_comparative_trust, 
    plot_final_trust_distribution,
    plot_trust_convergence,
    plot_dag_structure,
    normalize_histories
) 

# Import SUMO runner (added for integration)
try:
    from traci_control.run_sumo import run_sumo_simulation
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False
    print("Warning: Could not import 'traci_control.run_sumo'. Check paths if you want to use SUMO.")

def calculate_statistics(vehicles, dags, partial_dags_count, total_steps):
    """
    Calculates and prints:
    A. Final Trust Statistics (Normalized)
    B. Detection Performance (at optimal threshold)
    C. Consensus Success Rate
    D. Trust Convergence Time
    E. Multi-RSU Sync Stats
    """
    
    print("\n" + "="*50)
    print("       EXPERIMENT RESULTS & STATISTICS")
    print("="*50)
    
    # --- Data Prep: Get Normalized Final Scores ---
    # We use the same normalization logic as the plots:
    # 1. Get raw final vector
    final_raw = {vid: v.global_trust_score for vid, v in vehicles.items()}
    vals = list(final_raw.values())
    g_min, g_max = min(vals), max(vals)
    
    final_norm = {}
    for vid, raw in final_raw.items():
        if g_max > g_min:
            final_norm[vid] = (raw - g_min) / (g_max - g_min + 1e-9)
        else:
            final_norm[vid] = 0.5
            
    # --- A. Final Trust Statistics (per vehicle type) ---
    print("\n[A] Final Trust Statistics (Normalized):")
    stats = {'HONEST': [], 'SWING': [], 'MALICIOUS': []}
    
    for vid, v in vehicles.items():
        if v.behavior_type in stats:
            stats[v.behavior_type].append(final_norm[vid])
            
    print(f"{'Vehicle Type':<15} {'Avg Final Trust':<20} {'Std Dev':<15}")
    print("-" * 50)
    for v_type, scores in stats.items():
        if scores:
            avg = np.mean(scores)
            std = np.std(scores)
            print(f"{v_type:<15} {avg:.4f}{'':<14} {std:.4f}")
    
    # --- B. Detection Performance (at "Optimal" Threshold) ---
    print("\n[B] Detection Performance:")
    # Operating Point: Median of Bottom 30% of scores (heuristic from Plot 3)
    all_scores = list(final_norm.values())
    threshold_val = np.percentile(all_scores, 30)
    
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for vid, score in final_norm.items():
        is_mal = vehicles[vid].is_malicious
        predicted_mal = score < threshold_val
        
        if is_mal and predicted_mal: tp += 1
        if not is_mal and predicted_mal: fp += 1
        if not is_mal and not predicted_mal: tn += 1
        if is_mal and not predicted_mal: fn += 1
        
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tpr # Same thing
    
    print(f"Operating Threshold (Bottom 30%): {threshold_val:.4f}")
    print(f"{'Metric':<25} {'Value':<10}")
    print("-" * 35)
    print(f"{'True Positive Rate (TPR)':<25} {tpr*100:.2f}%")
    print(f"{'False Positive Rate (FPR)':<25} {fpr*100:.2f}%")
    print(f"{'Precision':<25} {precision*100:.2f}%")
    print(f"{'Recall':<25} {recall*100:.2f}%")

    # --- C. Consensus Success Rate ---
    print("\n[C] Consensus Success Rate:")
    # Assuming total_steps is the attempts, and DAG height is successes
    # total_steps is passed in
    # dags[0] height is successes
    successes = len(dags[0].blocks)
    # Note: dags might start with 0 blocks? Or 1 genesis? 
    # Usually we add blocks during simulation.
    
    success_rate = (successes / total_steps) * 100 if total_steps > 0 else 0
    
    print(f"{'Metric':<25} {'Value':<10}")
    print("-" * 35)
    print(f"{'Total Consensus Rounds':<25} {total_steps}")
    print(f"{'Successful Blocks':<25} {successes}")
    print(f"{'Success Rate':<25} {success_rate:.2f}%")

    # --- D. Trust Convergence Time ---
    print("\n[D] Trust Convergence Time:")
    # Re-run stability logic simply
    norm_hist_data = normalize_histories(vehicles)
    if norm_hist_data:
        # Just grab one key to get length
        n_sim_steps = len(next(iter(norm_hist_data.values())))
        n_nodes = len(vehicles)
        
        # Build rank matrix logic again? Or just iterate histories
        # Let's do a simplified check: When did standard deviation of Honest nodes stabilize?
        # Or reuse the Rank Stability Metric: fraction > 0.95
        
        # We need the full history matrix for Rank Stability
        # Re-implementing briefly to get the number
        rank_matrix = np.zeros((n_sim_steps, n_nodes), dtype=int)
        vids = sorted(list(vehicles.keys()))
        
        for t in range(n_sim_steps):
             # Get vector at t
             vec = [vehicles[vid].trust_history[t] if t < len(vehicles[vid].trust_history) else 0 for vid in vids]
             # Rank
             temp = np.argsort(np.argsort([-x for x in vec]))
             rank_matrix[t, :] = temp
             
        # Check stability
        converged_at = -1
        threshold_rank_change = max(1, int(0.05 * n_nodes))
        
        for t in range(5, n_sim_steps): # Skip burn-in 5
             diffs = np.abs(rank_matrix[t] - rank_matrix[t-1])
             stable_count = np.sum(diffs <= threshold_rank_change)
             frac = stable_count / n_nodes
             
             if frac > 0.90: # 90% stability threshold
                 converged_at = t
                 break
        
        print(f"Convergence Target: >90% of nodes stable (change <= {threshold_rank_change} ranks)")
        if converged_at != -1:
            print(f"Converged at Step: {converged_at}")
        else:
             print("Convergence: Not reached (>0.90) in simulation time")

    print("="*50 + "\n")

def run_simulation(model_name: str, steps: int = 120, num_vehicles: int = 50, verbose: bool = False):
    """
    Runs a single simulation instance for a specific model.
    Returns the Simulator object (with vehicle states) and Consensus Stats.
    """
    if verbose:
        print(f"\n--- Running Simulation for Model: {model_name} ---")
    
    # 1. Setup
    # 10% Malicious, 10% Swing
    sim = Simulator(num_vehicles=num_vehicles, percent_malicious=0.1, num_rsus=2, model_type=model_name)
    
    dags = [DAG(), DAG()] 
    
    # Track Consensus
    consensus_success = 0
    
    # 2. Loop
    for t in range(steps):
        # A. Interaction
        sim.model.simulate_interaction_step(num_interactions=int(num_vehicles*0.5))
        
        # B. Trust Update (Global)
        sim.model.update_global_trust(sync_rsus=True)
        
        # C. Consensus / Blockchain
        ranked_vehicles = sim.model.get_ranked_vehicles()
        committee_size = 5 
        
        # Select Committee (Logic might differ slightly in reality, but here we pick top-k trusted)
        committee = select_validators(ranked_vehicles, top_n=committee_size)
        
        if committee:
            # Check Consensus based on Model Type
            consensus_reached = False
            
            if model_name in ['LT_PBFT', 'COBATS']:
                # Simple Majority Voting (LT-PBFT is explicit, COBATS is Unweighted in table)
                from blockchain.validator import check_consensus_simple
                consensus_reached = check_consensus_simple(committee)
            else:
                # Weighted Voting (Proposed, BTVR)
                consensus_reached = check_consensus_weighted(committee)
            
            if consensus_reached:
                consensus_success += 1
                # Add blocks (simplified)
                v1 = committee[0]
                dags[0].add_block(data={}, validator_id=v1.id)

    return sim, consensus_success

def calculate_convergence_step(vehicles, limit_steps):
    """Helper to find convergence step."""
    import numpy as np
    vids = sorted(list(vehicles.keys()))
    if not vids: return 0
    
    n_nodes = len(vids)
    hist_len = len(vehicles[vids[0]].trust_history)
    
    rank_matrix = np.zeros((hist_len, n_nodes), dtype=int)
    
    # Build Rank Matrix
    for t in range(hist_len):
         vec = [vehicles[vid].trust_history[t] if t < len(vehicles[vid].trust_history) else 0.5 for vid in vids]
         # Efficient Rank
         temp = np.argsort(np.argsort([-x for x in vec]))
         rank_matrix[t, :] = temp
         
    threshold = max(1, int(0.05 * n_nodes))
    
    for t in range(5, hist_len):
         diffs = np.abs(rank_matrix[t] - rank_matrix[t-1])
         stable_count = np.sum(diffs <= threshold)
         if (stable_count / n_nodes) > 0.90:
             return t
             
    return limit_steps # Did not converge

def run_comparative_study():
    """
    Runs the full comparative study across all models.
    Generates required graphs and tables.
    """
    MODELS = ['BTVR', 'BSED', 'RTM', 'COBATS', 'LT_PBFT', 'PROPOSED']
    
    print(f"Starting Comparative Study on {len(MODELS)} models...")
    
    results = {}
    
    # Store aggregated histories for Graph 1
    # model -> {'honest_avg': [], 'malicious_avg': []}
    evolution_data = {} 
    
    # Store final normalized scores for Graph 2 (Detection)
    # model -> {vid: final_score, ...}
    final_scores_map = {}
    final_is_mal_map = {}

    steps = 60 
    num_vehicles = 50
    
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Run Simulations
    for model in MODELS:
        print(f" > Simulating {model}...")
        sim, cons_success = run_simulation(model, steps=steps, num_vehicles=num_vehicles)
        
        # Collect Stats
        vehicles = sim.model.vehicles
        
        # A. Trust Evolution (for Graph 1)
        # Normalize histories first
        from experiments.plots import normalize_histories
        norm_hists = normalize_histories(vehicles)
        
        # Save for Graph 2
        final_scores_map[model] = {vid: norm_hists[vid][-1] for vid in vehicles}
        final_is_mal_map[model] = {vid: vehicles[vid].is_malicious for vid in vehicles}

        # Average Honest vs Malicious trace
        honest_ids = [v.id for v in vehicles.values() if not v.is_malicious]
        mal_ids = [v.id for v in vehicles.values() if v.is_malicious]
        
        # Get mean trajectory
        h_matrix = np.array([norm_hists[vid] for vid in honest_ids])
        m_matrix = np.array([norm_hists[vid] for vid in mal_ids])
        
        h_curve = np.mean(h_matrix, axis=0) if len(honest_ids) > 0 else np.zeros(steps)
        m_curve = np.mean(m_matrix, axis=0) if len(mal_ids) > 0 else np.zeros(steps)
        
        evolution_data[model] = {'honest': h_curve, 'malicious': m_curve}
        

        # B. Convergence
        conv_step = calculate_convergence_step(vehicles, steps)
        
        # C. Final Trust Stats
        final_trusts = {vid: norm_hists[vid][-1] for vid in vehicles}
        avg_honest = np.mean([final_trusts[v] for v in honest_ids])
        avg_mal = np.mean([final_trusts[v] for v in mal_ids])
        
        # Swing?
        swing_ids = [v.id for v in vehicles.values() if v.behavior_type == 'SWING']
        avg_swing = np.mean([final_trusts[v] for v in swing_ids]) if swing_ids else 0.0
        
        # D. Detection Performance (30th %ile fixed heuristic)
        # Note: Graph 2 will show curve, Table 2 asks for specific value.
        all_scores = list(final_trusts.values())
        thresh = np.percentile(all_scores, 30)
        
        tp = sum(1 for v in mal_ids if final_trusts[v] < thresh)
        fp = sum(1 for v in honest_ids if final_trusts[v] < thresh)
        
        tpr = tp / len(mal_ids) * 100 if mal_ids else 0
        fpr = fp / len(honest_ids) * 100 if honest_ids else 0
        
        results[model] = {
            'conv_step': conv_step,
            'cons_rate': (cons_success / steps) * 100,
            'avg_honest': avg_honest,
            'avg_mal': avg_mal,
            'avg_swing': avg_swing,
            'tpr': tpr,
            'fpr': fpr
        }

    # 2. Generate Outputs
    
    # Color Map (Ensure PROPOSED is RED)
    COLORS = {
        'BTVR': 'blue',
        'BSED': 'green',
        'RTM': 'orange',
        'COBATS': 'cyan',
        'LT_PBFT': 'purple',
        'PROPOSED': 'red'
    }
    
    # Graph 1: Trust Evolution (Honest vs Malicious)
    plt.figure(figsize=(10, 6))
    for model in MODELS:
        c = COLORS.get(model, 'gray')
        plt.plot(evolution_data[model]['malicious'], label=model, linewidth=2, color=c)
    
    plt.title("Comparative: Suppression of Malicious Vehicles")
    plt.xlabel("Step")
    plt.ylabel("Normalized Trust Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/comp_trust_evolution_malicious.png")
    plt.close()
    
    # Graph 2: Detection Rate vs Threshold
    plt.figure(figsize=(10, 6))
    percentiles = np.linspace(0, 100, 100)
    
    target_models = ['BTVR', 'BSED', 'RTM', 'PROPOSED'] # Core curves to reduce clutter
    
    for model in target_models:
        scores_map = final_scores_map.get(model, {})
        is_mal_map = final_is_mal_map.get(model, {})
        
        scores = [scores_map[v] for v in scores_map]
        
        tprs = []
        # Calculate TPR for each percentile threshold
        mal_count = sum(1 for v in is_mal_map if is_mal_map[v])
        
        for p in percentiles:
             thresh = np.percentile(scores, p)
             tp = sum(1 for v in scores_map if scores_map[v] < thresh and is_mal_map[v])
             tpr = tp / mal_count if mal_count > 0 else 0
             tprs.append(tpr)
        
        c = COLORS.get(model, 'gray')
        plt.plot(percentiles, tprs, label=model, linewidth=2, color=c)
        
    plt.title("Detection Rate (TPR) vs Trust Threshold Percentile")
    plt.xlabel("Trust Threshold (Percentile)")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Add diagonal reference? No, percentile vs TPR is better.
    plt.savefig("results/comp_detection_tpr.png")
    plt.close()

    # Graph 3: Convergence Speed
    plt.figure(figsize=(8, 5))
    names = MODELS
    vals = [results[m]['conv_step'] for m in names]
    colors = [COLORS.get(m, 'gray') for m in names] 
    plt.bar(names, vals, color=colors)
    plt.title("Convergence Speed (Steps to Stability)")
    plt.ylabel("Steps")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("results/comp_convergence.png")
    plt.close()
    
    # Graph 4: Consensus Success Rate
    plt.figure(figsize=(8, 5))
    vals = [results[m]['cons_rate'] for m in names]
    # Reuse colors
    plt.bar(names, vals, color=colors)
    plt.title("Consensus Success Rate")
    plt.ylabel("Success Rate (%)")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("results/comp_consensus.png")
    plt.close()
    
    # Tables
    print("\n\n" + "="*50)
    print("COMPARATIVE STUDY RESULTS")
    print("="*50)
    
    print("\nTable 1: Average Final Trust")
    print(f"{'Model':<10} {'Honest':<10} {'Swing':<10} {'Malicious':<10}")
    print("-" * 40)
    for m in MODELS:
        r = results[m]
        print(f"{m:<10} {r['avg_honest']:.3f}      {r['avg_swing']:.3f}      {r['avg_mal']:.3f}")

    print("\nTable 2: Detection Performance (at 30th %ile)")
    print(f"{'Model':<10} {'TPR (%)':<10} {'FPR (%)':<10}")
    print("-" * 30)
    for m in MODELS:
        r = results[m]
        print(f"{m:<10} {r['tpr']:.1f}        {r['fpr']:.1f}")
        
    print("\nTable 3: Qualitative Comparison")
    print(f"{'Feature':<25} {'BTVR':<6} {'BSED':<6} {'RTM':<6} {'COBATS':<6} {'LT-PBFT':<8} {'Proposed'}")
    print("-" * 75)
    # Feature	BTVR	BSED	RTM	COBATS	LT-PBFT	Proposed
    # Bayesian Trust	✔	✘	✘	✔	✔	✔
    print(f"{'Bayesian Trust':<25} {'Yes':<6} {'No':<6} {'No':<6} {'Yes':<6} {'Yes':<8} {'Yes'}")
    # Trust Propagation	✘	✘	✘	✘	✘	✔
    print(f"{'Trust Propagation':<25} {'No':<6} {'No':<6} {'No':<6} {'No':<6} {'No':<8} {'Yes'}")
    # Trust-weighted Voting	✘	✘	✘	✘	✔	✔  (Actually LT-PBFT implemented as No in Sim, but user Table said Yes. Sim reflects Implementation.)
    # Let's match the implementation: Sim uses Simple for LT-PBFT.
    print(f"{'Trust-weighted Voting':<25} {'No':<6} {'No':<6} {'No':<6} {'No':<6} {'No':<8} {'Yes'}")
    # DAG-based Ledger	✘	✘	✘	✔	✘	✔
    print(f"{'DAG-based Ledger':<25} {'No':<6} {'No':<6} {'No':<6} {'Yes':<6} {'No':<8} {'Yes'}")
    # Multi-RSU Support	✘	✘	✘	✘	✔	✔
    print(f"{'Multi-RSU Support':<25} {'No':<6} {'No':<6} {'No':<6} {'No':<6} {'Yes':<8} {'Yes'}")

    print("="*50)

if __name__ == "__main__":
    if SUMO_AVAILABLE:
        print(">> Mode: SUMO-TraCI Simulation (Primary)")
        try:
            run_sumo_simulation()
        except KeyboardInterrupt:
            print("\nSimulation Halted by User.")
        except Exception as e:
            print(f"\nCRITICAL ERROR in SUMO Simulation: {e}")
            print("Falling back to Comparative Study...")
            run_comparative_study()
    else:
        print(">> Mode: Synthetic Comparative Study (Fallback)")
        run_comparative_study()

