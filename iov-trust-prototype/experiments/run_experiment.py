"""
Main Experiment Runner.

Integrates Trust Model, Simulator, and Mock Blockchain.
"""
import sys
import os

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
    normalize_histories
) 

def calculate_statistics(vehicles, dags, partial_dags_count, total_steps):
    """
    Calculates and prints:
    A. Final Trust Statistics (Normalized)
    B. Detection Performance (at optimal threshold)
    C. Consensus Success Rate
    D. Trust Convergence Time
    E. Multi-RSU Sync Stats
    """
    import numpy as np
    
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

def run():
    # 1. Setup
    print("Initializing Experiment (BayesTrust + VehicleRank)...")
    # 10% Malicious, 10% Swing
    # NEW: 2 RSUs
    sim = Simulator(num_vehicles=200, percent_malicious=0.1, num_rsus=20)
    #sim = Simulator(num_vehicles=30, percent_malicious=0.15, percent_swing=0.10, num_rsus=2)
    
    # NEW: Multiple DAGs (one per RSU/Region)
    dags = [DAG(), DAG()] 
    
    SIMULATION_STEPS = 120  # Matches Fig 8 timeline
    
    # Identify a specific Swing Attacker and a specific Observer for analysis
    swing_candidates = [v for v in sim.model.vehicles.values() if v.behavior_type == 'SWING']
    honest_candidates = [v for v in sim.model.vehicles.values() if v.behavior_type == 'HONEST']
    
    target_swing = swing_candidates[0] if swing_candidates else None
    observer = honest_candidates[0] if honest_candidates else None
    
    swing_global_history = []
    swing_local_history = []
    
    if target_swing:
        print(f"Tracking Swing Attacker: {target_swing.id}")
    
    # 2. Loop
    for t in range(SIMULATION_STEPS):
        # A. Run Trust Simulation Step
        # The simulator updates trust internally
        sim.model.simulate_interaction_step(num_interactions=40)
        
        # Consumes reports AND syncs RSUs
        sim.model.update_global_trust(sync_rsus=True)
        
        # Capture History for Swing Plot
        if target_swing:
            # 1. Capture current Global Trust (NORMALIZED for visualization)
            # t_norm = (t - min) / (max - min)
            # We need the full vector to normalize
            all_scores = [v.global_trust_score for v in sim.model.vehicles.values()]
            g_min, g_max = min(all_scores), max(all_scores)
            
            raw_val = target_swing.global_trust_score
            if g_max > g_min:
                norm_val = (raw_val - g_min) / (g_max - g_min + 1e-9)
            else:
                norm_val = 0.5
                
            swing_global_history.append(norm_val)
            
            # 2. Capture Local Trust from the Observer's perspective
            if observer:
                # Use sliding window for Plot 3 (Swing Analysis)
                # Graph 6 Change: Option A (best): Reduce sliding window to 10
                swing_local_history.append(observer.get_windowed_local_trust(target_swing.id, window_size=10))
        
        # B. Blockchain Logic - MULTI-DAG (Section IV: Trust-DBFT)
        # 1. Rank & Select Committee
        ranked_vehicles = sim.model.get_ranked_vehicles()
        committee_size = 5 # config c
        committee = select_validators(ranked_vehicles, top_n=committee_size)
        
        if committee:
            # Check Weighted Consensus (Abstracted Section IV-B)
            # Assuming proposal is "Good" - will the committee pass it?
            consensus_reached = check_consensus_weighted(committee)
            
            if consensus_reached:
                # Primary Validator (DAG 1)
                v1 = committee[0] # Leader
                snapshot1 = {v.id: v.global_trust_score for v in ranked_vehicles}
                dags[0].add_block(data=snapshot1, validator_id=v1.id)

                if len(committee) > 1:
                    v2 = committee[1] # Backup/Second Region Leader
                    snapshot2 = {v.id: v.global_trust_score for v in ranked_vehicles} 
                    dags[1].add_block(data=snapshot2, validator_id=v2.id)

                # 3. MERGE DAGs (Cross-Shard/Region Sync)
                dags[0].merge_with(dags[1])
                dags[1].merge_with(dags[0])
            else:
                # print(f"Step {t}: Consensus Failed.")
                pass
            
        if t % 10 == 0:
            leader_id = committee[0].id if committee else 'None'
            print(f"Step {t}: Leader = {leader_id} | DAG Size: {len(dags[0].blocks)}")

    # 3. Analysis
    print("Simulation Loop Complete.")
    print(f"DAG 1 Height: {len(dags[0].blocks)}")
    print(f"DAG 2 Height: {len(dags[1].blocks)}")
    
    # 4. Plotting
    
    # 4. Plotting
    print("Generating Plots...")
    plot_trust_evolution(sim.model.vehicles)
    plot_detection_metrics(sim.model.vehicles)
    plot_comparative_trust(sim.model.vehicles)
    plot_final_trust_distribution(sim.model.vehicles)
    plot_trust_convergence(sim.model.vehicles)
    
    if target_swing and observer:
        plot_swing_analysis(swing_global_history, swing_local_history)
        
    # 5. Print Statistics Table
    calculate_statistics(sim.model.vehicles, dags, 0, SIMULATION_STEPS)

if __name__ == "__main__":
    run()
