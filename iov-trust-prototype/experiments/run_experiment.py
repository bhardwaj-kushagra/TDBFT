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
    plot_trust_convergence
)

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
    
if __name__ == "__main__":
    run()
