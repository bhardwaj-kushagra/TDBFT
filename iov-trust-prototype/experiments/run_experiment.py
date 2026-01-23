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
from blockchain.validator import select_validators
from experiments.plots import (
    plot_trust_evolution, 
    plot_detection_metrics, 
    plot_swing_analysis,
    plot_comparative_trust, 
    plot_final_trust_distribution
)

def run():
    # 1. Setup
    print("Initializing Experiment...")
    # 10% Malicious, 10% Swing
    # NEW: 2 RSUs
    sim = Simulator(num_vehicles=30, percent_malicious=0.15, percent_swing=0.10, num_rsus=2)
    
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
            # 1. Capture current Global Trust
            swing_global_history.append(target_swing.global_trust_score)
            
            # 2. Capture Local Trust from the Observer's perspective
            if observer:
                # get_local_trust returns computed trust based on interactions so far
                swing_local_history.append(observer.get_local_trust(target_swing.id))
        
        # B. Blockchain Logic - MULTI-DAG
        # 1. Rank & Select Validators
        ranked_vehicles = sim.model.get_ranked_vehicles()
        validators = select_validators(ranked_vehicles, top_n=3)
        
        if validators:
            # 2. Multiple blocks could be proposed by different validators in different regions
            # For simplicity: Validator 1 -> DAG 1, Validator 2 -> DAG 2
            
            # Primary Validator (DAG 1)
            v1 = validators[0]
            snapshot1 = {v.id: v.global_trust_score for v in ranked_vehicles}
            dags[0].add_block(data=snapshot1, validator_id=v1.id)

            if len(validators) > 1:
                v2 = validators[1]
                snapshot2 = {v.id: v.global_trust_score for v in ranked_vehicles} # In real world, might differ slightly
                dags[1].add_block(data=snapshot2, validator_id=v2.id)

            # 3. MERGE DAGs (Cross-Shard/Region Sync)
            # Both DAGs learn about each other's blocks
            dags[0].merge_with(dags[1])
            dags[1].merge_with(dags[0])
            
        if t % 10 == 0:
            print(f"Step {t}: Top Validator = {validators[0].id if validators else 'None'} | DAG Size: {len(dags[0].blocks)}")

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
    
    if target_swing and observer:
        plot_swing_analysis(swing_global_history, swing_local_history)
    
if __name__ == "__main__":
    run()
