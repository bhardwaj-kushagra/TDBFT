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
from experiments.plots import plot_trust_evolution, plot_detection_metrics

def run():
    # 1. Setup
    print("Initializing Experiment...")
    # 10% Malicious, 10% Swing
    sim = Simulator(num_vehicles=30, percent_malicious=0.10, percent_swing=0.10)
    dag = DAG()
    
    SIMULATION_STEPS = 50
    
    # 2. Loop
    for t in range(SIMULATION_STEPS):
        # A. Run Trust Simulation Step
        # The simulator updates trust internally
        sim.model.simulate_interaction_step(num_interactions=40)
        sim.model.update_global_trust()
        
        # B. Blockchain Logic
        # 1. Rank & Select Validators
        ranked_vehicles = sim.model.get_ranked_vehicles()
        validators = select_validators(ranked_vehicles, top_n=3)
        
        if validators:
            # 2. Primary validator (highest rank) creates block
            # For this prototype, just pick the first one
            primary_validator = validators[0]
            
            # Data to store: Snapshot of all global trust scores
            trust_snapshot = {v.id: v.global_trust_score for v in ranked_vehicles}
            
            # 3. Append to DAG
            dag.add_block(data=trust_snapshot, validator_id=primary_validator.id)
            
        if t % 10 == 0:
            print(f"Step {t}: Top Validator = {validators[0].id if validators else 'None'}")

    # 3. Analysis
    print("Simulation Loop Complete.")
    print(f"DAG Height: {len(dag.blocks)}")
    
    # 4. Plotting
    plot_trust_evolution(sim.model.vehicles)
    plot_detection_metrics(sim.model.vehicles)
    
if __name__ == "__main__":
    run()
