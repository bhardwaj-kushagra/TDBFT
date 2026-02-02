"""
Simulation runner (Non-SUMO).

Provides a simple interface to run trust simulations without SUMO.
"""
from .trust_model import TrustModel

class Simulator:
    def __init__(self, num_vehicles=20, percent_malicious=0.2, percent_swing=0.1, num_rsus=2, model_type='PROPOSED', attack_intensity=0.8):
        self.model = TrustModel(num_vehicles, percent_malicious, percent_swing, num_rsus, model_type=model_type, attack_intensity=attack_intensity)
        
    def run(self, steps=100, interactions_per_step=20):
        """
        Runs the simulation for a number of steps.
        
        Args:
            steps (int): Number of time steps (epochs).
            interactions_per_step (int): Random interactions per step.
        """
        print(f"Starting simulation: {len(self.model.vehicles)} vehicles, {steps} steps.")
        
        for t in range(steps):
            # 1. Simulate Interactions
            self.model.simulate_interaction_step(interactions_per_step)
            
            # 2. Update Global Trust
            self.model.update_global_trust()
            
            # (Optional) Log progress
            if t % 10 == 0:
                print(f"Step {t}/{steps} complete.")
                
        print("Simulation complete.")

    def get_results(self):
        """Return the internal model for analysis."""
        return self.model
