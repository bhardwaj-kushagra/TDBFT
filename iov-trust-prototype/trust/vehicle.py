"""
Vehicle Entity Module.

Defines the Vehicle class which maintains its own trust parameters
towards other vehicles (Local Trust) and its assigned Global Trust.
"""
from typing import Dict, Tuple
from .bayesian import compute_trust, update_parameters

class Vehicle:
    BEHAVIOR_HONEST = 'HONEST'
    BEHAVIOR_MALICIOUS = 'MALICIOUS'
    BEHAVIOR_SWING = 'SWING'

    def __init__(self, vehicle_id: str, behavior_type: str = 'HONEST'):
        """
        Initialize a vehicle.
        
        Args:
            vehicle_id (str): Unique identifier.
            behavior_type (str): 'HONEST', 'MALICIOUS', or 'SWING'.
        """
        self.id = vehicle_id
        self.behavior_type = behavior_type
        self.is_malicious = (behavior_type != self.BEHAVIOR_HONEST)
        
        # Local Trust Database: vehicle_id -> (alpha, beta)
        # We initialize with alpha=1, beta=1 (Trust=0.5) for unknown vehicles.
        self.interactions: Dict[str, Tuple[float, float]] = {}
        
        # The global trust score assigned to this vehicle by the RSU
        self.global_trust_score = 0.5
        
        # History of global trust score for plotting
        self.trust_history = [0.5]

    def perform_action(self, step_count: int) -> bool:
        """
        Determines the outcome of an interaction initiated by another vehicle towards this vehicle.
        Returns True for a cooperative/good interaction, False for malicious/bad.
        """
        import random
        
        if self.behavior_type == self.BEHAVIOR_HONEST:
            # 99% success rate (accidents happen)
            return random.random() < 0.99
            
        elif self.behavior_type == self.BEHAVIOR_MALICIOUS:
            # Constant Malicious: 30% success rate (to hide slightly), or 0%?
            # Let's say it drops packets 80% of the time -> 20% success.
            return random.random() < 0.20
            
        elif self.behavior_type == self.BEHAVIOR_SWING:
            # Swing Attacker: Oscillates behavior.
            # Example: Good for 50 steps, Bad for 50 steps
            cycle_length = 50
            is_good_phase = (step_count // cycle_length) % 2 == 0
            
            if is_good_phase:
                return random.random() < 0.99
            else:
                return random.random() < 0.10 # Bad phase
        
        return True

    def get_local_trust(self, target_id: str) -> float:
        """Calculate local trust for a specific target vehicle."""
        alpha, beta = self.interactions.get(target_id, (1.0, 1.0))
        return compute_trust(alpha, beta)

    def record_interaction(self, target_id: str, is_positive: bool):
        """
        Update knowledge about a target vehicle based on an interaction (Section III-A).
        """
        alpha, beta = self.interactions.get(target_id, (1.0, 1.0))
        new_alpha, new_beta = update_parameters(alpha, beta, is_positive)
        self.interactions[target_id] = (new_alpha, new_beta)
        
    def get_interaction_stats(self, target_id: str):
        """Returns direct access to alpha/beta for RSU collection."""
        return self.interactions.get(target_id, (1.0, 1.0))

    def __repr__(self):
        return f"<Vehicle {self.id} | Malicious: {self.is_malicious} | Trust: {self.global_trust_score:.2f}>"
