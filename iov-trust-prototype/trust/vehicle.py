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

    def __init__(self, vehicle_id: str, behavior_type: str = 'HONEST', model_type: str = 'PROPOSED'):
        """
        Initialize a vehicle.
        
        Args:
            vehicle_id (str): Unique identifier.
            behavior_type (str): 'HONEST', 'MALICIOUS', or 'SWING'.
            model_type (str): Trust model to use.
        """
        self.id = vehicle_id
        self.behavior_type = behavior_type
        self.model_type = model_type
        self.is_malicious = (behavior_type != self.BEHAVIOR_HONEST)
        
        # Local Trust Database: vehicle_id -> (alpha, beta) or other params
        # Bayesian (PROPOSED, BTVR, COBATS, LT_PBFT): (alpha, beta)
        # BSED: (correct_count, total_count)
        self.interactions: Dict[str, Tuple[float, float]] = {}
        
        # RTM specific: vehicle_id -> current_trust
        self.rtm_trust: Dict[str, float] = {}

        # New: Interaction Logs for Sliding Window Analysis
        # vehicle_id -> list of booleans (outcomes)
        self.interaction_logs: Dict[str, list] = {}
        
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
        """Calculate local trust for a specific target vehicle based on Model Type."""
        if self.model_type == 'RTM':
            # RTM: Simple average table
            return self.rtm_trust.get(target_id, 0.5)

        elif self.model_type == 'BSED':
            # BSED: Behavior Score = Correct / Total
            correct, total = self.interactions.get(target_id, (0.0, 0.0))
            if total == 0: return 0.5
            return correct / total

        else:
            # Bayesian (Default for PROPOSED, BTVR, COBATS, LT_PBFT)
            alpha, beta = self.interactions.get(target_id, (1.0, 1.0))
            return compute_trust(alpha, beta)

    def record_interaction(self, target_id: str, is_positive: bool):
        """
        Update knowledge about a target vehicle based on an interaction (Section III-A).
        """
        # Log for sliding window (always keep for analysis)
        if target_id not in self.interaction_logs:
            self.interaction_logs[target_id] = []
        self.interaction_logs[target_id].append(is_positive)

        # Update Model specific state
        if self.model_type == 'RTM':
            # RTM: trust = (old + outcome)/2
            # Initialize if not present
            if target_id not in self.rtm_trust:
                self.rtm_trust[target_id] = 0.5
            
            outcome_val = 1.0 if is_positive else 0.0
            self.rtm_trust[target_id] = (self.rtm_trust[target_id] + outcome_val) / 2.0

        elif self.model_type == 'BSED':
            # BSED: update counts
            c, t = self.interactions.get(target_id, (0.0, 0.0))
            t += 1.0
            if is_positive: c += 1.0
            self.interactions[target_id] = (c, t)

        else:
            # Bayesian
            alpha, beta = self.interactions.get(target_id, (1.0, 1.0))
            new_alpha, new_beta = update_parameters(alpha, beta, is_positive)
            self.interactions[target_id] = (new_alpha, new_beta)

    def get_windowed_local_trust(self, target_id: str, window_size: int = 20) -> float:
        """
        Calculate local trust using only the recent window of interactions.
        Uses simplistic ratio or re-calculated Beta expectation over the window.
        """
        logs = self.interaction_logs.get(target_id, [])
        if not logs:
            return 0.5 # Default
            
        window = logs[-window_size:]
        positives = sum(1 for x in window if x)
        negatives = len(window) - positives
        
        # Re-apply Beta expectation logic to just this window
        # Prior is still 1, 1 for the window scope
        w_alpha = 1.0 + positives
        w_beta = 1.0 + negatives
        return w_alpha / (w_alpha + w_beta)

    def get_interaction_stats(self, target_id: str):
        """Returns direct access to alpha/beta for RSU collection."""
        return self.interactions.get(target_id, (1.0, 1.0))

    def __repr__(self):
        return f"<Vehicle {self.id} | Malicious: {self.is_malicious} | Trust: {self.global_trust_score:.2f}>"
