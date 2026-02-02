"""
Vehicle Entity Module.

Defines the Vehicle class which maintains its own trust parameters
towards other vehicles (Local Trust) and its assigned Global Trust.
"""

from typing import Dict, Tuple
import random
from .models.base import TrustStrategy
from .models.strategies import get_strategy

class Vehicle:
    BEHAVIOR_HONEST = 'HONEST'
    BEHAVIOR_MALICIOUS = 'MALICIOUS'
    BEHAVIOR_SWING = 'SWING'
    
    # Configuration
    SWING_CYCLE_LENGTH = 50

    def __init__(self, vehicle_id: str, behavior_type: str = 'HONEST', model_type: str = 'PROPOSED', attack_intensity: float = 0.8):
        """
        Initialize a vehicle.
        
        Args:
            vehicle_id (str): Unique identifier.
            behavior_type (str): 'HONEST', 'MALICIOUS', or 'SWING'.
            model_type (str): Trust model to use.
            attack_intensity (str): Probability of bad behavior when acting explicitly malicious.
                                    0.2 = Low, 0.5 = Medium, 0.9 = High.
        """
        self.id = vehicle_id
        self.behavior_type = behavior_type
        self.model_type = model_type
        self.attack_intensity = attack_intensity
        self.is_malicious = (behavior_type != self.BEHAVIOR_HONEST)
        
        # Strategy for Trust Logic
        self.strategy: TrustStrategy = get_strategy(model_type)

        # Storage for different models
        # Bayesian/BTVR/COBATS: {target_id: (alpha, beta)}
        # BSED: {target_id: (correct, total)}
        self.interactions: Dict[str, Tuple[float, float]] = {}
        
        # RTM specific: {target_id: float}
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
        
        if self.behavior_type == self.BEHAVIOR_HONEST:
            # 99% success rate (accidents happen)
            return random.random() < 0.99
            
        elif self.behavior_type == self.BEHAVIOR_MALICIOUS:
            # Drop packets with probability = attack_intensity
            # e.g. Intensity 0.8 => 20% success (cooperation)
            return random.random() > self.attack_intensity
            
        elif self.behavior_type == self.BEHAVIOR_SWING:
            # Swing Attacker: Oscillates behavior.
            # Intensity can affect the bad phase severity
            is_good_phase = (step_count // self.SWING_CYCLE_LENGTH) % 2 == 0
            
            if is_good_phase:
                return random.random() < 0.99
            else:
                # In bad phase, behave maliciously with intensity
                return random.random() > self.attack_intensity
        
        return True

    def get_local_trust(self, target_id: str) -> float:
        """Calculate local trust for a specific target vehicle based on Model Type."""
        return self.strategy.get_local_trust(self, target_id)

    def record_interaction(self, target_id: str, is_positive: bool):
        """
        Update knowledge about a target vehicle based on an interaction (Section III-A).
        """
        # Log for sliding window (always keep for analysis)
        if target_id not in self.interaction_logs:
            self.interaction_logs[target_id] = []
        self.interaction_logs[target_id].append(is_positive)

        # Delegate to Strategy
        self.strategy.record_interaction(self, target_id, is_positive)

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

    def get_trust_reports(self) -> Dict:
        """
        Returns the appropriate trust report dictionary based on model type.
        Used by RSU for aggregation.
        """
        return self.strategy.get_trust_reports(self)

    def __repr__(self):
        return f"<Vehicle {self.id} | Malicious: {self.is_malicious} | Trust: {self.global_trust_score:.2f}>"
