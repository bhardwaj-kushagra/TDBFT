"""
Roadside Unit (RSU) Module.

Refactored to implement Section III-B and III-C (VehicleRank).
Replaces simple averaging with iterative trust propagation (PageRank-like).
"""

from typing import Dict, List, Tuple
import numpy as np
from .models import get_strategy, TrustStrategy

class RSU:
    def __init__(self, rsu_id: str, model_type: str = 'PROPOSED'):
        """
        Args:
            rsu_id (str): Unique identifier.
            model_type (str): Trust model to use.
        """
        self.id = rsu_id
        self.model_type = model_type
        self.strategy: TrustStrategy = get_strategy(model_type)
        
        # VehicleRank params
        self.alpha_damping = 0.85 # Damping factor alpha in eq T = alpha*S + ...
        self.max_iter = 100
        self.tol = 1e-6
        
        # Current Global Trust Vector (t)
        # Map: vehicle_id -> float
        self.global_trust_vector: Dict[str, float] = {}

    def compute_vehiclerank(self, all_vehicle_ids: List[str], adjacency_reports: Dict[str, Dict[str, Tuple[float, float]]]):
        """
        Computes Global Trust based on model type.
        Delegated to Strategy.
        """
        self.global_trust_vector = self.strategy.compute_global_trust(self, all_vehicle_ids, adjacency_reports)


    def get_global_trust(self, target_id: str) -> float:
        """Computes current global trust for a target."""
        return self.global_trust_vector.get(target_id, 0.5)

    def incorporate_peer_knowledge(self, other_knowledge: Dict[str, float]):
        """
        Merges trust knowledge from another RSU. 
        For VehicleRank, we simulate consensus by averaging the final vectors.
        (Abstracted Consensus Trust Update - Section IV)
        """
        for vid, score in other_knowledge.items():
            my_score = self.global_trust_vector.get(vid, 0.5)
            # Weighted average or simple average 
            self.global_trust_vector[vid] = (my_score + score) / 2.0
