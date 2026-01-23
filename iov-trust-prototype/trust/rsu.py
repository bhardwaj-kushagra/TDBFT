"""
Roadside Unit (RSU) Module.

Refactored to implement Section III-B and III-C (VehicleRank).
Replaces simple averaging with iterative trust propagation (PageRank-like).
"""
from typing import Dict, List, Tuple
import numpy as np
from .bayesian import compute_trust

class RSU:
    def __init__(self, rsu_id: str):
        """
        Args:
            rsu_id (str): Unique identifier.
        """
        self.id = rsu_id
        
        # VehicleRank params
        self.alpha_damping = 0.85 # Damping factor alpha in eq T = alpha*S + ...
        self.max_iter = 100
        self.tol = 1e-6
        
        # Current Global Trust Vector (t)
        # Map: vehicle_id -> float
        self.global_trust_vector: Dict[str, float] = {}

    def compute_vehiclerank(self, all_vehicle_ids: List[str], adjacency_reports: Dict[str, Dict[str, Tuple[float, float]]]):
        """
        Implements Global Trust Computation (VehicleRank) as per Section III-C.
        
        Steps:
        1. Construct Trust Graph (Adjacency Matrix M) from local trusts mij.
        2. Normalize rows to get Stochastic Matrix S (wij).
        3. Compute PageRank: t = t * T.
        
        Args:
            all_vehicle_ids: List of all participating vehicles.
            adjacency_reports: Dict[reporter_id -> {target_id -> (alpha, beta)}]
        """
        n = len(all_vehicle_ids)
        if n == 0:
            return

        # Map ID -> Index
        id_to_idx = {vid: i for i, vid in enumerate(all_vehicle_ids)}
        
        # 1. Build Trust Matrix M (mij)
        M = np.zeros((n, n))
        
        for reporter_id, targets in adjacency_reports.items():
            if reporter_id not in id_to_idx:
                continue
            i = id_to_idx[reporter_id]
            
            for target_id, stats in targets.items():
                if target_id not in id_to_idx:
                    continue
                j = id_to_idx[target_id]
                
                # Compute mij (Local Trust) - Section III-A
                alpha, beta = stats
                mij = compute_trust(alpha, beta)
                M[i, j] = mij
                
        # 2. Normalize to get S (wij) - Section III-B
        # wij = mij / Sum(mik)
        row_sums = M.sum(axis=1, keepdims=True)
        # Handle division by zero for nodes with no outgoing edges
        row_sums[row_sums == 0] = 1.0 
        S = M / row_sums
        
        # 3. Iterative PageRank (Section III-C)
        # Initial trust vector t(0) = 1/n
        t_vec = np.ones(n) / n
        
        # Damping term (1-alpha) * e * t_tilde^T
        # Assuming static reputation t_tilde is uniform 1/n -> teleport matrix E is all 1/n
        # The standard PageRank update is: t_new = alpha * t * S + (1-alpha) * (1/n)
        
        teleport_val = (1.0 - self.alpha_damping) / n
        
        for _ in range(self.max_iter):
            t_new = self.alpha_damping * np.dot(t_vec, S) + teleport_val
            
            # Check convergence
            if np.linalg.norm(t_new - t_vec, 1) < self.tol:
                t_vec = t_new
                break
            t_vec = t_new
            
        # Store results
        for vid, idx in id_to_idx.items():
            self.global_trust_vector[vid] = t_vec[idx]

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
