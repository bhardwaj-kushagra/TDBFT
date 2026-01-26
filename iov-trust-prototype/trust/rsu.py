"""
Roadside Unit (RSU) Module.

Refactored to implement Section III-B and III-C (VehicleRank).
Replaces simple averaging with iterative trust propagation (PageRank-like).
"""
from typing import Dict, List, Tuple
import numpy as np
from .bayesian import compute_trust

class RSU:
    def __init__(self, rsu_id: str, model_type: str = 'PROPOSED'):
        """
        Args:
            rsu_id (str): Unique identifier.
            model_type (str): Trust model to use.
        """
        self.id = rsu_id
        self.model_type = model_type
        
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
        PROPOSED/LT_PBFT: VehicleRank
        Others: Simple Averaging of Local Trusts
        
        Args:
            all_vehicle_ids: List of all participating vehicles.
            adjacency_reports: Dict[reporter_id -> {target_id -> (alpha, beta) or other}]
        """
        n = len(all_vehicle_ids)
        if n == 0:
            return

        # Map ID -> Index
        id_to_idx = {vid: i for i, vid in enumerate(all_vehicle_ids)}
        
        # 1. Build Trust Matrix M (mij)
        M = np.zeros((n, n))
        
        # Collect all reports into M
        for reporter_id, targets in adjacency_reports.items():
            if reporter_id not in id_to_idx:
                continue
            i = id_to_idx[reporter_id]
            
            for target_id, stats in targets.items():
                if target_id not in id_to_idx:
                    continue
                j = id_to_idx[target_id]
                
                # Compute mij (Local Trust) based on mode
                if self.model_type == 'BSED':
                    # Stats is (correct, total)
                    c, t = stats
                    mij = c / t if t > 0 else 0.5
                elif self.model_type == 'RTM':
                    # Stats is trust_value directly (passed as tuple for compatibility?) 
                    # Actually Vehicle.interactions stores tuples. RTM stores scalar in rtm_trust.
                    # We need to ensure trust_model passes the right thing.
                    # Assuming we adapted trust_model to pass whatever is needed.
                    mij = stats[0] # IF packed as tuple
                else: 
                     # Bayesian models
                    alpha, beta = stats
                    mij = compute_trust(alpha, beta)
                
                M[i, j] = mij
                
        # ---------------------------------------------------------
        # ALGORITHM BRANCHING
        # ---------------------------------------------------------
        
        if self.model_type in ['PROPOSED', 'LT_PBFT']: 
            # === VehicleRank Logic ===
            
            # 2. Normalize to get S (wij)
            row_sums = M.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0 
            S = M / row_sums
            
            # 3. Iterative PageRank
            t_vec = np.ones(n) / n
            teleport_val = (1.0 - self.alpha_damping) / n
            
            for _ in range(self.max_iter):
                t_new = self.alpha_damping * np.dot(t_vec, S) + teleport_val
                if np.linalg.norm(t_new - t_vec, 1) < self.tol:
                    t_vec = t_new
                    break
                t_vec = t_new
            
            # Store results
            for vid, idx in id_to_idx.items():
                self.global_trust_vector[vid] = t_vec[idx]
        
        else:
            # === Simple Averaging (BTVR, BSED, RTM, COBATS) ===
            # Global Trust of j = Average of M[:, j] (ignoring zeros? or just sum/count)
            # Typically: Average of all reporters who reported interactions.
            
            for j in range(n):
                vid = all_vehicle_ids[j]
                col = M[:, j]
                # Filter out zero entries (assuming 0.0 means no interaction/neutral in sparse matrix, 
                # but careful if valid trust is 0.0. Here valid trust is [0,1]. 
                # Better: check adjacency_reports structure for existence)
                
                # Re-scan efficiently? Or just iterate reports again.
                # Actually, M is dense zeros. 
                # Let's compute average based on non-zero entries or adjacency list.
                
                # Better: Compute from adjacency list to be accurate about "who reported"
                total_trust = 0.0
                count = 0
                
                # Reverse lookup: who reported on vid?
                # This is O(V^2) naively. M is O(N^2) anyway.
                # Use M column. But wait, we initialized M with 0.
                # If trust can be 0, we can't distinguish "Untrusted" from "No Report".
                # For this prototype, assume trust > 0. Or assume 0.5 is default.
                
                # Let's iterate adjacency_reports to stay correct
                for reporter_id, targets in adjacency_reports.items():
                     if vid in targets:
                         # Re-retrieve trust
                         if reporter_id not in id_to_idx: continue
                         i = id_to_idx[reporter_id]
                         total_trust += M[i, j]
                         count += 1
                
                if count > 0:
                    self.global_trust_vector[vid] = total_trust / count
                else:
                    self.global_trust_vector[vid] = 0.5

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
