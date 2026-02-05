"""
Concrete Trust Strategies.

Implements the logic for PROPOSED, BTVR, BSED, RTM, COBATS, LT_PBFT.
Each model is implemented as a distinct class to allow for independent experimentation
and logic divergence during comparison.
"""
from typing import Dict, List, Any, Tuple
import numpy as np
from .base import TrustStrategy
from trust.bayesian import compute_trust, update_parameters

# ==============================================================================
# Helper for Matrix Operations (Shared math, but logic remains separate)
# ==============================================================================
def _run_pagerank(M: np.ndarray, n: int, damping: float, max_iter: int, tol: float) -> np.ndarray:
    """Helper to run the PageRank math on a matrix M."""
    # Normalize rows
    row_sums = M.sum(axis=1, keepdims=True)
    
    # Fix dangling nodes (rows with zero sum) by redistributing mass uniformly
    # Create the transition matrix S
    S = np.zeros_like(M)
    
    # Avoid division by zero
    non_zero_rows = (row_sums != 0).flatten()
    S[non_zero_rows] = M[non_zero_rows] / row_sums[non_zero_rows]
    
    # For dangling nodes (row_sum == 0), S[i] = 1/n
    # This prevents trust leakage
    zero_rows = ~non_zero_rows
    if np.any(zero_rows):
        S[zero_rows] = 1.0 / n
    
    # Iterative Power Method
    t_vec = np.ones(n) / n
    teleport_val = (1.0 - damping) / n
    
    for _ in range(max_iter):
        t_new = damping * np.dot(t_vec, S) + teleport_val
        if np.linalg.norm(t_new - t_vec, 1) < tol:
            t_vec = t_new
            break
        t_vec = t_new
    return t_vec

# ==============================================================================
# 1. PROPOSED (T-DBFT)
# ==============================================================================
class ProposedStrategy(TrustStrategy):
    """
    The Main Proposed Model (T-DBFT).
    Local: Bayesian (Alpha/Beta).
    Global: VehicleRank (PageRank-like) to mitigate sparse attacks.
    """
    def record_interaction(self, vehicle, target_id: str, is_positive: bool):
        alpha, beta = vehicle.interactions.get(target_id, (1.0, 1.0))
        new_alpha, new_beta = update_parameters(alpha, beta, is_positive)
        vehicle.interactions[target_id] = (new_alpha, new_beta)

    def get_local_trust(self, vehicle, target_id: str) -> float:
        alpha, beta = vehicle.interactions.get(target_id, (1.0, 1.0))
        return compute_trust(alpha, beta)

    def get_trust_reports(self, vehicle) -> Dict:
        return vehicle.interactions

    def compute_global_trust(self, rsu, all_ids: List[str], reports: Dict) -> Dict[str, float]:
        n = len(all_ids)
        if n == 0: return {}
        
        id_to_idx = {vid: i for i, vid in enumerate(all_ids)}
        M = np.zeros((n, n))
        
        # Build Matrix M for VehicleRank
        for reporter_id, targets in reports.items():
            if reporter_id not in id_to_idx: continue
            i = id_to_idx[reporter_id]
            for target_id, (alpha, beta) in targets.items():
                if target_id not in id_to_idx: continue
                j = id_to_idx[target_id]
                M[i, j] = compute_trust(alpha, beta)
        
        # Add small self-loop epsilon to ensure connectivity/trust in self (Fixes Issue 9)
        epsilon = 0.0001
        np.fill_diagonal(M, M.diagonal() + epsilon)
                
        # Run PageRank (Core feature of Proposed Model)
        # Using RSU params which can be tuned specifically for PROPOSED
        t_vec = _run_pagerank(M, n, rsu.alpha_damping, rsu.max_iter, rsu.tol)
            
        result = {}
        for vid, idx in id_to_idx.items():
            result[vid] = t_vec[idx]
        return result


# ==============================================================================
# 2. LT_PBFT
# ==============================================================================
class LtPbftStrategy(TrustStrategy):
    """
    Lightweight Trust-Based PBFT.
    
    NOTE: In the reference code, this shared logic with PROPOSED.
    We separate it here. If LT_PBFT should use standard averaging instead 
    of VehicleRank, you can replace the `compute_global_trust` body below.
    Currently: Implements VehicleRank (per legacy code matches).
    """
    def record_interaction(self, vehicle, target_id: str, is_positive: bool):
        alpha, beta = vehicle.interactions.get(target_id, (1.0, 1.0))
        new_alpha, new_beta = update_parameters(alpha, beta, is_positive)
        vehicle.interactions[target_id] = (new_alpha, new_beta)

    def get_local_trust(self, vehicle, target_id: str) -> float:
        alpha, beta = vehicle.interactions.get(target_id, (1.0, 1.0))
        return compute_trust(alpha, beta)

    def get_trust_reports(self, vehicle) -> Dict:
        return vehicle.interactions

    def compute_global_trust(self, rsu, all_ids: List[str], reports: Dict) -> Dict[str, float]:
        # Legacy Assumption: LT_PBFT was using the same aggregation as Proposed.
        # This code is now independent. 
        # TODO: If LT_PBFT should be "weaker", replace this with simple averaging.
        n = len(all_ids)
        if n == 0: return {}
        
        id_to_idx = {vid: i for i, vid in enumerate(all_ids)}
        M = np.zeros((n, n))
        
        for reporter_id, targets in reports.items():
            if reporter_id not in id_to_idx: continue
            i = id_to_idx[reporter_id]
            for target_id, (alpha, beta) in targets.items():
                if target_id not in id_to_idx: continue
                j = id_to_idx[target_id]
                M[i, j] = compute_trust(alpha, beta)
                
        # Run PageRank
        t_vec = _run_pagerank(M, n, rsu.alpha_damping, rsu.max_iter, rsu.tol)
            
        result = {}
        for vid, idx in id_to_idx.items():
            result[vid] = t_vec[idx]
        return result


# ==============================================================================
# 3. BTVR (Bayesian Trust-based Voting Resource)
# ==============================================================================
class BtvrStrategy(TrustStrategy):
    """
    BTVR: Standard Bayesian Local + Simple Averaging Global.
    """
    def record_interaction(self, vehicle, target_id: str, is_positive: bool):
        alpha, beta = vehicle.interactions.get(target_id, (1.0, 1.0))
        new_alpha, new_beta = update_parameters(alpha, beta, is_positive)
        vehicle.interactions[target_id] = (new_alpha, new_beta)

    def get_local_trust(self, vehicle, target_id: str) -> float:
        alpha, beta = vehicle.interactions.get(target_id, (1.0, 1.0))
        return compute_trust(alpha, beta)

    def get_trust_reports(self, vehicle) -> Dict:
        return vehicle.interactions

    def compute_global_trust(self, rsu, all_ids: List[str], reports: Dict) -> Dict[str, float]:
        # Logic: Simple Average of all available reports
        scores = {vid: [] for vid in all_ids}
        
        for reporter_id, targets in reports.items():
            for target_id, (alpha, beta) in targets.items():
                if target_id in scores:
                    scores[target_id].append(compute_trust(alpha, beta))
                    
        result = {}
        for vid, val_list in scores.items():
            if val_list:
                result[vid] = sum(val_list) / len(val_list)
            else:
                result[vid] = 0.5
        return result


# ==============================================================================
# 4. COBATS
# ==============================================================================
class CobatsStrategy(TrustStrategy):
    """
    COBATS: Confidence-based Trust.
    Currently behaves like BTVR (Average). 
    TODO: Add confidence-weighting based on total interactions (alpha+beta).
    """
    def record_interaction(self, vehicle, target_id: str, is_positive: bool):
        alpha, beta = vehicle.interactions.get(target_id, (1.0, 1.0))
        new_alpha, new_beta = update_parameters(alpha, beta, is_positive)
        vehicle.interactions[target_id] = (new_alpha, new_beta)

    def get_local_trust(self, vehicle, target_id: str) -> float:
        # COBATS often considers uncertainty, but basic scalar is mean
        alpha, beta = vehicle.interactions.get(target_id, (1.0, 1.0))
        return compute_trust(alpha, beta)

    def get_trust_reports(self, vehicle) -> Dict:
        return vehicle.interactions

    def compute_global_trust(self, rsu, all_ids: List[str], reports: Dict) -> Dict[str, float]:
        # Logic: Simple Average (Same baseline as BTVR for now)
        scores = {vid: [] for vid in all_ids}
        
        for reporter_id, targets in reports.items():
            for target_id, (alpha, beta) in targets.items():
                if target_id in scores:
                    scores[target_id].append(compute_trust(alpha, beta))
                    
        result = {}
        for vid, val_list in scores.items():
            if val_list:
                result[vid] = sum(val_list) / len(val_list)
            else:
                result[vid] = 0.5
        return result


# ==============================================================================
# 5. BSED (Behavior-based)
# ==============================================================================
class BsedStrategy(TrustStrategy):
    """
    BSED: Event-based / Count-based Trust.
    Local: Ratio (Correct / Total).
    Global: Simple Averaging.
    """
    def record_interaction(self, vehicle, target_id: str, is_positive: bool):
        # vehicle.interactions stores {target_id: (correct, total)}
        c, t = vehicle.interactions.get(target_id, (0.0, 0.0))
        t += 1.0
        if is_positive: c += 1.0
        vehicle.interactions[target_id] = (c, t)

    def get_local_trust(self, vehicle, target_id: str) -> float:
        c, t = vehicle.interactions.get(target_id, (0.0, 0.0))
        if t == 0: return 0.5
        return c / t

    def get_trust_reports(self, vehicle) -> Dict:
        return vehicle.interactions

    def compute_global_trust(self, rsu, all_ids: List[str], reports: Dict) -> Dict[str, float]:
        scores = {vid: [] for vid in all_ids}
        
        for reporter_id, targets in reports.items():
            for target_id, (c, t) in targets.items():
                if target_id in scores:
                    val = c/t if t > 0 else 0.5
                    scores[target_id].append(val)
                    
        result = {}
        for vid, val_list in scores.items():
            if val_list:
                result[vid] = sum(val_list) / len(val_list)
            else:
                result[vid] = 0.5
        return result


# ==============================================================================
# 6. RTM (Reputation Trust Management)
# ==============================================================================
class RtmStrategy(TrustStrategy):
    """
    RTM: Recursive/Direct Trust Update.
    Local: score = (old + outcome)/2.
    Global: Simple Averaging.
    """
    def record_interaction(self, vehicle, target_id: str, is_positive: bool):
        # vehicle.rtm_trust stores {target_id: float}
        current = vehicle.rtm_trust.get(target_id, 0.5)
        outcome = 1.0 if is_positive else 0.0
        vehicle.rtm_trust[target_id] = (current + outcome) / 2.0

    def get_local_trust(self, vehicle, target_id: str) -> float:
        return vehicle.rtm_trust.get(target_id, 0.5)

    def get_trust_reports(self, vehicle) -> Dict:
        return vehicle.rtm_trust

    def compute_global_trust(self, rsu, all_ids: List[str], reports: Dict) -> Dict[str, float]:
        scores = {vid: [] for vid in all_ids}
        
        for reporter_id, targets in reports.items():
            for target_id, val in targets.items():
                if target_id in scores:
                    scores[target_id].append(val)
                    
        result = {}
        for vid, val_list in scores.items():
            if val_list:
                result[vid] = sum(val_list) / len(val_list)
            else:
                result[vid] = 0.5
        return result

# ==============================================================================
# Factory
# ==============================================================================
def get_strategy(model_type: str) -> TrustStrategy:
    """Returns a fresh instance of the requested strategy."""
    if model_type == 'PROPOSED':
        return ProposedStrategy()
    elif model_type == 'LT_PBFT':
        return LtPbftStrategy()
    elif model_type == 'BTVR':
        return BtvrStrategy()
    elif model_type == 'COBATS':
        return CobatsStrategy()
    elif model_type == 'BSED':
        return BsedStrategy()
    elif model_type == 'RTM':
        return RtmStrategy()
    else:
        # Default fallback
        print(f"Warning: Unknown model_type '{model_type}', defaulting to ProposedStrategy.")
        return ProposedStrategy()
