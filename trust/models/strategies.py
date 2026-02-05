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
# 2. PBFT (Baseline)
# ==============================================================================
class PbftStrategy(TrustStrategy):
    """
    PBFT / BFT Baseline.
    Trust-agnostic. All nodes are trusted equally.
    Used to simulate classical BFT where 2/3 majority rules with no trust memory.
    """
    def record_interaction(self, vehicle, target_id: str, is_positive: bool):
        # PBFT does not track trust
        pass

    def get_local_trust(self, vehicle, target_id: str) -> float:
        # Everyone is 1.0 (or equal)
        return 1.0

    def get_trust_reports(self, vehicle) -> Dict:
        return {}

    def compute_global_trust(self, rsu, all_ids: List[str], reports: Dict) -> Dict[str, float]:
        # Return equal score for all to represent "No Trust Ranking"
        # Validator selection will be random/arbitrary among these
        return {vid: 1.0 for vid in all_ids}


# ==============================================================================
# 3. LT_PBFT (Lightweight Trust PBFT)
# ==============================================================================
class LtPbftStrategy(TrustStrategy):
    """
    Lightweight Trust-Based PBFT.
    Uses ONLY local trust (Bayesian) + Simple Averaging.
    Explicitly weaker than PBFT (adds overhead) and Proposed (no Graph trust).
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
        # REPLACED: Use Simple Averaging instead of PageRank
        # This matches "Lightweight" and ensures it's distinguishable from Proposed
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
# 4. BTVR (Bayesian Trust-based Voting Resource)
# ==============================================================================
class BtvrStrategy(TrustStrategy):
    """
    BTVR: Standard Bayesian Local + Simple Averaging Global.
    A baseline for "Basic Trust".
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
        # Logic: Simple Average
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
# 5. COBATS (Representative / Lite)
# ==============================================================================
class CobatsStrategy(TrustStrategy):
    """
    COBATS-lite: Confidence-based Trust.
    Uses Interaction Count as Confidence Weight.
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
        # Weighted Average based on Confidence (Total Interactions)
        # Weight = log(alpha + beta) to prevent dominance by old nodes? 
        # Or simple alpha + beta. Using linear sum for "Confidence".
        
        weighted_sums = {vid: 0.0 for vid in all_ids}
        total_weights = {vid: 0.0 for vid in all_ids}
        
        for reporter_id, targets in reports.items():
            for target_id, (alpha, beta) in targets.items():
                if target_id in weighted_sums:
                    trust_val = compute_trust(alpha, beta)
                    weight = alpha + beta # Confidence metric
                    
                    weighted_sums[target_id] += trust_val * weight
                    total_weights[target_id] += weight
                    
        result = {}
        for vid in all_ids:
            if total_weights[vid] > 0:
                result[vid] = weighted_sums[vid] / total_weights[vid]
            else:
                result[vid] = 0.5
        return result


# ==============================================================================
# 6. BSED (Behavior-based, Representative / Simplified)
# ==============================================================================
class BsedStrategy(TrustStrategy):
    """
    BSED-lite: Event-based / Count-based Trust.
    Includes "Data Consistency" check to penalize outliers.
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
        # 1. Initial Simple Average
        scores = {vid: [] for vid in all_ids}
        for reporter_id, targets in reports.items():
            for target_id, (c, t) in targets.items():
                if target_id in scores:
                    val = c/t if t > 0 else 0.5
                    scores[target_id].append(val)
        
        initial_means = {}
        for vid, val_list in scores.items():
            initial_means[vid] = sum(val_list) / len(val_list) if val_list else 0.5

        # 2. Credibility Weighting (Outlier Penalty)
        # If a reporter deviates significantly from the initial mean, reduce their weight.
        reporter_weights = {vid: 1.0 for vid in all_ids}
        
        for reporter_id, targets in reports.items():
            deviations = []
            for target_id, (c, t) in targets.items():
                val = c/t if t > 0 else 0.5
                consensus = initial_means.get(target_id, 0.5)
                deviations.append(abs(val - consensus))
            
            if deviations:
                avg_dev = sum(deviations) / len(deviations)
                # If avg deviation is high (>0.3), credibility drops
                if avg_dev > 0.3:
                    reporter_weights[reporter_id] = 0.5 # Weak penalty
                if avg_dev > 0.5:
                    reporter_weights[reporter_id] = 0.1 # Strong penalty

        # 3. Weighted Aggregation
        weighted_sums = {vid: 0.0 for vid in all_ids}
        total_weights = {vid: 0.0 for vid in all_ids}

        for reporter_id, targets in reports.items():
            w = reporter_weights.get(reporter_id, 1.0)
            for target_id, (c, t) in targets.items():
                if target_id in weighted_sums:
                    val = c/t if t > 0 else 0.5
                    weighted_sums[target_id] += val * w
                    total_weights[target_id] += w
                    
        result = {}
        for vid in all_ids:
            if total_weights[vid] > 0:
                result[vid] = weighted_sums[vid] / total_weights[vid]
            else:
                result[vid] = 0.5
        return result


# ==============================================================================
# 7. RTM (Reputation Trust Management - Stabilized)
# ==============================================================================
class RtmStrategy(TrustStrategy):
    """
    RTM: Recursive/Direct Trust Update with Smoothing.
    Fixed Instability: Uses alpha-smoothing instead of simple average.
    trust_new = alpha * trust_old + (1 - alpha) * observation
    """
    def record_interaction(self, vehicle, target_id: str, is_positive: bool):
        # vehicle.rtm_trust stores {target_id: float}
        current = vehicle.rtm_trust.get(target_id, 0.5)
        outcome = 1.0 if is_positive else 0.0
        
        # Smoothing factor alpha. Typical range [0.7, 0.9].
        # Higher alpha = more memory, less volatility.
        alpha = 0.8 
        
        vehicle.rtm_trust[target_id] = alpha * current + (1.0 - alpha) * outcome

    def get_local_trust(self, vehicle, target_id: str) -> float:
        return vehicle.rtm_trust.get(target_id, 0.5)

    def get_trust_reports(self, vehicle) -> Dict:
        return vehicle.rtm_trust

    def compute_global_trust(self, rsu, all_ids: List[str], reports: Dict) -> Dict[str, float]:
        # RTM typically transmits the reputation value directly.
        # Global aggregation is Average.
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
    elif model_type == 'PBFT' or model_type == 'BFT':
        return PbftStrategy()
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
