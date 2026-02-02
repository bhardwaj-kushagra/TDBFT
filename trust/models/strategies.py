"""
Concrete Trust Strategies.

Implements the logic for PROPOSED, BTVR, BSED, RTM, COBATS, LT_PBFT.
"""
from typing import Dict, List, Any, Tuple
import numpy as np
from .base import TrustStrategy
from trust.bayesian import compute_trust, update_parameters

# ==============================================================================
# 1. PROPOSED / LT_PBFT (Bayesian + VehicleRank)
# ==============================================================================
class BayesianVehicleRankStrategy(TrustStrategy):
    """
    Used by PROPOSED and LT_PBFT.
    Local: Bayesian (Alpha/Beta).
    Global: VehicleRank (PageRank-like).
    """
    def record_interaction(self, vehicle, target_id: str, is_positive: bool):
        # vehicle.interactions stores {target_id: (alpha, beta)}
        alpha, beta = vehicle.interactions.get(target_id, (1.0, 1.0))
        new_alpha, new_beta = update_parameters(alpha, beta, is_positive)
        vehicle.interactions[target_id] = (new_alpha, new_beta)

    def get_local_trust(self, vehicle, target_id: str) -> float:
        alpha, beta = vehicle.interactions.get(target_id, (1.0, 1.0))
        return compute_trust(alpha, beta)

    def get_trust_reports(self, vehicle) -> Dict:
        return vehicle.interactions

    def compute_global_trust(self, rsu, all_ids: List[str], reports: Dict) -> Dict[str, float]:
        # VehicleRank Logic
        n = len(all_ids)
        if n == 0: return {}
        
        id_to_idx = {vid: i for i, vid in enumerate(all_ids)}
        M = np.zeros((n, n))
        
        # Build Matrix
        for reporter_id, targets in reports.items():
            if reporter_id not in id_to_idx: continue
            i = id_to_idx[reporter_id]
            for target_id, (alpha, beta) in targets.items():
                if target_id not in id_to_idx: continue
                j = id_to_idx[target_id]
                M[i, j] = compute_trust(alpha, beta)
                
        # Normalize
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 
        S = M / row_sums
        
        # Iterative PageRank
        t_vec = np.ones(n) / n
        teleport_val = (1.0 - rsu.alpha_damping) / n
        
        for _ in range(rsu.max_iter):
            t_new = rsu.alpha_damping * np.dot(t_vec, S) + teleport_val
            if np.linalg.norm(t_new - t_vec, 1) < rsu.tol:
                t_vec = t_new
                break
            t_vec = t_new
            
        result = {}
        for vid, idx in id_to_idx.items():
            result[vid] = t_vec[idx]
        return result


# ==============================================================================
# 2. BTVR / COBATS (Bayesian + Simple Averaging)
# ==============================================================================
class BayesianAveragingStrategy(TrustStrategy):
    """
    Used by BTVR and COBATS.
    Local: Bayesian (Alpha/Beta).
    Global: Simple Averaging of local trust scores.
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
        # Simple Averaging
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
# 3. BSED (Count-based + Simple Averaging)
# ==============================================================================
class BsedStrategy(TrustStrategy):
    """
    Used by BSED.
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
# 4. RTM (Direct Update + Simple Averaging)
# ==============================================================================
class RtmStrategy(TrustStrategy):
    """
    Used by RTM.
    Local: Recursive update score = (old + outcome)/2.
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
    if model_type in ['PROPOSED', 'LT_PBFT']:
        return BayesianVehicleRankStrategy()
    elif model_type in ['BTVR', 'COBATS']:
        return BayesianAveragingStrategy()
    elif model_type == 'BSED':
        return BsedStrategy()
    elif model_type == 'RTM':
        return RtmStrategy()
    else:
        # Default fallback
        return BayesianVehicleRankStrategy()
