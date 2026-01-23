"""
Roadside Unit (RSU) Module.

The RSU is responsible for aggregating local trust scores from vehicles
to form a Global Trust rating. It applies a forgetting factor to 
decay old data.
"""
from typing import Dict, List
from .bayesian import compute_trust

class RSU:
    def __init__(self, rsu_id: str, forgetting_factor: float = 0.95):
        """
        Args:
            rsu_id (str): Unique identifier.
            forgetting_factor (float): Factor to decay past cumulative trust (0 < lambda <= 1).
                                      0.95 means retain 95% of past info, 5% decay.
        """
        self.id = rsu_id
        self.forgetting_factor = forgetting_factor
        
        # Global Trust Database: vehicle_id -> (cumulative_alpha, cumulative_beta)
        self.global_knowledge: Dict[str, List[float]] = {}

    def aggregate_trust_reports(self, target_id: str, reports: List[float]):
        """
        Aggregates trust reports (local trust scores) from reporting vehicles.
        
        In this prototype, we simplify aggregation:
        New Global Alpha += Sum of (Reporter Trust * Reported Alpha) ? 
        
        A simpler approach for this prototype:
        1. Convert reports back to implied (alpha, beta) or just weight them.
        2. Update the global (alpha, beta) for the target.
        
        SIMPLIFIED STRATEGY:
        Global Trust is updated by treating the average of reports as a 'batch interaction'.
        
        new_alpha = old_alpha * lambda + sum(reported_alphas)
        new_beta  = old_beta  * lambda + sum(reported_betas)
        """
        
        # Init if not exists
        if target_id not in self.global_knowledge:
            self.global_knowledge[target_id] = [1.0, 1.0] # alpha, beta

        current_alpha, current_beta = self.global_knowledge[target_id]
        
        # Apply forgetting factor to history
        current_alpha *= self.forgetting_factor
        current_beta *= self.forgetting_factor
        
        # Aggregate new evidence
        # In a real system, we'd weight by repuation of reporter. 
        # Here we just sum the evidence provided by reports.
        # But reports are usually just "scores".
        # Let's assume reports are effectively vote counts.
        
        # Alternative Logic:
        # report > 0.5 is a "positive" vote, < 0.5 is "negative"
        pos_votes = 0.0
        neg_votes = 0.0
        
        for score in reports:
            if score > 0.5:
                pos_votes += 1.0
            elif score < 0.5:
                neg_votes += 1.0
            # 0.5 is neutral, ignore
            
        # Update global parameters
        current_alpha += pos_votes
        current_beta += neg_votes
        
        # Store back
        self.global_knowledge[target_id] = [current_alpha, current_beta]

    def get_global_trust(self, target_id: str) -> float:
        """Computes current global trust for a target."""
        if target_id not in self.global_knowledge:
            return 0.5
        alpha, beta = self.global_knowledge[target_id]
        return compute_trust(alpha, beta)

    def incorporate_peer_knowledge(self, other_knowledge: Dict[str, List[float]]):
        """
        Merges trust knowledge from another RSU.
        Uses a weighted average approach.
        """
        for vid, params in other_knowledge.items():
            other_alpha, other_beta = params
            
            if vid not in self.global_knowledge:
                self.global_knowledge[vid] = [1.0, 1.0]
                
            my_params = self.global_knowledge[vid]
            
            # Merge Strategy: Average the parameters (Consensus)
            # new = (mine + other) / 2
            my_params[0] = (my_params[0] + other_alpha) / 2
            my_params[1] = (my_params[1] + other_beta) / 2

