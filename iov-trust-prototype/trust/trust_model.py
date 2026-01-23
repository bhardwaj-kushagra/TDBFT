"""
Trust Model Simulator/Orchestrator.

Manages the simulation environment, vehicles, and RSUs.
Orchestrates the cycle of: Interaction -> Local Update -> Global Aggregation.
"""
import random
from typing import List, Dict
from .vehicle import Vehicle
from .rsu import RSU

class TrustModel:
    def __init__(self, num_vehicles: int, percent_malicious: float = 0.1, percent_swing: float = 0.05):
        self.vehicles: Dict[str, Vehicle] = {}
        self.rsu = RSU(rsu_id="RSU-01")
        self.step_count = 0
        
        num_malicious = int(num_vehicles * percent_malicious)
        num_swing = int(num_vehicles * percent_swing)
        
        # Create Vehicles
        for i in range(num_vehicles):
            vid = f"V{i:03d}"
            
            if i < num_malicious:
                behavior = Vehicle.BEHAVIOR_MALICIOUS
            elif i < num_malicious + num_swing:
                behavior = Vehicle.BEHAVIOR_SWING
            else:
                behavior = Vehicle.BEHAVIOR_HONEST
                
            self.vehicles[vid] = Vehicle(vid, behavior_type=behavior)

    def simulate_interaction_step(self, num_interactions: int = 50):
        """
        Randomly picks pairs of vehicles to interact.
        Malicious vehicles behave badly (send false messages / drop packets).
        """
        self.step_count += 1
        all_ids = list(self.vehicles.keys())
        
        for _ in range(num_interactions):
            observer_id = random.choice(all_ids)
            target_id = random.choice(all_ids)
            
            if observer_id == target_id:
                continue
            
            target = self.vehicles[target_id]
            observer = self.vehicles[observer_id]
            
            # Determine interaction outcome based on target's behavior
            is_good_interaction = target.perform_action(self.step_count)
            
            # Record local trust update
            observer.record_interaction(target.id, is_positive=is_good_interaction)

    def update_global_trust(self):
        """
        RSU collects reports from all vehicles about all other vehicles.
        """
        all_ids = list(self.vehicles.keys())
        
        for subject_id in all_ids:
            # Collect reports ABOUT subject_id FROM everyone else
            reports = []
            for reporter_id in all_ids:
                if reporter_id == subject_id:
                    continue
                
                reporter = self.vehicles[reporter_id]
                # If reporter has interacted with subject
                if subject_id in reporter.interactions:
                    trust_score = reporter.get_local_trust(subject_id)
                    reports.append(trust_score)
            
            # RSU processes these reports
            if reports:
                self.rsu.aggregate_trust_reports(subject_id, reports)
                
            # Update the subject vehicle's stored global score (for reference/validation)
            new_global = self.rsu.get_global_trust(subject_id)
            self.vehicles[subject_id].global_trust_score = new_global
            self.vehicles[subject_id].trust_history.append(new_global)

    def get_ranked_vehicles(self) -> List[Vehicle]:
        """Returns list of vehicles sorted by global trust (descending)."""
        v_list = list(self.vehicles.values())
        v_list.sort(key=lambda v: v.global_trust_score, reverse=True)
        return v_list
