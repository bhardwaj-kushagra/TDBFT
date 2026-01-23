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
    def __init__(self, num_vehicles: int, percent_malicious: float = 0.1, percent_swing: float = 0.05, num_rsus: int = 2):
        self.vehicles: Dict[str, Vehicle] = {}
        self.rsus: List[RSU] = [RSU(rsu_id=f"RSU-{i+1:02d}") for i in range(num_rsus)]
        
        # Vehicle -> RSU assignment (Round Robin for simplicity)
        self.vehicle_rsu_map: Dict[str, RSU] = {}
        
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
            
            # Assign to RSU
            assigned_rsu = self.rsus[i % len(self.rsus)]
            self.vehicle_rsu_map[vid] = assigned_rsu

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

    def update_global_trust(self, sync_rsus: bool = True):
        """
        1. Vehicles report to assigned RSUs.
        2. RSUs update local knowledge.
        3. (Optional) RSUs sync with each other to form Global Consensus.
        """
        all_ids = list(self.vehicles.keys())
        
        # Phase 1: Reporting to assigned RSUs
        for subject_id in all_ids:
            # Gather reports about subject_id
            for reporter_id in all_ids:
                if reporter_id == subject_id:
                    continue
                
                reporter = self.vehicles[reporter_id]
                if subject_id in reporter.interactions:
                    trust_score = reporter.get_local_trust(subject_id)
                    
                    # Reporter sends this to ITS OWN assigned RSU
                    my_rsu = self.vehicle_rsu_map[reporter_id]
                    my_rsu.aggregate_trust_reports(subject_id, [trust_score])
        
        # Phase 2: RSU Synchronization (Consensus)
        if sync_rsus:
            # Simple Gossiping: Everyone merges with everyone
            # In simulation, we just iterate and merge
            for rsu in self.rsus:
                for other_rsu in self.rsus:
                    if rsu.id != other_rsu.id:
                        rsu.incorporate_peer_knowledge(other_rsu.global_knowledge)

        # Phase 3: Push Final Scores back to vehicles (for simulation/plotting)
        # We assume after Sync, all RSUs have converged (or close enough).
        # We take the score from the vehicle's assigned RSU as the authority.
        for vid, v in self.vehicles.items():
            assigned_rsu = self.vehicle_rsu_map[vid]
            new_global = assigned_rsu.get_global_trust(vid)
            v.global_trust_score = new_global
            v.trust_history.append(new_global)

    def get_ranked_vehicles(self) -> List[Vehicle]:
        """Returns list of vehicles sorted by global trust (descending)."""
        v_list = list(self.vehicles.values())
        v_list.sort(key=lambda v: v.global_trust_score, reverse=True)
        return v_list
