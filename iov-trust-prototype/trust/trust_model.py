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
    def __init__(self, num_vehicles: int, percent_malicious: float = 0.1, percent_swing: float = 0.05, num_rsus: int = 2, model_type: str = 'PROPOSED', attack_intensity: float = 0.8):
        self.vehicles: Dict[str, Vehicle] = {}
        self.rsus: List[RSU] = [RSU(rsu_id=f"RSU-{i+1:02d}", model_type=model_type) for i in range(num_rsus)]
        
        # Vehicle -> RSU assignment (Round Robin for simplicity)
        self.vehicle_rsu_map: Dict[str, RSU] = {}
        
        self.step_count = 0
        self.model_type = model_type
        
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
                
            self.vehicles[vid] = Vehicle(vid, behavior_type=behavior, model_type=model_type, attack_intensity=attack_intensity)
            
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
        Orchestrates Section III-C (VehicleRank) and Section IV (Consensus Sync).
        
        1. Collect all local trust reports (Section III-A/B).
        2. RSUs compute VehicleRank (Section III-C).
        3. RSUs sync finalized scores (Section IV).
        """
        all_ids = list(self.vehicles.keys())
        
        # Phase 1: Data Collection
        # Gather the full trust graph (Adjacency Matrix inputs)
        # In a real scenario, vehicles send reports to RSU.
        # Here we simulate RSU pulling observations from its assigned region.
        
        for rsu in self.rsus:
            # RSU builds its view of the graph
            # This RSU manages a subset of vehicles, but needs full graph for PageRank?
            # Usually PageRank is global. 
            # Simplified: Each RSU collects ALL info (shared ledger/gossiping) 
            # OR computes based on partial view.
            # Paper implies RSUs are Edge nodes. Let's assume full visibility for prototype.
            
            adjacency_reports = {}
            for reporter_id in all_ids:
                reporter = self.vehicles[reporter_id]
                adjacency_reports[reporter_id] = reporter.interactions
                
            # Phase 2: Compute VehicleRank (Section III-C)
            rsu.compute_vehiclerank(all_ids, adjacency_reports)
        
        # Phase 3: RSU Synchronization (Consensus) - Section IV
        if sync_rsus:
            # Sync the computed Global Trust Vectors
            for rsu in self.rsus:
                for other_rsu in self.rsus:
                    if rsu.id != other_rsu.id:
                        rsu.incorporate_peer_knowledge(other_rsu.global_trust_vector)

        # Phase 4: Push Final Scores back to vehicles
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
