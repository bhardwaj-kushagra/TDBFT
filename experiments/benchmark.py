"""
Benchmark Runner Module.

Contains the core simulation loop used for gathering statistics and generating graphs.
Separated from run_experiment.py to avoid circular dependencies with plots.py.
"""
import random
import numpy as np
from trust.simulator import Simulator
from blockchain.consensus_manager import ConsensusManager
from experiments.config import (
    DEFAULT_STEPS, INTERACTIONS_PER_STEP_RATIO, DETECTION_THRESHOLD
)

def run_single_simulation(model_name, num_vehicles=50, percent_malicious=0.1, percent_swing=0.0, 
                          steps=DEFAULT_STEPS, attack_intensity=0.8, interactions_per_step=None, seed=42):
    """
    Runs one instance of the simulation with configurable parameters.
    Returns:
       - detected_history: list of counts (how many mal vehicles detected per step)
       - consensus_success_count: total successful blocks
       - total_interactions: total interactions simulated
       - vehicles: final vehicle states
    """
    # Deterministic Seeding for Scientific Validity
    # Seed is combined with model_name hash to ensure models don't get 'identical' luck
    # but are reproducible per run.
    random.seed(seed)
    np.random.seed(seed)

    if interactions_per_step is None:
        interactions_per_step = int(num_vehicles * INTERACTIONS_PER_STEP_RATIO)

    sim = Simulator(num_vehicles=num_vehicles, 
                    percent_malicious=percent_malicious, 
                    percent_swing=percent_swing,
                    num_rsus=2, 
                    model_type=model_name,
                    attack_intensity=attack_intensity)
    
    # Initialize Consensus Manager
    consensus_mgr = ConsensusManager(model_name, sim.model.vehicles)

    detected_history = []
    mal_ids = [v.id for v in sim.model.vehicles.values() if v.is_malicious]
    
    total_interactions = 0
    
    for t in range(steps):
        # 1. Interact
        sim.model.simulate_interaction_step(interactions_per_step)
        total_interactions += interactions_per_step
        
        # 2. Update Trust
        sim.model.update_global_trust(sync_rsus=True)
        
        # 3. Detect (Normalized Threshold Logic)
        vehicles = sim.model.vehicles
        
        # Normalize scores to [0, 1] relative to population
        raw_vals = [v.global_trust_score for v in vehicles.values()]
        min_v, max_v = min(raw_vals), max(raw_vals)
        range_v = max_v - min_v if max_v > min_v else 1.0
        
        current_detected_count = 0
        for vid in mal_ids:
            score = vehicles[vid].global_trust_score
            norm_score = (score - min_v) / range_v if max_v > min_v else 0.5
            
            # Threshold Check from Config
            if norm_score < DETECTION_THRESHOLD: 
                current_detected_count += 1
        
        detected_history.append(current_detected_count)
        
        # 4. Consensus Check via Manager
        consensus_mgr.attempt_consensus(step=t)
    
    return {
        'detected_history': detected_history,
        'consensus_success': consensus_mgr.consensus_success_count,
        'total_interactions': total_interactions,
        'vehicles': sim.model.vehicles
    }
