"""
SUMO Control Script.

Uses TraCI to:
1. Connect to SUMO simulation.
2. Retrieve vehicle positions.
3. Determine proximity-based interactions.
4. Feed results into the Trust Model (replacing random interactions).

Prerequisites:
- SUMO installed and in PATH.
- SUMO_HOME environment variable set.
"""
import os
import sys
import math
import random
import time

# Ensure SUMO_HOME is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    # Set a default if possible or warn
    # sys.exit("please declare environment variable 'SUMO_HOME'")
    pass

try:
    import traci
except ImportError:
    # sys.exit("Could not import traci. Check SUMO_HOME and python path.")
    traci = None

# Fix path to allow importing modules from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trust.simulator import Simulator
from blockchain.consensus_manager import ConsensusManager
from experiments.config import (
    SUMO_CONFIG_PATH, SUMO_STEPS, INTERACTION_RANGE, INTERACTION_PROBABILITY,
    SUMO_VEHICLE_COUNT, DEFAULT_MALICIOUS_PERCENT, DEFAULT_SWING_PERCENT
)
from experiments.plots import (
    plot_trust_evolution, 
    plot_detection_metrics, 
    plot_swing_analysis,
    plot_comparative_trust, 
    plot_final_trust_distribution
)

def run_sumo_simulation():
    """
    Main loop for SUMO-driven simulation.
    """
    print("Initializing SUMO Experiment...")
    
    if traci is None:
        print("Error: TraCI not imported. Is SUMO_HOME set?")
        return

    # 1. Setup Trust Model
    sim = Simulator(num_vehicles=SUMO_VEHICLE_COUNT, 
                    percent_malicious=DEFAULT_MALICIOUS_PERCENT, 
                    percent_swing=DEFAULT_SWING_PERCENT, 
                    num_rsus=2)
    
    # Initialize Consensus Manager (replaces manual DAG/Validator logic)
    consensus_mgr = ConsensusManager('PROPOSED', sim.model.vehicles)
    
    # Mapping SUMO ID -> TrustModel ID (e.g., 'veh0' -> 'V000')
    sumo_to_trust_map = {}
    available_trust_ids = list(sim.model.vehicles.keys())
    available_trust_ids.sort() # Ensure deterministic order V000, V001...
    
    # Tracking for plots
    swing_candidates = [v for v in sim.model.vehicles.values() if v.behavior_type == 'SWING']
    honest_candidates = [v for v in sim.model.vehicles.values() if v.behavior_type == 'HONEST']
    target_swing = swing_candidates[0] if swing_candidates else None
    observer = honest_candidates[0] if honest_candidates else None
    swing_global_history = []
    swing_local_history = []
    
    if target_swing:
        print(f"Tracking Swing Attacker: {target_swing.id}")

    # 2. Start SUMO
    sumoBinary = "sumo-gui" 
    
    # Auto-start and quit-on-end are crtical to avoid hanging
    sumoCmd = [sumoBinary, "-c", SUMO_CONFIG_PATH, "--start", "--quit-on-end"]
    
    print(f"Starting connection to SUMO with {SUMO_CONFIG_PATH}...")
    try:
        traci.start(sumoCmd)
        print("TraCI Connected successfully.")
    except Exception as e:
        print(f"Error starting SUMO (check if sumo is in PATH): {e}")
        return

    step = 0
    
    try:
        while step < SUMO_STEPS:
            traci.simulationStep()
            sim.model.step_count = step
            
            # --- Map New Vehicles & Subscribe Context ---
            active_sumo_ids = traci.vehicle.getIDList()
            for s_id in active_sumo_ids:
                if s_id not in sumo_to_trust_map:
                    if available_trust_ids:
                        t_id = available_trust_ids.pop(0)
                        sumo_to_trust_map[s_id] = t_id
                        
                        # Optimization: Subscribe to context (Neighbors within Range)
                        # flags=0 to get default aggregation, but we just need IDs mainly
                        # VAR_POSITION tells SUMO what to return (we don't strictly use it if we trust the subscription)
                        traci.vehicle.subscribeContext(
                            s_id, 
                            traci.constants.CMD_GET_VEHICLE_VARIABLE, 
                            INTERACTION_RANGE, 
                            [traci.constants.VAR_POSITION]
                        )

            # --- Simulate Interactions via Context Subscription (O(N*k)) ---
            processed_pairs = set() # To avoid double processing A-B and B-A
            
            for s_id in active_sumo_ids:
                if s_id not in sumo_to_trust_map: continue
                
                # Get neighbors from subscription
                # Returns dict: {neighbor_id: {VAR_POSITION: ...}}
                neighbors = traci.vehicle.getContextSubscriptionResults(s_id)
                
                if neighbors:
                    trust_id_a = sumo_to_trust_map[s_id]
                    
                    for neighbor_id in neighbors:
                        if neighbor_id == s_id or neighbor_id not in sumo_to_trust_map:
                            continue
                            
                        trust_id_b = sumo_to_trust_map[neighbor_id]
                        
                        # Sort pair to ensure uniqueness for current step
                        pair_key = tuple(sorted((trust_id_a, trust_id_b)))
                        
                        if pair_key in processed_pairs:
                            continue
                            
                        processed_pairs.add(pair_key)
                        
                        # Interaction Logic
                        if random.random() < INTERACTION_PROBABILITY:
                            observer_v = sim.model.vehicles[trust_id_a]
                            target_v = sim.model.vehicles[trust_id_b]
                            
                            # Bidirectional Interaction
                            is_good_b = target_v.perform_action(sim.model.step_count)
                            observer_v.record_interaction(target_v.id, is_positive=is_good_b)
                            
                            is_good_a = observer_v.perform_action(sim.model.step_count)
                            target_v.record_interaction(observer_v.id, is_positive=is_good_a)

            # --- Trust Updates ---
            # --- Data Collection for Swing Plot ---
            if target_swing:
                swing_global_history.append(target_swing.global_trust_score)
                if observer:
                     # Calculate local trust based on observer's history w.r.t target (Windowed)
                     swing_local_history.append(observer.get_windowed_local_trust(target_swing.id, window_size=10))

            # --- Blockchain Consensus (via Manager) ---
            consensus_mgr.attempt_consensus(step)

            # --- Trust Updates ---
            sim.model.update_global_trust(sync_rsus=True)

            step += 1
            # More frequent logging to detect stalls
            if step % 50 == 0:
                print(f"SUMO Step {step}/{SUMO_STEPS} | Active Vehicles: {len(active_sumo_ids)}")

        traci.close()
        print("SUMO Simulation Finished.")
        
        # --- Plotting ---
        print("Generating Plots...")
        plot_trust_evolution(sim.model.vehicles)
        plot_detection_metrics(sim.model.vehicles)
        plot_comparative_trust(sim.model.vehicles)
        plot_final_trust_distribution(sim.model.vehicles)
        
        if target_swing and observer:
            plot_swing_analysis(swing_global_history, swing_local_history)

    except Exception as e:
        print(f"Error during simulation: {e}")
        try:
            traci.close()
        except:
            pass

if __name__ == "__main__":
    run_sumo_simulation()
