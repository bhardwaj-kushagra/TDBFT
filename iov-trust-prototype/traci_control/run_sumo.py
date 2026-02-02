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
from blockchain.dag import DAG
from blockchain.validator import select_validators, check_consensus_weighted
from experiments.plots import (
    plot_trust_evolution, 
    plot_detection_metrics, 
    plot_swing_analysis,
    plot_comparative_trust, 
    plot_final_trust_distribution
)

INTERACTION_RANGE = 100.0  # meters
INTERACTION_PROBABILITY = 0.3 # Chance to interact if in range
SUMO_STEPS = 500 # Duration

def get_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def run_sumo_simulation():
    """
    Main loop for SUMO-driven simulation.
    """
    print("Initializing SUMO Experiment...")
    
    if traci is None:
        print("Error: TraCI not imported. Is SUMO_HOME set?")
        return

    # 1. Setup Trust Model
    # 20 Vehicles, similar params to run_experiment
    sim = Simulator(num_vehicles=20, percent_malicious=0.15, percent_swing=0.10, num_rsus=2)
    dags = [DAG(), DAG()]
    
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
    # Look for config in ../sumo/config.sumocfg
    # options: "sumo-gui" (visual) or "sumo" (faster/headless)
    sumoBinary = "sumo-gui" 
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(base_dir, "..", "sumo", "config.sumocfg"))
    
    # Auto-start and quit-on-end are crtical to avoid hanging
    sumoCmd = [sumoBinary, "-c", config_path, "--start", "--quit-on-end"]
    
    print(f"Starting connection to SUMO with {config_path}...")
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
            
            # --- Map New Vehicles ---
            active_sumo_ids = traci.vehicle.getIDList()
            for s_id in active_sumo_ids:
                if s_id not in sumo_to_trust_map:
                    if available_trust_ids:
                        t_id = available_trust_ids.pop(0)
                        sumo_to_trust_map[s_id] = t_id
                        # print(f"Mapped SUMO:{s_id} -> Trust:{t_id}")
            
            # --- Simulate Interactions based on Proximity ---
            vehicle_positions = {}
            for s_id in active_sumo_ids:
                if s_id in sumo_to_trust_map:
                    try:
                        vehicle_positions[sumo_to_trust_map[s_id]] = traci.vehicle.getPosition(s_id)
                    except Exception:
                        pass
            
            # Pairwise check (O(N^2) but N is small ~20)
            v_ids = list(vehicle_positions.keys())
            for i in range(len(v_ids)):
                for j in range(i + 1, len(v_ids)):
                    vid_a = v_ids[i]
                    vid_b = v_ids[j]
                    
                    pos_a = vehicle_positions[vid_a]
                    pos_b = vehicle_positions[vid_b]
                    
                    if get_distance(pos_a, pos_b) < INTERACTION_RANGE:
                        if random.random() < INTERACTION_PROBABILITY:
                            # TRIGGER INTERACTION
                            
                            observer_v = sim.model.vehicles[vid_a]
                            target_v = sim.model.vehicles[vid_b]
                            
                            # A -> B Interaction (A observes B)
                            # B acts based on its behavior profile
                            is_good_b = target_v.perform_action(sim.model.step_count)
                            observer_v.record_interaction(target_v.id, is_positive=is_good_b)
                            
                            # B -> A Interaction (B observes A)
                            # A acts based on its behavior profile
                            is_good_a = observer_v.perform_action(sim.model.step_count)
                            target_v.record_interaction(observer_v.id, is_positive=is_good_a)

            # --- Trust Updates ---
            # --- Data Collection for Swing Plot ---
            if target_swing:
                swing_global_history.append(target_swing.global_trust_score)
                if observer:
                     # Calculate local trust based on observer's history w.r.t target (Windowed)
                     # Using window_size=10 per latest fix
                     swing_local_history.append(observer.get_windowed_local_trust(target_swing.id, window_size=10))

            # --- Blockchain Consensus ---
            ranked_vehicles = sim.model.get_ranked_vehicles()
            committee_size = 5
            committee = select_validators(ranked_vehicles, top_n=committee_size)
            
            if committee:
                consensus_reached = check_consensus_weighted(committee)
                if consensus_reached:
                    v1 = committee[0]
                    snapshot1 = {v.id: v.global_trust_score for v in ranked_vehicles}
                    dags[0].add_block(data=snapshot1, validator_id=v1.id)
                    
                    if len(committee) > 1:
                        v2 = committee[1]
                        dags[1].add_block(data=snapshot1, validator_id=v2.id)
                    
                    dags[0].merge_with(dags[1])
                    dags[1].merge_with(dags[0])

            # --- Trust Updates ---
            sim.model.update_global_trust(sync_rsus=True)

            step += 1
            # More frequent logging to detect stalls
            if step % 10 == 0:
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
