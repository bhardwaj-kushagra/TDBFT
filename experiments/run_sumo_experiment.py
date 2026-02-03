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
    import traci.constants as tc
    SUMO_AVAILABLE = True
except ImportError:
    # sys.exit("Could not import traci. Check SUMO_HOME and python path.")
    traci = None
    # Create a dummy class to satisfy static analysis when SUMO is missing
    class MockConstants:
        CMD_GET_VEHICLE_VARIABLE = 0
        VAR_POSITION = 0
    tc = MockConstants()
    SUMO_AVAILABLE = False

# Fix path to allow importing modules from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trust.simulator import Simulator
from blockchain.consensus_manager import ConsensusManager
from experiments.config import (
    SUMO_CONFIG_PATH, SUMO_STEPS, INTERACTION_RANGE, INTERACTION_PROBABILITY,
    SUMO_VEHICLE_COUNT, DEFAULT_MALICIOUS_PERCENT, DEFAULT_SWING_PERCENT,
    MODELS, RESULTS_DIR
)
from experiments.plots import (
    plot_trust_evolution, 
    plot_detection_metrics, 
    plot_swing_analysis,
    plot_comparative_trust, 
    plot_final_trust_distribution
)
import matplotlib.pyplot as plt

def run_sumo_logic(model_name, label=""):
    """
    Executes a SINGLE instance of SUMO simulation for a specific model.
    Returns the final vehicle state and detection history.
    """
    print(f"\n--- Starting SUMO Instance: {model_name} ---")
    
    if traci is None:
        print("Error: TraCI not imported. Is SUMO_HOME set?")
        return None

    # 1. Setup Trust Model
    sim = Simulator(num_vehicles=SUMO_VEHICLE_COUNT, 
                    percent_malicious=DEFAULT_MALICIOUS_PERCENT, 
                    percent_swing=DEFAULT_SWING_PERCENT, 
                    num_rsus=2,
                    model_type=model_name)
    
    # Initialize Consensus Manager
    consensus_mgr = ConsensusManager(model_name, sim.model.vehicles)
    
    # Mapping SUMO ID -> TrustModel ID (e.g., 'veh0' -> 'V000')
    sumo_to_trust_map = {}
    available_trust_ids = list(sim.model.vehicles.keys())
    available_trust_ids.sort() 
    
    # Tracking for plots
    swing_candidates = [v for v in sim.model.vehicles.values() if v.behavior_type == 'SWING']
    honest_candidates = [v for v in sim.model.vehicles.values() if v.behavior_type == 'HONEST']
    target_swing = swing_candidates[0] if swing_candidates else None
    observer = honest_candidates[0] if honest_candidates else None
    
    swing_global_history = []
    swing_local_history = []
    
    # Metric tracking for comparison
    detection_history = []
    mal_ids = [v.id for v in sim.model.vehicles.values() if v.is_malicious]

    if target_swing:
        print(f"Tracking Swing Attacker: {target_swing.id}")

    # 2. Start SUMO
    sumoBinary = "sumo-gui"
    sumoCmd = [sumoBinary, "-c", SUMO_CONFIG_PATH, "--start", "--quit-on-end"]
    
    try:
        traci.start(sumoCmd)
        print("TraCI Connected successfully.")
    except Exception as e:
        print(f"Error starting SUMO (check if sumo is in PATH): {e}")
        return None

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
                        traci.vehicle.subscribeContext(
                            s_id, 
                            tc.CMD_GET_VEHICLE_VARIABLE, 
                            INTERACTION_RANGE, 
                            [tc.VAR_POSITION]
                        )

            # --- Simulate Interactions ---
            processed_pairs = set()
            for s_id in active_sumo_ids:
                if s_id not in sumo_to_trust_map: continue
                neighbors = traci.vehicle.getContextSubscriptionResults(s_id)
                
                if neighbors:
                    trust_id_a = sumo_to_trust_map[s_id]
                    for neighbor_id in neighbors:
                        if neighbor_id == s_id or neighbor_id not in sumo_to_trust_map: continue
                        trust_id_b = sumo_to_trust_map[neighbor_id]
                        pair_key = tuple(sorted((trust_id_a, trust_id_b)))
                        if pair_key in processed_pairs: continue
                        processed_pairs.add(pair_key)
                        
                        if random.random() < INTERACTION_PROBABILITY:
                            observer_v = sim.model.vehicles[trust_id_a]
                            target_v = sim.model.vehicles[trust_id_b]
                            # Interactions
                            observer_v.record_interaction(target_v.id, target_v.perform_action(step))
                            target_v.record_interaction(observer_v.id, observer_v.perform_action(step))

            # --- Logic Updates ---
            if target_swing:
                swing_global_history.append(target_swing.global_trust_score)
                if observer:
                     swing_local_history.append(observer.get_windowed_local_trust(target_swing.id, window_size=10))

            consensus_mgr.attempt_consensus(step)
            sim.model.update_global_trust(sync_rsus=True)
            
            # Record Detection Count
            # Simple threshold check for the plot
            detected_count = 0
            for vid in mal_ids:
                 score = sim.model.vehicles[vid].global_trust_score
                 # Rough threshold for quick metric, actual logic is in plots
                 if score < 0.4: 
                     detected_count += 1
            detection_history.append(detected_count)

            step += 1
            if step % 50 == 0:
                print(f"SUMO Step {step}/{SUMO_STEPS} | Active Vehicles: {len(active_sumo_ids)}")

        traci.close()
        print(f"SUMO Simulation Finished for {model_name}.")
        
        return {
            'model': model_name,
            'vehicles': sim.model.vehicles,
            'detection_history': detection_history,
            'swing_history': (swing_global_history, swing_local_history) if target_swing else None
        }

    except Exception as e:
        print(f"Error during simulation: {e}")
        try: traci.close()
        except: pass
        return None

def run_sumo_simulation(compare=False):
    """
    Entry point for SUMO mode.
    If compare=True, runs all models sequentially.
    """
    if not compare:
        # --- SINGLE MODE ---
        print(">> Running Single SUMO Instance (model='PROPOSED')")
        result = run_sumo_logic('PROPOSED')
        if result:
            print("Generating Single Run Plots...")
            plot_trust_evolution(result['vehicles'])
            plot_detection_metrics(result['vehicles'])
            plot_final_trust_distribution(result['vehicles'])
            
            if result['swing_history']:
                gh, lh = result['swing_history']
                plot_swing_analysis(gh, lh)
    
    else:
        # --- COMPARE MODE ---
        print(f">> Running Comparative SUMO Suite (Models: {MODELS})")
        results = {}
        
        for model in MODELS:
            res = run_sumo_logic(model)
            if res:
                results[model] = res
        
        # Comparative Plotting
        if results:
            print("Generating Comparative SUMO Plots...")
            plt.figure(figsize=(10, 6))
            for model, data in results.items():
                plt.plot(data['detection_history'], label=model)
                
            plt.title("SUMO: Malicious Detection Over Time by Model")
            plt.xlabel("Simulation Step")
            plt.ylabel("Detected Count")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            os.makedirs(RESULTS_DIR, exist_ok=True)
            outfile = os.path.join(RESULTS_DIR, "sumo_comparative_detection.png")
            plt.savefig(outfile)
            plt.close()
            print(f"Saved {outfile}")

if __name__ == "__main__":
    run_sumo_simulation()
