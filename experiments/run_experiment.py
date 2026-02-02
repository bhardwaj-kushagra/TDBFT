"""
Main Experiment Runner.

Integrates Trust Model, Simulator, and Mock Blockchain.
Supports:
1. SUMO-based realistic simulation (Default).
2. Comparative Study (Paper Graphs 1-7) via --paper flag.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

# Fix path to allow importing modules from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trust.simulator import Simulator
from blockchain.consensus_manager import ConsensusManager
from experiments.config import (
    MODELS, COLORS, LINE_STYLES, LINE_WIDTHS, get_style,
    DEFAULT_STEPS, INTERACTIONS_PER_STEP_RATIO, DETECTION_THRESHOLD,
    RESULTS_DIR
)
from experiments.plots import (
    plot_trust_evolution, 
    plot_detection_metrics, 
    plot_swing_analysis,
    plot_comparative_trust, 
    plot_final_trust_distribution,
    plot_dag_structure,
    normalize_histories
) 

# Import SUMO runner (added for integration)
try:
    from experiments.run_sumo_experiment import run_sumo_simulation
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
# (Moved to experiments.config)
# See imports above.


# ==========================================
# SIMULATION ENGINE
# ==========================================
def run_single_simulation(model_name, num_vehicles=50, percent_malicious=0.1, percent_swing=0.0, 
                          steps=DEFAULT_STEPS, attack_intensity=0.8, interactions_per_step=None):
    """
    Runs one instance of the simulation with configurable parameters.
    Returns:
       - detected_history: list of counts (how many mal vehicles detected per step)
       - consensus_success_count: total successful blocks
       - total_interactions: total interactions simulated
       - vehicles: final vehicle states
    """
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

# ==========================================
# GRAPH GENERATION (PAPER ARTIFACTS)
# ==========================================

def generate_graph_1(out_dir="results"):
    """Graph 1: IoV Traffic (Interactions) vs Malicious Detected"""
    print("Generating Graph 1...")
    steps = 80 
    interactions_per_step = 50 
    
    plt.figure(figsize=(10, 6))
    
    for model in MODELS:
        res = run_single_simulation(model, steps=steps, interactions_per_step=interactions_per_step, percent_malicious=0.1)
        
        # X-axis: Interactions (Step * IntPerStep)
        x_axis = [i * interactions_per_step for i in range(1, steps + 1)]
        y_axis = res['detected_history']
        
        plt.plot(x_axis, y_axis, **get_style(model))
        
    plt.title("IoV Traffic vs Malicious Vehicle Behavior")
    plt.xlabel("Total Number of Interactions")
    plt.ylabel("Total Number of Malicious Vehicles Detected")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{out_dir}/graph1_traffic_vs_detection.png")
    plt.close()

def generate_graph_2(out_dir="results"):
    """Graph 2: Interactions vs Malicious Vehicles (10%, 20%, 30%)"""
    print("Generating Graph 2...")
    ratios = [0.1, 0.2, 0.3]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    steps = 60
    
    for i, ratio in enumerate(ratios):
        ax = axes[i]
        for model in MODELS:
            res = run_single_simulation(model, steps=steps, percent_malicious=ratio, interactions_per_step=50)
            x_axis = [k * 50 for k in range(1, steps + 1)]
            y_axis = res['detected_history']
            
            style = get_style(model)
            ax.plot(x_axis, y_axis, **style)
            
        ax.set_title(f"Attacker Ratio: {int(ratio*100)}%")
        ax.set_xlabel("Interactions")
        if i == 0:
            ax.set_ylabel("Detected Malicious Vehicles")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(f"{out_dir}/graph2_robustness_scenarios.png")
    plt.close()

def generate_graph_3_and_4(out_dir="results"):
    """Graph 3 & 4: TPS and Average Throughput vs Network Size"""
    print("Generating Graph 3 & 4...")
    sizes = [10, 20, 30, 40, 50, 60]
    results_tps = {m: [] for m in MODELS}
    steps = 50
    
    for N in sizes:
        for model in MODELS:
            res = run_single_simulation(model, num_vehicles=N, steps=steps)
            success_count = res['consensus_success']
            
            # Theoretical Complexity Penalty
            if model == 'LT_PBFT':
                overhead = (N ** 2) / 20.0 # O(N^2)
            elif model == 'COBATS':
                overhead = (N * 2) / 10.0
            elif model == 'PROPOSED':
                overhead = (N) / 10.0 # O(N)
            else:
                 overhead = (N * 1.5) / 10.0
            
            raw_throughput = success_count / steps 
            tps = (raw_throughput * 1000) / (10 + overhead)
            results_tps[model].append(tps)
            
    # Graph 3: Line Plot
    plt.figure(figsize=(10, 6))
    for model in MODELS:
        plt.plot(sizes, results_tps[model], **get_style(model))
    plt.title("Throughput (TPS) vs Network Size")
    plt.xlabel("Network Size (Number of Vehicles)")
    plt.ylabel("Throughput (TPS)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{out_dir}/graph3_throughput_line.png")
    plt.close()
    
    # Graph 4: Bar Chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(sizes))
    width = 0.12 
    for i, model in enumerate(MODELS):
        offset = (i - len(MODELS)/2) * width + width/2
        c = COLORS.get(model, 'gray')
        plt.bar(x + offset, results_tps[model], width, label=model, color=c)
    plt.title("Average Throughput vs Network Size")
    plt.xlabel("Network Size")
    plt.ylabel("Average Throughput (TPS)")
    plt.xticks(x, sizes)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{out_dir}/graph4_throughput_bar.png")
    plt.close()

def generate_graph_5(out_dir="results"):
    """Graph 5: Swing Attack Success vs Intensity"""
    print("Generating Graph 5...")
    intensities = [0.2, 0.5, 0.8, 0.95]
    labels_x = ['Low', 'Medium', 'High', 'Very High']
    res_success = {m: [] for m in MODELS}
    
    for intensity in intensities:
        for model in MODELS:
            sim_res = run_single_simulation(model, percent_malicious=0.0, percent_swing=0.2, 
                                            attack_intensity=intensity, steps=80)
            total_rounds = 80
            failures = total_rounds - sim_res['consensus_success']
            rate = (failures / total_rounds) * 100
            res_success[model].append(rate)

    plt.figure(figsize=(10, 6))
    for model in MODELS:
        plt.plot(labels_x, res_success[model], **get_style(model))
    plt.title("Success Rate of Swing Attacks vs Attack Intensity")
    plt.xlabel("Attack Intensity")
    plt.ylabel("Attack Success Rate (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{out_dir}/graph5_swing_attack_success.png")
    plt.close()
    return intensities, labels_x

def generate_graph_6(intensities, labels_x, out_dir="results"):
    """Graph 6: Internal Attack Success vs Intensity"""
    print("Generating Graph 6...")
    res_success = {m: [] for m in MODELS}
    
    for intensity in intensities:
        for model in MODELS:
            sim_res = run_single_simulation(model, percent_malicious=0.2, percent_swing=0.0,
                                            attack_intensity=intensity, steps=80)
            total = 80
            failures = total - sim_res['consensus_success']
            rate = (failures / total) * 100
            res_success[model].append(rate)
            
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels_x))
    width = 0.12
    for i, model in enumerate(MODELS):
        offset = (i - len(MODELS)/2) * width + width/2
        c = COLORS.get(model, 'gray')
        plt.bar(x + offset, res_success[model], width, label=model, color=c)
    plt.title("Internal Attack Challenge: Success Rate vs Intensity")
    plt.xlabel("Attack Intensity")
    plt.ylabel("Attack Success Rate (%)")
    plt.xticks(x, labels_x)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{out_dir}/graph6_internal_attack_bar.png")
    plt.close()

def generate_graph_7(out_dir="results"):
    """Graph 7: Trust Convergence Stability"""
    print("Generating Graph 7...")
    steps = 100
    plt.figure(figsize=(10, 6))
    
    for model in MODELS:
        sim = Simulator(model_type=model, num_vehicles=50, percent_malicious=0.1)
        history_ranks = [] 
        
        for t in range(steps):
             sim.model.simulate_interaction_step(25)
             sim.model.update_global_trust(sync_rsus=True)
             
             vehicles = sim.model.vehicles
             scores = [(v.id, v.global_trust_score) for v in vehicles.values()]
             scores.sort(key=lambda x: x[1], reverse=True)
             
             rank_map = {vid: i for i, (vid, score) in enumerate(scores)}
             history_ranks.append(rank_map)
             
        threshold = 5 
        y_axis = [0.0] 
        
        for t in range(1, steps):
            prev = history_ranks[t-1]
            curr = history_ranks[t]
            stable_count = 0
            for vid in prev:
                change = abs(prev[vid] - curr[vid])
                if change <= threshold:
                    stable_count += 1
            y_axis.append(stable_count / 50.0)
            
        plt.plot(range(steps), y_axis, **get_style(model))

    plt.title("Trust/Rank Convergence Stability")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Fraction of Nodes with Stable Rank")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{out_dir}/graph7_convergence_stability.png")
    plt.close()

def run_paper_suite():
    """Run all paper graph generations."""
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    print("========================================")
    print("STARTING PAPER COMPARATIVE STUDY (Graphs 1-7)")
    print("========================================")
    
    generate_graph_1(out_dir)
    generate_graph_2(out_dir)
    generate_graph_3_and_4(out_dir)
    intensities, labels = generate_graph_5(out_dir)
    generate_graph_6(intensities, labels, out_dir)
    generate_graph_7(out_dir)
    
    print("\nAll Paper Graphs generated in /results.")

# ==========================================
# MAIN ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="IoV Trust Simulation Runner")
    parser.add_argument('-p', '--paper', action='store_true', help="Run the full Comparative Study (Graphs 1-7)")
    parser.add_argument('-s', '--sumo', action='store_true', help="Force SUMO mode (requires SUMO installed)")
    parser.add_argument('-n', '--no-sumo', action='store_true', help="Disable SUMO check and run Comparative Study")
    
    args = parser.parse_args()
    
    # 1. Paper Mode (Explicit)
    if args.paper:
        run_paper_suite()
        return

    # 2. SUMO Mode (Explicit or Available)
    if args.sumo or (SUMO_AVAILABLE and not args.no_sumo):
        if not SUMO_AVAILABLE:
            print("Error: SUMO is not available or traci import failed.")
            sys.exit(1)
        
        print(">> Mode: SUMO-TraCI Simulation")
        try:
            run_sumo_simulation()
        except KeyboardInterrupt:
            print("\nSimulation aborted by user.")
        except Exception as e:
            print(f"Error during SUMO simulation: {e}")
            import traceback
            traceback.print_exc()
        return

    # 3. Fallback: Run Paper Suite as default "Simulation" if no SUMO
    print(">> Mode: Comparative Study (Fallback - No SUMO detected)")
    run_paper_suite()

if __name__ == "__main__":
    main()

