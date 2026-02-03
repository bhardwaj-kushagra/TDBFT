"""
Main Experiment Runner.
Orchestrates the simulation based on Mode (Default/SUMO) and Scope (Single/Compare).

Modes:
 1. Default (Python-based Random Simulation)
    - Single: Runs one 'PROPOSED' model simulation.
    - Compare: Runs the full Paper Suite (Graphs 1-7).
    
 2. SUMO (TraCI-based Realistic Simulation)
    - Single: Runs one 'PROPOSED' model in SUMO.
    - Compare: Runs all models sequentially in SUMO and plots comparison.
"""
import sys
import os
import argparse

# Allow imports from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.config import MODELS, RESULTS_DIR
from experiments.benchmark import run_single_simulation
from experiments.plots import (
    run_paper_suite, 
    plot_trust_evolution, 
    plot_detection_metrics, 
    plot_final_trust_distribution
)

# Optional SUMO import
try:
    from experiments.run_sumo_experiment import run_sumo_simulation
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False


def run_default_demo():
    """Mode 2B: Default Python Simulation (Single Run)"""
    print(">> Mode: Default Random Simulation (Single Model: PROPOSED)")
    
    # Run using the centralized benchmark logic
    # Using 100 steps for a good demo
    res = run_single_simulation('PROPOSED', steps=100)
    
    print("Simulation Complete. Generating plots...")
    plot_trust_evolution(res['vehicles'])
    plot_detection_metrics(res['vehicles'])
    plot_final_trust_distribution(res['vehicles'])
    
    print(f"Check {RESULTS_DIR} for plots.")


def main():
    parser = argparse.ArgumentParser(description="IoV Trust Simulation Runner")
    
    # Mode Selection: Default vs SUMO
    parser.add_argument('-s', '--sumo', action='store_true', 
                        help="Run in SUMO Mode (requires TraCI). If excluded, runs Default Python Sim.")
    
    # Scope Selection: Single vs Compare
    parser.add_argument('-c', '--compare', action='store_true', 
                        help="Run Comparative Suite (All Models). Default is Single Run (PROPOSED).")
    
    args = parser.parse_args()
    
    # =========================================================
    # MODE 1: SUMO
    # =========================================================
    if args.sumo:
        if not SUMO_AVAILABLE:
            print("Error: SUMO libraries not found. Is SUMO_HOME set?")
            print("Falling back to logic check... TraCI import failed.")
            sys.exit(1)
            
        if args.compare:
            # 1A: SUMO Compare
            print(">> Mode: SUMO Comparative Suite")
            run_sumo_simulation(compare=True)
        else:
            # 1B: SUMO Single
            print(">> Mode: SUMO Single Run")
            run_sumo_simulation(compare=False)
            
    # =========================================================
    # MODE 2: DEFAULT (Python Random)
    # =========================================================
    else:
        if args.compare:
            # 2A: Default Compare (Paper Graphs)
            print(">> Mode: Default Comparative Study (Paper Graphs 1-7)")
            run_paper_suite() # From plots.py
        else:
            # 2B: Default Single (Demo)
            run_default_demo()

if __name__ == "__main__":
    main()

