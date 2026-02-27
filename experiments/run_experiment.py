"""
Main Experiment Runner.
Orchestrates the simulation based on Mode (Default/SUMO) and Scope (Single/Compare).

Modes:
 1. Default (Python-based Random Simulation)
    - Single: Runs one 'PROPOSED' model simulation with solo plots.
    - Compare: Regenerates ALL graphs (comparative + proposed-solo + DAG).

 2. SUMO (TraCI-based Realistic Simulation)
    - Single: Runs one 'PROPOSED' model in SUMO.
    - Compare: Runs all models sequentially in SUMO and plots comparison.
"""
import sys
import os
import argparse

# Allow imports from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from experiments.run_sumo_experiment import run_sumo_simulation, SUMO_AVAILABLE
except ImportError:
    SUMO_AVAILABLE = False
    def run_sumo_simulation(compare=False):
        print("SUMO modules not found.")

from experiments.config import RESULTS_DIR
from experiments.benchmark import run_single_simulation
from experiments.plots import (
    run_all_graphs,
    plot_proposed_trust_evolution,
    plot_proposed_detection_metrics,
    plot_proposed_rank_distribution,
    plot_proposed_trust_convergence,
    plot_proposed_trust_normalized,
)


def run_default_demo():
    """Mode 2B: Default Python Simulation (Single Run — PROPOSED only plots)."""
    print(">> Mode: Default Random Simulation (Single Model: PROPOSED)")
    res = run_single_simulation('PROPOSED', steps=100)
    out = RESULTS_DIR
    os.makedirs(out, exist_ok=True)
    print("Simulation complete. Generating proposed-solo plots ...")
    plot_proposed_trust_evolution(res['vehicles'], out)
    plot_proposed_detection_metrics(res['vehicles'], out)
    plot_proposed_rank_distribution(res['vehicles'], out)
    plot_proposed_trust_convergence(res['vehicles'], out)
    plot_proposed_trust_normalized(res['vehicles'], out)
    print(f"\nDone. Check {os.path.abspath(out)} for plots.")


def main():
    parser = argparse.ArgumentParser(description="IoV Trust Simulation Runner")

    parser.add_argument('-s', '--sumo', action='store_true',
                        help="Run in SUMO Mode (requires TraCI). If excluded, runs Default Python Sim.")
    parser.add_argument('-c', '--compare', action='store_true',
                        help="Regenerate ALL graphs (comparative + proposed-solo + DAG).")

    args = parser.parse_args()

    # MODE 1: SUMO
    if args.sumo:
        if not SUMO_AVAILABLE:
            print("Error: SUMO libraries not found. Is SUMO_HOME set?")
            sys.exit(1)
        if args.compare:
            print(">> Mode: SUMO Comparative Suite")
            run_sumo_simulation(compare=True)
        else:
            print(">> Mode: SUMO Single Run")
            run_sumo_simulation(compare=False)
    # MODE 2: DEFAULT (Python Random)
    else:
        if args.compare:
            print(">> Mode: Full Graph Suite (all comparative + proposed-solo + DAG)")
            run_all_graphs()
        else:
            run_default_demo()

if __name__ == "__main__":
    main()
