# IoV Trust Management Prototype - Developer Guide

**Date:** January 23, 2026

---

## 1. Project Overview

This repository acts as a simulation sandbox for the **VehicleRank** and **Trust-DBFT** protocols. It separates the "Trust Logic" (Real math) from the "Infrastructure" (Mocked Blockchain/Simulation).

### Directory Structure

```text
./
├── trust/              # CORE ALGORITHMS
│   ├── models/         # STRATEGY PATTERN (New)
│   │   ├── strategies.py # ProposedStrategy, BTVRStrategy, etc.
│   │   └── base.py       # TrustStrategy Interface
│   ├── bayesian.py     # BayesTrust (Eq III-A)
│   ├── rsu.py          # Context for Strategies
│   ├── vehicle.py      # Node state & behavior
│   └── trust_model.py  # Orchestrator
├── blockchain/         # MOCKED LEDGER
│   ├── dag.py          # Graph data structure (with TCW)
│   ├── consensus_manager.py # Bridge between Trust & DAG
│   └── validator.py    # Committee & Consensus (Eq IV-A/B)
├── experiments/        # RUNTIME
│   ├── run_experiment.py # Main entry point (CLI)
│   ├── run_sumo_experiment.py # TraCI integration
│   └── plots.py        # Matplotlib visualization
└── results/            # OUTPUT
```

---

## 2. Code Walkthrough

### A. `trust/models/strategies.py` (The Heart)

**Implements:** Section III (Global Trust Logic).

*   **`ProposedStrategy`**: Implements the full **VehicleRank** algorithm (Power Iteration).
*   **`BTVRStrategy`**: Implements the baseline Beta-based averaging.
*   **Usage**: The system picks the strategy at runtime using the Factory pattern in `trust/models/__init__.py`.

### B. `trust/bayesian.py`

**Implements:** Section III-A (Local Trust Formulation).

* **Function**: `compute_trust(alpha, beta)`
* **Logic**: Returns (alpha / (alpha + beta)).
* **Usage**: Called by RSUs to convert raw interaction counts $(y, n)$ into basic probability scores $m_{ij}$.

### C. `trust/rsu.py`

**Implements:** Section III-C (Context for VehicleRank).

*   **Class**: `RSU`
*   **Role**: Acts as the Context in the Strategy Pattern. It holds the `strategy` instance and delegates the heavy lifting via `self.strategy.compute_global_trust(...)`.

### `trust/vehicle.py`

**Implements:** Simulation Entities.

* **Behaviors**:
  * `HONEST`: 99% cooperation.
  * `MALICIOUS`: 20% cooperation (Packet drop/Bad data).
  * `SWING`: Alternates behavior in 50-step cycles using a per-vehicle random phase offset (so attackers don’t flip in sync). The Proposed model’s VehicleRank aggregation helps damp oscillations relative to simple averaging baselines.

### C. `blockchain/validator.py`

**Implements:** Section IV (Trust-DBFT).

* **Method**: `select_validators(ranked, top_n)`
  * Returns the top-c vehicles sorted by global trust.
* **Method**: `check_consensus_weighted(committee)`
  * Simulates a voting round.
  * Calculates `approval_mass` vs `total_mass`.
  * Returns `True` only if weighted approval >= 66%.

### D. `experiments/run_experiment.py`

**Implements:** Simulation Lifecycle.

1. **Setup**: Spawns 30 vehicles (15% Malicious, 10% Swing).
2. **Loop**:
   * **Simulate**: Random interactions occur.
   * **Update**: `trust_model.update_global_trust()` runs VehicleRank.
   * **Consensus**: Committee is formed; Weighted vote occurs.
   * **Commit**: If consensus passes, blocks are added to DAG.
   * **Merge**: Multi-region DAGs (RSU 1 & RSU 2) sync.

--- & CLI.

*   **Commands**:
    *   `python ... -s`: Run with SUMO.
    *   `python ... -c`: Run comparative suite.
*   **Flow**:
    1.  Initializes `Simulator` (which builds Logic).
    2.  Initializes `ConsensusManager` (which builds DAG).
    3.  Loops through steps -> Interactions -> Trust Updates -> Block Creation

### command

```powershell
# From the project root
python experiments/run_experiment.py
```

### Output Interpretation

1. **Console Logs**:
   * `Step X: Leader = V012`: Shows which node is currently most trusted.
   * `DAG Size`: Growth indicates successful consensus rounds. Stagnation means consensus is rejecting blocks (likely due to malicious committee members being outvoted but present).
2. **Plots (`results/`)**:
   * `comparative_trust_evolution.png`: Compare the curves for Honest vs Malicious nodes. Honest should stabilize high; Malicious should converge low.
   * `swing_analysis.png`: Check if the Global Trust curve (purple) successfully dampens the oscillations of the Local Trust curve (orange).

---

## 4. Customization

* **Change Attacks**: Edit `experiments/run_experiment.py` -> `percent_malicious` or `percent_swing`.
* **Adjust Physics**: Edit `trust/bayesian.py` -> Base priors $a, b$.
* **Adjust Algo**: Edit `trust/rsu.py` -> `alpha_damping` (set lower if convergence is too slow).
