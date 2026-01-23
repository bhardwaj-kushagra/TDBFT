# IoV Trust Management Prototype - Developer Guide

**Date:** January 23, 2026

---

## 1. Project Overview

This repository acts as a simulation sandbox for the **VehicleRank** and **Trust-DBFT** protocols. It separates the "Trust Logic" (Real math) from the "Infrastructure" (Mocked Blockchain/Simulation).

### Directory Structure

```
iov-trust-prototype/
├── trust/              # CORE ALGORITHMS
│   ├── bayesian.py     # BayesTrust (Eq III-A)
│   ├── rsu.py          # VehicleRank (Eq III-C)
│   ├── vehicle.py      # Node state & behavior
│   └── trust_model.py  # Orchestrator
├── blockchain/         # MOCKED LEDGER
│   ├── dag.py          # Graph data structure
│   └── validator.py    # Committee & Consensus (Eq IV-A/B)
├── experiments/        # RUNTIME
│   ├── run_experiment.py # Main entry point
│   └── plots.py        # Matplotlib visualization
└── results/            # OUTPUT
```

---

## 2. Code Walkthrough

### A. `trust/bayesian.py`
**Implements:** Section III-A (Local Trust Formulation).
*   **Function**: `compute_trust(alpha, beta)`
*   **Logic**: Returns (alpha / (alpha + beta)).
*   **Usage**: Called by RSUs to convert raw interaction counts $(y, n)$ into basic probability scores $m_{ij}$.

### B. `trust/rsu.py`
**Implements:** Section III-C (VehicleRank).
*   **Class**: `RSU`
*   **Key Method**: `compute_vehiclerank(all_ids, reports)`
*   **Algorithm**:
    1.  **Build Matrix**: Converts local trust reports into an $N \times N$ matrix.
    2.  **Normalize**: Normalizes rows so they sum to 1.
    3.  **Power Iteration**:
        ```python
        t_new = alpha * dot(t, S) + (1-alpha) * teleport
        ```
    4.  Runs until `norm(t_new - t) < 1e-6`.

### `trust/vehicle.py`
**Implements:** Simulation Entities.
*   **Behaviors**:
    *   `HONEST`: 99% cooperation.
    *   `MALICIOUS`: 20% cooperation (Packet drop/Bad data).
    *   `SWING`: Alternates behavior every 50 steps to fool simple averaging models. VehicleRank detects this by analyzing the entire graphs structure rather than just history.

### C. `blockchain/validator.py`
**Implements:** Section IV (Trust-DBFT).
*   **Method**: `select_validators(ranked, top_n)`
    *   Returns the top-c vehicles sorted by global trust.
*   **Method**: `check_consensus_weighted(committee)`
    *   Simulates a voting round.
    *   Calculates `approval_mass` vs `total_mass`.
    *   Returns `True` only if weighted approval >= 66%.

### D. `experiments/run_experiment.py`
**Implements:** Simulation Lifecycle.
1.  **Setup**: Spawns 30 vehicles (15% Malicious, 10% Swing).
2.  **Loop**:
    *   **Simulate**: Random interactions occur.
    *   **Update**: `trust_model.update_global_trust()` runs VehicleRank.
    *   **Consensus**: Committee is formed; Weighted vote occurs.
    *   **Commit**: If consensus passes, blocks are added to DAG.
    *   **Merge**: Multi-region DAGs (RSU 1 & RSU 2) sync.

---

## 3. How to Run

### Requirements
*   Python 3.8+
*   `numpy`
*   `matplotlib`

### command
```powershell
# From the parent directory
python iov-trust-prototype/experiments/run_experiment.py
```

### Output Interpretation
1.  **Console Logs**:
    *   `Step X: Leader = V012`: Shows which node is currently most trusted.
    *   `DAG Size`: Growth indicates successful consensus rounds. Stagnation means consensus is rejecting blocks (likely due to malicious committee members being outvoted but present).
2.  **Plots (`results/`)**:
    *   `comparative_trust_evolution.png`: Compare the curves for Honest vs Malicious nodes. Honest should stabilize high; Malicious should converge low.
    *   `swing_analysis.png`: Check if the Global Trust curve (purple) successfully dampens the oscillations of the Local Trust curve (orange).

---

## 4. Customization

*   **Change Attacks**: Edit `experiments/run_experiment.py` -> `percent_malicious` or `percent_swing`.
*   **Adjust Physics**: Edit `trust/bayesian.py` -> Base priors $a, b$.
*   **Adjust Algo**: Edit `trust/rsu.py` -> `alpha_damping` (set lower if convergence is too slow).
