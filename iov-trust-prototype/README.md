# IoV Trust Management Prototype

A research prototype for an Internet of Vehicles (IoV) trust management system.
Combines Bayesian trust computation with a lightweight DAG-based blockchain for validator selection.

## Modules

### 1. Trust Engine (`trust/`)
- Implements Beta Reputation System (Bayesian).
- `compute_trust = alpha / (alpha + beta)`
- Local vehicle trust generation.
- RSU Global Trust Aggregation with forgetting factor.

### 2. Mock Blockchain (`blockchain/`)
- Mocks a DAG (Directed Acyclic Graph) structure.
- **Not a real blockchain**: No hashing, mining, or cryptography.
- Purpose: To simulate the storage of trust updates and selection of validators.
- Validator Selection: Top-N vehicles by Global Trust Score are allowed to append blocks.

### 3. Simulation (`experiments/`)
- `run_experiment.py`: Main entry point. Simulates random interactions (or SUMO interactions in future) to drive trust updates.
- Generates plots in `results/`.

### 4. SUMO Integration (`sumo/`, `traci_control/`)
- Placeholders for future integration with Eclipse SUMO via TraCI.
- Will replace random interaction logic with proximity-based logic.

## How to Run

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the experiment:
   ```bash
   python experiments/run_experiment.py
   ```

3. View results:
   - Check console output for validator selection logs.
   - Open `results/trust_evolution.png` to see trust scores over time.

## Architecture Flow

1. **Interaction**: Vehicles interact (simulated success/failure based on malicious status).
2. **Local Update**: Observer updates $\alpha, \beta$ for Target.
3. **Aggregation**: RSU collects local trusts, updates Global Trust using forgetting factor.
4. **Ranking**: Vehicles ranked by Global Trust.
5. **Consensus (Mock)**: Top validators selected to record state in DAG.
