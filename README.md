# IoV Trust Management Prototype (T-DBFT & VehicleRank)

This repository contains the simulation code for the research work on **Bayesian-Optimized Trust-Driven Consensus Framework over a DAG-Enabled Consortium Blockchain for IoV**.

The framework integrates Bayesian trust evaluation, a hybrid DPoS-DBFT consensus mechanism (Trust-DBFT), and a DAG-based ledger to ensure secure and scalable message sharing in Internet of Vehicles (IoV) environments.

## Key Features

- **Bayesian Trust Model**: Updates local trust based on interaction history ($\alpha, \beta$).
- **VehicleRank**: Iterative global trust computation to mitigate Sybil attacks.
- **Trust-DBFT Consensus**: Weighted voting mechanism where voting power is proportional to trust.
- **DAG Ledger**: Parallel block creation across different RSU clusters with merging capability.

## Repository Structure

- **`trust/`**: Core trust logic including Bayesian inference and VehicleRank algorithm.
- **`blockchain/`**: Mocked blockchain implementation featuring DAG structure and Validator selection.
- **`experiments/`**: Simulation entry points and plotting scripts for algorithm validation.
- **`sumo/`**: Integration with SUMO traffic simulator for realistic mobility scenarios.
- **`results/`**: Simulation outputs (plots, metrics).

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Experiment (Algorithm Validation)

```bash
python experiments/run_experiment.py
```

This runs the simulation using random mobility and interactions to validate the trust model and consensus logic.
Check `results/` for output plots like `trust_evolution.png`.

### 3. Run with SUMO (Realistic Mobility)

*Note: Requires SUMO to be installed and `SUMO_HOME` environment variable set.*

```bash
python experiments/run_sumo_experiment.py
```

## Documentation

- [Code Guide](docs/CODE_GUIDE.md): Detailed walkthrough of the codebase and its modules.
- [Project Documentation](docs/PROJECT_DOCUMENTATION.md): Overview of the project goals and conventions.
- [Theory](docs/THEORY.md): Theoretical background of the trust models and consensus algorithms used.
- [Technical Implementation Details](docs/TECHNICAL_IMPLEMENTATION_DETAILS.md): Deep technical analysis.

