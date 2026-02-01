# GitHub Copilot Instructions – IoV Trust Management Prototype

## Project Goal
Research prototype for Internet of Vehicles (IoV) trust management.
Focus: Bayesian trust computation + validator selection. “Blockchain” is a mocked DAG used for role assignment/history.

## Big Picture Flow
1) Simulate vehicle interactions
2) Update **local trust** via Beta reputation ($t=\alpha/(\alpha+\beta)$)
3) RSUs aggregate to **global trust** (with forgetting factor)
4) Rank vehicles by global trust
5) Select Top-N as validators
6) Append trust updates to a mocked DAG “blockchain”
7) Optional: replace random interactions with SUMO/TraCI mobility-driven interactions

## Key Directories
- `trust/`: core trust logic (keep deterministic/pure Python)
  - `vehicle.py`: vehicle state (alpha/beta, local/global trust), attacker behaviors
  - `bayesian.py`: Beta reputation update + trust formula
  - `rsu.py`: global aggregation + forgetting factor
  - `trust_model.py`: orchestrates updates + penalties
  - `simulator.py`: non-SUMO interaction generator (fast experiments/plots)

- `blockchain/`: intentionally lightweight mock chain (readability > realism)
  - `dag.py`: DAG of trust updates (no hashing/mining/crypto)
  - `block.py`: record structure for trust updates
  - `validator.py`: select Top-N by global trust; only validators append

- `sumo/`: SUMO config artifacts (`*.net.xml`, `*.rou.xml`, `*.sumocfg`)—typically generated, not hand-edited
- `traci_control/`: TraCI bridge
  - `run_sumo.py`: reads vehicle IDs/positions and triggers interactions; **must reuse** `trust/` logic

- `experiments/`: entry points + analysis
  - `run_experiment.py`: main script (simulate → trust updates → ranking/validators → DAG append)
  - `plots.py`: saves plots under `results/`

## Developer Workflow
- Default (no SUMO): `python experiments/run_experiment.py`
- SUMO (optional):
  - `netgenerate --grid --grid.number=5 -o sumo/net.net.xml`
  - `python randomTrips.py -n sumo/net.net.xml -o sumo/routes.rou.xml -e 7200`
  - `python traci_control/run_sumo.py`

## Project Conventions
- Trust logic is “real”; blockchain layer is mocked/abstract.
- Initial trust defaults to 0.5 via alpha=1, beta=1.
- SUMO/TraCI code must not duplicate or embed trust computations.

## Avoid
- Real blockchain frameworks, PoS/PBFT implementations, smart contracts, cryptography-heavy designs.
- Mixing SUMO mobility logic into `trust/` modules.
- Over-engineering the DAG layer.
This should:

Simulate interactions

Update trust

Rank vehicles

Generate plots

SUMO integration (optional / later)

SUMO is used only for mobility realism.

Typical commands:

netgenerate --grid --grid.number=5 -o sumo/net.net.xml
python randomTrips.py -n sumo/net.net.xml -o sumo/routes.rou.xml -e 7200
python traci_control/run_sumo.py


If SUMO is unavailable, the project must still run end-to-end.

Project-Specific Conventions

Trust logic is real, blockchain logic is mocked

No OMNeT++, no packet-level simulation

No real consensus algorithms

SUMO must never contain trust logic

Trust updates happen periodically (simulated time)

Initial trust defaults to 0.5 (alpha=1, beta=1)

What NOT to Do

Do not introduce real blockchain frameworks

Do not implement PoS, PBFT, or smart contracts

Do not mix SUMO logic into trust modules

Do not over-engineer DAG or cryptography

This repository prioritizes clarity, trends, and explainability over completeness.

When in Doubt

Favor:

Readable Python

Clear separation of concerns

Correct trust trends over exact numeric matching

This is a prototype for demonstration and evaluation, not production software.# GitHub Copilot Instructions – IoV Trust Management Prototype

## Big Picture Architecture

This project is a **research prototype** for an Internet of Vehicles (IoV) trust management system.
It focuses on **Bayesian trust computation and validator selection**, not a full blockchain implementation.

High-level flow:
1. Vehicles interact (simulated)
2. Local trust is updated using a Bayesian (alpha/beta) reputation model
3. RSUs aggregate local trust into global trust using a forgetting factor
4. Vehicles are ranked by global trust
5. Top-N trusted vehicles act as **blockchain validators**
6. Trust updates are stored in a **mocked DAG-based blockchain**
7. SUMO (via TraCI) later replaces random interactions with mobility-based ones

Blockchain, consensus, PoS/PBFT are **abstracted** and intentionally lightweight.

---

## Core Components and Responsibilities

### `trust/`
Core trust logic. This is the **most important** directory.

- `vehicle.py`
  - Defines the Vehicle object
  - Holds alpha, beta, local_trust, global_trust
  - Encodes malicious / swing attacker behavior

- `bayesian.py`
  - Implements Beta reputation logic
  - Trust = alpha / (alpha + beta)
  - No Bayesian optimization or GP models are used

- `rsu.py`
  - Aggregates local trust into global trust
  - Applies a forgetting factor to prevent rapid trust recovery

- `trust_model.py`
  - Orchestrates local + global trust updates
  - Applies penalties for malicious behavior

- `simulator.py`
  - Generates interactions without SUMO
  - Used for fast experimentation and plotting

Trust logic should be **pure Python and deterministic**.

---

### `blockchain/`
Mocked blockchain layer used only for **role assignment and history tracking**.

- `dag.py`
  - Implements a DAG of trust updates
  - Each node references previous trust states
  - No hashing, mining, or cryptography

- `block.py`
  - Lightweight block structure for trust records

- `validator.py`
  - Selects top-N vehicles by global trust
  - Only validators are allowed to append DAG nodes

This layer must remain **simple and readable**.

---

### `sumo/`
SUMO configuration files.

- `net.net.xml`
- `routes.rou.xml`
- `config.sumocfg`

These files are usually **generated via CLI tools**, not edited manually.

---

### `traci_control/`
SUMO ↔ Python bridge.

- `run_sumo.py`
  - Uses TraCI to read vehicle IDs and positions
  - Triggers interactions based on distance
  - Feeds interactions into existing trust logic

TraCI code must **reuse** trust logic, not duplicate it.

---

### `experiments/`
Entry points and analysis.

- `run_experiment.py`
  - Main executable script
  - Runs simulation, ranks vehicles, selects validators
  - Stores trust updates in DAG

- `plots.py`
  - Generates:
    - Trust vs time
    - Malicious detection rate vs threshold

Plots are saved under `results/`.

---

## Developer Workflow (Important)

### Run without SUMO (default workflow)
Used for most development and debugging.

```bash
python experiments/run_experiment.py
```
Use the following fixed color and style scheme across all plots:

BTVR      → Blue, dashed line
BSED      → Green, dotted line
RTM       → Orange, dash-dot line
COBATS    → Cyan, dashed line
LT-PBFT   → Purple, dotted line
PROPOSED → Red, solid line, linewidth=3, prominent markers
