# Technical Implementation Specification: IoV Trust & Consensus Prototype

**Version:** 1.0.0  
**Target Audience:** Research Scientists, System Architects, Advanced Developers  
**Scope:** Deep technical analysis of the codebase, algorithms, mathematical models, and data structures used in the `iov-trust-prototype`.

---

## 0. Document Conventions, Code Map, and Invariants

This document is written as an implementation specification: it describes what the code does (including simplifications) and how values flow across modules.

### 0.1. Module Map (Runtime Ownership)

The system is intentionally split into three layers:

- **Trust layer (`trust/`)**: deterministic numerical logic and state updates for vehicles and RSUs.
- **Consensus/storage layer (`blockchain/`)**: mocked ledger structure (DAG) + validator committee selection.
- **Experiment/drivers (`experiments/`, `traci_control/`)**: orchestration, metrics, and plotting.

Key runtime objects (long-lived across the simulation run):

- `trust.vehicle.Vehicle`: holds local interaction state and global trust time-series.
- `trust.rsu.RSU`: produces and synchronizes global trust vectors.
- `trust.trust_model.TrustModel`: orchestrates interaction + RSU update cycles.
- `blockchain.dag.DAG`: stores accepted trust snapshots as blocks.

### 0.2. Naming and Identity Invariants

- Vehicle identifiers are strings like `V000`, `V001`, ... created in `TrustModel.__init__`.
- RSU identifiers are strings like `RSU-01`, `RSU-02`, ...
- A vehicle’s maliciousness is tracked in two related fields:
    - `Vehicle.behavior_type` ∈ {`HONEST`, `MALICIOUS`, `SWING`}
    - `Vehicle.is_malicious` is `True` for both `MALICIOUS` and `SWING`.

### 0.3. Determinism vs Randomness

- The trust computations (Beta update, VehicleRank matrix math, weighted voting check) are deterministic given the same interaction outcomes.
- Interaction outcomes are stochastic (Python `random`), and **there is no global seed set by default**. Reproducible runs require calling `random.seed(...)` (and `numpy.random.seed(...)` if extended) in the driver.

### 0.4. Scope Notes (Intended Simplifications)

This is a research prototype. Several components are deliberately simplified:

- The DAG does **not** implement tip-selection algorithms (e.g., MCMC walks). It uses a simplified “current tips” list.
- “Consensus” is abstracted as a weighted threshold check; there is no message-level PBFT state machine.
- RSU “synchronization” is modeled by pairwise vector averaging.

## 1. Trust Inference Model (Mathematical Core)

The system utilizes a **Bayesian Inference** model based on the Beta Probability Density Function (PDF) to estimate the trustworthiness of vehicles. This is deterministic processing implemented in `trust/bayesian.py`.

### 1.1. Local Trust Computation
For any observer vehicle $i$ monitoring target vehicle $j$, trust is modeled as the probability that $j$ will cooperate in the next interaction.

*   **Underlying Distribution:** Beta($\alpha, \beta$)
*   **Parameters:**
    *   $\alpha$ (alpha): Cumulative count of **positive/cooperative** interactions + 1 (prior).
    *   $\beta$ (beta): Cumulative count of **negative/malicious** interactions + 1 (prior).
*   **Expectation Value (Trust Score):**
    $$ T_{i,j} = E[\text{Beta}(\alpha, \beta)] = \frac{\alpha}{\alpha + \beta} $$
*   **Uncertainty (Variance):**
    $$ \sigma^2 = \frac{\alpha \beta}{(\alpha+\beta)^2 (\alpha+\beta+1)} $$
    *(Note: While calculated in theory, the current implementation primarily uses the Expectation Value for decision making.)*

### 1.2. Parameter Update Logic
Upon an interaction between Observer $i$ and Target $j$ with outcome $O \in \{0, 1\}$ (0=Failure, 1=Success):

$$ \alpha_{new} = \alpha_{old} + O $$
$$ \beta_{new} = \beta_{old} + (1 - O) $$

**Code Reference:** `trust/bayesian.py` -> `update_parameters`

### 1.3. Implementation Details: `trust/bayesian.py`

#### 1.3.1. `compute_trust(alpha: float, beta: float) -> float`

- Primary return: `alpha / (alpha + beta)`.
- Edge case: if `alpha + beta == 0`, returns `0.5`.
    - In the current system, the normal prior is `(alpha, beta) = (1.0, 1.0)`, so this edge case is mainly defensive.

#### 1.3.2. `update_parameters(current_alpha, current_beta, is_positive, weight=1.0) -> tuple`

- Adds `weight` to `alpha` for positives and to `beta` for negatives.
- Current callers use the default `weight=1.0`.
- This makes it trivial to extend the system to confidence-weighted observations (e.g., weight based on channel quality, distance, RSU confidence).

### 1.4. Where Alpha/Beta Live (State Ownership)

The Bayesian parameters are stored per-observer-per-target in `Vehicle.interactions`:

- Type: `Dict[str, Tuple[float, float]]`
- Key: `target_id`
- Value: `(alpha, beta)` for Bayesian-mode models, or model-specific tuples for other baselines.

The state is not centralized: each vehicle maintains its own local view of others.

---

## 2. Global Trust Aggregation (VehicleRank Algorithm)

Local trust scores ($T_{i,j}$) are sparse and subjective. The RSU (Roadside Unit) aggregates these into a Global Trust Vector ($T_{global}$) using a modified PageRank algorithm called **VehicleRank**.

**Code Reference:** `trust/rsu.py` -> `compute_vehiclerank`

### 2.1. Adjacency Matrix Construction ($M$)
The RSU constructs an $N \times N$ matrix $M$ where $M_{ij} = T_{i,j}$ (Local Trust of $j$ observed by $i$).
*   If no interaction exists, $M_{ij} = 0$.

### 2.2. Row Normalization (Stochastic Matrix $S$)
To create a valid transition probability matrix, rows are normalized:
$$ S_{ij} = \frac{M_{ij}}{\sum_{k} M_{ik}} $$
*   *Handling Dangling Nodes:* If a row sum is 0 (vehicle has no outgoing trust opinions), it is treated as a uniform distribution ($1/N$).

### 2.3. Iterative Power Method
The Global Trust Vector $\vec{t}$ is the principal eigenvector of $S$. It is computed iteratively:

$$ \vec{t}_{k+1} = d (S^T \vec{t}_k) + \frac{1-d}{N}\vec{1} $$

*   **Damping Factor ($d$):** `self.alpha_damping = 0.85` (Standard PageRank probability of following a link vs. teleporting).
*   **Convergence Tolerance:** `self.tol = 1e-6`.
*   **Max Iterations:** 100.

**Significance:** This effectively "weights" a vehicle's opinion by its own reputation. A highly trusted vehicle's local opinions have more impact on the global score of others.

### 2.4. Model Branching in `RSU.compute_vehiclerank`

The RSU supports multiple “model types” via branching:

- `PROPOSED`, `LT_PBFT`: uses VehicleRank (PageRank-like propagation).
- `BTVR`, `BSED`, `RTM`, `COBATS`: uses simple column-wise aggregation (average of received reports).

Important implementation detail:

- The RSU builds a dense matrix `M` initialized to zeros.
- For the simple-averaging branch, the code *does not* treat `M[i, j] = 0.0` as a guaranteed report; it re-checks `adjacency_reports` to detect whether `vid` was actually present in the reporter’s dictionary.
- Default for “no reports”: `0.5`.

### 2.5. RSU Synchronization Semantics (`incorporate_peer_knowledge`)

The RSU-to-RSU “consensus” is modeled by vector fusion:

- For every `(vid, score)` in the peer vector, update:
    $$ T_{new} = \frac{T_{mine} + T_{peer}}{2} $$
- This is a symmetric averaging, executed pairwise across RSUs in `TrustModel.update_global_trust(sync_rsus=True)`.

**Implementation note (important for the paper):**

- The doc/paper may refer to a “forgetting factor.” In the current code, there is **no explicit exponential forgetting factor** applied inside the RSU. Instead, “forgetting” happens implicitly by:
    - Recomputing trust each step from the evolving local trust graph.
    - Optional windowing at the local-trust level (see `Vehicle.get_windowed_local_trust`).

---

## 3. Adversary Modeling (Behavior Classes)

Defined in `trust/vehicle.py`, the system models three distinct adversary profiles.

### 3.1. Honest Vehicle
*   **Behavior:** Cooperates with probability $P_{coop} = 0.99$.
*   **Failure Rate:** 1% (Simulating natural packet loss or noise).

### 3.2. Constant Malicious (DoS/Blackhole)
*   **Behavior:** Cooperates with probability $P_{coop} = 0.20$.
*   **Strategy:** Consistently drops packets or sends falsified data.
*   **Detection Signature:** Trust score rapidly converges to $\approx 0.2$.

### 3.3. Swing Attacker (On-Off / Oscillating)
*   **Strategy:** Builds trust, then exploits it, then rebuilds.
*   **Implementation:**
    ```python
    cycle_length = 50
    phase = (step_count // cycle_length) % 2
    probability = 0.99 if phase == 0 else 0.10
    ```
*   **Detection Challenge:** Requires the sliding window or forgetting factor in the RSU aggregation to detect the sudden drop.

### 3.4. Behavior Implementation: `Vehicle.perform_action(step_count) -> bool`

- The outcome is sampled using Python’s `random.random()`.
- Honest vehicles: `random.random() < 0.99`.
- Malicious vehicles: `random.random() < 0.20`.
- Swing attackers:
    - `cycle_length = 50`
    - Good phase: `< 0.99`
    - Bad phase: `< 0.10`

**Implication:** Swing attackers are marked as malicious in the evaluation (they contribute to detection positives), but their outcomes can temporarily mimic honest nodes.

### 3.5. Local Trust Windowing (Detection Aid)

`Vehicle.get_windowed_local_trust(target_id, window_size=20)` computes a trust estimate from only the last `window_size` interaction outcomes.

- It uses the same Beta expectation form but reconstitutes parameters from the window:
    - `w_alpha = 1 + positives_in_window`
    - `w_beta = 1 + negatives_in_window`
    - `trust = w_alpha / (w_alpha + w_beta)`

This is currently used explicitly in SUMO runs to log swing-local vs swing-global trends.

---

## 4. Hybrid Consensus Architecture (DAG + Weighted DBFT)

The system employs a novel hybrid architecture: **DAG for Storage** + **Reputation-Weighted Voting for Security**.

### 4.1. The Data Structure: Directed Acyclic Graph (DAG)
**Code Reference:** `blockchain/dag.py`

Unlike a linear blockchain (Linked List), the DAG allows multiple tips.
*   **Node (Block):** Contains `data`, `validator_id`, `parents` (list of hashes).
*   **Tip Management:**
    *   *Current Implementation:* `self.tips` list tracks the latest added blocks.
    *   *Insertion Rule:* New blocks verify and reference **all** currently known tips as parents.
    *   *Result:* This creates a "confluence" structure, keeping the DAG relatively narrow and linear-like to prevent excessive branching (tangle).

### 4.2. Committee Selection (The "Miners")
**Code Reference:** `blockchain/validator.py` -> `select_validators`

Instead of Hash power (PoW) or Stake (PoS), we use **Trust**.
1.  RSU sorts all vehicles by `global_trust_score`.
2.  Top $K$ vehicles are selected to form the **Consensus Committee**.
    *   *Parameter:* `top_n = 3` (configurable).

### 4.3. Weighted Consensus (The "Vote")
**Code Reference:** `blockchain/validator.py` -> `check_consensus_weighted`

To append a block, the committee must vote.
*   **Standard PBFT:** One Node = One Vote.
*   **Our Implementation (Trust-Weighted):** One Node = Voting Power equal to Global Trust.

**Validity Condition:**
$$ \sum_{v \in C} (T_v \cdot \text{Vote}_v) \ge \frac{2}{3} \sum_{v \in C} T_v $$

Where:
*   $C$: The Committee.
*   $T_v$: Global Trust Score of vehicle $v$.
*   $\text{Vote}_v$: 1 if Approve, 0 if Reject.

**Security Implication:** Even if a malicious node enters the Top $K$ (e.g., a Swing Attacker in good phase), its vote has less weight than a perfectly honest node ($T \approx 0.99$).

### 4.4. Block Structure (`blockchain/block.py`)

Each DAG node is a `Block` with fields:

- `id`: first 8 chars of a UUID4 (`str(uuid.uuid4())[:8]`). This is not a cryptographic hash.
- `timestamp`: `time.time()` (UNIX epoch float).
- `data`: a payload; typically a snapshot dict `{vehicle_id: global_trust_score}` or `{}` for simplified tests.
- `validator_id`: the committee member who created the block.
- `parents`: list of parent block IDs referenced by this block.

### 4.5. DAG Tip Tracking and “Linear-ish” Behavior

In `blockchain/dag.py`:

- `self.tips` is a list of block IDs that are considered “current tips”.
- On `add_block(...)`:
    - `parents = list(self.tips)`
    - After inserting the new block, the implementation sets `self.tips = [new_block.id]`.

**Consequence:** Although the structure is called a DAG, this policy produces a *narrow* DAG that behaves close to a linear chain (unless merge logic introduces multiple tips).

### 4.6. DAG Merge Semantics (`DAG.merge_with`)

`merge_with(other_dag)` merges block dictionaries and then merges tip sets:

- Blocks: union by block ID.
- Tips: `combined_tips = list(set(self.tips + other_dag.tips))` and then keep only last 5 entries: `self.tips = combined_tips[-5:]`.

**Implementation note:**

- This does not compute real “tip” ancestry (i.e., whether a tip has children). The prototype treats tips as a bounded list representing recent heads.

---

## 5. Simulation Lifecycle (Orchestration)

The `experiments/run_experiment.py` script drives the system through discrete time steps.

### Step-by-Step Execution Trace

1.  **Initialization (`TrustModel.__init__`)**:
    *   Instantiates $N$ vehicles.
    *   Assigns behaviors (Honest/Malicious/Swing) based on percent distributions.
    *   Initializes Adjacency Matrix $M$ with uniform priors.

2.  **Interaction Loop (Per Step)**:
    *   **Selection:** Random pairs $(i, j)$ selected (or via SUMO proximity).
    *   **Action:** $j$ performs action (Success/Fail).
    *   **Update:** $i$ updates local $\alpha, \beta$ for $j$.

3.  **Aggregation Phase (`simulation_step % 10 == 0`)**:
    *   RSU pulls all local trust data.
    *   Runs **VehicleRank** to compute new Global Trust Vector.
    *   RSU applies **Forgetting Factor** (decay) to old global scores to allow recovery/punishment.
    *   Vehicles are re-ranked.

4.  **Consensus Phase**:
    *   Top $K$ vehicles identified.
    *   Leader proposes a block (snapshot of trust scores).
    *   Committee performs Weighted Voting.
    *   If Pass: Block appends to DAG.

5.  **Metrics Recording**:
    *   Current trust scores of all nodes recorded for `matplotlib` plotting.

### 5.1. Core Orchestrator: `trust/trust_model.py`

#### 5.1.1. Initialization Path: `TrustModel.__init__`

- Creates a dictionary `self.vehicles: Dict[str, Vehicle]`.
- Creates `self.rsus: List[RSU]` of length `num_rsus`.
- Assigns vehicles to RSUs via round-robin mapping `self.vehicle_rsu_map[vid] = rsu`.
- Vehicle behavior assignment is deterministic by index ordering:
    - First `num_malicious` vehicles become `MALICIOUS`.
    - Next `num_swing` vehicles become `SWING`.
    - Remaining vehicles become `HONEST`.

#### 5.1.2. Interaction Generation: `simulate_interaction_step(num_interactions)`

For each of `num_interactions` iterations:

1. Choose `observer_id` uniformly from all IDs.
2. Choose `target_id` uniformly from all IDs.
3. If same, skip.
4. Compute outcome using `target.perform_action(self.step_count)`.
5. Apply `observer.record_interaction(target.id, is_positive=is_good_interaction)`.

This produces an Erdős–Rényi-like stream of observations over time.

#### 5.1.3. Global Update and RSU Sync: `update_global_trust(sync_rsus=True)`

This performs four phases:

1. For each RSU: build `adjacency_reports` by pulling `reporter.interactions` from *all* vehicles (prototype assumes full visibility).
2. Each RSU calls `rsu.compute_vehiclerank(all_ids, adjacency_reports)`.
3. If `sync_rsus=True`: each RSU incorporates every other RSU’s vector via `incorporate_peer_knowledge` (pairwise averaging).
4. Push back final scores to vehicles:
     - For vehicle `vid`, pick its assigned RSU.
     - Set `v.global_trust_score = assigned_rsu.get_global_trust(vid)`.
     - Append into `v.trust_history`.

### 5.2. Simulation Wrapper: `trust/simulator.py`

`Simulator.run(steps, interactions_per_step)` is a lightweight loop:

- For each time step:
    - `simulate_interaction_step(interactions_per_step)`
    - `update_global_trust()`
- Logs progress every 10 steps.

### 5.3. Local Trust Storage Semantics: `trust/vehicle.py`

Vehicles store multiple representations depending on `model_type`:

- `PROPOSED`, `BTVR`, `COBATS`, `LT_PBFT` (Bayesian local state):
    - `interactions[target_id] = (alpha, beta)`
    - Updated via `update_parameters(...)`.
- `BSED` (behavior score):
    - `interactions[target_id] = (correct_count, total_count)`.
- `RTM`:
    - state stored in `rtm_trust[target_id]` as a scalar updated by `trust = (trust + outcome)/2`.

In all cases, `interaction_logs[target_id]` stores the raw boolean outcomes over time for analysis and windowed re-estimation.

---

## 6. Key Configuration Parameters

| Parameter | Value | Location | Description |
| :--- | :--- | :--- | :--- |
| `alpha_damping` | 0.85 | `trust/rsu.py` | PageRank damping factor. |
| `max_iter` | 100 | `trust/rsu.py` | Max iterations for centrality convergence. |
| `tol` | 1e-6 | `trust/rsu.py` | Convergence tolerance for floating point math. |
| `top_n` | 3 | `experiments/run_experiment.py` | Size of Consensus Committee. |
| `cycle_length` | 50 | `trust/vehicle.py` | Period of oscillation for Swing Attackers. |
| `interaction_rate` | 50 | `trust/trust_model.py` | Number of interactions per time step. |
| `consensus_threshold` | 0.66 (2/3) | `blockchain/validator.py` | Required weighted majority for block approval. |

---

## 7. Experiment Driver (`experiments/run_experiment.py`) — Comparative Study Pipeline

The experiment script contains both single-model simulation and a comparative multi-model evaluation pipeline.

### 7.1. Core Utilities: `calculate_statistics(...)`

This routine computes printed “Results & Statistics”:

- **Normalization:** Final global trust scores are min-max normalized across vehicles:
    $$ t' = \frac{t - \min(t)}{\max(t) - \min(t) + \epsilon} $$
    using `1e-9` as the stabilizer.
- **Detection threshold:** uses a heuristic “bottom 30%” threshold (`np.percentile(all_scores, 30)`) to label low-trust nodes as predicted malicious.
- **Consensus success rate:** approximated as `len(dags[0].blocks) / total_steps`.
- **Convergence:** builds a rank matrix and checks when >90% of nodes have rank changes within 5% of N.

### 7.2. Single Run Helper: `run_simulation(model_name, steps, num_vehicles, verbose)`

Per step:

1. Generate interactions: `num_interactions = int(num_vehicles * 0.5)`.
2. Update global trust with RSU sync.
3. Form committee: `committee_size = 5`.
4. Run consensus check:
     - `LT_PBFT`, `COBATS`: uses `check_consensus_simple` (unweighted).
     - Others: uses `check_consensus_weighted`.
5. If consensus is reached, a block is appended to `dags[0]`:
     - data payload currently `{}` in this helper.

### 7.3. Comparative Study: `run_comparative_study()`

- Models evaluated: `['BTVR', 'BSED', 'RTM', 'COBATS', 'LT_PBFT', 'PROPOSED']`.
- Stores:
    - evolution data (per model) for trust evolution plots.
    - final normalized scores for detection plots.
    - convergence and consensus metrics.

**Implementation note:** The driver treats these models as “mode switches” through shared code paths (Vehicle record/get methods + RSU aggregation branch).

---

## 8. Plotting and Metrics Formalization (`experiments/plots.py`)

Plotting utilities are not “just visualization”; they define evaluation metrics that can influence reported results.

### 8.1. Min-Max Normalization per Time Step

`normalize_histories(vehicles)` normalizes scores **row-wise** (per time step):

- Build a raw matrix of shape `(n_steps, n_vehicles)`.
- For each time step $t$, normalize across vehicles using that step’s min/max.

This makes VehicleRank’s *relative* outputs comparable over time even if absolute scale drifts.

### 8.2. Trust Evolution Plot

`plot_trust_evolution`:

- Uses `burn_in = 5` and plots individual traces with low alpha.
- Adds median trend lines for honest and malicious groups.
- Adds a “Top 30% committee cutoff” trend line using the 70th percentile at each step.

### 8.3. Detection Metrics Plot

`plot_detection_metrics`:

- Sweeps threshold percentiles from 0 to 100.
- Labels nodes as malicious if `score < thresh`.
- Computes TPR and FPR against the known ground truth `v.is_malicious`.
- Highlights the 30th percentile operating point.

### 8.4. Additional Plots (Distribution, Convergence, DAG)

The plotting module also contains:

- Final rank distribution plots.
- Trust convergence plots.
- DAG structure plots (requires `networkx` if enabled in that plotting routine).

---

## 9. SUMO/TraCI Integration (`traci_control/run_sumo.py`) — Mobility-Driven Observations

The SUMO driver replaces random pairing with proximity-based interactions using TraCI.

### 9.1. TraCI Initialization and Environment Requirements

- Attempts to import `traci` from SUMO tools directory using `SUMO_HOME`.
- Uses `sumo-gui` by default and loads `sumo/config.sumocfg`.

### 9.2. ID Mapping: SUMO Vehicles → Trust Vehicles

- SUMO vehicle IDs (e.g., `veh0`) are mapped to TrustModel IDs (e.g., `V000`).
- Mapping order is deterministic: trust IDs are sorted and assigned sequentially as SUMO IDs appear.

### 9.3. Proximity Interaction Model

Key constants:

- `INTERACTION_RANGE = 100.0` meters
- `INTERACTION_PROBABILITY = 0.3` per pair per step
- `SUMO_STEPS = 500`

Algorithm per simulation step:

1. Retrieve all active SUMO vehicles: `traci.vehicle.getIDList()`.
2. Build Trust-ID position map using `traci.vehicle.getPosition(sumo_id)`.
3. Pairwise O($N^2$) scan of mapped vehicles:
     - If distance < range and random < probability:
         - Execute bidirectional interactions (A observes B and B observes A).

This gives mobility realism while keeping trust logic fully reused.

### 9.4. Ordering Note (Consensus vs Trust Update)

Within the SUMO loop, consensus is executed **before** `sim.model.update_global_trust(sync_rsus=True)`.

- Practically: blocks appended at step $t$ may reflect trust scores from the previous global update.
- This ordering is acceptable in a prototype, but must be stated explicitly in a technical report.

---

## 10. Complexity and Performance Profile (Big-O)

Let $N$ = number of vehicles.

- Random interaction generation in `TrustModel.simulate_interaction_step(k)` is O($k$).
- VehicleRank build of matrix `M` is O($N^2$) in worst case due to iteration over reports (dense target loops in practice).
- Power iteration for VehicleRank is O($I \cdot N^2$) where $I$ is iterations until convergence (bounded by 100).
- SUMO proximity scan is O($N^2$) per step (explicit nested loops), but tuned for small N (~20).

---

## 11. Validation Checks and Known Limitations (Implementation Reality)

This section documents current “as-coded” behavior that matters for future extension:

- **No cryptographic integrity:** Block IDs are not hashes; DAG provides audit structure only.
- **No true PBFT message phases:** Weighted check approximates a commit rule but does not model prepare/commit rounds.
- **RSU sync is averaging:** There is no adversarial RSU model or byzantine RSU handling.
- **Tip set correctness is simplified:** DAG tips are tracked as a bounded list and do not enforce real ancestry-based tip definition.
- **Random seed not fixed:** Results can vary across runs unless seeds are set.

---

## 12. Extension Hooks (Where to Add “Next Research” Without Refactoring)

Concrete integration points that preserve architecture separation:

- Add interaction weighting via `update_parameters(..., weight=...)` based on distance/SNR.
- Implement explicit forgetting factor inside `RSU` as:
    $$ T_{new} = \lambda T_{old} + (1-\lambda) T_{computed} $$
    while keeping VehicleRank unchanged.
- Replace DAG tip policy with a real tip selection heuristic while keeping validator gating.
- Add an adversarial voting model where malicious nodes sometimes vote YES on bad proposals (instead of always NO), and measure how the weighted threshold reacts.
