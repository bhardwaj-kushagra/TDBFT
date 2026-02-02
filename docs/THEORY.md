# IoV Trust Management Prototype - Theoretical Framework

**Date:** January 23, 2026
**Based on:** Research Paper Sections III & IV ("Secure and Reliable Vehicular Information Sharing...")

---

## 1. Introduction

This document outlines the theoretical foundations of the implemented Trust Management System for the Internet of Vehicles (IoV). The system is designed to secure vehicular communication by identifying malicious actors through a two-layer trust model: **Bayesian Local Trust (BayesTrust)** and **Global Trust Propagation (VehicleRank)**, culminating in a **Trust-Weighted Consensus (Trust-DBFT)**.

## 2. System Model (Section III)

The system treats the vehicular network as a directed, weighted graph $G = (V, E)$, where:
*   $V$ is the set of vehicles.
*   $E$ represents interactions between vehicles.
*   Weights $w_{ij}$ represent the trust vehicle $i$ places in vehicle $j$.

### A. Local Trust Value (BayesTrust) - Section III-A

Trust is established through direct interactions. Peer-to-peer trust is calculated using Beta Probability Distribution.

*   **Interaction Tracking**:
    For every pair $(v_i, v_j)$, we track:
    *   $n_{ij}$: Total number of interactions.
    *   $y_{ij}$: Number of cooperative (successful) interactions.

*   **Bayesian Formulation**:
    We use a Beta distribution prior with hyperparameters $a=1, b=1$ (representing initial uncertainty). The posterior distribution after observing interactions is:
    $$ Beta(y_{ij} + 1, n_{ij} - y_{ij} + 1) $$

*   **Local Trust Score ($m_{ij}$)**:
    The expected value (mean) of the posterior distribution defines the local trust:
    $$ m_{ij} = \frac{y_{ij} + 1}{n_{ij} + 2} $$
    
    *   **Range**: $[0, 1]$.
    *   **Interpretation**: Probability that the next interaction with $v_j$ will be honest.

### B. Trust Graph Construction - Section III-B

RSUs (Roadside Units) aggregate these local scores to build a global view.

1.  **Adjacency Matrix**: Local trust scores form a matrix $M$ where $M_{ij} = m_{ij}$.
2.  **Normalization**: To model trust propagation as a stochastic process, edges are normalized to create a transition matrix $S$:
    $$ w_{ij} = \frac{m_{ij}}{\sum_{k \in V} m_{ik}} $$
    *   $w_{ij}$ represents the portion of $v_i$'s "trust capital" allocated to $v_j$.

### C. Global Trust Computation (VehicleRank) - Section III-C

To prevent attackers from boosting each other's scores (collusion) or swinging their behavior, we implement **VehicleRank**, an iterative algorithm inspired by PageRank.

*   **Concept**: A vehicle is trusted if it is trusted by other highly trusted vehicles.
*   **Formula**:
    The Global Trust Vector $t$ is computed iteratively:
    $$ t^{(k+1)} = \alpha \cdot S^T \cdot t^{(k)} + (1 - \alpha) \cdot E $$
    
    *   $\alpha$: Damping factor (typically 0.85). Controls the trade-off between propagation and the "teleport" probability.
    *   $S$: Stochastic trust matrix (from previous step).
    *   $E$: Teleport vector (uniform distribution $1/N$). Prevents trust sinks in the graph.
    
*   **Convergence**: The iteration continues until the change in trust scores is below a tolerance ($\epsilon = 10^{-6}$).

---

## 3. Trust-Based Committee Consensus (Section IV)

Traditional Proof-of-Work is too slow for IoV, and standard PBFT is vulnerable if >33% of nodes are malicious. We implement **Trust-DBFT**, which weighs consensus power by reputation.

### A. Committee Selection - Section IV-A

Instead of random selection, validators (Consensus Committee) are chosen based on Global Trust Ranking.

1.  **Ranking**: Vehicles are sorted descending by score from VehicleRank.
2.  **Selection**: The top $c$ vehicles form the consensus committee $C$.
3.  **Leader**: The vehicle with the maximum trust score $v_{max}$ is designated as the Primary/Leader.

### B. Trust-Weighted Voting - Section IV-B

In classical BFT, 1 Node = 1 Vote. In Trust-DBFT, **1 Unit of Trust = 1 Vote**.

*   **Voting Logic**:
    When a block is proposed:
    *   Honest committee members vote $s_i = 1$ (Approve).
    *   Malicious members vote $s_i = 0$ (Reject/DoS).

*   **Consensus Condition**:
    A block is committed if the weighted trust of approvers exceeds 2/3 of the committee's total trust mass:
    $$ \sum_{v_i \in C} (t_i \cdot s_i) \ge \frac{2}{3} \sum_{v_i \in C} t_i $$

This ensures that even if malicious nodes enter the committee, they cannot disrupt consensus unless they control >33% of the *network reputation*, which is mathematically difficult under VehicleRank.

---

## 4. Blockchain & DAG Integration

*   **Structure**: Trust updates are stored in a DAG (Directed Acyclic Graph) rather than a linear chain to allow parallel writes from different network regions.
*   **Role**: The DAG serves as the immutable ledger of trust history, ensuring past behaviors cannot be erased.

---
**References**:
1.  Bayesian Inference for Beta Distributions.
2.  PageRank Algorithm (Brin & Page, 1998).
3.  Practical Byzantine Fault Tolerance (PBFT) - Castro & Liskov.
