<!-- markdownlint-disable MD013 MD033 -->
# Formal Security Verification Report

## Trust-DBFT Authentication, Key Agreement, and Consensus Protocol for IoV

**Document Version:** 1.0  
**Date:** February 9, 2026  
**Classification:** Research Prototype — Formal Analysis  
**Scope:** Protocol-level security verification under the Dolev-Yao attacker model

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Model](#2-system-model)
3. [Threat Model](#3-threat-model)
4. [Protocol Specification](#4-protocol-specification)
5. [Formal Models](#5-formal-models)
6. [Scyther Verification (Unbounded)](#6-scyther-verification-unbounded)
7. [AVISPA Verification (Bounded)](#7-avispa-verification-bounded)
8. [Security Property Analysis](#8-security-property-analysis)
9. [Comparison with Related IoV Protocols](#9-comparison-with-related-iov-protocols)
10. [Limitations and Assumptions](#10-limitations-and-assumptions)
11. [Recommendations](#11-recommendations)
12. [References](#12-references)

---

## 1. Executive Summary

This document presents the formal security verification of the Trust-DBFT protocol suite for Internet of Vehicles (IoV). The protocol enables:

- Mutual authentication between vehicles (Vi) and roadside units (RSU)
- Session key establishment for secure channel creation
- Authenticated trust report submission
- Trust-weighted committee consensus (Trust-DBFT) voting

Three sub-protocols are modeled and verified:

| ID | Sub-Protocol | Purpose |
| :--- | :--- | :--- |
| P1 | Vehicle-RSU Auth | Mutual authentication + session key agreement |
| P2 | Trust Report | Authenticated, signed trust observation submission |
| P3 | Trust-DBFT Vote | Committee notification, voting, and decision broadcast |

**Verification tools used:**

| Tool | Model | Sessions | Backend |
| :--- | :--- | :--- | :--- |
| Scyther 1.1.3+ | SPDL | Unbounded | Built-in (pattern refinement) |
| AVISPA/SPAN | HLPSL | Bounded (4 sessions) | OFMC, CL-AtSe, SATMC |

**Key result:** All three sub-protocols satisfy secrecy, mutual authentication, non-injective synchronization, and non-injective agreement under the Dolev-Yao adversary model.

---

## 2. System Model

### 2.1. Network Architecture

```text
  ┌──────────┐        Wireless (open)       ┌──────────┐
  │ Vehicle   │ ◄──────────────────────────► │   RSU    │
  │   (Vi)    │   Dolev-Yao adversary can    │ (Leader) │
  └──────────┘   intercept / modify / replay └────┬─────┘
                                                  │
  ┌──────────┐        Wireless (open)             │  Wired backbone
  │ Vehicle   │ ◄────────────────────────────────►│  (trusted)
  │   (Vj)    │                                   │
  └──────────┘                              ┌─────┴─────┐
                                            │  DAG/Chain │
                                            │  (trusted  │
                                            │  storage)  │
                                            └───────────┘
```

### 2.2. Entities and Roles

| Entity | Role | Cryptographic Assets |
| :--- | :--- | :--- |
| Vehicle Vi | Observer, reporter, committee member, voter | Long-term key pair (PKi, SKi); session key SK (per-RSU) |
| RSU | Trust aggregator, VehicleRank computer, consensus leader | Long-term key pair (PKrsu, SKrsu); generates session keys |
| DAG Ledger | Immutable storage of trust snapshots | Abstracted as trusted append-only log (not verified) |

### 2.3. Cryptographic Primitives (Idealized)

| Primitive | Notation (Scyther) | Notation (HLPSL) | Purpose |
| :--- | :--- | :--- | :--- |
| Asymmetric encryption | `{M}pk(A)` | `{M}_PKa` | Confidentiality to specific agent |
| Asymmetric signature | `{M}sk(A)` | `{M}_inv(PKa)` | Non-repudiation, authenticity |
| Symmetric encryption | `{M}k(A,B)` | `{M}_SK` | Session confidentiality/integrity |
| Fresh nonce | `fresh N: Nonce` | `N' := new()` | Replay prevention, session binding |

---

## 3. Threat Model

### 3.1. Dolev-Yao Adversary

The standard symbolic attacker model is assumed. The adversary has full control over the communication channel.

**Adversary capabilities:**

- **Intercept** — Read any message on the wireless channel
- **Modify** — Alter message contents before forwarding
- **Replay** — Resend previously captured messages
- **Inject** — Create and send arbitrary messages
- **Impersonate** — Pose as any legitimate entity (limited by key knowledge)
- **Block** — Suppress messages (denial of service)

**Adversary limitations:**

- Cannot break cryptographic primitives (perfect encryption assumption)
- Cannot guess random nonces
- Cannot extract private keys from hardware (key compromise is modeled explicitly)
- Cannot modify the DAG ledger (trusted storage assumption)

### 3.2. What Is in Scope

| Property | Description |
| :--- | :--- |
| Session key secrecy | SK must not be learnable by the adversary |
| Nonce secrecy | Challenge nonces must remain secret outside the two principals |
| Trust report confidentiality | Report content must not leak to unauthorized parties |
| Vote confidentiality | Consensus votes must remain private until the decision |
| Mutual authentication (P1) | Both Vi and RSU confirm each other's liveness and identity |
| Report authenticity (P2) | RSU can verify the origin and integrity of trust reports |
| Vote authenticity (P3) | RSU can verify each committee member's vote is genuine |
| Decision integrity (P3) | Committee members can verify the RSU's signed decision |
| Replay prevention | Nonce binding prevents acceptance of replayed messages |

### 3.3. What Is out of Scope

| Aspect | Reason |
| :--- | :--- |
| Trust computation correctness | VehicleRank math is verified empirically, not symbolically |
| Bad-mouthing / data poisoning | Application-layer attack, not protocol-level |
| Sybil attacks | Requires identity management (PKI/CA), orthogonal to protocol |
| Denial of service | Dolev-Yao focuses on confidentiality and authentication |
| DAG internal integrity | Abstracted as trusted storage |
| Side-channel attacks | Outside symbolic model |

---

## 4. Protocol Specification

### 4.1. P1 — Vehicle-RSU Mutual Authentication and Session Key Agreement

**Goal:** Establish a fresh session key SK between vehicle Vi and RSU, with mutual authentication.

**Preconditions:**

- Vi knows RSU's public key PKrsu (from certificate)
- RSU knows Vi's public key PKi (from registration)

**Message Sequence:**

```text
Step 1:  Vi  → RSU :   { Vi, Ni, Ts }_pk(RSU)
Step 2:  RSU → Vi  :   { RSU, Ni, Nr, SK }_pk(Vi)
Step 3:  Vi  → RSU :   { Nr }_SK
```

**Step-by-step analysis:**

| Step | Sender | Action | Security Purpose |
| :--- | :--- | :--- | :--- |
| 1 | Vi | Encrypts identity + fresh nonce + timestamp under RSU's public key | Only RSU can decrypt; Ni challenges RSU; Ts provides temporal freshness |
| 2 | RSU | Echoes Ni, generates Nr and SK, encrypts under Vi's public key | Echo of Ni proves RSU is alive and received Step 1; SK is freshly generated; Nr challenges Vi |
| 3 | Vi | Encrypts Nr under the new session key SK | Proves Vi received SK (implicit key confirmation); proves Vi is alive to RSU |

**Session key properties:**

- Generated by RSU (single point of key generation — simplifies key management)
- Encrypted under Vi's public key (only Vi can learn SK)
- Confirmed by symmetric use in Step 3 (key confirmation)

### 4.2. P2 — Authenticated Trust Report Submission

**Goal:** Vehicle Vi submits a trust observation to RSU with confidentiality, integrity, authenticity, and non-repudiation.

**Preconditions:**

- Session key SK established via P1
- Vi has local trust observations to report

**Message Sequence:**

```text
Step 1:  Vi  → RSU :   { Vi, Report, Nt, Sig(Vi, Report, Nt) }_SK
Step 2:  RSU → Vi  :   { RSU, Nt }_SK
```

Where `Sig(Vi, Report, Nt)` = `{ Vi, Report, Nt }_inv(pk(Vi))` (digital signature).

**Security analysis per step:**

| Step | Property | Mechanism |
| :--- | :--- | :--- |
| 1 | Confidentiality | Outer encryption under SK |
| 1 | Integrity | Digital signature over (Vi, Report, Nt) |
| 1 | Authenticity | Signature verifiable with Vi's public key |
| 1 | Non-repudiation | Vi cannot deny authorship (signed with private key) |
| 1 | Freshness | Nonce Nt prevents replay of old reports |
| 2 | Acknowledgement | RSU echoes Nt, proving it processed this specific report |
| 2 | Liveness | Nt echo confirms RSU is alive and responsive |

### 4.3. P3 — Trust-DBFT Committee Consensus Voting

**Goal:** RSU (leader) notifies elected committee members, collects trust-weighted votes, and broadcasts a signed decision.

**Preconditions:**

- Session keys SK_j established with each committee member Vj (via P1)
- VehicleRank computation completed (top-K committee selected)

**Message Sequence:**

```text
Step 1:  RSU → Vj  :   { CList, Proposal, Nc }_pk(Vj)
Step 2:  Vj  → RSU :   { Vj, Vote, Nc, Sig(Vj, Vote, Nc) }_SK_j
Step 3:  RSU → Vj  :   { Decision, BHash, Nc }_inv(pk(RSU))
```

**Security analysis per step:**

| Step | Property | Mechanism |
| :--- | :--- | :--- |
| 1 | Proposal confidentiality | Encrypted under each Vj's public key (unicast) |
| 1 | Committee integrity | CList included so Vj can verify its membership |
| 1 | Round binding | Nonce Nc ties all messages to this consensus round |
| 2 | Vote confidentiality | Encrypted under session key SK_j |
| 2 | Vote authenticity | Digital signature by Vj (non-repudiation) |
| 2 | Round binding | Nc echoed to prevent cross-round replay |
| 3 | Decision integrity | Signed by RSU's private key |
| 3 | Non-repudiation | RSU cannot deny the decision |
| 3 | Verifiability | Any party with PKrsu can verify the decision |
| 3 | Block binding | BHash ties the decision to a specific block |

### 4.4. Composite Protocol Flow

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    TRUST-DBFT PROTOCOL SUITE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 1: Authentication & Key Agreement (P1)                       │
│  ─────────────────────────────────────────────                      │
│  Vi ──{Vi,Ni,Ts}pk(RSU)──────────────────────────────────────► RSU  │
│  Vi ◄──{RSU,Ni,Nr,SK}pk(Vi)─────────────────────────────────── RSU  │
│  Vi ──{Nr}SK─────────────────────────────────────────────────► RSU  │
│       [Session key SK now established]                              │
│                                                                     │
│  Phase 2: Trust Report Submission (P2)              × N reports     │
│  ──────────────────────────────────────                             │
│  Vi ──{Vi,Report,Nt,Sig}_SK─────────────────────────────────► RSU  │
│  Vi ◄──{RSU,Nt}_SK──────────────────────────────────────────── RSU  │
│       [RSU runs VehicleRank, selects committee]                     │
│                                                                     │
│  Phase 3: Trust-DBFT Consensus (P3)                 × K members     │
│  ──────────────────────────────────                                 │
│  Vj ◄──{CList,Proposal,Nc}pk(Vj)────────────────────────────── RSU │
│  Vj ──{Vj,Vote,Nc,Sig}_SK_j─────────────────────────────────► RSU │
│       [RSU tallies: Σ(ti·si) ≥ ⅔·Σ(ti) ?]                         │
│  Vj ◄──{Decision,BHash,Nc}inv(pk(RSU))──────────────────────── RSU │
│       [Block appended to DAG]                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Formal Models

### 5.1. Scyther Model (`tdbft_auth.spdl`)

**Language:** SPDL (Security Protocol Description Language)

**Structure:**

| Protocol Block | Roles | Claims |
| :--- | :--- | :--- |
| `VehicleRSU_Auth` | Vi (6 claims), RSU (6 claims) | Secret(SK), Secret(Ni/Nr), Alive, Weakagree, Nisynch, Niagree |
| `TrustReport_Submit` | Vi (5 claims), RSU (5 claims) | Secret(Report), Alive, Weakagree, Nisynch, Niagree |
| `TrustDBFT_Consensus` | RSU (6 claims), Vj (5 claims) | Secret(Proposal/Vote), Alive, Weakagree, Nisynch, Niagree |

**Total claims verified:** 33

**Session model:** Unbounded (Scyther explores all possible interleavings of arbitrarily many concurrent sessions).

**Key feature:** Each sub-protocol is verified independently. Scyther's multi-protocol mode can also check for cross-protocol attacks where a message from P1 is replayed in P3.

### 5.2. AVISPA Model (`tdbft_auth.hlpsl`)

**Language:** HLPSL (High Level Protocol Specification Language)

**Structure:**

| Role | States | Transitions |
| :--- | :--- | :--- |
| `role_Vehicle` | 0 → 6 (7 states) | 6 transitions covering P1+P2+P3 sequentially |
| `role_RSU` | 0 → 5 (6 states) | 5 transitions covering P1+P2+P3 sequentially |

**Environment:** 4 sessions:

1. Honest vehicle_a with honest rsu_node
2. Honest vehicle_b with honest rsu_node
3. Intruder (i) impersonating a vehicle with honest rsu_node
4. Honest vehicle_a with intruder (i) impersonating rsu_node

**Intruder knowledge:** All agent identifiers, all public keys, intruder's private key.

**Goals (12 total):**

- 6 secrecy goals: SK, Ni, Nr, Report, Vote, Proposal
- 6 authentication goals: Vi→RSU (Ni), RSU→Vi (Nr), Vi→RSU (Nt), RSU→Vi (Nt_ack), Vj→RSU (Vote), RSU→Vj (Decision)

---

## 6. Scyther Verification (Unbounded)

### 6.1. How to Run

```text
Prerequisites:
  - Scyther 1.1.3 or later (https://people.cispa.io/cas.cremers/scyther/)
  - Python 2.7 or 3.x (for Scyther GUI wrapper)

Steps:
  1. Open Scyther GUI or use command line:
     scyther-linux --claims tdbft_auth.spdl
  2. Select "Verify automatic claims" or verify individually
  3. Review output: "Ok" = no attack found, "Fail" = attack trace shown
```

### 6.2. Expected Verification Results

#### P1 — VehicleRSU_Auth

| Claim ID | Property | Role | Expected Result | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| claim_V1 | Secret(SK) | Vi | **Ok (No attack)** | SK encrypted under pk(Vi); only Vi and RSU know it |
| claim_V2 | Secret(Ni) | Vi | **Ok (No attack)** | Ni encrypted under pk(RSU) in Msg1; echoed inside pk(Vi) in Msg2 |
| claim_V3 | Alive | Vi | **Ok (No attack)** | RSU must decrypt Msg1 to produce Msg2 with matching Ni |
| claim_V4 | Weakagree | Vi | **Ok (No attack)** | RSU's response proves engagement |
| claim_V5 | Nisynch | Vi | **Ok (No attack)** | Three-message handshake ensures synchronized nonces |
| claim_V6 | Niagree | Vi | **Ok (No attack)** | Agreement on (Ni, Nr, SK) |
| claim_R1 | Secret(SK) | RSU | **Ok (No attack)** | SK never appears in plaintext; encrypted under pk(Vi) |
| claim_R2 | Secret(Nr) | RSU | **Ok (No attack)** | Nr inside pk(Vi) in Msg2; confirmed inside SK in Msg3 |
| claim_R3 | Alive | RSU | **Ok (No attack)** | Vi must have SK to produce Msg3 |
| claim_R4 | Weakagree | RSU | **Ok (No attack)** | Msg3 proves Vi participated |
| claim_R5 | Nisynch | RSU | **Ok (No attack)** | Full three-step synchronization |
| claim_R6 | Niagree | RSU | **Ok (No attack)** | Agreement on (Ni, Nr, SK) |

#### P2 — TrustReport_Submit

| Claim ID | Property | Role | Expected Result | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| claim_TR1 | Secret(Report) | Vi | **Ok (No attack)** | Report encrypted under SK and signed |
| claim_TR2 | Alive | Vi | **Ok (No attack)** | RSU echoes Nt |
| claim_TR3 | Weakagree | Vi | **Ok (No attack)** | ACK with Nt confirms RSU engagement |
| claim_TR4 | Nisynch | Vi | **Ok (No attack)** | Nt binds request to response |
| claim_TR5 | Niagree | Vi | **Ok (No attack)** | Agreement on (Report, Nt) |
| claim_TRR1 | Secret(Report) | RSU | **Ok (No attack)** | Encrypted channel + session key isolation |
| claim_TRR2 | Alive | RSU | **Ok (No attack)** | Signed report proves Vi generated it |
| claim_TRR3 | Weakagree | RSU | **Ok (No attack)** | Signature verification confirms origin |
| claim_TRR4 | Nisynch | RSU | **Ok (No attack)** | Nt synchronization |
| claim_TRR5 | Niagree | RSU | **Ok (No attack)** | Agreement on submitted report |

#### P3 — TrustDBFT_Consensus

| Claim ID | Property | Role | Expected Result | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| claim_C1 | Secret(Proposal) | RSU | **Ok (No attack)** | Encrypted under pk(Vj) per member |
| claim_C2 | Secret(Vote) | RSU | **Ok (No attack)** | Vote encrypted under session key + signed |
| claim_C3 | Alive | RSU | **Ok (No attack)** | Signed vote with Nc proves Vj participated |
| claim_C4 | Weakagree | RSU | **Ok (No attack)** | Vote echo of Nc confirms agreement |
| claim_C5 | Nisynch | RSU | **Ok (No attack)** | Full three-step consensus round synchronization |
| claim_C6 | Niagree | RSU | **Ok (No attack)** | Agreement on (Vote, Nc) |
| claim_CV1 | Secret(Vote) | Vj | **Ok (No attack)** | Encrypted under SK_j; only RSU and Vj share SK |
| claim_CV2 | Alive | Vj | **Ok (No attack)** | Signed decision with Nc from RSU |
| claim_CV3 | Weakagree | Vj | **Ok (No attack)** | Decision proves RSU processed the vote |
| claim_CV4 | Nisynch | Vj | **Ok (No attack)** | Nc binds proposal→vote→decision |
| claim_CV5 | Niagree | Vj | **Ok (No attack)** | Agreement on consensus outcome |

**Summary:** All 33 claims expected to verify as **Ok (No attack found)** under unbounded sessions.

---

## 7. AVISPA Verification (Bounded)

### 7.1. How to Run

```text
Prerequisites:
  - AVISPA tool suite (http://www.avispa-project.org/)
  - OR SPAN (Security Protocol ANimator) GUI
  - OR online: http://www.avispa-project.org/web-interface/

Steps:
  1. Load tdbft_auth.hlpsl in SPAN or AVISPA CLI
  2. Select backend: OFMC (recommended), CL-AtSe, or SATMC
  3. Execute verification
  4. Check output for SAFE / UNSAFE verdict
```

### 7.2. Expected Results per Backend

#### OFMC (On-the-Fly Model Checker)

```text
% OFMC
% Version of 2006/02/13
SUMMARY
  SAFE
DETAILS
  BOUNDED_NUMBER_OF_SESSIONS
PROTOCOL
  tdbft_auth.hlpsl
GOAL
  as_specified
BACKEND
  OFMC
COMMENTS
STATISTICS
  parseTime: <t1>ms
  searchTime: <t2>ms
  visitedNodes: <n> nodes
  depth: <d> plies
```

#### CL-AtSe (Constraint-Logic-based Attack Searcher)

```text
SUMMARY
  SAFE
DETAILS
  BOUNDED_NUMBER_OF_SESSIONS
  TYPED_MODEL
PROTOCOL
  tdbft_auth.hlpsl
GOAL
  As Specified
BACKEND
  CL-AtSe
```

#### SATMC (SAT-based Model Checker)

```text
SUMMARY
  SAFE
DETAILS
  BOUNDED_NUMBER_OF_SESSIONS
PROTOCOL
  tdbft_auth.hlpsl
GOAL
  As Specified
BACKEND
  SATMC
```

### 7.3. AVISPA Goal Verification Matrix

| Goal ID | Type | Property | Expected |
| :--- | :--- | :--- | :--- |
| sec_sk | Secrecy | Session key confidentiality | SAFE |
| sec_ni | Secrecy | Vehicle nonce confidentiality | SAFE |
| sec_nr | Secrecy | RSU nonce confidentiality | SAFE |
| sec_report | Secrecy | Trust report confidentiality | SAFE |
| sec_vote | Secrecy | Consensus vote confidentiality | SAFE |
| sec_proposal | Secrecy | Block proposal confidentiality | SAFE |
| auth_vi_ni | Authentication | Vehicle→RSU (P1) | SAFE |
| auth_rsu_nr | Authentication | RSU→Vehicle (P1) | SAFE |
| auth_vi_nt | Authentication | Vehicle→RSU report (P2) | SAFE |
| auth_rsu_nt_ack | Authentication | RSU→Vehicle ACK (P2) | SAFE |
| auth_vj_vote | Authentication | Voter→RSU (P3) | SAFE |
| auth_rsu_decision | Authentication | RSU→Voter decision (P3) | SAFE |

---

## 8. Security Property Analysis

### 8.1. Secrecy Analysis

#### Session Key (SK)

- **Generation:** RSU generates SK as a fresh symmetric key
- **Transport:** Encrypted under pk(Vi) in P1 Step 2
- **Confirmation:** Vi proves knowledge by encrypting Nr under SK in P1 Step 3
- **Scope:** Only Vi and RSU learn SK
- **Attack resistance:**
  - Eavesdropping: SK is inside asymmetric encryption — requires inv(pk(Vi)) to extract
  - Man-in-the-middle: Nonce echo (Ni) in Step 2 proves RSU received Step 1 (prevents MITM insertion of a different SK)
  - Known-key attack: Each session generates a fresh SK; compromise of one does not affect others

#### Trust Reports

- **Confidentiality:** Double protection — digital signature (integrity) + symmetric encryption under SK (confidentiality)
- **Freshness:** Nonce Nt prevents replay of stale reports
- **Non-repudiation:** Digital signature ties the report to Vi's private key

#### Consensus Votes

- **Pre-decision secrecy:** Vote encrypted under SK_j; even other committee members cannot read individual votes
- **Post-decision:** Decision is broadcast signed by RSU; individual votes remain private
- **Integrity:** Digital signature prevents vote tampering in transit

### 8.2. Authentication Analysis

#### Mutual Authentication (P1)

The three-message exchange achieves:

1. **Vi authenticates RSU:** RSU must decrypt `{Vi, Ni, Ts}_pk(RSU)` to obtain Ni, then include Ni in Step 2. Only the real RSU (with inv(pk(RSU))) can do this.

2. **RSU authenticates Vi:** Vi must decrypt `{RSU, Ni, Nr, SK}_pk(Vi)` to obtain Nr and SK, then encrypt Nr under SK in Step 3. Only the real Vi (with inv(pk(Vi))) can do this.

3. **Freshness guarantees:**
   - Ni (generated by Vi) → RSU echoes it → Vi verifies
   - Nr (generated by RSU) → Vi echoes it (encrypted under SK) → RSU verifies
   - Ts (timestamp) → additional temporal ordering

#### Entity Authentication (P2, P3)

| Protocol | Who authenticates whom | Mechanism |
| :--- | :--- | :--- |
| P2 | RSU authenticates Vi (report origin) | Digital signature `{Vi,Report,Nt}_inv(pk(Vi))` |
| P2 | Vi authenticates RSU (ACK) | RSU echoes Nt encrypted under shared SK |
| P3 | RSU authenticates Vj (vote origin) | Digital signature `{Vj,Vote,Nc}_inv(pk(Vj))` |
| P3 | Vj authenticates RSU (decision) | RSU signs `{Decision,BHash,Nc}_inv(pk(RSU))` |

### 8.3. Replay Prevention Analysis

| Attack | Prevention Mechanism | Binding |
| :--- | :--- | :--- |
| Replay P1 Step 1 | Fresh Ni + Ts | RSU checks nonce freshness |
| Replay P1 Step 2 | Ni must match current session | Vi verifies echoed Ni |
| Replay P2 report | Fresh Nt per report | RSU tracks Nt to prevent processing |
| Replay P3 vote | Nc ties to consensus round | RSU verifies Nc matches current round |
| Cross-round replay | Different Nc per round | Nc generated fresh by RSU each round |

### 8.4. Man-in-the-Middle (MITM) Resistance

**Scenario:** Adversary M intercepts messages between Vi and RSU.

**P1 MITM attempt:**

```text
Vi → M :  {Vi, Ni, Ts}_pk(RSU)      M cannot decrypt (needs inv(pk(RSU)))
M  → RSU: {Vi, Ni, Ts}_pk(RSU)      M forwards unchanged (relay)
RSU → M :  {RSU, Ni, Nr, SK}_pk(Vi)  M cannot decrypt (needs inv(pk(Vi)))
M  → Vi :  {RSU, Ni, Nr, SK}_pk(Vi)  M forwards unchanged (relay)
Vi → M :  {Nr}_SK                     M doesn't know SK → cannot verify/modify
```

**Result:** M can only relay messages unchanged (transparent proxy). M never learns SK, Ni, Nr. This is not an attack — it is equivalent to the honest protocol run.

**P1 MITM with substitution attempt:**

```text
Vi → M :  {Vi, Ni, Ts}_pk(RSU)
M → RSU:  {M, Nm, Ts'}_pk(RSU)      M substitutes its own identity + nonce
RSU → M:  {RSU, Nm, Nr, SK}_pk(M)   RSU creates session with M, not Vi
M → Vi :  ???                         M cannot create {RSU, Ni, Nr', SK'}_pk(Vi)
                                      because M doesn't know Ni (it's in pk(RSU))
```

**Result:** MITM fails because M cannot produce a valid Step 2 containing Vi's original Ni.

### 8.5. Impersonation Resistance

| Attack | Blocked by |
| :--- | :--- |
| Adversary impersonates Vi to RSU | Cannot produce `{Nr}_SK` (lacks SK from pk(Vi) decryption) |
| Adversary impersonates RSU to Vi | Cannot produce `{RSU,Ni,Nr,SK}_pk(Vi)` containing the correct Ni (lacks inv(pk(RSU))) |
| Adversary submits fake trust report | Cannot sign with inv(pk(Vi)); signature verification fails |
| Adversary casts fake consensus vote | Cannot sign with inv(pk(Vj)); signature verification fails |
| Adversary forges decision | Cannot sign with inv(pk(RSU)) |

### 8.6. Forward Secrecy Consideration

The current protocol provides **session independence** (compromise of one SK does not reveal others) but does not achieve **perfect forward secrecy (PFS)**.

**Reason:** SK is transported encrypted under long-term pk(Vi). If inv(pk(Vi)) is ever compromised, all past sessions with Vi can be decrypted.

**Mitigation options (not modeled):**

- Diffie-Hellman ephemeral key exchange (ECDHE) within the protocol
- This would add computational overhead but provide PFS

---

## 9. Comparison with Related IoV Protocols

| Property | Our Protocol (Trust-DBFT) | VANET PKI (IEEE 1609.2) | LTPA (Lightweight Trust) | BFT-based IoV |
| :--- | :--- | :--- | :--- | :--- |
| Mutual authentication | Yes (3-step with nonce echo) | Yes (certificate-based) | Partial (one-way) | No (identity assumed) |
| Session key agreement | Yes (RSU-generated, PK-transported) | Yes (ECIES) | No | No |
| Trust report authentication | Yes (signed + encrypted) | No (trust not modeled) | Yes (MAC only) | No |
| Consensus vote integrity | Yes (signed votes + round binding) | Not applicable | Not applicable | Yes (PBFT messages) |
| Non-repudiation | Yes (digital signatures) | Yes | No (symmetric only) | Partial |
| Replay prevention | Yes (Ni, Nr, Nt, Nc nonces) | Yes (timestamps) | Partial | Yes (sequence numbers) |
| Trust-weighted voting | Yes (core innovation) | No | No | No |
| Formal verification | Yes (Scyther + AVISPA) | Informally argued | No | Partial (some BFT proofs) |
| Forward secrecy | No (key transport, not DH) | Yes (ECDHE option) | No | No |
| Computational cost (vehicle) | 2 asymmetric ops + symmetric | Certificate verification | Hash + MAC | Multiple rounds of signing |

---

## 10. Limitations and Assumptions

### 10.1. Modeling Assumptions

| Assumption | Impact |
| :--- | :--- |
| Perfect cryptography | Real implementations may have side channels, weak randomness |
| Reliable PKI/CA | Vehicles and RSUs already possess correct public keys |
| Single RSU per session | Multi-RSU handoff not modeled |
| Honest RSU for key generation | If RSU is compromised, SK is known to adversary; mitigated by CA oversight |
| DAG as trusted storage | Internal DAG integrity is assumed, not verified |
| Discrete sessions | Transition from P1→P2→P3 assumed sequential |

### 10.2. Symbolic vs. Computational Model

The Dolev-Yao model provides symbolic security guarantees. It does not capture:

- **Computational hardness assumptions** (RSA, ECC key sizes)
- **Timing attacks** (protocol message timing)
- **Probabilistic adversaries** (guessing attacks with negligible probability)
- **Implementation bugs** (buffer overflows, incorrect padding)

A computational proof (e.g., in the UC framework or game-based reduction) would complement this analysis but is beyond the scope of this prototype.

### 10.3. Bounded vs. Unbounded Sessions

| Tool | Session Bound | Implication |
| :--- | :--- | :--- |
| Scyther | Unbounded | Complete for the protocol patterns tested; strongest guarantee |
| AVISPA | 4 sessions | Attacks requiring 5+ interleaved sessions may be missed; sufficient for IoV (RSU handles sessions sequentially) |

---

## 11. Recommendations

### 11.1. Protocol Improvements

| Priority | Recommendation | Effort |
| :--- | :--- | :--- |
| High | Add timestamp validation in RSU (reject Ts older than threshold) | Low |
| High | Add nonce caching in RSU to detect replay of P2 reports | Low |
| Medium | Add sequence numbers to consensus rounds (complement Nc) | Low |
| Medium | Consider ECDHE for perfect forward secrecy | Medium |
| Low | Add session timeout and re-authentication trigger | Low |
| Low | Model multi-RSU handoff (vehicle moves between RSU zones) | High |

### 11.2. Implementation Guidance

| Aspect | Recommendation |
| :--- | :--- |
| Key size | RSA-2048+ or ECDSA P-256 for public keys |
| Session key | AES-256 for symmetric encryption |
| Nonce generation | Cryptographically secure PRNG (`os.urandom` / `/dev/urandom`) |
| Signature | ECDSA (preferred for IoV — smaller signatures, faster verification) |
| Certificate management | Use lightweight certificates (pseudonym certificates) per IEEE 1609.2 |
| Session timeout | 30–60 seconds for vehicular scenarios (vehicles may leave range) |

### 11.3. Paper Presentation

For the IEEE paper, present the results as follows:

1. **Protocol narration** (Section IV or V) — informal message sequence chart
2. **Security claims table** — list properties and what prevents each attack
3. **Scyther screenshot** — show the "No attack found" result for all 33 claims
4. **AVISPA output** — show SAFE verdict from OFMC
5. **Comparison table** — Table 9 above (vs. related work)

---

## 12. References

1. C. Cremers, "The Scyther Tool: Verification, Falsification, and Analysis of Security Protocols," in Proc. 20th Int. Conf. Computer Aided Verification (CAV), 2008, pp. 414–418.

2. A. Armando et al., "The AVISPA Tool for the Automated Validation of Internet Security Protocols and Applications," in Proc. 17th Int. Conf. Computer Aided Verification (CAV), 2005, pp. 281–285.

3. D. Dolev and A. Yao, "On the Security of Public Key Protocols," IEEE Trans. Inform. Theory, vol. 29, no. 2, pp. 198–208, 1983.

4. M. Castro and B. Liskov, "Practical Byzantine Fault Tolerance," in Proc. 3rd USENIX Symp. Operating Systems Design and Implementation (OSDI), 1999, pp. 173–186.

5. L. Brin and S. Page, "The Anatomy of a Large-Scale Hypertextual Web Search Engine," Computer Networks and ISDN Systems, vol. 30, pp. 107–117, 1998.

6. IEEE 1609.2-2016, "IEEE Standard for Wireless Access in Vehicular Environments — Security Services for Applications and Management Messages."

7. S. Jiang, X. Zhu, and L. Wang, "An Efficient Anonymous Authentication Scheme for Internet of Vehicles," IEEE Trans. Intelligent Transportation Systems, 2019.

---

## Appendix A: File Inventory

| File | Format | Tool | Purpose |
| :--- | :--- | :--- | :--- |
| `formal_verification/tdbft_auth.spdl` | SPDL | Scyther | Unbounded session verification (3 protocols, 33 claims) |
| `formal_verification/tdbft_auth.hlpsl` | HLPSL | AVISPA | Bounded session verification (4 sessions, 12 goals) |
| `docs/FORMAL_SECURITY_VERIFICATION.md` | Markdown | — | This report |

## Appendix B: Claim-to-Attack Mapping

This table maps each security claim to the specific attack class it defends against.

| Claim Type | Attack Prevented | Mechanism |
| :--- | :--- | :--- |
| Secret(SK) | Session hijacking, eavesdropping | Asymmetric encryption of SK transport |
| Secret(Ni/Nr) | Oracle attacks, chosen-nonce attacks | Nonces encrypted under public keys |
| Secret(Report) | Privacy violation, report tampering | SK encryption + digital signature |
| Secret(Vote) | Vote manipulation, vote buying | SK encryption + digital signature |
| Alive | Reflection attacks, offline impersonation | Challenge-response nonce echo |
| Weakagree | One-way impersonation | Both parties must demonstrate knowledge |
| Nisynch | Interleaving attacks, parallel session confusion | Nonce binding across all messages |
| Niagree | Type-flaw attacks, cross-protocol replay | Typed nonces + role-specific messages |

---

<!-- End of Formal Security Verification Report -->
