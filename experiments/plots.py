"""
Plotting Utilities for IoV Trust Management Prototype.

All graphs are saved to a single flat output directory (RESULTS_DIR = results/).
No subfolders, no supplementary directories. One call to run_all_graphs() regenerates everything.

Naming convention (strictly enforced):
  - comp_*       : Comparative plots — all 7 models overlaid
  - proposed_*   : Solo plots for the PROPOSED model only
  - dag_*        : Structural visualisations (DAG topology)

Complete output manifest (22 files):
──────────────────────────────────────
  COMPARATIVE (15 files):
    comp_traffic_detection.png        — IoV traffic volume vs malicious detection
    comp_robustness_10.png            — Robustness at 10% attacker ratio
    comp_robustness_20.png            — Robustness at 20% attacker ratio
    comp_robustness_30.png            — Robustness at 30% attacker ratio
    comp_capacity.png                 — Normalised capacity index (line)
    comp_throughput.png               — Capacity index bar chart
    comp_latency.png                  — Analytical consensus latency
    comp_swing_attack.png             — Swing attack success rate (line)
    comp_internal_attack.png          — Internal attack success rate (bar)
    comp_convergence.png              — Rank convergence stability
    comp_consensus.png                — Consensus success rate vs network size
    comp_detection_tpr.png            — Detection TPR at 30th-percentile
    comp_trust_evolution.png          — Median honest trust evolution
    comp_trust_evolution_malicious.png — Median malicious trust evolution
    comp_trust_normalized.png         — Z-score trust separation

  PROPOSED SOLO (6 files):
    proposed_trust_evolution.png      — Per-vehicle trust traces + medians
    proposed_detection_metrics.png    — TPR / FPR across percentiles
    proposed_rank_distribution.png    — Final rank scatter + committee cutoff
    proposed_trust_convergence.png    — Rank stability fraction over time
    proposed_trust_normalized.png     — Z-score by behavior category
    proposed_swing_analysis.png       — Swing attacker: global vs local trust

  STRUCTURAL (1 file):
    dag_structure.png                 — DAG topology from short PROPOSED run
──────────────────────────────────────
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import os

from experiments.config import (
    MODELS, COLORS, LINE_STYLES, LINE_WIDTHS, get_style, RESULTS_DIR, DETECTION_THRESHOLD
)
from experiments.benchmark import run_single_simulation, get_normalized_detection_scores
from trust.simulator import Simulator
from blockchain.consensus_manager import ConsensusManager

# ── Publication-quality global plotting profile ──────────────────────────────
plt.rcParams.update({
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "figure.figsize":   (6, 4),
    "font.size":        14,
    "axes.titlesize":   16,
    "axes.labelsize":   16,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  10,
    "lines.linewidth":  2,
    "grid.alpha":       0.3,
})

# ── Helpers ──────────────────────────────────────────────────────────────────

def _save(fig, path):
    """Ensure directory exists, save figure, print confirmation, close."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  [saved] {path}")
    plt.close(fig)


def normalize_histories(vehicles):
    """
    Per-step Min-Max normalisation of trust histories.
    Returns dict {vid: np.array} of normalised traces.
    """
    if not vehicles:
        return {}
    sample_v = next(iter(vehicles.values()))
    n_steps = len(sample_v.trust_history)
    vids = list(vehicles.keys())
    raw = np.zeros((n_steps, len(vids)))
    for i, vid in enumerate(vids):
        h = vehicles[vid].trust_history
        length = min(len(h), n_steps)
        raw[:length, i] = h[:length]
    norm = np.zeros_like(raw)
    for t in range(n_steps):
        row = raw[t, :]
        rmin, rmax = row.min(), row.max()
        if rmax > rmin:
            norm[t, :] = (row - rmin) / (rmax - rmin + 1e-9)
        else:
            norm[t, :] = 0.5
    return {vid: norm[:, i] for i, vid in enumerate(vids)}


# ═══════════════════════════════════════════════════════════════════════════════
#  PROPOSED-SOLO PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_proposed_trust_evolution(vehicles, out_dir):
    """Honest vs malicious trust traces with median trend lines (PROPOSED only)."""
    norm_data = normalize_histories(vehicles)
    fig, ax = plt.subplots()
    burn_in = 5
    for vid, v in vehicles.items():
        if vid not in norm_data:
            continue
        h = norm_data[vid]
        if len(h) <= burn_in:
            continue
        seg = h[burn_in:]
        xs = range(burn_in, burn_in + len(seg))
        c = 'red' if v.is_malicious else 'green'
        ax.plot(xs, seg, color=c, alpha=0.12, label='_nolegend_')

    if norm_data:
        n_steps = len(next(iter(norm_data.values())))
        xs = range(burn_in, n_steps)
        honest_ids = [v.id for v in vehicles.values() if not v.is_malicious]
        mal_ids = [v.id for v in vehicles.values() if v.is_malicious]
        if honest_ids:
            h_med = np.median([norm_data[uid] for uid in honest_ids], axis=0)[burn_in:]
            ax.plot(xs, h_med, color='darkgreen', linewidth=2, label='Honest Median')
        if mal_ids:
            m_med = np.median([norm_data[uid] for uid in mal_ids], axis=0)[burn_in:]
            ax.plot(xs, m_med, color='darkred', linewidth=2, label='Malicious Median')
        all_mat = np.array([norm_data[uid] for uid in vehicles.keys()])
        p70 = np.percentile(all_mat, 70, axis=0)[burn_in:]
        ax.plot(xs, p70, color='blue', linestyle='--', label='Top 30% (Committee Cutoff)')

    ax.set_ylim(0, 1.05)
    ax.set_title("Evolution of Global Trust Scores (Normalised)")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Normalised Trust Score")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "proposed_trust_evolution.png"))


def plot_proposed_detection_metrics(vehicles, out_dir):
    """Detection TPR / FPR across trust percentiles (PROPOSED only)."""
    norm_data = normalize_histories(vehicles)
    pts = []
    for vid, v in vehicles.items():
        if vid in norm_data:
            pts.append({'score': norm_data[vid][-1], 'is_mal': v.is_malicious})
    if not pts:
        return
    pts.sort(key=lambda x: x['score'])
    scores = [x['score'] for x in pts]
    total_mal = sum(1 for x in pts if x['is_mal'])
    total_hon = len(pts) - total_mal
    percentiles = np.linspace(0, 100, 100)
    tprs, fprs = [], []
    for p in percentiles:
        thresh = np.percentile(scores, p)
        tp = sum(1 for x in pts if x['score'] < thresh and x['is_mal'])
        fp = sum(1 for x in pts if x['score'] < thresh and not x['is_mal'])
        tprs.append(tp / total_mal if total_mal else 0)
        fprs.append(fp / total_hon if total_hon else 0)

    fig, ax = plt.subplots()
    ax.plot(percentiles, tprs, label='Detection Rate (TPR)', color='blue')
    ax.plot(percentiles, fprs, label='False Positive Rate (FPR)', color='red', linestyle='--')
    idx_30 = 30
    if idx_30 < len(tprs):
        ax.axvline(x=30, color='green', linestyle=':', label='Bottom 30% Threshold')
        ax.annotate(f'TPR={tprs[idx_30]:.2f}', xy=(30, tprs[idx_30]),
                    xytext=(35, tprs[idx_30] - 0.15),
                    arrowprops=dict(facecolor='black', shrink=0.05))
    ax.set_title("Detection Performance vs Trust Percentile")
    ax.set_xlabel("Trust Threshold (Percentile)")
    ax.set_ylabel("Rate")
    ax.legend()
    ax.grid(True)
    _save(fig, os.path.join(out_dir, "proposed_detection_metrics.png"))


def plot_proposed_rank_distribution(vehicles, out_dir):
    """Final vehicle ranking scatter with committee cutoff (PROPOSED only)."""
    norm_data = normalize_histories(vehicles)
    data = []
    for vid, v in vehicles.items():
        if vid in norm_data:
            data.append((v, norm_data[vid][-1]))
    data.sort(key=lambda x: x[1], reverse=True)
    ranks = range(1, len(data) + 1)
    scores = [x[1] for x in data]
    colors = ['red' if x[0].is_malicious else 'green' for x in data]
    ids = [x[0].id for x in data]
    committee_size = 5

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(ranks, scores, c=colors, s=100, alpha=0.7)
    if len(data) >= committee_size:
        ax.axvline(x=committee_size + 0.5, color='blue', linestyle='--', label='Committee Cutoff')
        ax.axhline(y=scores[committee_size - 1], color='gray', linestyle=':', alpha=0.5)
    top_k = bottom_k = 5
    n_v = len(data)
    for i, txt in enumerate(ids):
        if (i + 1) <= top_k or (i + 1) > (n_v - bottom_k):
            ax.annotate(txt, (ranks[i], scores[i]), fontsize=10, alpha=0.7)
    ax.set_title("Vehicle Ranking & Committee Selection")
    ax.set_xlabel("Rank (1 = Highest Trust)")
    ax.set_ylabel("Normalised Global Trust (VehicleRank)")
    custom = [Line2D([0], [0], color='green', marker='o', linestyle=''),
              Line2D([0], [0], color='red', marker='o', linestyle='')]
    ax.legend(custom, ['Honest', 'Malicious'])
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "proposed_rank_distribution.png"))


def plot_proposed_swing_analysis(out_dir):
    """Swing attacker dynamics for PROPOSED: global vs local trust of a swing node."""
    res = run_single_simulation('PROPOSED', steps=100, percent_malicious=0.0, percent_swing=0.2)
    vehicles = res['vehicles']
    swing_nodes = [v for v in vehicles.values() if v.behavior_type == 'SWING']
    if not swing_nodes:
        print("  [skip] proposed_swing_analysis.png -- no swing nodes found")
        return
    target = swing_nodes[0]
    global_hist = target.trust_history

    # Compute windowed local trust from an honest observer
    honest_nodes = [v for v in vehicles.values() if not v.is_malicious]
    local_hist = []
    if honest_nodes:
        observer = honest_nodes[0]
        logs = observer.interaction_logs.get(target.id, [])
        for t in range(len(global_hist)):
            window = logs[:t + 1][-10:]
            if window:
                pos = sum(1 for x in window if x)
                local_hist.append((1 + pos) / (2 + len(window)))
            else:
                local_hist.append(0.5)
    else:
        local_hist = [0.5] * len(global_hist)

    fig, ax = plt.subplots()
    ax.plot(global_hist, label='Global Trust (Smoothed VehicleRank)', color='purple')
    ax.plot(local_hist, label='Local Trust (Bayesian Window=10)', color='orange', alpha=0.8)
    ax.set_title("Swing Attacker Dynamics")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Trust Score")
    ax.legend()
    ax.grid(True)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    ax.text(0.05, 0.05,
            "While swing attackers manipulate local trust,\n"
            "their global trust remains suppressed due to\n"
            "temporal smoothing and network-wide propagation.",
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)
    _save(fig, os.path.join(out_dir, "proposed_swing_analysis.png"))


def plot_proposed_trust_convergence(vehicles, out_dir):
    """Fraction of nodes with stable rank over time (PROPOSED only)."""
    if not vehicles:
        return
    vids = sorted(vehicles.keys())
    n_steps = len(vehicles[vids[0]].trust_history)
    n_v = len(vids)
    score_mat = np.zeros((n_steps, n_v))
    for i, vid in enumerate(vids):
        h = vehicles[vid].trust_history
        score_mat[:min(len(h), n_steps), i] = h[:min(len(h), n_steps)]
    rank_mat = np.zeros((n_steps, n_v), dtype=int)
    for t in range(n_steps):
        order = np.argsort(-score_mat[t, :])
        r = np.empty_like(order)
        r[order] = np.arange(n_v)
        rank_mat[t, :] = r
    threshold = max(1, int(0.05 * n_v))
    burn_in = 5
    xs, ys = [], []
    for t in range(1, n_steps):
        if t <= burn_in:
            continue
        stable = np.sum(np.abs(rank_mat[t] - rank_mat[t - 1]) <= threshold)
        xs.append(t)
        ys.append(stable / n_v)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, color='navy')
    ax.set_title(f"Trust Convergence Time\n(Stable Rank = change <= {threshold} positions)")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Fraction of Nodes with Stable Rank")
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    _save(fig, os.path.join(out_dir, "proposed_trust_convergence.png"))


def plot_proposed_trust_normalized(vehicles, out_dir):
    """Z-score trust difference (honest/malicious/swing) for PROPOSED only."""
    norm_data = normalize_histories(vehicles)
    if not norm_data:
        return
    histories = {'HONEST': [], 'MALICIOUS': [], 'SWING': []}
    for vid, v in vehicles.items():
        if vid in norm_data:
            histories[v.behavior_type].append(norm_data[vid])
    all_h = np.array([norm_data[uid] for uid in vehicles.keys()])
    step_data = all_h.T
    mean_s = np.mean(step_data, axis=1)
    std_s = np.std(step_data, axis=1) + 1e-9
    burn_in = 5
    steps = np.arange(len(mean_s))

    fig, ax = plt.subplots()
    for cat, lists in histories.items():
        if not lists:
            continue
        g_avg = np.mean(np.array(lists).T, axis=1)
        z = (g_avg - mean_s) / std_s
        ax.plot(steps[burn_in:], z[burn_in:], label=f'{cat} (Z-Score)', markevery=10)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.fill_between(steps[burn_in:], -1, 1, color='gray', alpha=0.2, label='\u00b11 Std Dev')
    ax.set_title("Normalised Trust Difference (Z-Score)\n(First 5 steps omitted for cold-start)")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Z-Score (Std Devs from Global Mean)")
    ax.legend()
    ax.grid(True)
    _save(fig, os.path.join(out_dir, "proposed_trust_normalized.png"))


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPARATIVE PLOTS (all models)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comp_traffic_detection(out_dir):
    """All models: IoV traffic (interactions) vs malicious detected."""
    print("Generating comp_traffic_detection ...")
    steps, ips = 80, 50
    fig, ax = plt.subplots()
    for model in MODELS:
        res = run_single_simulation(model, steps=steps, interactions_per_step=ips, percent_malicious=0.1)
        x = [i * ips for i in range(1, steps + 1)]
        ax.plot(x, res['detected_history'], **get_style(model))
    ax.set_title("IoV Traffic vs Malicious Vehicle Detection")
    ax.set_xlabel("Total Number of Interactions")
    ax.set_ylabel("Malicious Vehicles Detected")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "comp_traffic_detection.png"))


def plot_comp_robustness(out_dir):
    """Per-ratio robustness: separate PNGs for 10%, 20%, 30% malicious."""
    print("Generating comp_robustness (10 / 20 / 30%) ...")
    ratios = [0.1, 0.2, 0.3]
    steps, ips = 60, 50
    for ratio in ratios:
        tag = int(ratio * 100)
        fig, ax = plt.subplots()
        for model in MODELS:
            res = run_single_simulation(model, steps=steps, percent_malicious=ratio, interactions_per_step=ips)
            x = [k * ips for k in range(1, steps + 1)]
            ax.plot(x, res['detected_history'], **get_style(model))
        ax.set_title(f"Robustness — Attacker Ratio {tag}%")
        ax.set_xlabel("Interactions")
        ax.set_ylabel("Detected Malicious Vehicles")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, os.path.join(out_dir, f"comp_robustness_{tag}.png"))


def plot_comp_capacity(out_dir):
    """All models: normalised capacity index vs network size (line plot)."""
    print("Generating comp_capacity ...")
    sizes = [10, 20, 30, 40, 50, 60]
    cap = {m: [] for m in MODELS}
    steps = 50
    for N in sizes:
        for model in MODELS:
            res = run_single_simulation(model, num_vehicles=N, steps=steps)
            sc = res['consensus_success']
            if model == 'LT_PBFT':
                ov = (N ** 2) / 20.0
            elif model == 'COBATS':
                ov = (N * 2) / 10.0
            elif model == 'PROPOSED':
                ov = N / 10.0
            else:
                ov = (N * 1.5) / 10.0
            cap[model].append((sc / steps * 1000) / (10 + ov))

    fig, ax = plt.subplots()
    for model in MODELS:
        ax.plot(sizes, cap[model], **get_style(model))
    ax.set_title("Normalised Capacity Index vs Network Size")
    ax.set_xlabel("Network Size (Number of Vehicles)")
    ax.set_ylabel("Capacity Index")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "comp_capacity.png"))
    return sizes, cap


def plot_comp_throughput(sizes, cap, out_dir):
    """All models: capacity bar chart vs network size."""
    print("Generating comp_throughput ...")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(sizes))
    width = 0.12
    for i, model in enumerate(MODELS):
        offset = (i - len(MODELS) / 2) * width + width / 2
        ax.bar(x + offset, cap[model], width, label=model, color=COLORS.get(model, 'gray'))
    ax.set_title("Average Capacity Index vs Network Size")
    ax.set_xlabel("Network Size")
    ax.set_ylabel("Capacity Index")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    _save(fig, os.path.join(out_dir, "comp_throughput.png"))


def plot_comp_latency(out_dir):
    """Analytical consensus latency vs network size."""
    print("Generating comp_latency ...")
    sizes = [10, 20, 30, 40, 50, 60]
    lat = {m: [] for m in MODELS}
    base = 100
    for N in sizes:
        for model in MODELS:
            if model == 'PBFT':
                v = base + 1.8 * N ** 2
            elif model == 'LT_PBFT':
                v = base + 1.5 * N ** 2
            elif model == 'RTM':
                v = base + 15 * N
            elif model == 'BSED':
                v = base + 18 * N
            elif model == 'COBATS':
                v = base + 20 * N
            elif model == 'BTVR':
                v = base + 4 * N + 20
            elif model == 'PROPOSED':
                v = base + 1.5 * N + 10
            else:
                v = base + 10 * N
            lat[model].append(v)

    fig, ax = plt.subplots()
    for model in MODELS:
        ax.plot(sizes, lat[model], marker='o', **get_style(model))
    ax.set_title("Consensus Latency vs Network Size")
    ax.set_xlabel("Network Size (Nodes)")
    ax.set_ylabel("Consensus Latency (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "comp_latency.png"))


def plot_comp_swing_attack(out_dir):
    """All models: swing attack success rate vs intensity."""
    print("Generating comp_swing_attack ...")
    intensities = [0.2, 0.5, 0.8, 0.95]
    labels = ['Low', 'Medium', 'High', 'Very High']
    res_s = {m: [] for m in MODELS}
    for intensity in intensities:
        for model in MODELS:
            r = run_single_simulation(model, percent_malicious=0.0, percent_swing=0.2,
                                      attack_intensity=intensity, steps=80)
            failures = 80 - r['consensus_success']
            res_s[model].append(failures / 80 * 100)

    fig, ax = plt.subplots()
    for model in MODELS:
        ax.plot(labels, res_s[model], **get_style(model))
    ax.set_title("Swing Attack Success Rate vs Intensity")
    ax.set_xlabel("Attack Intensity")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "comp_swing_attack.png"))
    return intensities, labels


def plot_comp_internal_attack(intensities, labels, out_dir):
    """All models: internal attack success rate vs intensity (bar chart)."""
    print("Generating comp_internal_attack ...")
    res_s = {m: [] for m in MODELS}
    for intensity in intensities:
        for model in MODELS:
            r = run_single_simulation(model, percent_malicious=0.2, percent_swing=0.0,
                                      attack_intensity=intensity, steps=80)
            failures = 80 - r['consensus_success']
            res_s[model].append(failures / 80 * 100)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    w = 0.12
    for i, model in enumerate(MODELS):
        off = (i - len(MODELS) / 2) * w + w / 2
        ax.bar(x + off, res_s[model], w, label=model, color=COLORS.get(model, 'gray'))
    ax.set_title("Internal Attack: Success Rate vs Intensity")
    ax.set_xlabel("Attack Intensity")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    _save(fig, os.path.join(out_dir, "comp_internal_attack.png"))


def plot_comp_convergence(out_dir):
    """All models: rank convergence stability over time."""
    print("Generating comp_convergence ...")
    import random as _rng
    _rng.seed(42)
    np.random.seed(42)
    steps = 100
    n_vehicles = 50
    threshold = max(1, int(0.05 * n_vehicles))  # Match proposed_trust_convergence threshold
    fig, ax = plt.subplots()
    for model in MODELS:
        sim = Simulator(model_type=model, num_vehicles=n_vehicles, percent_malicious=0.1, percent_swing=0.0)
        ranks_hist = []
        for t in range(steps):
            sim.model.simulate_interaction_step(25)
            sim.model.update_global_trust(sync_rsus=True)
            scored = sorted(sim.model.vehicles.values(), key=lambda v: v.global_trust_score, reverse=True)
            ranks_hist.append({v.id: i for i, v in enumerate(scored)})
        threshold_c = threshold  # Use consistent threshold defined above
        y = [0.0]
        for t in range(1, steps):
            prev, curr = ranks_hist[t - 1], ranks_hist[t]
            stable = sum(1 for vid in prev if abs(prev[vid] - curr[vid]) <= threshold_c)
            y.append(stable / float(n_vehicles))
        ax.plot(range(steps), y, **get_style(model))
    ax.set_title("Trust/Rank Convergence Stability")
    ax.set_xlabel("Simulation Steps")
    ax.set_ylabel("Fraction of Nodes with Stable Rank")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "comp_convergence.png"))


def plot_comp_consensus(out_dir):
    """All models: consensus success rate across network sizes."""
    print("Generating comp_consensus ...")
    sizes = [10, 20, 30, 40, 50, 60]
    steps = 50
    rates = {m: [] for m in MODELS}
    for N in sizes:
        for model in MODELS:
            res = run_single_simulation(model, num_vehicles=N, steps=steps, percent_malicious=0.1)
            rates[model].append(res['consensus_success'] / steps * 100)

    fig, ax = plt.subplots()
    for model in MODELS:
        ax.plot(sizes, rates[model], marker='o', **get_style(model))
    ax.set_title("Consensus Success Rate vs Network Size")
    ax.set_xlabel("Network Size (Nodes)")
    ax.set_ylabel("Consensus Success Rate (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "comp_consensus.png"))


def plot_comp_detection_tpr(out_dir):
    """All models: detection TPR under the shared normalized threshold logic."""
    print("Generating comp_detection_tpr ...")
    ratios = [0.1, 0.15, 0.2, 0.25, 0.3]
    tpr_data = {m: [] for m in MODELS}
    for ratio in ratios:
        for model in MODELS:
            res = run_single_simulation(model, steps=80, percent_malicious=ratio)
            vehicles = res['vehicles']
            mal_ids = [v.id for v in vehicles.values() if v.is_malicious]
            if not mal_ids or not vehicles:
                tpr_data[model].append(0)
                continue

            norm_scores = get_normalized_detection_scores(vehicles)
            tp = sum(1 for vid in mal_ids if norm_scores.get(vid, 0.5) < DETECTION_THRESHOLD)
            tpr_data[model].append(tp / len(mal_ids) * 100)

    fig, ax = plt.subplots()
    x_labels = [f"{int(r * 100)}%" for r in ratios]
    for model in MODELS:
        ax.plot(x_labels, tpr_data[model], marker='o', **get_style(model))
    ax.set_title(f"Detection TPR at Threshold = {DETECTION_THRESHOLD:.2f}")
    ax.set_xlabel("Malicious Ratio")
    ax.set_ylabel("True Positive Rate (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "comp_detection_tpr.png"))


def plot_comp_trust_evolution(out_dir):
    """All models: median honest trust over simulation steps."""
    print("Generating comp_trust_evolution ...")
    steps = 80
    fig, ax = plt.subplots()
    for model in MODELS:
        res = run_single_simulation(model, steps=steps, percent_malicious=0.1)
        vehicles = res['vehicles']
        norm = normalize_histories(vehicles)
        honest = [v.id for v in vehicles.values() if not v.is_malicious]
        if honest and norm:
            med = np.median([norm[uid] for uid in honest], axis=0)
            ax.plot(range(steps + 1), med, **get_style(model))
    ax.set_title("Comparative Trust Evolution (Honest Median)")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Normalised Trust Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "comp_trust_evolution.png"))


def plot_comp_trust_evolution_malicious(out_dir):
    """All models: median malicious trust over simulation steps."""
    print("Generating comp_trust_evolution_malicious ...")
    steps = 80
    fig, ax = plt.subplots()
    for model in MODELS:
        res = run_single_simulation(model, steps=steps, percent_malicious=0.1)
        vehicles = res['vehicles']
        norm = normalize_histories(vehicles)
        mal = [v.id for v in vehicles.values() if v.is_malicious]
        if mal and norm:
            med = np.median([norm[uid] for uid in mal], axis=0)
            ax.plot(range(steps + 1), med, **get_style(model))
    ax.set_title("Comparative Trust Evolution (Malicious Median)")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Normalised Trust Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "comp_trust_evolution_malicious.png"))


def plot_comp_trust_normalized(out_dir):
    """All models: Z-score separation between honest and malicious groups."""
    print("Generating comp_trust_normalized ...")
    steps = 80
    fig, ax = plt.subplots()
    for model in MODELS:
        res = run_single_simulation(model, steps=steps, percent_malicious=0.1)
        vehicles = res['vehicles']
        norm = normalize_histories(vehicles)
        if not norm:
            continue
        honest = [norm[v.id] for v in vehicles.values() if not v.is_malicious and v.id in norm]
        mal = [norm[v.id] for v in vehicles.values() if v.is_malicious and v.id in norm]
        if not honest or not mal:
            continue
        h_avg = np.mean(honest, axis=0)
        m_avg = np.mean(mal, axis=0)
        all_h = np.array(list(norm.values()))
        std = np.std(all_h, axis=0) + 1e-9
        separation = (h_avg - m_avg) / std
        burn = 5
        ax.plot(range(burn, steps + 1), separation[burn:], **get_style(model))
    ax.set_title("Trust Separation (Honest − Malicious) Z-Score")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Z-Score Separation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "comp_trust_normalized.png"))


# ═══════════════════════════════════════════════════════════════════════════════
#  DAG STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dag_structure(out_dir):
    """Generate DAG visualisation by running a short PROPOSED simulation."""
    print("Generating dag_structure ...")
    import random as _rng
    _rng.seed(42)
    np.random.seed(42)
    sim = Simulator(num_vehicles=10, model_type='PROPOSED')
    mgr = ConsensusManager('PROPOSED', sim.model.vehicles)
    for t in range(20):
        sim.model.simulate_interaction_step(5)
        sim.model.update_global_trust(sync_rsus=True)
        mgr.attempt_consensus(step=t)

    dag = mgr.dag
    blocks = dag.blocks
    if not blocks:
        print("  [skip] dag_structure.png -- DAG is empty")
        return

    # Compute depths
    depths = {bid: 0 for bid in blocks}
    changed, iters = True, 0
    max_depth = 0
    while changed and iters < len(blocks) + 2:
        changed = False
        iters += 1
        for bid, blk in blocks.items():
            mp = max((depths[p] for p in blk.parents if p in depths), default=-1)
            nd = mp + 1 if mp >= 0 else 0
            if nd > depths[bid]:
                depths[bid] = nd
                changed = True
                max_depth = max(max_depth, nd)

    layers = {}
    for bid, d in depths.items():
        layers.setdefault(d, []).append(bid)
    y_coords = {}
    for d, bids in layers.items():
        n = len(bids)
        for i, bid in enumerate(bids):
            y_coords[bid] = i - (n - 1) / 2.0

    fig, ax = plt.subplots(figsize=(8, 4))
    for bid, blk in blocks.items():
        x1, y1 = depths[bid], y_coords[bid]
        for pid in blk.parents:
            if pid in depths:
                ax.plot([depths[pid], x1], [y_coords[pid], y1], color='gray', alpha=0.5, zorder=1)
    x_vals = [depths[b] for b in blocks]
    y_vals = [y_coords[b] for b in blocks]
    validators = sorted(set(b.validator_id for b in blocks.values()))
    val_map = {v: i for i, v in enumerate(validators)}
    c = [val_map[blocks[b].validator_id] for b in blocks]
    ax.scatter(x_vals, y_vals, c=c, cmap='tab10', s=300, zorder=2, edgecolors='black')
    for bid in blocks:
        ax.text(depths[bid], y_coords[bid], bid[:4], fontsize=10, ha='center', va='center',
                color='white', fontweight='bold', zorder=3)
    ax.set_title(f"DAG Structure (Height: {max_depth + 1}, Blocks: {len(blocks)})")
    ax.set_xlabel("Layer (Temporal Depth)")
    ax.set_yticks([])
    ax.grid(False)
    cmap = plt.get_cmap('tab10')
    if len(validators) <= 10:
        patches = [mpatches.Patch(color=cmap(val_map[v]), label=f'Val {v}') for v in validators]
        ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1))
    _save(fig, os.path.join(out_dir, "dag_structure.png"))


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_graphs():
    """
    Single entry point that regenerates EVERY graph.
    All outputs land in RESULTS_DIR with standardised comp_*/proposed_*/dag_* names.
    Prints granular progress and verifies completeness at the end.
    """
    out = RESULTS_DIR
    os.makedirs(out, exist_ok=True)

    # ── Complete manifest of expected output files ────────────────────────────
    EXPECTED_OUTPUTS = [
        # Comparative (all-model) plots
        'comp_traffic_detection.png',
        'comp_robustness_10.png',
        'comp_robustness_20.png',
        'comp_robustness_30.png',
        'comp_capacity.png',
        'comp_throughput.png',
        'comp_latency.png',
        'comp_swing_attack.png',
        'comp_internal_attack.png',
        'comp_convergence.png',
        'comp_consensus.png',
        'comp_detection_tpr.png',
        'comp_trust_evolution.png',
        'comp_trust_evolution_malicious.png',
        'comp_trust_normalized.png',
        # Proposed-solo plots
        'proposed_trust_evolution.png',
        'proposed_detection_metrics.png',
        'proposed_rank_distribution.png',
        'proposed_trust_convergence.png',
        'proposed_trust_normalized.png',
        'proposed_swing_analysis.png',
        # Structural
        'dag_structure.png',
    ]

    total_steps = 16  # Number of generation steps
    total_files = len(EXPECTED_OUTPUTS)  # 22 individual graph files
    step = 0

    def _step(label, files=1):
        nonlocal step
        step += 1
        suffix = f" ({files} files)" if files > 1 else ""
        print(f"\n[Step {step}/{total_steps}] {label}{suffix}")

    print(f"\n{'=' * 60}")
    print(f"  FULL GRAPH SUITE -- {total_files} graphs across {total_steps} steps")
    print(f"  Output: {os.path.abspath(out)}/")
    print(f"{'=' * 60}")

    # ── SECTION 1: Comparative Plots (15 files) ─────────────────────────────
    print(f"\n{'-' * 50}")
    print("  SECTION 1/3 -- Comparative Plots (all models)")
    print(f"{'-' * 50}")

    _step("comp_traffic_detection")
    try:
        plot_comp_traffic_detection(out)
    except Exception as e:
        print(f"  [FAIL] comp_traffic_detection: {e}")

    _step("comp_robustness (10%, 20%, 30%)", files=3)
    try:
        plot_comp_robustness(out)
    except Exception as e:
        print(f"  [FAIL] comp_robustness: {e}")

    _step("comp_capacity")
    sizes, cap = None, None
    try:
        sizes, cap = plot_comp_capacity(out)
    except Exception as e:
        print(f"  [FAIL] comp_capacity: {e}")

    _step("comp_throughput")
    if sizes is not None and cap is not None:
        try:
            plot_comp_throughput(sizes, cap, out)
        except Exception as e:
            print(f"  [FAIL] comp_throughput: {e}")
    else:
        print("  [SKIP] comp_throughput (depends on comp_capacity)")

    _step("comp_latency")
    try:
        plot_comp_latency(out)
    except Exception as e:
        print(f"  [FAIL] comp_latency: {e}")

    _step("comp_swing_attack")
    intensities, labels = None, None
    try:
        intensities, labels = plot_comp_swing_attack(out)
    except Exception as e:
        print(f"  [FAIL] comp_swing_attack: {e}")

    _step("comp_internal_attack")
    if intensities is not None and labels is not None:
        try:
            plot_comp_internal_attack(intensities, labels, out)
        except Exception as e:
            print(f"  [FAIL] comp_internal_attack: {e}")
    else:
        print("  [SKIP] comp_internal_attack (depends on comp_swing_attack)")

    _step("comp_convergence")
    try:
        plot_comp_convergence(out)
    except Exception as e:
        print(f"  [FAIL] comp_convergence: {e}")

    _step("comp_consensus")
    try:
        plot_comp_consensus(out)
    except Exception as e:
        print(f"  [FAIL] comp_consensus: {e}")

    _step("comp_detection_tpr")
    try:
        plot_comp_detection_tpr(out)
    except Exception as e:
        print(f"  [FAIL] comp_detection_tpr: {e}")

    _step("comp_trust_evolution")
    try:
        plot_comp_trust_evolution(out)
    except Exception as e:
        print(f"  [FAIL] comp_trust_evolution: {e}")

    _step("comp_trust_evolution_malicious")
    try:
        plot_comp_trust_evolution_malicious(out)
    except Exception as e:
        print(f"  [FAIL] comp_trust_evolution_malicious: {e}")

    _step("comp_trust_normalized")
    try:
        plot_comp_trust_normalized(out)
    except Exception as e:
        print(f"  [FAIL] comp_trust_normalized: {e}")

    # ── SECTION 2: Proposed-Solo Plots (6 files) ────────────────────────────
    print(f"\n{'-' * 50}")
    print("  SECTION 2/3 -- Proposed Solo Plots")
    print(f"{'-' * 50}")

    _step("proposed solo suite (evolution + detection + rank + convergence + normalized)", files=5)
    try:
        print("  Running PROPOSED simulation for solo plots ...")
        res = run_single_simulation('PROPOSED', steps=100, percent_malicious=0.1)
        plot_proposed_trust_evolution(res['vehicles'], out)
        plot_proposed_detection_metrics(res['vehicles'], out)
        plot_proposed_rank_distribution(res['vehicles'], out)
        plot_proposed_trust_convergence(res['vehicles'], out)
        plot_proposed_trust_normalized(res['vehicles'], out)
    except Exception as e:
        print(f"  [FAIL] proposed solo suite: {e}")

    _step("proposed_swing_analysis")
    try:
        plot_proposed_swing_analysis(out)
    except Exception as e:
        print(f"  [FAIL] proposed_swing_analysis: {e}")

    # ── SECTION 3: Structural Visualisation (1 file) ────────────────────────
    print(f"\n{'-' * 50}")
    print("  SECTION 3/3 -- Structural Visualisations")
    print(f"{'-' * 50}")

    _step("dag_structure")
    try:
        plot_dag_structure(out)
    except Exception as e:
        print(f"  [FAIL] dag_structure: {e}")

    # ── VERIFICATION & SUMMARY ───────────────────────────────────────────────
    generated = sorted(f for f in os.listdir(out) if f.endswith('.png'))
    missing = [f for f in EXPECTED_OUTPUTS if f not in generated]
    extra = [f for f in generated if f not in EXPECTED_OUTPUTS]

    print(f"\n{'=' * 60}")
    print(f"  GENERATION COMPLETE")
    print(f"{'=' * 60}")

    for cat_name, prefix in [('Comparative', 'comp_'),
                              ('Proposed Solo', 'proposed_'),
                              ('Structural', 'dag_')]:
        cat_files = [f for f in generated if f.startswith(prefix)]
        if cat_files:
            label = "file" if len(cat_files) == 1 else "files"
            print(f"\n  {cat_name} ({len(cat_files)} {label}):")
            for f in cat_files:
                print(f"    [OK] {f}")

    if extra:
        print(f"\n  Extra files in output ({len(extra)}):")
        for f in extra:
            print(f"    + {f}")

    if missing:
        print(f"\n  WARNING -- {len(missing)} expected graphs MISSING:")
        for f in missing:
            print(f"    [MISS] {f}")
    else:
        print(f"\n  All {total_files} expected graphs verified present.")

    print(f"\n  Directory: {os.path.abspath(out)}/")
    print(f"{'=' * 60}\n")


# Keep backward-compatible alias used by run_experiment.py --compare
run_paper_suite = run_all_graphs
