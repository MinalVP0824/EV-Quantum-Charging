"""Matplotlib figures used by the notebook, the CLI and the Streamlit app."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BASE = "#c9c7bf"
COLORS = {
    "FCFS": "#eb6834",
    "Classical optimum": "#1baf7a",
    "Quantum (XY-QAOA)": "#2a78d6",
    "Quantum (X-QAOA)": "#4a3aa7",
}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 110,
})


def forecast_plot(fc, window_hours, use="pred"):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    hist = fc.history
    day = fc.target_day
    ax.plot(hist["timestamp"], hist["demand_kw"], color=INK2, lw=1.5, label="Observed load")
    ax.plot(day["timestamp"], day["actual"], color=INK2, lw=1.5, ls=":", label="Actual (held out)")
    ax.fill_between(day["timestamp"], day["p10"], day["p90"], color=COLORS["Quantum (XY-QAOA)"],
                    alpha=0.15, lw=0, label="P10 to P90 band")
    ax.plot(day["timestamp"], day["pred"], color=COLORS["Quantum (XY-QAOA)"], lw=2, label="AI forecast")
    ts = pd.to_datetime(day["timestamp"])
    sel = ts.dt.hour.isin(window_hours)
    ax.axvspan(ts[sel].min(), ts[sel].max() + pd.Timedelta(hours=1), color="#eda100", alpha=0.12,
               lw=0, label="Scheduling window")
    ax.set_ylabel("Feeder load (kW)")
    ax.set_title("Demand forecast feeding the optimizer", loc="left")
    ax.legend(ncol=5, loc="upper left", fontsize=8.5, bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    return fig


def load_plot(inst, metrics: dict):
    names = list(metrics)
    x = np.arange(inst.n_slots)
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k, name in enumerate(names):
        pos = x - 0.4 + w / 2 + k * w
        ev = np.array(metrics[name]["ev_load_kw"])
        base = np.array(inst.base_load)
        ax.bar(pos, base, w * 0.92, color=BASE, edgecolor="white", lw=1,
               label="Forecast base load" if k == 0 else None)
        ax.bar(pos, ev, w * 0.92, bottom=base, color=COLORS.get(name, INK2), edgecolor="white",
               lw=1, label=f"EV load, {name}")
        for px, tot in zip(pos, base + ev):
            ax.text(px, tot + 1, f"{tot:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, inst.slot_labels)
    ax.set_ylabel("Total load (kW)")
    ax.set_title("Load per time slot", loc="left")
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return fig


def schedule_grid(inst, assignments: dict):
    names = list(assignments)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 2.4), squeeze=False)
    for ax, name in zip(axes[0], names):
        a = assignments[name]
        S, T = len(inst.stations), inst.n_slots
        ax.set_xlim(0, T)
        ax.set_ylim(0, S)
        ax.grid(False)
        for s in range(S):
            for t in range(T):
                ax.add_patch(plt.Rectangle((t + 0.04, S - s - 1 + 0.08), 0.92, 0.84,
                                           facecolor="#f4f3ef", edgecolor="none"))
        for v, slots in a.items():
            for s, t in slots:
                ax.add_patch(plt.Rectangle((t + 0.04, S - s - 1 + 0.08), 0.92, 0.84,
                                           facecolor=COLORS.get(name, INK2), alpha=0.9, edgecolor="none"))
                label = inst.vehicles[v].name.split(" ", 1)
                ax.text(t + 0.5, S - s - 0.5, label[0] + "\n" + label[1], ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
        ax.set_xticks(np.arange(T) + 0.5, inst.slot_labels)
        ax.set_yticks(np.arange(S) + 0.5, [st.name for st in reversed(inst.stations)])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(name, loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def convergence_plot(results: dict):
    """One panel per solver; the cost scales differ too much to share an axis."""
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 3.3), squeeze=False)
    for ax, (name, r) in zip(axes[0], results.items()):
        ax.plot(np.arange(1, len(r.history) + 1), r.history, lw=1.8, color=COLORS.get(name, INK2))
        ax.axhline(r.optimal_energy, color=COLORS["Classical optimum"], lw=1.2, ls="--")
        ax.text(len(r.history), r.optimal_energy, "exact optimum ", ha="right", va="bottom",
                fontsize=8, color=COLORS["Classical optimum"])
        ax.set_xlabel("Cost evaluation (COBYLA, depth grows 1 to p)")
        ax.set_ylabel("CVaR cost (Rs)")
        ax.set_title(f"{name}, p={r.p}", loc="left", fontsize=10)
    fig.suptitle("Hybrid quantum-classical training loop", x=0.01, ha="left", fontweight="bold",
                 color=INK, fontsize=11)
    fig.tight_layout()
    return fig


def distribution_plot(r, energies, feasible, top=20):
    idx = np.argsort(r.probs)[::-1][:top]
    opt = np.isclose(energies, r.optimal_energy) & feasible
    cols = [COLORS["Classical optimum"] if opt[k] else (COLORS["Quantum (XY-QAOA)"] if feasible[k] else BASE)
            for k in idx]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(range(len(idx)), r.probs[idx] * 100, color=cols, edgecolor="white")
    ax.set_xticks(range(len(idx)), [f"{energies[k]:.0f}" for k in idx], rotation=60, fontsize=7.5)
    ax.set_xlabel("Schedule cost (Rs) of the most likely measured states")
    ax.set_ylabel("Probability (%)")
    ax.set_title(f"Measurement distribution, {r.mixer}-mixer QAOA p={r.p}", loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS["Classical optimum"], label="Optimal schedule"),
                       Patch(color=COLORS["Quantum (XY-QAOA)"], label="Valid schedule"),
                       Patch(color=BASE, label="Breaks a constraint")], fontsize=8.5)
    fig.tight_layout()
    return fig


def mixer_comparison_plot(rows: pd.DataFrame):
    """rows: columns mixer, p, prob_feasible, prob_optimal."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = [f"{m}-mixer p={p}" for m, p in zip(rows["mixer"], rows["p"])]
    y = np.arange(len(rows))
    ax.barh(y + 0.2, rows["prob_feasible"] * 100, 0.38, color=COLORS["Quantum (XY-QAOA)"], label="P(valid schedule)")
    ax.barh(y - 0.2, rows["prob_optimal"] * 100, 0.38, color=COLORS["Classical optimum"], label="P(optimal schedule)")
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Probability per shot (%, log scale)")
    ax.set_title("Why the constraint-preserving mixer matters", loc="left")
    ax.legend(fontsize=8.5, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.25))
    fig.tight_layout()
    return fig
