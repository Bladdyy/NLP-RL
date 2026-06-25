"""
Download data from W&B and create publication-quality plots.

1. MLP vs Transformer (State mode) at depths 8/16/32 on ant_u4_maze.
2. Transformer configuration comparison (None, State, StateGoal, StateActor, Full)
   at depth 8 with identical hyperparameters.
3. Tokenization comparison (patches, semantic, per-dim) - Full transformer at d=8.
4. Pooling comparison (cls, mean, flatten) - semantic tokenization, Full at d=8.
5. Pre-LN vs Post-LN comparison - State mode, semantic, flatten at d=8.
6. Embedding normalization comparison (Base, L2+learnable T, SIGReg, Weak SIGReg)
   - State mode, semantic, cls at d=8.
7. Appendix: actor loss, critic loss, and success rate for early transformer runs,
   showing the training instability that motivated stabilization efforts.
8. Appendix: temperature parameter and time at goal for the failed L2+learnable T run,
   illustrating why training collapsed.
9. Deep comparison: time at goal for the crashed 64-State run vs 32-State-2 and 32-MLP.

Outputs:
  paper_figures/wandb_mlp_vs_state.pdf (.png)
  paper_figures/wandb_transformer_modes_compare.pdf (.png)
  paper_figures/wandb_tokenization_compare.pdf (.png)
  paper_figures/wandb_pooling_compare.pdf (.png)
  paper_figures/wandb_norm_compare.pdf (.png)
  paper_figures/wandb_embed_norm_compare.pdf (.png)
  paper_figures/wandb_losses_appendix.pdf (.png)
  paper_figures/wandb_temperature_appendix.pdf (.png)
  paper_figures/wandb_deep_comparison.pdf (.png)
"""

import json
import os
from pathlib import Path

import numpy as np
import wandb
from matplotlib import pyplot as plt
import matplotlib

# ── Style ─────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
})

OUT_DIR = Path("paper_figures")
OUT_DIR.mkdir(exist_ok=True)

# ── Run selection ─────────────────────────────────────────────────────────
# (entity, project, run_id) for each of the 6 target runs.
RUNS = {
    "8-MLP":    {"id": "mlat3yp4",  "label": "MLP d=8",   "arch": "MLP",   "depth": 8},
    "16-MLP":   {"id": "vc6blv0f",  "label": "MLP d=16",  "arch": "MLP",   "depth": 16},
    "32-MLP":   {"id": "mkhc4fov",  "label": "MLP d=32",  "arch": "MLP",   "depth": 32},
    "8-State":  {"id": "5q1mo7o3",  "label": "State d=8",  "arch": "State", "depth": 8},
    "16-State-2": {"id": "25r8gv9i",  "label": "State d=16", "arch": "State", "depth": 16},
    "32-State-2": {"id": "i42kndl4",  "label": "State d=32", "arch": "State", "depth": 32},
}

# Transformer configuration comparison (depth 8, same transformer hyperparams)
RUNS_TRANSFORMER_MODES = {
    "None":     {"id": "mlat3yp4",  "label": "None (MLP)",       "mode": "None"},
    "State":    {"id": "5q1mo7o3",  "label": "State",           "mode": "State"},
    "StateGoal": {"id": "xto6342e", "label": "StateGoal",      "mode": "StateGoal"},
    "StateActor": {"id": "31hkna6n", "label": "StateActor",    "mode": "StateActor"},
    "Full":     {"id": "ecjre8b2",  "label": "Full",           "mode": "Full"},
}

# Tokenization comparison (Full transformer, depth=8, lr=3e-4, skip=4)
# Note: per_dim uses flatten pooling while the others use cls
RUNS_TOKENIZATION = {
    "patches": {"id": "oh6hlnps",  "label": "Patches",            "tokenization": "patches", "pooling": "cls"},
    "semantic": {"id": "8hko0hmv", "label": "Semantic",          "tokenization": "semantic", "pooling": "cls"},
    "per_dim":  {"id": "4xgqytdm", "label": "Per-dim",           "tokenization": "per_dim",  "pooling": "flatten"},
}

# Pooling comparison (Full transformer, depth=8, semantic tokenization)
# Note: cls/mean use lr=3e-4, flatten uses lr=1e-4
RUNS_POOLING = {
    "cls":     {"id": "8hko0hmv",  "label": "Semantic + CLS",       "tokenization": "semantic", "pooling": "cls"},
    "mean":    {"id": "tnbnd8n9",  "label": "Semantic + Mean",      "tokenization": "semantic", "pooling": "mean"},
    "flatten": {"id": "mgb5erec",  "label": "Semantic + Flatten",   "tokenization": "semantic", "pooling": "flatten"},
}

# Pre-norm vs post-norm comparison (State mode, semantic, flatten, d=8, lr=1e-4)
RUNS_NORM = {
    "pre":  {"id": "jc0saw08",  "label": "Pre-LN",   "norm": "pre"},
    "post": {"id": "0q9rnsrk",  "label": "Post-LN",  "norm": "post"},
}

# Embedding normalization comparison (State, semantic, cls, d=8, lr=1e-4)
RUNS_EMBED_NORM = {
    "base":       {"id": "5q1mo7o3",  "label": "Base",           "embed_norm": "base"},
    "l2":         {"id": "3i9ngrhw",  "label": "L2 + learnable T", "embed_norm": "l2"},
    "sigreg":     {"id": "9h2o51br",  "label": "SIGReg",         "embed_norm": "sigreg"},
    "weak_sigreg": {"id": "uia7633l", "label": "Weak SIGReg",    "embed_norm": "weak_sigreg"},
}

# Early-run loss analysis (appendix): shows instability in early transformer runs
RUNS_LOSSES = [
    {"id": "8hko0hmv", "label": "Semantic + CLS",     "color": "#D55E00"},
    {"id": "oh6hlnps", "label": "Patches + CLS",      "color": "#0072B2"},
    {"id": "tnbnd8n9", "label": "Semantic + Mean",    "color": "#009E73"},
    {"id": "sal1orpj", "label": "State (raw)",        "color": "#CC79A7"},
]

ENTITY = "oskarkulinski-mimuw"
PROJECT = "NLPRL"

# ── Colour / style scheme ────────────────────────────────────────────────
# We want clear visual grouping: MLP = blues (solid), State = oranges/reds (dashed)
ARCH_STYLES = {
    "MLP":   {"color_fn": lambda d: plt.cm.Blues({8: 0.48, 16: 0.72, 32: 0.90}[d]), "ls": "-"},
    "State": {"color_fn": lambda d: plt.cm.Oranges({8: 0.48, 16: 0.72, 32: 0.90}[d]), "ls": "-"},
}

# Distinct colour palette for transformer-mode comparison
MODE_COLORS = {
    "None":       "#333333",
    "State":      "#E69F00",
    "StateGoal":  "#56B4E9",
    "StateActor": "#009E73",
    "Full":       "#CC79A7",
}

# Colour palette for tokenization comparison
TOKENIZATION_COLORS = {
    "patches": "#0072B2",
    "semantic": "#D55E00",
    "per_dim":  "#009E73",
}

# Colour palette for pooling comparison
POOLING_COLORS = {
    "cls":     "#0072B2",
    "mean":    "#D55E00",
    "flatten": "#009E73",
}

# Colour palette for norm comparison
NORM_COLORS = {
    "pre":  "#0072B2",
    "post": "#D55E00",
}

# Colour palette for embedding normalization comparison
EMBED_NORM_COLORS = {
    "base":       "#333333",
    "l2":         "#0072B2",
    "sigreg":     "#D55E00",
    "weak_sigreg": "#009E73",
}


def download_all(force=False):
    """Download eval/episode_success for each run, return dict keyed by run name."""
    cache_path = OUT_DIR / "_wandb_cache.json"
    if cache_path.exists() and not force:
        print(f"  Using cached data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    api = wandb.Api()
    data = {}
    for name, info in RUNS.items():
        print(f"  Downloading {name} ...", end=" ", flush=True)
        run = api.run(f"{ENTITY}/{PROJECT}/{info['id']}")
        hist = run.history(pandas=False)
        # Filter down to just (epoch, success) pairs
        successes = []
        for h in hist:
            step = h.get("_step")  # epoch index
            succ = h.get("eval/episode_success")
            if step is not None and succ is not None:
                successes.append((step, succ))
        successes.sort(key=lambda x: x[0])
        data[name] = {
            "steps": [s[0] for s in successes],
            "values": [s[1] for s in successes],
        }
        print(f"  {len(successes)} points, final={successes[-1][1]:.1f}")

    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


# ── Plotting ──────────────────────────────────────────────────────────────

def make_plot(data, smooth=3):
    """Plot MLP vs State at depths 8/16/32 - single column width."""

    # ICML single column = 3.25 in, golden ratio height ~ 2.6 in
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), constrained_layout=True)

    # Plot each run
    for name, info in RUNS.items():
        series = data[name]
        steps = np.array(series["steps"]) + 1  # 1-indexed epochs
        values = np.array(series["values"])

        # Simple moving-average smoothing (window centred, edge-padded)
        if smooth > 1:
            kernel = np.ones(smooth) / smooth
            values_pad = np.pad(values, (smooth // 2, smooth // 2), mode="edge")
            values = np.convolve(values_pad, kernel, mode="valid")[:len(values)]

        style = ARCH_STYLES[info["arch"]]
        color = style["color_fn"](info["depth"])
        ls = style["ls"]

        ax.plot(steps, values, color=color, ls=ls, lw=1.2,
                label=info["label"])

    # Labels and limits
    ax.set_xlabel("Env Steps (M)")
    ax.set_ylabel("Time at goal")
    ax.set_xlim(0, 105)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="y")

    # Legend
    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="black", framealpha=0.9, handlelength=2.5)

    save_fig(fig, "wandb_mlp_vs_state")
    return fig


def save_fig(fig, name):
    for ext in [".pdf", ".png"]:
        path = OUT_DIR / f"{name}{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=350)
        print(f"  Saved {path}")
    plt.close(fig)


def download_transformer_modes(force=False):
    """Download eval/episode_success for transformer-mode comparison runs."""
    cache_path = OUT_DIR / "_wandb_cache_transformer_modes.json"
    if cache_path.exists() and not force:
        print(f"  Using cached data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    api = wandb.Api()
    data = {}
    for name, info in RUNS_TRANSFORMER_MODES.items():
        print(f"  Downloading {name} ...", end=" ", flush=True)
        run = api.run(f"{ENTITY}/{PROJECT}/{info['id']}")
        hist = run.history(pandas=False)
        successes = []
        for h in hist:
            step = h.get("_step")
            succ = h.get("eval/episode_success")
            if step is not None and succ is not None:
                successes.append((step, succ))
        successes.sort(key=lambda x: x[0])
        data[name] = {
            "steps": [s[0] for s in successes],
            "values": [s[1] for s in successes],
        }
        print(f"  {len(successes)} points, final={successes[-1][1]:.1f}")

    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def make_transformer_modes_plot(data, smooth=3):
    """Plot comparing transformer modes (None, State, StateGoal, StateActor, Full)
    - all at depth 8 with identical transformer hyperparameters."""

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), constrained_layout=True)

    # Line styles: dashed for None (MLP baseline), solid for transformer modes
    linestyles = {
        "None":       (0, ()),          # solid
        "State":      (0, (3, 1.5)),    # dotted
        "StateGoal":  (0, (5, 2)),      # dashed
        "StateActor": (0, (1, 1)),      # densely dotted
        "Full":       (0, ()),          # solid
    }

    for name in RUNS_TRANSFORMER_MODES:
        series = data[name]
        steps = np.array(series["steps"]) + 1
        values = np.array(series["values"])

        if smooth > 1:
            kernel = np.ones(smooth) / smooth
            values_pad = np.pad(values, (smooth // 2, smooth // 2), mode="edge")
            values = np.convolve(values_pad, kernel, mode="valid")[:len(values)]

        ax.plot(steps, values, color=MODE_COLORS[name], ls=linestyles[name], lw=1.2,
                label=RUNS_TRANSFORMER_MODES[name]["label"])

    ax.set_xlabel("Env Steps (M)")
    ax.set_ylabel("Time at goal")
    ax.set_xlim(0, 105)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="y")

    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="black", framealpha=0.9, handlelength=2.5)

    save_fig(fig, "wandb_transformer_modes_compare")
    return fig


def download_tokenization(force=False):
    """Download eval/episode_success for tokenization comparison runs."""
    cache_path = OUT_DIR / "_wandb_cache_tokenization.json"
    if cache_path.exists() and not force:
        print(f"  Using cached data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    api = wandb.Api()
    data = {}
    for name, info in RUNS_TOKENIZATION.items():
        print(f"  Downloading {name} ...", end=" ", flush=True)
        run = api.run(f"{ENTITY}/{PROJECT}/{info['id']}")
        hist = run.history(pandas=False)
        successes = []
        for h in hist:
            step = h.get("_step")
            succ = h.get("eval/episode_success")
            if step is not None and succ is not None:
                successes.append((step, succ))
        successes.sort(key=lambda x: x[0])
        data[name] = {
            "steps": [s[0] for s in successes],
            "values": [s[1] for s in successes],
        }
        print(f"  {len(successes)} points, final={successes[-1][1]:.1f}")

    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def download_pooling(force=False):
    """Download eval/episode_success for pooling comparison runs."""
    cache_path = OUT_DIR / "_wandb_cache_pooling.json"
    if cache_path.exists() and not force:
        print(f"  Using cached data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    api = wandb.Api()
    data = {}
    for name, info in RUNS_POOLING.items():
        print(f"  Downloading {name} ...", end=" ", flush=True)
        run = api.run(f"{ENTITY}/{PROJECT}/{info['id']}")
        hist = run.history(pandas=False)
        successes = []
        for h in hist:
            step = h.get("_step")
            succ = h.get("eval/episode_success")
            if step is not None and succ is not None:
                successes.append((step, succ))
        successes.sort(key=lambda x: x[0])
        data[name] = {
            "steps": [s[0] for s in successes],
            "values": [s[1] for s in successes],
        }
        print(f"  {len(successes)} points, final={successes[-1][1]:.1f}")

    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def make_tokenization_plot(data, smooth=3):
    """Plot comparing tokenization strategies (patches, semantic, per-dim)
    with the Full transformer at depth 8."""

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), constrained_layout=True)

    linestyles = {
        "patches":  (0, ()),
        "semantic": (0, (4, 1.5)),
        "per_dim":  (0, (1, 1)),
    }

    for name in RUNS_TOKENIZATION:
        series = data[name]
        steps = np.array(series["steps"]) + 1
        values = np.array(series["values"])

        if smooth > 1:
            kernel = np.ones(smooth) / smooth
            values_pad = np.pad(values, (smooth // 2, smooth // 2), mode="edge")
            values = np.convolve(values_pad, kernel, mode="valid")[:len(values)]

        ax.plot(steps, values, color=TOKENIZATION_COLORS[name], ls=linestyles[name], lw=1.2,
                label=RUNS_TOKENIZATION[name]["label"])

    ax.set_xlabel("Env Steps (M)")
    ax.set_ylabel("Time at goal")
    ax.set_xlim(0, 105)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="y")

    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="black", framealpha=0.9, handlelength=2.5)

    save_fig(fig, "wandb_tokenization_compare")
    return fig


def make_pooling_plot(data, smooth=3):
    """Plot comparing pooling strategies (cls, mean, flatten)
    with semantic tokenization, Full transformer at depth 8."""

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), constrained_layout=True)

    linestyles = {
        "cls":     (0, ()),
        "mean":    (0, (4, 1.5)),
        "flatten": (0, (1, 1)),
    }

    for name in RUNS_POOLING:
        series = data[name]
        steps = np.array(series["steps"]) + 1
        values = np.array(series["values"])

        if smooth > 1:
            kernel = np.ones(smooth) / smooth
            values_pad = np.pad(values, (smooth // 2, smooth // 2), mode="edge")
            values = np.convolve(values_pad, kernel, mode="valid")[:len(values)]

        ax.plot(steps, values, color=POOLING_COLORS[name], ls=linestyles[name], lw=1.2,
                label=RUNS_POOLING[name]["label"])

    ax.set_xlabel("Env Steps (M)")
    ax.set_ylabel("Time at goal")
    ax.set_xlim(0, 105)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="y")

    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="black", framealpha=0.9, handlelength=2.5)

    save_fig(fig, "wandb_pooling_compare")
    return fig


def download_norm(force=False):
    """Download eval/episode_success for pre-norm vs post-norm comparison runs."""
    cache_path = OUT_DIR / "_wandb_cache_norm.json"
    if cache_path.exists() and not force:
        print(f"  Using cached data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    api = wandb.Api()
    data = {}
    for name, info in RUNS_NORM.items():
        print(f"  Downloading {name} ...", end=" ", flush=True)
        run = api.run(f"{ENTITY}/{PROJECT}/{info['id']}")
        hist = run.history(pandas=False)
        successes = []
        for h in hist:
            step = h.get("_step")
            succ = h.get("eval/episode_success")
            if step is not None and succ is not None:
                successes.append((step, succ))
        successes.sort(key=lambda x: x[0])
        data[name] = {
            "steps": [s[0] for s in successes],
            "values": [s[1] for s in successes],
        }
        print(f"  {len(successes)} points, final={successes[-1][1]:.1f}")

    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def make_norm_plot(data, smooth=3):
    """Plot comparing Pre-LN vs Post-LN normalization
    with State mode, semantic tokenization, flatten pooling at depth 8."""

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), constrained_layout=True)

    linestyles = {
        "pre":  (0, ()),
        "post": (0, (4, 1.5)),
    }

    for name in RUNS_NORM:
        series = data[name]
        steps = np.array(series["steps"]) + 1
        values = np.array(series["values"])

        if smooth > 1:
            kernel = np.ones(smooth) / smooth
            values_pad = np.pad(values, (smooth // 2, smooth // 2), mode="edge")
            values = np.convolve(values_pad, kernel, mode="valid")[:len(values)]

        ax.plot(steps, values, color=NORM_COLORS[name], ls=linestyles[name], lw=1.2,
                label=RUNS_NORM[name]["label"])

    ax.set_xlabel("Env Steps (M)")
    ax.set_ylabel("Time at goal")
    ax.set_xlim(0, 105)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="y")

    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="black", framealpha=0.9, handlelength=2.5)

    save_fig(fig, "wandb_norm_compare")
    return fig


def download_embed_norm(force=False):
    """Download eval/episode_success for embedding normalization comparison runs."""
    cache_path = OUT_DIR / "_wandb_cache_embed_norm.json"
    if cache_path.exists() and not force:
        print(f"  Using cached data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    api = wandb.Api()
    data = {}
    for name, info in RUNS_EMBED_NORM.items():
        print(f"  Downloading {name} ...", end=" ", flush=True)
        run = api.run(f"{ENTITY}/{PROJECT}/{info['id']}")
        hist = run.history(pandas=False)
        successes = []
        for h in hist:
            step = h.get("_step")
            succ = h.get("eval/episode_success")
            if step is not None and succ is not None:
                successes.append((step, succ))
        successes.sort(key=lambda x: x[0])
        data[name] = {
            "steps": [s[0] for s in successes],
            "values": [s[1] for s in successes],
        }
        print(f"  {len(successes)} points, final={successes[-1][1]:.1f}")

    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def make_embed_norm_plot(data, smooth=3):
    """Plot comparing embedding normalization strategies
    (Base, L2+learnable T, SIGReg, Weak SIGReg)
    with State mode, semantic tokenization, cls pooling at depth 8."""

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), constrained_layout=True)

    linestyles = {
        "base":       (0, ()),
        "l2":         (0, (3, 1.5)),
        "sigreg":     (0, (5, 2)),
        "weak_sigreg": (0, (1, 1)),
    }

    for name in RUNS_EMBED_NORM:
        series = data[name]
        steps = np.array(series["steps"]) + 1
        values = np.array(series["values"])

        if smooth > 1:
            kernel = np.ones(smooth) / smooth
            values_pad = np.pad(values, (smooth // 2, smooth // 2), mode="edge")
            values = np.convolve(values_pad, kernel, mode="valid")[:len(values)]

        ax.plot(steps, values, color=EMBED_NORM_COLORS[name], ls=linestyles[name], lw=1.2,
                label=RUNS_EMBED_NORM[name]["label"])

    ax.set_xlabel("Env Steps (M)")
    ax.set_ylabel("Time at goal")
    ax.set_xlim(0, 105)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="y")

    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="black", framealpha=0.9, handlelength=2.5)

    save_fig(fig, "wandb_embed_norm_compare")
    return fig


def download_losses(force=False):
    """Download actor_loss, critic_loss and eval/episode_success for early-run analysis."""
    cache_path = OUT_DIR / "_wandb_cache_losses.json"
    if cache_path.exists() and not force:
        print(f"  Using cached data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    api = wandb.Api()
    data = {}
    for info in RUNS_LOSSES:
        label = info["label"]
        print(f"  Downloading {label} ...", end=" ", flush=True)
        run = api.run(f"{ENTITY}/{PROJECT}/{info['id']}")
        hist = run.history(pandas=False)
        steps = []
        actor_losses = []
        critic_losses = []
        successes = []
        for h in hist:
            step = h.get("_step")
            al = h.get("training/actor_loss")
            cl = h.get("training/critic_loss")
            succ = h.get("eval/episode_success")
            if step is not None:
                steps.append(step)
                actor_losses.append(al if al is not None else float("nan"))
                critic_losses.append(cl if cl is not None else float("nan"))
                successes.append(succ if succ is not None else float("nan"))
        data[label] = {
            "steps": steps,
            "actor_loss": actor_losses,
            "critic_loss": critic_losses,
            "success": successes,
        }
        print(f"  {len(steps)} points")

    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def make_losses_plot(data, smooth=3):
    """Plot actor and critic loss for early transformer runs
    to illustrate training instability (losses rising after ~20-40 epochs)."""

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4), constrained_layout=True)

    for info in RUNS_LOSSES:
        label = info["label"]
        color = info["color"]
        series = data[label]
        steps = np.array(series["steps"]) + 1

        # Actor loss (solid)
        vals = np.array(series["actor_loss"])
        if smooth > 1 and len(vals) >= smooth:
            kernel = np.ones(smooth) / smooth
            vals_pad = np.pad(vals, (smooth // 2, smooth // 2), mode="edge")
            vals = np.convolve(vals_pad, kernel, mode="valid")[:len(vals)]
        ax.plot(steps, vals, color=color, ls="-", lw=1.2,
                label=f"{label} (actor)")

        # Critic loss (dashed)
        vals = np.array(series["critic_loss"])
        if smooth > 1 and len(vals) >= smooth:
            kernel = np.ones(smooth) / smooth
            vals_pad = np.pad(vals, (smooth // 2, smooth // 2), mode="edge")
            vals = np.convolve(vals_pad, kernel, mode="valid")[:len(vals)]
        ax.plot(steps, vals, color=color, ls="--", lw=1.2,
                label=f"{label} (critic)")

    ax.axvspan(30, 50, color="red", alpha=0.08, zorder=0)

    ax.set_xlabel("Env Steps (M)")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, 105)
    ax.set_ylim(3, 5)
    ax.grid(True, which="both", axis="y")
    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="black", framealpha=0.9, handlelength=2.5, fontsize=6.5)

    save_fig(fig, "wandb_losses_appendix")
    return fig


def download_temperature(force=False):
    """Download log_temperature and eval/episode_success for the temperature run."""
    cache_path = OUT_DIR / "_wandb_cache_temperature.json"
    if cache_path.exists() and not force:
        print(f"  Using cached data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/3i9ngrhw")
    hist = run.history(pandas=False)
    steps = []
    log_temps = []
    successes = []
    for h in hist:
        step = h.get("_step")
        lt = h.get("training/log_temperature")
        succ = h.get("eval/episode_success")
        if step is not None:
            steps.append(step)
            log_temps.append(lt if lt is not None else float("nan"))
            successes.append(succ if succ is not None else float("nan"))

    data = {
        "steps": steps,
        "log_temperature": log_temps,
        "success": successes,
    }
    print(f"  Downloaded {len(steps)} points")

    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def make_temperature_plot(data, smooth=3):
    """Plot temperature parameter and time at goal for the failed L2+learnable T run
    to illustrate why it collapsed."""

    fig, (ax_temp, ax_succ) = plt.subplots(1, 2, figsize=(6.5, 2.75),
                                           constrained_layout=True, sharex=True)
    steps = np.array(data["steps"]) + 1

    # Temperature (log scale, left y)
    log_temps = np.array(data["log_temperature"])
    if smooth > 1 and len(log_temps) >= smooth:
        kernel = np.ones(smooth) / smooth
        pad = np.pad(log_temps, (smooth // 2, smooth // 2), mode="edge")
        log_temps = np.convolve(pad, kernel, mode="valid")[:len(log_temps)]
    ax_temp.plot(steps, log_temps, color="#D55E00", ls="-", lw=1.2,
                 label="log temperature")
    ax_temp.set_xlabel("Env Steps (M)")
    ax_temp.set_ylabel("log Temperature")
    ax_temp.set_xlim(0, 105)
    ax_temp.grid(True, which="both", axis="y")
    ax_temp.legend(loc="lower left", frameon=True, fancybox=False,
                   edgecolor="black", framealpha=0.9)

    # Success rate
    succ = np.array(data["success"])
    if smooth > 1 and len(succ) >= smooth:
        kernel = np.ones(smooth) / smooth
        pad = np.pad(succ, (smooth // 2, smooth // 2), mode="edge")
        succ = np.convolve(pad, kernel, mode="valid")[:len(succ)]
    ax_succ.plot(steps, succ, color="#0072B2", ls="-", lw=1.2,
                 label="time at goal")
    ax_succ.set_xlabel("Env Steps (M)")
    ax_succ.set_ylabel("Time at goal")
    ax_succ.set_xlim(0, 105)
    ax_succ.set_ylim(bottom=0)
    ax_succ.grid(True, which="both", axis="y")
    ax_succ.legend(loc="upper left", frameon=True, fancybox=False,
                   edgecolor="black", framealpha=0.9)

    save_fig(fig, "wandb_temperature_appendix")
    return fig


def download_deep_comparison(force=False):
    """Download eval/episode_success for the deep/crashed run comparison."""
    cache_path = OUT_DIR / "_wandb_cache_deep.json"
    if cache_path.exists() and not force:
        print(f"  Using cached data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    api = wandb.Api()
    runs_info = [
        ("64-State (crashed)", "xxja73yf"),
        ("32-State-2", "i42kndl4"),
        ("32-MLP", "mkhc4fov"),
    ]
    data = {}
    for label, rid in runs_info:
        print(f"  Downloading {label} ...", end=" ", flush=True)
        run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        hist = run.history(pandas=False)
        successes = []
        for h in hist:
            step = h.get("_step")
            succ = h.get("eval/episode_success")
            if step is not None and succ is not None:
                successes.append((step, succ))
        successes.sort(key=lambda x: x[0])
        data[label] = {
            "steps": [s[0] for s in successes],
            "values": [s[1] for s in successes],
        }
        print(f"  {len(successes)} points, final={successes[-1][1]:.1f}")

    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def make_deep_comparison_plot(data, smooth=3):
    """Plot time at goal for the crashed 64-State run vs 32-State-2 and 32-MLP."""
    
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), constrained_layout=True)

    styles = {
        "64-State (crashed)": {"color": "#CC79A7", "ls": (0, (4, 1.5)), "label": "64-State (crashed)"},
        "32-State-2":           {"color": "#0072B2", "ls": "-", "label": "32-State"},
        "32-MLP":             {"color": "#333333", "ls": "-", "label": "32-MLP"},
    }

    for label in data:
        series = data[label]
        steps = np.array(series["steps"]) + 1
        values = np.array(series["values"])

        if smooth > 1:
            kernel = np.ones(smooth) / smooth
            vals_pad = np.pad(values, (smooth // 2, smooth // 2), mode="edge")
            values = np.convolve(vals_pad, kernel, mode="valid")[:len(values)]

        sty = styles[label]
        ax.plot(steps, values, color=sty["color"], ls=sty["ls"], lw=1.2, label=sty["label"])

    ax.set_xlabel("Env Steps (M)")
    ax.set_ylabel("Time at goal")
    ax.set_xlim(0, 105)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="y")
    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="black", framealpha=0.9, handlelength=2.5)

    save_fig(fig, "wandb_deep_comparison")
    return fig


if __name__ == "__main__":
    print("=== Step 1: Downloading data from W&B (MLP vs State) ===")
    data = download_all(force=False)

    print("\n=== Step 2: Plotting MLP vs State ===")
    make_plot(data, smooth=3)

    print("\n=== Step 3: Downloading data from W&B (transformer modes) ===")
    data_tm = download_transformer_modes(force=False)

    print("\n=== Step 4: Plotting transformer modes ===")
    make_transformer_modes_plot(data_tm, smooth=3)

    print("\n=== Step 5: Downloading data from W&B (tokenization) ===")
    data_tok = download_tokenization(force=False)

    print("\n=== Step 6: Plotting tokenization comparison ===")
    make_tokenization_plot(data_tok, smooth=3)

    print("\n=== Step 7: Downloading data from W&B (pooling) ===")
    data_pool = download_pooling(force=False)

    print("\n=== Step 8: Plotting pooling comparison ===")
    make_pooling_plot(data_pool, smooth=3)

    print("\n=== Step 9: Downloading data from W&B (norm) ===")
    data_norm = download_norm(force=False)

    print("\n=== Step 10: Plotting norm comparison ===")
    make_norm_plot(data_norm, smooth=3)

    print("\n=== Step 11: Downloading data from W&B (embed norm) ===")
    data_en = download_embed_norm(force=False)

    print("\n=== Step 12: Plotting embedding normalization comparison ===")
    make_embed_norm_plot(data_en, smooth=3)

    print("\n=== Step 13: Downloading data from W&B (losses appendix) ===")
    data_losses = download_losses(force=False)

    print("\n=== Step 14: Plotting losses appendix ===")
    make_losses_plot(data_losses, smooth=3)

    print("\n=== Step 15: Downloading data from W&B (temperature appendix) ===")
    data_temp = download_temperature(force=False)

    print("\n=== Step 16: Plotting temperature appendix ===")
    make_temperature_plot(data_temp, smooth=3)

    print("\n=== Step 17: Downloading data from W&B (deep comparison) ===")
    data_deep = download_deep_comparison(force=False)

    print("\n=== Step 18: Plotting deep comparison ===")
    make_deep_comparison_plot(data_deep, smooth=3)

    print(f"\nDone. Files in {OUT_DIR.resolve()}/")
