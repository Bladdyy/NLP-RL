"""
Download data from W&B and create publication-quality plots.

1. Comparison of pretrained encoders vs trainable encoder
   (MLP 8 x exact cls hybrid, x = minilm, bge, e5, gte, trainable).
2. Comparison of description_type (MLP 8 BGE cls hybrid, differing only
   in description_type: exact / coordinates / high_level).
3. Comparison of text_pooling (TR 8 BGE exact, differing only in
   text_pooling: cls / mean / token).
4. Comparison of MLP vs Transformer BASE runs
   (MLP 8/16/32 BASE vs TR 8/16/32 BASE).  TR 16 BASE is TBD.
5. Comparison of MLP x BGE exact cls hybrid vs MLP x Base (all 3 sizes).
6. Comparison of TR x BGE exact cls hybrid vs TR x Base (all 3 sizes).
   Runs for plots 6 will be added later; the code is left commented out.

Outputs:
  paper_figures/wandb_encoder_compare.pdf
  paper_figures/wandb_description_type_compare.pdf
  paper_figures/wandb_pooling_compare.pdf
  paper_figures/wandb_mlp_vs_transformer.pdf
  paper_figures/wandb_mlp_bge_vs_base.pdf
  paper_figures/wandb_tr_bge_vs_base.pdf
"""

import json
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

ENTITY = "oskarkulinski-mimuw"
PROJECT = "NLP"

# =====================================================================
# Run definitions
# =====================================================================

# ── Plot 1: Pretrained encoders vs trainable encoder ──────────────────
RUNS_ENCODER = {
    "minilm":    {"id": "x4s6uus3",  "label": "MINILM"},
    "bge":       {"id": "48zypgf9",  "label": "BGE"},
    "e5":        {"id": "wr4rm24g",  "label": "E5"},
    "gte":       {"id": "50zimqt8",  "label": "GTE"},
    "trainable": {"id": "c326gyrg",  "label": "Trainable"},
}

# ── Plot 2: Description type comparison ───────────────────────────────
RUNS_DESC_TYPE = {
    "exact":       {"id": "48zypgf9",  "label": "Exact"},
    "coordinates": {"id": "udruzpqh",  "label": "Coordinates"},
    "high_level":  {"id": "76lgjrzy",  "label": "High-level"},
}

# ── Plot 3: Pooling comparison (text_pooling) ─────────────────────────
# (Will be filled once the missing runs are available.)
RUNS_POOLING = {}

# ── Plot 4: MLP vs Transformer BASE runs ──────────────────────────────
RUNS_MLP_VS_TR = {
    "MLP 8":  {"id": "f7xnk9yj",  "label": "MLP 8",   "arch": "MLP", "depth": 8},
    "MLP 16": {"id": "vc6blv0f",  "label": "MLP 16",  "arch": "MLP", "depth": 16},
    "MLP 32": {"id": "mkhc4fov",  "label": "MLP 32",  "arch": "MLP", "depth": 32},
    "TR 8":   {"id": "5q1mo7o3",  "label": "TR 8",    "arch": "TR",  "depth": 8},
    # "TR 16":  {"id": "???",       "label": "TR 16",   "arch": "TR",  "depth": 16},  # TBD
    "TR 32":  {"id": "i42kndl4",  "label": "TR 32",   "arch": "TR",  "depth": 32},
}

# ── Plot 5: MLP BGE hybrid vs MLP Base (3 sizes) ──────────────────────
RUNS_MLP_BGE_VS_BASE = {
    "MLP 8 BGE exact cls hybrid":  {"id": "48zypgf9",  "label": "MLP 8 BGE"},
    "MLP 8 BASE":                  {"id": "f7xnk9yj",  "label": "MLP 8 Base"},
    "MLP 16 BGE exact cls hybrid": {"id": "1nxtxnnf",  "label": "MLP 16 BGE"},
    "MLP 16 BASE":                 {"id": "vc6blv0f",  "label": "MLP 16 Base"},
    "MLP 32 BGE exact cls hybrid": {"id": "wgkso04c",  "label": "MLP 32 BGE"},
    "MLP 32 BASE":                 {"id": "mkhc4fov",  "label": "MLP 32 Base"},
}

# ── Plot 6: TR BGE hybrid vs TR Base (3 sizes) ────────────────────────
# (Runs to be added; leave empty for now.)
RUNS_TR_BGE_VS_BASE = {}

# =====================================================================
# Colour schemes
# =====================================================================

# Plot 1: encoder colours
ENCODER_COLORS = {
    "minilm":    "#0072B2",
    "bge":       "#D55E00",
    "e5":        "#009E73",
    "gte":       "#CC79A7",
    "trainable": "#333333",
}

# Plot 2: description type colours
DESC_TYPE_COLORS = {
    "exact":       "#0072B2",
    "coordinates": "#D55E00",
    "high_level":  "#009E73",
}

# Plot 4: MLP vs TR — blues for MLP, oranges for TR
ARCH_COLORS = {
    "MLP": {"color_fn": lambda d: plt.cm.Blues({8: 0.48, 16: 0.72, 32: 0.90}[d]), "ls": "-"},
    "TR":  {"color_fn": lambda d: plt.cm.Oranges({8: 0.48, 16: 0.72, 32: 0.90}[d]), "ls": "-"},
}

# Plot 5: MLP BGE vs Base — solid for BGE, dashed for Base
MLP_BGE_BASE_STYLES = {
    "MLP 8 BGE":  {"color": "#D55E00", "ls": "-"},
    "MLP 8 Base": {"color": "#0072B2", "ls": "--"},
    "MLP 16 BGE": {"color": "#D55E00", "ls": "-"},
    "MLP 16 Base":{"color": "#0072B2", "ls": "--"},
    "MLP 32 BGE": {"color": "#D55E00", "ls": "-"},
    "MLP 32 Base":{"color": "#0072B2", "ls": "--"},
}


# =====================================================================
# Helpers
# =====================================================================

def download_runs(runs_dict, cache_name, force=False):
    """Download eval/episode_success for a dict of runs, cache result."""
    cache_path = OUT_DIR / f"_wandb_cache_{cache_name}.json"
    if cache_path.exists() and not force:
        print(f"  Using cached data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    api = wandb.Api()
    data = {}
    for key, info in runs_dict.items():
        print(f"  Downloading {key} ...", end=" ", flush=True)
        run = api.run(f"{ENTITY}/{PROJECT}/{info['id']}")
        hist = run.history(pandas=False)
        successes = []
        for h in hist:
            step = h.get("_step")
            succ = h.get("eval/episode_success")
            if step is not None and succ is not None:
                successes.append((step, succ))
        successes.sort(key=lambda x: x[0])
        data[key] = {
            "steps": [s[0] for s in successes],
            "values": [s[1] for s in successes],
        }
        print(f"  {len(successes)} points, final={successes[-1][1]:.1f}")

    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def smooth_series(values, window=3):
    """Simple moving-average smoothing (centred, edge-padded)."""
    if window <= 1 or len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def make_plot(data, runs_dict, color_map, linestyle_map, filename, ylabel="Time at goal"):
    """Generic single-axis line plot."""
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), constrained_layout=True)

    for key in runs_dict:
        series = data[key]
        steps = np.array(series["steps"]) + 1
        values = smooth_series(np.array(series["values"]))

        color = color_map[key]
        ls = linestyle_map.get(key, "-")
        ax.plot(steps, values, color=color, ls=ls, lw=1.2, label=runs_dict[key]["label"])

    ax.set_xlabel("Env Steps (M)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 105)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="y")
    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="black", framealpha=0.9, handlelength=2.5)

    save_fig(fig, filename)


def save_fig(fig, name):
    for ext in [".pdf", ".png"]:
        path = OUT_DIR / f"{name}{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=350)
        print(f"  Saved {path}")
    plt.close(fig)


# =====================================================================
# Plotting functions
# =====================================================================

def make_encoder_plot(data):
    """Plot #1: Pretrained encoders vs trainable encoder."""
    linestyles = {k: "-" for k in RUNS_ENCODER}
    make_plot(data, RUNS_ENCODER, ENCODER_COLORS, linestyles, "wandb_encoder_compare")


def make_description_type_plot(data):
    """Plot #2: Description type comparison."""
    linestyles = {k: "-" for k in RUNS_DESC_TYPE}
    make_plot(data, RUNS_DESC_TYPE, DESC_TYPE_COLORS, linestyles, "wandb_description_type_compare")


def make_mlp_vs_transformer_plot(data):
    """Plot #4: MLP vs Transformer BASE runs."""
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), constrained_layout=True)

    for key, info in RUNS_MLP_VS_TR.items():
        series = data[key]
        steps = np.array(series["steps"]) + 1
        values = smooth_series(np.array(series["values"]))

        style = ARCH_COLORS[info["arch"]]
        color = style["color_fn"](info["depth"])
        ax.plot(steps, values, color=color, ls="-", lw=1.2, label=info["label"])

    ax.set_xlabel("Env Steps (M)")
    ax.set_ylabel("Time at goal")
    ax.set_xlim(0, 105)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="y")
    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="black", framealpha=0.9, handlelength=2.5)
    save_fig(fig, "wandb_mlp_vs_transformer")


def make_mlp_bge_vs_base_plot(data):
    """Plot #5: MLP x BGE exact cls hybrid vs MLP x Base (3 sizes).

    BGE variants use green (solid), Base variants use blue (dashed).
    Each size gets its own shade (light → dark for larger depth).
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.75), constrained_layout=True)

    # Distinct colour families: Base = blues, BGE = greens
    base_depth_colors = {8: plt.cm.Blues(0.55), 16: plt.cm.Blues(0.72), 32: plt.cm.Blues(0.90)}
    bge_depth_colors  = {8: plt.cm.Greens(0.45), 16: plt.cm.Greens(0.65), 32: plt.cm.Greens(0.85)}

    for key, info in RUNS_MLP_BGE_VS_BASE.items():
        series = data[key]
        steps = np.array(series["steps"]) + 1
        values = smooth_series(np.array(series["values"]))

        # Determine depth from label
        lbl = info["label"]
        if "8" in lbl:
            depth = 8
        elif "16" in lbl:
            depth = 16
        else:
            depth = 32

        is_bge = "BGE" in lbl
        color = bge_depth_colors[depth] if is_bge else base_depth_colors[depth]
        ls = "-"
        lw = 1.4 if is_bge else 1.0
        ax.plot(steps, values, color=color, ls=ls, lw=lw, label=lbl)

    ax.set_xlabel("Env Steps (M)")
    ax.set_ylabel("Time at goal")
    ax.set_xlim(0, 105)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="y")
    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="black", framealpha=0.9, handlelength=2.5)
    save_fig(fig, "wandb_mlp_bge_vs_base")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    print("=== Plot 1: Pretrained encoders vs trainable encoder ===")
    data_enc = download_runs(RUNS_ENCODER, "encoder", force=False)
    make_encoder_plot(data_enc)

    print("\n=== Plot 2: Description type comparison ===")
    data_desc = download_runs(RUNS_DESC_TYPE, "description_type", force=False)
    make_description_type_plot(data_desc)

    print("\n=== Plot 3: Pooling comparison (TBD — missing runs) ===")
    if RUNS_POOLING:
        data_pool = download_runs(RUNS_POOLING, "pooling", force=False)
        # make_pooling_plot(data_pool)
    else:
        print("  Skipped — no pooling runs defined yet.")

    print("\n=== Plot 4: MLP vs Transformer BASE runs ===")
    data_mlptr = download_runs(RUNS_MLP_VS_TR, "mlp_vs_tr", force=False)
    make_mlp_vs_transformer_plot(data_mlptr)

    print("\n=== Plot 5: MLP x BGE exact cls hybrid vs MLP x Base ===")
    data_mlp_bge = download_runs(RUNS_MLP_BGE_VS_BASE, "mlp_bge_vs_base", force=False)
    make_mlp_bge_vs_base_plot(data_mlp_bge)

    print("\n=== Plot 6: TR x BGE exact cls hybrid vs TR x Base (TBD) ===")
    if RUNS_TR_BGE_VS_BASE:
        data_tr_bge = download_runs(RUNS_TR_BGE_VS_BASE, "tr_bge_vs_base", force=False)
        # make_tr_bge_vs_base_plot(data_tr_bge)
    else:
        print("  Skipped — runs to be added later.")

    print(f"\nDone. Files in {OUT_DIR.resolve()}/")