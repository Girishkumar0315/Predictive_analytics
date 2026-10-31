"""Publication-quality plotting helpers (Matplotlib + Seaborn)."""
from __future__ import annotations

import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIG_DIR, RES_DIR

PALETTE = {"blue": "#1f5fa8", "orange": "#e07b00", "green": "#2a9d8f", "red": "#c0392b",
           "grey": "#7f8c8d", "purple": "#7b4ea3", "gold": "#c9a227"}
CAT = ["#1f5fa8", "#e07b00", "#2a9d8f", "#c0392b", "#7b4ea3", "#c9a227", "#5d6d7e", "#d35d9e"]
SEQ5 = ["#a50026", "#f08a5d", "#d9d9d9", "#7fb7d6", "#1a5fa8"]     # diverging, low -> high

_REGISTRY: dict = {"figures": [], "tables": []}


def set_style():
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid.axis": "both", "grid.alpha": 0.35,
        "figure.constrained_layout.use": True,
    })


def reset_registry():
    _REGISTRY["figures"].clear(); _REGISTRY["tables"].clear()
    _flush()


def _flush():
    (RES_DIR / "registry.json").write_text(json.dumps(_REGISTRY, indent=1))


def savefig(fig, name: str, caption: str, objective: str = ""):
    """Save a 300-dpi PNG and register the caption (used later by the report builder)."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=300)
    entry = {"file": f"{name}.png", "caption": caption, "objective": objective}
    _REGISTRY["figures"] = [f for f in _REGISTRY["figures"] if f["file"] != entry["file"]] + [entry]
    _flush()
    return path


def save_table(df, name: str, caption: str, objective: str = "", index: bool = False):
    tdir = RES_DIR / "tables"; tdir.mkdir(exist_ok=True)
    df.to_csv(tdir / f"{name}.csv", index=index)
    entry = {"file": f"{name}.csv", "caption": caption, "objective": objective}
    _REGISTRY["tables"] = [t for t in _REGISTRY["tables"] if t["file"] != entry["file"]] + [entry]
    _flush()
