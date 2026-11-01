"""Fig. 2 of the report: mean block reads vs recall@10 at n = 1M (static, print)."""
import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "sweeps_n1000000_s0.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "../fig_reads_recall.svg"

rows = defaultdict(list)
with open(src) as f:
    for r in csv.DictReader(f):
        rows[r["config"]].append((float(r["recall_at_10"]), float(r["mean_block_reads"])))

# Categorical slots 1-4 of the reference palette (validated, light mode), plus a
# second encoding (dash + marker) so the series survive grayscale printing.
series = [
    ("A_diskann_vertex_search_id_layout", "A: DiskANN (vertex, ID)", "#2a78d6", (0, (4, 2)), "o"),
    ("B_block_search_id_layout", "B: block, ID", "#eb6834", (0, (1, 1.5)), "s"),
    ("C_starling_block_search_bnf", "C: Starling (block, BNF)", "#1baf7a", "solid", "^"),
    ("F_v2_implicit_blockmates_fixed_layout", "F: V2 pruning + BNF", "#eda100", (0, (6, 1.5, 1, 1.5)), "D"),
]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "font.size": 7.5, "svg.fonttype": "path", "axes.linewidth": 0.6,
})
fig, ax = plt.subplots(figsize=(3.4, 2.15))
ink, muted = "#0b0b0b", "#52514e"
for key, label, color, dash, marker in series:
    pts = sorted(p for p in rows[key] if p[0] >= 0.80)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.plot(xs, ys, color=color, linestyle=dash, linewidth=1.6, marker=marker,
            markersize=3.2, markeredgewidth=0, label=label, zorder=3)
ax.set_yscale("log")
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
ax.yaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 500]))
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

# Direct labels (identity is never color-alone): letter tags near each curve at
# recall 0.90; C and F overlap, so they share one tag below the curves.
import numpy as np
def y_at(key, x):
    pts = sorted(rows[key])
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    return float(np.exp(np.interp(x, xs, np.log(ys))))
for key, tag, dy in [("A_diskann_vertex_search_id_layout", "A", 1.32),
                     ("B_block_search_id_layout", "B", 0.76),
                     ("C_starling_block_search_bnf", "C, F", 0.62)]:
    ax.text(0.905, y_at(key, 0.905) * dy, tag, color=ink, fontsize=7.5,
            ha="center", va="center", zorder=4)
ax.set_xlim(0.80, 1.0)
ax.set_xlabel("Recall@10", color=ink)
ax.set_ylabel("Mean block reads per query", color=ink)
ax.grid(True, which="major", color="#e4e3df", linewidth=0.5, zorder=0)
ax.grid(True, which="minor", axis="y", color="#f0efec", linewidth=0.4, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(muted)
ax.tick_params(colors=muted, width=0.5, length=2.5)
ax.legend(loc="upper left", frameon=False, handlelength=2.6, labelcolor=ink, fontsize=7)
fig.tight_layout(pad=0.3)
fig.savefig(out)
fig.savefig(out.replace(".svg", ".png"), dpi=200)
print("wrote", out)
