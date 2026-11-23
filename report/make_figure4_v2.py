"""Figure 4 for report_v2: 40-task LIBERO success across epochs 1-3 and the epoch-3 suite breakdown."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "smooth-operator" / "experiment" / "adaptive_dwt_vla"
if not ROOT.exists():
    ROOT = HERE.parent / "experiment" / "adaptive_dwt_vla"
OUT = HERE / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.3,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.3,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
s_labels = ["Spatial", "Object", "Goal", "LIBERO-10", "Overall"]
STEPS = {1: "14500", 2: "29000", 3: "43500"}


def load_rates(tag):
    vals = []
    ts = tn = 0
    for s in suites:
        sc = nn = 0
        for tid in range(10):
            f = ROOT / f"results/eval/full40_{tag}__{s}_t{tid}.json"
            if not f.exists():
                return None
            d = json.load(open(f))
            sc += sum(int(e["success"]) for e in d["episodes"])
            nn += len(d["episodes"])
        vals.append(100 * sc / nn)
        ts += sc
        tn += nn
    vals.append(100 * ts / tn)
    return vals


specs = [
    ("dwt_fixed_slw@full_s0_dwt_fixed_slw", "Haar (soft + wt)", "#9dc4e4", "s", "--"),
    ("dwt_fixed@full_s0_dwt_fixed", "Haar (plain CE)", "#1f6fb2", "o", "-"),
    ("dct_fixed@full_s0_dct_fixed", "DCT, no BPE (CE)", "#e6a34a", "^", "-"),
    ("dct_fixed@full_s0_dct_fixed_sqrtw", "DCT, no BPE (sqrt-wt)", "#c96a07", "D", "--"),
    ("poly_fixed@full_s0_poly_fixed_sqrtw", "Gram poly (sqrt-wt)", "#2f8f5b", "P", "-"),
]
REF = ("ref_pi0fast_libero", "Full fine-tune (pi0fast-libero)", "#7a7a7a")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.85), gridspec_kw={"width_ratios": [1.05, 1.45]})

ref = load_rates(REF[0])
ax1.axhline(ref[-1], color=REF[2], linestyle=":", linewidth=1.2)
ax1.text(0.78, ref[-1] + 0.8, f"full fine-tune {ref[-1]:.1f}%", fontsize=6.5, color=REF[2], ha="left")

for prefix, lab, col, mk, ls in specs:
    pts = [(e, load_rates(f"{prefix}_step_{s}")) for e, s in STEPS.items()]
    pts = [(e, r[-1]) for e, r in pts if r is not None]
    xs, ys = zip(*pts)
    ax1.plot(xs, ys, color=col, marker=mk, linestyle=ls, linewidth=1.8, markersize=5.5, label=lab)
    if "poly" in prefix:
        for e, y in pts:
            ax1.text(e, y + 1.8, f"{y:.1f}%", ha="center", fontsize=6.8, color=col, fontweight="bold")
    elif "slw" in prefix:
        for e, y in pts:
            ax1.text(e, y - 4.2, f"{y:.1f}%", ha="center", fontsize=6.7, color=col)
    elif prefix == "dwt_fixed@full_s0_dwt_fixed":
        ax1.text(xs[-1] + 0.07, ys[-1] - 1.0, f"{ys[-1]:.1f}%", ha="left", fontsize=6.7, color=col)
    elif prefix == "dct_fixed@full_s0_dct_fixed":
        ax1.text(xs[-1] + 0.07, ys[-1] - 2.6, f"{ys[-1]:.1f}%", ha="left", fontsize=6.7, color=col)

ax1.set_xticks([1, 2, 3], ["Epoch 1\n(14.5k)", "Epoch 2\n(29k)", "Epoch 3\n(43.5k)"])
ax1.set_xlim(0.72, 3.42)
ax1.set_ylim(45, 98)
ax1.set_ylabel("Overall success (%, 400 eps)")
ax1.set_title("(a) Overall: Epochs 1 to 3")
ax1.grid(axis="y", alpha=0.3)

e3 = [s for s in specs if load_rates(f"{s[0]}_step_43500") is not None]
bars_spec = [(f"{p}_step_43500", lab, col) for p, lab, col, _, _ in e3] + [(REF[0], REF[1], REF[2])]
x = np.arange(len(s_labels))
bw = 0.19
n = len(bars_spec)
for j, (tag, lab, col) in enumerate(bars_spec):
    vals = load_rates(tag)
    offs = x + (j - (n - 1) / 2) * bw
    bars = ax2.bar(offs, vals, width=bw * 0.92, color=col, label=lab if tag == REF[0] else None)
    if "poly" in tag or tag == REF[0]:
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=6.0, color="#222")

ax2.axvline(3.5, color="#888", linestyle="--", linewidth=0.8)
ax2.set_xticks(x, s_labels)
ax2.set_ylim(0, 108)
ax2.set_title("(b) Epoch 3 breakdown by suite")
ax2.grid(axis="y", alpha=0.3)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, ncol=3, loc="upper center", bbox_to_anchor=(0.52, 1.10), frameon=False, fontsize=7.3)
fig.tight_layout(rect=[0, 0, 1, 0.88], pad=0.35)
fig.savefig(OUT / "epoch_suites_v2.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "epoch_suites_v2.pdf")
