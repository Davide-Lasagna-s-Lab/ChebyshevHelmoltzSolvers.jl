"""Plot recorded timings; never rerun or modify the numerical benchmark."""
from pathlib import Path
import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
source = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results/helmoltz-cpu.csv"
with source.open() as stream:
    rows = list(csv.DictReader(stream))
data = {(int(r["Ny"]), int(r["systems"]), r["mode"]): float(r["minimum_us"]) for r in rows}
ny = sorted({k[0] for k in data})
batches = sorted({k[1] for k in data})
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
colors = ["#7b1fa2", "#1976d2", "#ef6c00", "#388e3c", "#d32f2f", "#0097a7", "#5d4037"]
markers = ["o", "s", "^", "D", "v", "P", "X"]
for batch, color, marker in zip(batches, colors, markers):
    scalar = [data[n, batch, "scalar_contiguous"] for n in ny]
    batched = [data[n, batch, "batched"] for n in ny]
    axes[0].plot(ny, [t/batch for t in scalar], color=color, ls="--", marker=marker, ms=4,
                 markerfacecolor="none")
    axes[0].plot(ny, [t/batch for t in batched], color=color, marker=marker, ms=4)
    axes[1].plot(ny, [s/t for s, t in zip(scalar, batched)],
                 color=color, marker=marker, ms=4, label=f"{batch:,} systems")
for ax in axes:
    ax.set_xlabel("Chebyshev coefficient count $N_y$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ny, [str(n) for n in ny])
    ax.grid(alpha=0.2)
axes[0].set_yscale("log")
axes[0].set_ylabel("Solve time (µs/system)")
axes[0].set_title("Solid: batched · dashed: contiguous scalar")
axes[1].set_ylabel("Speedup $t_{scalar} / t_{batched}$")
axes[1].axhline(1, color="#616161", lw=0.8, ls="--")
axes[1].legend(frameon=False, fontsize=8, ncol=2)
fig.suptitle("CPU Helmholtz solves · Float64 · one Julia thread")
for suffix in ("png", "svg"):
    fig.savefig(source.parent / f"helmoltz-solves.{suffix}")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
colors = ["#7b1fa2", "#1976d2", "#ef6c00", "#388e3c", "#d32f2f", "#0097a7", "#5d4037"]
markers = ["o", "s", "^", "D", "v", "P", "X"]
for n, color, marker in zip(ny, colors, markers):
    for mode, ls in (("scalar_update", "--"), ("batched_update", "-")):
        axes[0].plot(batches, [data[n, b, mode]/1000 for b in batches],
                     color=color, ls=ls, marker=marker, ms=4)
    axes[1].plot(batches, [data[n, b, "scalar_update"]/data[n, b, "batched_update"] for b in batches],
                 color=color, marker=marker, ms=4, label=f"$N_y={n}$")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xticks(batches, [str(b) for b in batches])
    ax.set_xlabel("Number of systems")
    ax.grid(alpha=0.2)
axes[0].set_yscale("log")
axes[0].set_ylabel("Update time (ms/batch)")
axes[0].set_title("Solid: batched · dashed: scalar")
axes[1].set_ylabel("Speedup $t_{scalar} / t_{batched}$")
axes[1].axhline(1, color="#616161", lw=0.8, ls="--")
axes[1].legend(frameon=False, fontsize=8, ncol=2)
fig.suptitle("Operator updates include assembly, factorisation and validation")
for suffix in ("png", "svg"):
    fig.savefig(source.parent / f"helmoltz-updates.{suffix}")
plt.close(fig)

# Keep Matplotlib's generated SVG path data free of trailing whitespace.
for path in source.parent.glob("helmoltz-*.svg"):
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
