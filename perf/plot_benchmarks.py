"""Plot recorded timings; never rerun or modify the numerical benchmark."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / "results/helmoltz-cpu.csv").open() as stream:
    rows = list(csv.DictReader(stream))
data = {(int(r["Ny"]), int(r["systems"]), r["mode"]): float(r["median_us"]) for r in rows}
ny = sorted({k[0] for k in data})
batches = sorted({k[1] for k in data})
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})
styles = {"scalar_contiguous": ("Scalar, contiguous", "#616161", "o"),
          "scalar_strided": ("Scalar, strided", "#ef6c00", "s"),
          "batched": ("Batched, native layout", "#7b1fa2", "^"),
          "roundtrip": ("Batched + layout conversion", "#1976d2", "D")}
fig, axes = plt.subplots(2, len(batches), figsize=(10, 6), sharex=True, layout="constrained")
for j, batch in enumerate(batches):
    for mode, (label, color, marker) in styles.items():
        times = [data[n, batch, mode] for n in ny]
        axes[0, j].plot(ny, times, label=label, color=color, marker=marker, ms=4)
        if mode != "scalar_contiguous":
            speedup = [data[n, batch, "scalar_contiguous"]/t for n, t in zip(ny, times)]
            axes[1, j].plot(ny, speedup, color=color, marker=marker, ms=4)
    axes[0, j].set_title(f"{batch:,} systems")
    axes[0, j].set_yscale("log")
    axes[1, j].axhline(1, color="#616161", lw=0.8, ls="--")
    axes[1, j].set_xlabel("Chebyshev coefficient count $N_y$")
    axes[1, j].set_xticks(ny)
    for ax in axes[:, j]:
        ax.grid(alpha=0.2)
axes[0, 0].set_ylabel("Complete solve time (µs/batch)")
axes[1, 0].set_ylabel("Speedup over contiguous scalar loop")
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
fig.suptitle("CPU Helmholtz solves · Float64 · one Julia thread")
for suffix in ("png", "svg"):
    fig.savefig(root / f"results/helmoltz-solves.{suffix}")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
colors = ["#7b1fa2", "#1976d2", "#ef6c00"]
for n, color in zip(ny, colors):
    for mode, ls in (("scalar_update", "--"), ("batched_update", "-")):
        axes[0].plot(batches, [data[n, b, mode]/1000 for b in batches],
                     color=color, ls=ls, marker="o", ms=4)
    axes[1].plot(batches, [data[n, b, "batched_update"]/data[n, b, "scalar_update"] for b in batches],
                 color=color, marker="o", ms=4, label=f"$N_y={n}$")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xticks(batches, [str(b) for b in batches])
    ax.set_xlabel("Number of systems")
    ax.grid(alpha=0.2)
axes[0].set_yscale("log")
axes[0].set_ylabel("Update time (ms/batch)")
axes[0].set_title("Solid: batched · dashed: scalar")
axes[1].set_ylabel("Batched / scalar update time")
axes[1].axhline(1, color="#616161", lw=0.8, ls="--")
axes[1].legend(frameon=False)
fig.suptitle("Operator updates include assembly, factorisation and validation")
for suffix in ("png", "svg"):
    fig.savefig(root / f"results/helmoltz-updates.{suffix}")
plt.close(fig)

# Keep Matplotlib's generated SVG path data free of trailing whitespace.
for path in (root / "results").glob("helmoltz-*.svg"):
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
