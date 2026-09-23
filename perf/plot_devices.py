"""Plot saved device benchmarks without rerunning the solver."""
from pathlib import Path
from math import isqrt
import csv
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def batch_label(b):
    """Show square batch sizes as side counts, retaining arbitrary-size support."""
    n = isqrt(b)
    return rf"${n}^2$ systems" if n*n == b else f"{b:,} systems"


source = Path(sys.argv[1])
rows = list(csv.DictReader(source.open()))
samples = ', '.join(map(str, sorted({int(r['samples']) for r in rows})))
colors = ['#7b1fa2', '#1976d2', '#ef6c00', '#388e3c', '#d32f2f', '#0097a7']
markers = ['o', 's', '^', 'D', 'v', 'P']
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                     'font.size': 10, 'savefig.dpi': 200})
kinds = sorted({r['solver'] for r in rows}, reverse=True)
if not any(float(r['gpu_solve_seconds']) > 0 for r in rows):
    for operation in ('solve', 'update'):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout='constrained')
        for ax, kind in zip(axes, kinds):
            for b, color, marker in zip(sorted({int(r['systems']) for r in rows}), colors, markers):
                data = sorted((r for r in rows if r['solver']==kind and int(r['systems'])==b), key=lambda r:int(r['Ny']))
                if not data:
                    continue
                x = [int(r['Ny']) for r in data]
                ax.plot(x, [float(r[f'cpu_{operation}_seconds'])*1e6/b for r in data],
                        color=color, marker=marker, ms=4, label=batch_label(b))
            ax.set(yscale='log', xlabel=r'Coefficient count $N_y$', title=kind.capitalize())
            ax.set_xscale('log', base=2)
            ax.grid(alpha=.2)
        axes[0].set_ylabel(f'Minimum {operation} time (µs/system)')
        axes[1].legend(frameon=False, fontsize=8)
        fig.savefig(source.with_name(source.stem+f'-{operation}.png'))
        plt.close(fig)

if any(float(r['gpu_solve_seconds']) > 0 for r in rows):
    for operation in ("solve", "update"):
        for kind in kinds:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
            for b, color, marker in zip(sorted({int(r['systems']) for r in rows}), colors, markers):
                data = sorted((r for r in rows if r['solver']==kind and int(r['systems'])==b), key=lambda r:int(r['Ny']))
                if not data:
                    continue
                x = [int(r['Ny']) for r in data]
                axes[0].plot(x, [float(r[f"cpu_{operation}_seconds"])*1e6/b for r in data], color=color, ls='--', marker=marker, ms=3, mfc='none')
                axes[0].plot(x, [float(r[f"gpu_{operation}_seconds"])*1e6/b for r in data], color=color, marker=marker, ms=3)
                axes[1].plot(x, [float(r[f"{operation}_speedup"]) for r in data], color=color, marker=marker, ms=4, label=batch_label(b))
            for ax in axes:
                ax.set_xscale('log', base=2)
                ax.set_xlabel(r'Coefficient count $N_y$')
                ax.grid(alpha=.2)
            axes[0].set(yscale='log', ylabel=f'Minimum {operation} time (µs/system)', title='Solid: A100; dashed: batched CPU')
            axes[1].set_ylabel('Speedup: batched CPU / A100')
            axes[1].set_yscale('log')
            axes[1].axhline(1, color='grey', ls='--', lw=.8)
            axes[1].legend(frameon=False, fontsize=8,
                           loc='upper left', bbox_to_anchor=(1.02, 1))
            fig.suptitle(kind.capitalize()+f' {operation} · ComplexF64 · {samples} samples')
            fig.savefig(source.with_name(source.stem+f'-{kind}-{operation}.png'))
            plt.close(fig)
