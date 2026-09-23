# Initial validation and partial benchmark

Source: `2525db9c5bf59d3d6c028a553cf59ca2c482627c`.

The local CPU sweep completed all 48 Helmholtz configurations and some coupled
cases before it was deliberately stopped. The coupled CPU influence setup was
then changed to traverse contiguous systems in `92072b1`; published final
figures use the complete rerun under that revision, not this partial file.

The CPU and A100 logs record successful validation of the initial batched
coupled implementation. These files are retained as provenance, not combined
with the final timing curves.
