"""
Evidence for two README claims:
  1. XFOIL convergence rate is roughly flat across the swept alpha range.
  2. Rows-per-airfoil spread is driven by Reynolds multiplicity, not convergence.

Run: PYTHONPATH=. python scripts/dataset_diagnostics.py | tee logs/dataset_diagnostics.log
"""
import h5py, numpy as np, pandas as pd

H5 = "data/raw/polar_dataset_relabeled.h5"

with h5py.File(H5, "r") as f:
    attrs = dict(f.attrs)
    df = pd.DataFrame({k: f[k][:] for k in ["m", "p", "t", "alpha", "log_re"]})

n_runs = len(df.log_re.unique())          # one unique Re per XFOIL run
print(f"file: {H5}")
print(f"attrs n_samples={attrs['n_samples']}  n_failed={attrs['n_failed']}  total_rows={attrs['total_rows']}")
print(f"successful runs (unique log_re): {n_runs}")

# ── Claim 1: convergence rate vs alpha ───────────────────────────────────────
# Each successful run sweeps the same alpha grid. A run missing a given alpha
# means XFOIL failed to converge at that operating point.
grid = np.arange(attrs["alpha_start"], attrs["alpha_end"] + 1e-9, attrs["alpha_step"])
counts = df.groupby("alpha").size()
rate = (counts.reindex(np.round(grid, 3), fill_value=0) / n_runs * 100)

print("\nconvergence rate by alpha (% of successful runs producing a row):")
for a, r in rate.items():
    print(f"  alpha={a:6.1f}   {r:6.2f}%")
print(f"\n  min={rate.min():.2f}%  max={rate.max():.2f}%  mean={rate.mean():.2f}%")
print(f"  spread (max-min) = {rate.max() - rate.min():.2f} percentage points")

lo = rate[rate.index <= 5].mean()
hi = rate[rate.index >= 10].mean()
print(f"  mean rate at alpha<=5: {lo:.2f}%   at alpha>=10: {hi:.2f}%   difference: {hi-lo:+.2f} pp")
print(f"  NOTE: sweep ends at alpha={attrs['alpha_end']}; no post-stall points were ever requested.")

# ── Claim 2: rows per airfoil vs Reynolds multiplicity ───────────────────────
df["d1"] = np.rint(df.m * 100).astype(int)
df["d2"] = np.rint(df.p * 10).astype(int)
df["d34"] = np.rint(df.t * 100).astype(int)

g = df.groupby(["d1", "d2", "d34"]).agg(rows=("alpha", "size"), n_re=("log_re", "nunique"))
r = np.corrcoef(g.rows, g.n_re)[0, 1]

print(f"\nairfoils: {len(g)}")
print(f"rows per airfoil: min={g.rows.min()}  median={g.rows.median():.0f}  max={g.rows.max()}")
print(f"Re values per airfoil: min={g.n_re.min()}  median={g.n_re.median():.0f}  max={g.n_re.max()}")
print(f"corr(rows, n_re) = {r:.4f}")
print(f"mean rows per Re value = {(g.rows / g.n_re).mean():.1f}  (alpha grid has {len(grid)} points)")