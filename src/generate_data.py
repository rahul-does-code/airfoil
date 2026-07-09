"""
src/generate_data.py

Generate the surrogate training dataset via Latin Hypercube Sampling.
Runs XFOIL across the NACA 4-series + Re space, stores results in HDF5.

m/p/t are LHS-sampled continuously but then rounded to integer NACA 4-series
digits before being handed to XFOIL. The HDF5 output stores the *effective*
rounded m/p/t (what XFOIL actually simulated), not the pre-rounding
continuous samples, so engineered features describe the real geometry.
Datasets generated before this fix stored the pre-rounding continuous values
instead.

Usage:
    PYTHONPATH=. python src/generate_data.py --n_samples 2000 --output data/raw/polar_dataset.h5
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.stats.qmc import LatinHypercube, scale
from naca import naca_digits, digits_to_params

# Add project root to path so local imports are available.
sys.path.insert(0, str(Path(__file__).parent.parent))

# This project uses a local xfoil package that wraps the Homebrew XFOIL binary.
from xfoil import XFoil
from xfoil.model import Naca4


BOUNDS_LOW = [0.00, 0.0, 0.08, np.log10(2e5)]
BOUNDS_HIGH = [0.06, 0.7, 0.18, np.log10(3e6)]

ALPHA_START = -5.0
ALPHA_END = 15.0
ALPHA_STEP = 0.5

MAX_ITER = 100
MIN_CONVERGED_POINTS = 10


def sample_design_points(n_samples: int, seed: int = 42) -> np.ndarray:
    sampler = LatinHypercube(d=4, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    return scale(unit_samples, BOUNDS_LOW, BOUNDS_HIGH)


def run_one(xf: XFoil, m: float, p: float, t: float, log_re: float):
    d1, d2, t_digits = naca_digits(m, p, t)
    m_eff, p_eff, t_eff = digits_to_params(d1, d2, t_digits)

    d3, d4 = t_digits // 10, t_digits % 10

    xf.airfoil = Naca4(d1, d2, d3, d4)
    xf.Re = 10 ** log_re
    xf.M = 0.0
    xf.max_iter = MAX_ITER

    try:
        a, cl, cd, cm, _ = xf.aseq(ALPHA_START, ALPHA_END, ALPHA_STEP)
    except RuntimeError:
        return None

    if len(a) < MIN_CONVERGED_POINTS:
        return None

    return (
        np.array(a),
        np.array(cl),
        np.array(cd),
        np.array(cm),
        m_eff,
        p_eff,
        t_eff,
    )

def main(n_samples: int, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    design_points = sample_design_points(n_samples)
    print(f"Generated {n_samples} LHS design points.")
    print(f"Saving to {output_path}\n")

    xf = XFoil()
    records = []
    t0 = time.time()
    n_failed = 0

    for i, (m, p, t, log_re) in enumerate(design_points):
        result = run_one(xf, m, p, t, log_re)

        if result is None:
            n_failed += 1
            if (i + 1) % 100 == 0 or n_samples <= 100:
                elapsed = time.time() - t0
                print(f"  [{i + 1}/{n_samples}]  failed: {n_failed}  elapsed: {elapsed:.0f}s")
            continue

        alphas, cls, cds, cms, m_eff, p_eff, t_eff = result
        n_pts = len(alphas)

        records.append({
            "m": np.full(n_pts, m_eff),
            "p": np.full(n_pts, p_eff),
            "t": np.full(n_pts, t_eff),
            "log_re": np.full(n_pts, log_re),
            "alpha": alphas,
            "cl": cls,
            "cd": cds,
            "cm": cms,
        })

        if (i + 1) % 100 == 0 or n_samples <= 100:
            elapsed = time.time() - t0
            rate = (i + 1 - n_failed) / max(elapsed, 1e-6)
            eta = (n_samples - i - 1) / max(rate, 1e-6)
            print(
                f"  [{i + 1}/{n_samples}]  ok: {i + 1 - n_failed}  failed: {n_failed}  "
                f"rate: {rate:.1f}/s  ETA: {eta / 60:.1f}min"
            )

    if not records:
        raise RuntimeError("No successful XFOIL runs. Dataset is empty.")

    all_keys = ["m", "p", "t", "log_re", "alpha", "cl", "cd", "cm"]
    flat = {key: np.concatenate([record[key] for record in records]) for key in all_keys}
    total_rows = len(flat["alpha"])

    print(f"\nTotal data rows: {total_rows:,}  ({n_samples - n_failed} successful runs)")
    print(f"Failed runs: {n_failed} ({100 * n_failed / n_samples:.1f}%)")

    with h5py.File(output_path, "w") as f:
        for key, arr in flat.items():
            f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)

        f.attrs["n_samples"] = n_samples
        f.attrs["n_failed"] = n_failed
        f.attrs["total_rows"] = total_rows
        f.attrs["alpha_start"] = ALPHA_START
        f.attrs["alpha_end"] = ALPHA_END
        f.attrs["alpha_step"] = ALPHA_STEP
        f.attrs["xfoil_source"] = "Homebrew XFOIL binary wrapped by local Python xfoil package"
        f.attrs["features_are"] = "rounded discrete NACA parameters"
        
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Number of (airfoil, Re) LHS design points",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/polar_dataset.h5"),
        help="Output HDF5 file path",
    )
    args = parser.parse_args()
    main(args.n_samples, args.output)
