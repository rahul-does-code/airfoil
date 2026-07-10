from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from naca import naca_digits, digits_to_params


def load_h5(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with h5py.File(path, "r") as h5:
        data = {key: h5[key][...] for key in h5.keys()}
        attrs = dict(h5.attrs)

    return data, attrs


def write_h5(path: Path, data: dict[str, np.ndarray], attrs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as h5:
        for key, arr in data.items():
            h5.create_dataset(key, data=arr, compression="gzip", compression_opts=4)

        for key, value in attrs.items():
            h5.attrs[key] = value

        h5.attrs["features_are"] = "rounded discrete NACA parameters"
        h5.attrs["migration"] = (
            "m, p, t relabeled from original continuous LHS samples to the "
            "effective rounded NACA geometry sent to XFOIL. log_re, alpha, cl, "
            "cd, and cm are unchanged."
        )


def relabel_geometry(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {key: value.copy() for key, value in data.items()}

    m_old = data["m"]
    p_old = data["p"]
    t_old = data["t"]

    m_new = np.empty_like(m_old, dtype=np.float64)
    p_new = np.empty_like(p_old, dtype=np.float64)
    t_new = np.empty_like(t_old, dtype=np.float64)

    for i in range(len(m_old)):
        d1, d2, t_digits = naca_digits(float(m_old[i]), float(p_old[i]), float(t_old[i]))
        m_new[i], p_new[i], t_new[i] = digits_to_params(d1, d2, t_digits)

    assert np.all(p_new[m_new == 0] == 0)
    assert set(np.round(m_new * 100).astype(int)) <= set(range(0, 7))
    assert np.all((t_new >= 0.08) & (t_new <= 0.18))

    out["m"] = m_new
    out["p"] = p_new
    out["t"] = t_new

    return out


def print_shape_stats(data: dict[str, np.ndarray]) -> None:
    shapes = np.stack(
        [
            np.round(data["m"] * 100),
            np.round(data["p"] * 10),
            np.round(data["t"] * 100),
        ],
        axis=1,
    ).astype(int)

    unique_shapes, counts = np.unique(shapes, axis=0, return_counts=True)

    print(f"Total rows: {len(shapes):,}")
    print(f"Unique discrete airfoils: {len(unique_shapes):,}")
    print("Rows per airfoil:")
    print(f"  min:    {counts.min()}")
    print(f"  mean:   {counts.mean():.2f}")
    print(f"  median: {np.median(counts):.2f}")
    print(f"  max:    {counts.max()}")

    print("\nFull distribution:")
    values, freqs = np.unique(counts, return_counts=True)
    for value, freq in zip(values, freqs):
        print(f"  {value:>4} rows: {freq:>5} airfoils")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/polar_dataset.h5"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/polar_dataset_relabeled.h5"),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {args.input}")

    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {args.output}. "
            "Delete it manually if you want to regenerate it."
        )

    data, attrs = load_h5(args.input)
    relabeled = relabel_geometry(data)

    print_shape_stats(relabeled)
    write_h5(args.output, relabeled, attrs)

    print(f"\nWrote relabeled dataset → {args.output}")


if __name__ == "__main__":
    main()