from pathlib import Path

import h5py
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path("data/raw/polar_dataset.h5")
PLOTS_DIR = Path("data/plots")


def load_hdf5_dataset(path):
    with h5py.File(path, "r") as h5:
        data = {
            "m": h5["m"][:],
            "p": h5["p"][:],
            "t": h5["t"][:],
            "log_re": h5["log_re"][:],
            "alpha": h5["alpha"][:],
            "cl": h5["cl"][:],
            "cd": h5["cd"][:],
            "cm": h5["cm"][:],
        }

    df = pd.DataFrame(data)
    # (m, p, t) is the shape identifier used by preprocess.shape_level_split.
    # There's no NACA code string stored in this schema, so build a readable
    # per-shape label for grouping/legends instead.
    df["shape"] = df.apply(lambda r: f"m={r['m']:.3f} p={r['p']:.2f} t={r['t']:.3f}", axis=1)
    return df


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_hdf5_dataset(DATA_PATH)

    print("Dataset shape:", df.shape)
    print()
    print("Columns:")
    print(df.columns.tolist())
    print()
    print("Rows per shape:")
    print(df["shape"].value_counts().sort_index())
    print()
    print("Numerical summary:")
    print(df[["m", "p", "t", "log_re", "alpha", "cl", "cd", "cm"]].describe())
    print()

    bad_drag = df[df["cd"] <= 0]
    print("Rows with cd <= 0:", len(bad_drag))

    missing_alpha_counts = df.groupby("shape")["alpha"].count()
    print()
    print("Shapes with fewer than 21 alpha points:")
    print(missing_alpha_counts[missing_alpha_counts < 21])

    # Plot Cl vs alpha for each shape
    plt.figure()
    for shape, group in df.groupby("shape"):
        group = group.sort_values("alpha")
        plt.plot(group["alpha"], group["cl"], marker="o", linewidth=1, markersize=3, label=shape)

    plt.xlabel("Angle of attack, alpha (deg)")
    plt.ylabel("Lift coefficient, Cl")
    plt.title("Cl vs Alpha for LHS Airfoil Samples")
    plt.legend(fontsize=6, ncol=3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cl_vs_alpha.png", dpi=200)

    # Plot Cd vs alpha for each shape
    plt.figure()
    for shape, group in df.groupby("shape"):
        group = group.sort_values("alpha")
        plt.plot(group["alpha"], group["cd"], marker="o", linewidth=1, markersize=3, label=shape)

    plt.xlabel("Angle of attack, alpha (deg)")
    plt.ylabel("Drag coefficient, Cd")
    plt.title("Cd vs Alpha for LHS Airfoil Samples")
    plt.legend(fontsize=6, ncol=3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cd_vs_alpha.png", dpi=200)

    # Plot Cl/Cd vs alpha
    df["cl_over_cd"] = df["cl"] / df["cd"]

    plt.figure()
    for shape, group in df.groupby("shape"):
        group = group.sort_values("alpha")
        plt.plot(group["alpha"], group["cl_over_cd"], marker="o", linewidth=1, markersize=3, label=shape)

    plt.xlabel("Angle of attack, alpha (deg)")
    plt.ylabel("Cl / Cd")
    plt.title("Lift-to-Drag Ratio vs Alpha")
    plt.legend(fontsize=6, ncol=3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cl_over_cd_vs_alpha.png", dpi=200)

    print()
    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
