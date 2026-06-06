from pathlib import Path

import h5py
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path("data/raw/naca4_lhs_dataset.h5")
PLOTS_DIR = Path("data/plots")


def load_hdf5_dataset(path):
    with h5py.File(path, "r") as h5:
        data = {
            "naca": [x.decode() for x in h5["naca"][:]],
            "max_camber": h5["max_camber"][:],
            "camber_position": h5["camber_position"][:],
            "thickness": h5["thickness"][:],
            "Re": h5["Re"][:],
            "alpha": h5["alpha"][:],
            "CL": h5["CL"][:],
            "CD": h5["CD"][:],
            "CM": h5["CM"][:],
        }

    return pd.DataFrame(data)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_hdf5_dataset(DATA_PATH)

    print("Dataset shape:", df.shape)
    print()
    print("Columns:")
    print(df.columns.tolist())
    print()
    print("Rows per airfoil:")
    print(df["naca"].value_counts().sort_index())
    print()
    print("Numerical summary:")
    print(df[["max_camber", "camber_position", "thickness", "Re", "alpha", "CL", "CD", "CM"]].describe())
    print()

    bad_drag = df[df["CD"] <= 0]
    print("Rows with CD <= 0:", len(bad_drag))

    missing_alpha_counts = df.groupby("naca")["alpha"].count()
    print()
    print("Airfoils with fewer than 21 alpha points:")
    print(missing_alpha_counts[missing_alpha_counts < 21])

    # Plot CL vs alpha for each airfoil
    plt.figure()
    for naca, group in df.groupby("naca"):
        group = group.sort_values("alpha")
        plt.plot(group["alpha"], group["CL"], marker="o", linewidth=1, markersize=3, label=naca)

    plt.xlabel("Angle of attack, alpha (deg)")
    plt.ylabel("Lift coefficient, CL")
    plt.title("CL vs Alpha for LHS Airfoil Samples")
    plt.legend(fontsize=6, ncol=3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cl_vs_alpha.png", dpi=200)

    # Plot CD vs alpha for each airfoil
    plt.figure()
    for naca, group in df.groupby("naca"):
        group = group.sort_values("alpha")
        plt.plot(group["alpha"], group["CD"], marker="o", linewidth=1, markersize=3, label=naca)

    plt.xlabel("Angle of attack, alpha (deg)")
    plt.ylabel("Drag coefficient, CD")
    plt.title("CD vs Alpha for LHS Airfoil Samples")
    plt.legend(fontsize=6, ncol=3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cd_vs_alpha.png", dpi=200)

    # Plot CL/CD vs alpha
    df["CL_over_CD"] = df["CL"] / df["CD"]

    plt.figure()
    for naca, group in df.groupby("naca"):
        group = group.sort_values("alpha")
        plt.plot(group["alpha"], group["CL_over_CD"], marker="o", linewidth=1, markersize=3, label=naca)

    plt.xlabel("Angle of attack, alpha (deg)")
    plt.ylabel("CL / CD")
    plt.title("Lift-to-Drag Ratio vs Alpha")
    plt.legend(fontsize=6, ncol=3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cl_over_cd_vs_alpha.png", dpi=200)

    print()
    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
