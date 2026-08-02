"""
src/preprocess.py

Preprocess the raw XFOIL HDF5 dataset for model training.

Available feature sets:
    base8   = original physics-informed features
    polar9  = base8 + cl_linear_sq
    geom13  = base8 + drag-polar/geometric drag features
    re11    = base8 + Reynolds/skin-friction interaction features
    all16   = base8 + geom features + Reynolds interaction features

Targets before scaling:
    cl, log(cd), cm
"""

import argparse
import h5py
import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


FEATURE_SETS = {"base8", "polar9", "geom13", "re11", "all16"}


def load_raw(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        return {k: f[k][:] for k in f.keys()}


def feature_names_for(feature_set: str) -> list[str]:
    """Ordered feature name list for a feature set. Single source of truth for
    column ordering, shared by engineer_features() and the physics-informed
    loss (which needs to know which columns are alpha-dependent)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set={feature_set!r}. Choose from {sorted(FEATURE_SETS)}")

    names = [
        "sin_alpha",
        "cos_alpha",
        "log_re",
        "m_norm",
        "p_norm",
        "t_norm",
        "cl_linear",
        "t_over_c",
    ]

    if feature_set in {"polar9", "geom13", "all16"}:
        names.append("cl_linear_sq")

    if feature_set in {"geom13", "all16"}:
        names.extend(["abs_sin_alpha", "t_norm_sq", "m_t", "t_abs_alpha"])

    if feature_set in {"re11", "all16"}:
        names.extend(["log_re_t", "log_re_sq", "log_re_abs_sin_alpha"])

    return names


def engineer_features(data: dict, feature_set: str = "base8") -> tuple[np.ndarray, list[str]]:
    names = feature_names_for(feature_set)

    alpha_rad = np.deg2rad(data["alpha"])

    sin_a = np.sin(alpha_rad)
    cos_a = np.cos(alpha_rad)

    if "log_re" in data:
        log_re = data["log_re"]
    else:
        log_re = np.log10(data["Re"])

    m_norm = data["m"] / 0.09
    p_norm = data["p"] / 0.9
    t_norm = data["t"] / 0.21

    cl_linear = 2 * np.pi * sin_a
    t_over_c = data["t"]

    cl_linear_sq = cl_linear ** 2
    abs_sin_a = np.abs(sin_a)
    t_norm_sq = t_norm ** 2
    m_t = m_norm * t_norm
    t_abs_alpha = t_norm * abs_sin_a

    log_re_t = log_re * t_norm
    log_re_sq = log_re ** 2
    log_re_abs_sin_alpha = log_re * abs_sin_a

    all_features = {
        "sin_alpha": sin_a,
        "cos_alpha": cos_a,
        "log_re": log_re,
        "m_norm": m_norm,
        "p_norm": p_norm,
        "t_norm": t_norm,
        "cl_linear": cl_linear,
        "t_over_c": t_over_c,
        "cl_linear_sq": cl_linear_sq,
        "abs_sin_alpha": abs_sin_a,
        "t_norm_sq": t_norm_sq,
        "m_t": m_t,
        "t_abs_alpha": t_abs_alpha,
        "log_re_t": log_re_t,
        "log_re_sq": log_re_sq,
        "log_re_abs_sin_alpha": log_re_abs_sin_alpha,
    }

    X = np.stack([all_features[name] for name in names], axis=1).astype(np.float32)
    return X, names

def shape_level_split(data: dict, seed: int = 42):
    """
    Split by discrete NACA airfoil identity: rounded m/p/t digits.

    Re is deliberately excluded from identity. All alpha/Re rows for a
    geometry stay in one split, so the test set measures generalization to
    unseen geometry rather than unseen Reynolds numbers.
    """
    rng = np.random.default_rng(seed)

    shapes = np.stack(
        [
            np.round(data["m"] * 100),
            np.round(data["p"] * 10),
            np.round(data["t"] * 100),
        ],
        axis=1,
    ).astype(int)

    unique_shapes = np.unique(shapes, axis=0)
    rng.shuffle(unique_shapes)

    n_shapes = len(unique_shapes)
    n_train = int(0.70 * n_shapes)
    n_val = int(0.15 * n_shapes)

    train_shapes = set(map(tuple, unique_shapes[:n_train]))
    val_shapes = set(map(tuple, unique_shapes[n_train:n_train + n_val]))

    train_idx = []
    val_idx = []
    test_idx = []

    for i, shape in enumerate(map(tuple, shapes)):
        if shape in train_shapes:
            train_idx.append(i)
        elif shape in val_shapes:
            val_idx.append(i)
        else:
            test_idx.append(i)

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)

    split_shape_counts = {
        "train": len(train_shapes),
        "val": len(val_shapes),
        "test": n_shapes - len(train_shapes) - len(val_shapes),
    }

    split_row_counts = {
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx),
    }

    total_rows = len(shapes)

    print("Shape-level split:")
    for name in ["train", "val", "test"]:
        print(
            f"  {name:<5}: "
            f"{split_shape_counts[name]:>5} shapes "
            f"({split_shape_counts[name] / n_shapes:>6.2%}), "
            f"{split_row_counts[name]:>7} rows "
            f"({split_row_counts[name] / total_rows:>6.2%})"
        )

    return train_idx, val_idx, test_idx

def build_dataset(h5_path: Path, out_dir: Path, feature_set: str = "base8"):
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_raw(h5_path)

    X, feature_names = engineer_features(data, feature_set=feature_set)
    log_cd = np.log(np.clip(data["cd"], 1e-8, None))
    Y = np.stack([data["cl"], log_cd, data["cm"]], axis=1).astype(np.float32)

    train_idx, val_idx, test_idx = shape_level_split(data)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]

    scaler = StandardScaler()
    Y_train = scaler.fit_transform(Y_train).astype(np.float32)
    Y_val = scaler.transform(Y_val).astype(np.float32)
    Y_test = scaler.transform(Y_test).astype(np.float32)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "X_test.npy", X_test)

    np.save(out_dir / "Y_train.npy", Y_train)
    np.save(out_dir / "Y_val.npy", Y_val)
    np.save(out_dir / "Y_test.npy", Y_test)

    np.save(out_dir / "alpha_train.npy", data["alpha"][train_idx].astype(np.float32))
    np.save(out_dir / "alpha_val.npy", data["alpha"][val_idx].astype(np.float32))
    np.save(out_dir / "alpha_test.npy", data["alpha"][test_idx].astype(np.float32))

    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(out_dir / "feature_set.txt", "w") as f:
        f.write(feature_set + "\n")

    with open(out_dir / "feature_names.txt", "w") as f:
        for name in feature_names:
            f.write(name + "\n")

    print(f"Feature set: {feature_set}")
    print(f"Input dimension: {X.shape[1]}")
    print("Features:", ", ".join(feature_names))
    print(f"Train: {len(X_train):,}")
    print(f"Val:   {len(X_val):,}")
    print(f"Test:  {len(X_test):,}")
    print(f"Total: {len(X):,}")
    print(f"Saved preprocessed data → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=Path, default=Path("data/raw/polar_dataset_relabeled.h5"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--feature_set", choices=sorted(FEATURE_SETS), default="base8")
    args = parser.parse_args()

    import h5py
    with h5py.File(args.h5_path, "r") as _f:
        if "migration" not in _f.attrs:
            raise SystemExit(
                f"{args.h5_path} has no 'migration' attribute — this looks like the "
                "pre-relabel dataset. Preprocessing it reproduces the geometry-labeling "
                "bug (548 apparent shapes instead of 483). Run src/relabel_data.py first, "
                "or pass --h5_path data/raw/polar_dataset_relabeled.h5."
            )

    build_dataset(
        h5_path=args.h5_path,
        out_dir=args.out_dir,
        feature_set=args.feature_set,
    )
