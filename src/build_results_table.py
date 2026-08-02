"""
src/build_results_table.py

Assembles the full experiment results table across all feature sets and models.
Prints a formatted table ready for inclusion in the writeup.

Each model is scored against features engineered directly from the raw
dataset for its own feature set (not sliced from a single cached X_test.npy —
feature sets are not column-prefixes of each other, e.g. re11's columns 8-10
are completely different engineered quantities than geom13's).

Usage:
    PYTHONPATH=. python src/build_results_table.py
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from model import AirfoilMLP
from preprocess import load_raw, engineer_features, shape_level_split

RAW_H5 = Path("data/raw/polar_dataset_relabeled.h5")
MODELS = Path("models")

def prepare_data():
    """..."""
    import h5py
    with h5py.File(RAW_H5, "r") as _f:
        if "migration" not in _f.attrs:
            raise SystemExit(
                f"{RAW_H5} has no 'migration' attribute — this is the pre-relabel "
                "dataset. Re-splitting it yields 548 apparent shapes, so the test set "
                "overlaps training geometries and all metrics are leakage-contaminated."
            )
    data = load_raw(RAW_H5)

# ── Experiment registry ───────────────────────────────────────────────────────
# (display_name, kind, path, feature_set, input_dim)
EXPERIMENTS = [
    # Baselines
    ("Ridge base8",     "ridge",   MODELS / "baseline.pkl",              "base8",  8),
    ("Ridge geom13",    "ridge",   MODELS / "baseline_geom13.pkl",       "geom13", 13),
    # Neural — base8
    ("MLP base8",       "mlp",     MODELS / "mlp_physics0.001_wcd1_seed0.pt", "base8", 8),
    # Neural — geom13 seeds
    ("MLP geom13 s0",   "mlp",     MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed0.pt", "geom13", 13),
    ("MLP geom13 s1",   "mlp",     MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed1.pt", "geom13", 13),
    ("MLP geom13 s2",   "mlp",     MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed2.pt", "geom13", 13),
    ("MLP geom13 s3",   "mlp",     MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed3.pt", "geom13", 13),
    # Neural — other feature sets
    ("MLP polar9",      "mlp",     MODELS / "mlp_polar9_physics0.001_wcd1_seed0.pt",  "polar9", 9),
    ("MLP re11",        "mlp",     MODELS / "mlp_re11_physics0.001_wcd1_seed0.pt",   "re11",   11),
    ("MLP all16",       "mlp",     MODELS / "mlp_all16_physics0.001_wcd1_seed0.pt",  "all16",  16),
]


def infer_input_dim_from_state_dict(state_dict: dict) -> int:
    if "net.0.weight" not in state_dict:
        raise ValueError("Checkpoint does not contain net.0.weight; incompatible architecture.")
    return int(state_dict["net.0.weight"].shape[1])


def prepare_data():
    """Build Y_test/scaler once from raw data, and a per-feature-set X_test
    cache, independent of whatever data/processed/X_test.npy currently holds."""
    data = load_raw(RAW_H5)
    train_idx, _val_idx, test_idx = shape_level_split(data)

    log_cd = np.log(np.clip(data["cd"], 1e-8, None))
    Y = np.stack([data["cl"], log_cd, data["cm"]], axis=1).astype(np.float32)
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    scaler = StandardScaler()
    scaler.fit(Y_train)
    Y_test_s = scaler.transform(Y_test).astype(np.float32)

    return data, test_idx, Y_test_s, scaler


def x_test_for(data: dict, test_idx: np.ndarray, feature_set: str, cache: dict) -> np.ndarray:
    if feature_set not in cache:
        X, _names = engineer_features(data, feature_set=feature_set)
        cache[feature_set] = X[test_idx]
    return cache[feature_set]


def to_physical(Y_scaled, scaler):
    Y = scaler.inverse_transform(Y_scaled.copy())
    Y[:, 1] = np.exp(Y[:, 1])
    return Y


def metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


def eval_mlp(ckpt_path, X_tensor, input_dim):
    model = AirfoilMLP(input_dim=input_dim)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        return model(X_tensor).numpy()


def eval_ridge(pkl_path, X_np):
    with open(pkl_path, "rb") as f:
        ridge_models = pickle.load(f)
    return np.stack([m.predict(X_np) for m in ridge_models], axis=1)


def main():
    data, test_idx, Y_test_s, scaler = prepare_data()
    Y_true = to_physical(Y_test_s, scaler)
    x_test_cache: dict[str, np.ndarray] = {}

    # Header
    w = 18
    print(f"\n{'Model':<{w}} {'Cl R²':>7} {'Cl RMSE':>9} {'Cl MAE':>8} "
          f"{'Cd R²':>7} {'Cd RMSE':>9} {'Cd MAE':>8} "
          f"{'Cm R²':>7} {'Cm RMSE':>9} {'Cm MAE':>8}")
    print("-" * (w + 3 * 25))

    geom13_preds = []   # collect for ensemble row

    for name, kind, path, feature_set, dim in EXPERIMENTS:
        if not path.exists():
            print(f"{'  [missing] ' + name:<{w}}")
            continue

        X_np = x_test_for(data, test_idx, feature_set, x_test_cache)

        try:
            if kind == "mlp":
                state_dict = torch.load(path, map_location="cpu")
                ckpt_dim = infer_input_dim_from_state_dict(state_dict)
                if ckpt_dim != dim:
                    print(f"  SKIP {name}: registry says dim={dim} but checkpoint has "
                          f"input_dim={ckpt_dim} — fix the EXPERIMENTS entry")
                    continue
                model = AirfoilMLP(input_dim=dim)
                model.load_state_dict(state_dict)
                model.eval()
                with torch.no_grad():
                    pred_s = model(torch.from_numpy(X_np)).numpy()
            else:
                pred_s = eval_ridge(path, X_np)
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
            continue

        pred = to_physical(pred_s, scaler)

        # Collect geom13 seeds for ensemble
        if "geom13" in name and kind == "mlp":
            geom13_preds.append(pred_s)

        cl_r2, cl_rmse, cl_mae = metrics(Y_true[:, 0], pred[:, 0])
        cd_r2, cd_rmse, cd_mae = metrics(Y_true[:, 1], pred[:, 1])
        cm_r2, cm_rmse, cm_mae = metrics(Y_true[:, 2], pred[:, 2])

        print(f"{name:<{w}} {cl_r2:>7.4f} {cl_rmse:>9.5f} {cl_mae:>8.5f} "
              f"{cd_r2:>7.4f} {cd_rmse:>9.5f} {cd_mae:>8.5f} "
              f"{cm_r2:>7.4f} {cm_rmse:>9.5f} {cm_mae:>8.5f}")

    # ── Ensemble row ──────────────────────────────────────────────────────────
    if len(geom13_preds) >= 2:
        ens_pred = to_physical(np.mean(geom13_preds, axis=0), scaler)
        cl_r2, cl_rmse, cl_mae = metrics(Y_true[:, 0], ens_pred[:, 0])
        cd_r2, cd_rmse, cd_mae = metrics(Y_true[:, 1], ens_pred[:, 1])
        cm_r2, cm_rmse, cm_mae = metrics(Y_true[:, 2], ens_pred[:, 2])
        name = f"MLP geom13 ens{len(geom13_preds)}"
        print("-" * (w + 3 * 25))
        print(f"{name:<{w}} {cl_r2:>7.4f} {cl_rmse:>9.5f} {cl_mae:>8.5f} "
              f"{cd_r2:>7.4f} {cd_rmse:>9.5f} {cd_mae:>8.5f} "
              f"{cm_r2:>7.4f} {cm_rmse:>9.5f} {cm_mae:>8.5f}")

    print()


if __name__ == "__main__":
    main()
