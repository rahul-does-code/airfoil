"""
src/build_results_table.py

Assembles the full experiment results table across all feature sets and models.
Prints a formatted table ready for inclusion in the writeup.

Usage:
    PYTHONPATH=. python src/build_results_table.py
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from model import AirfoilMLP

PROCESSED  = Path("data/processed")
MODELS     = Path("models")

# ── Experiment registry ───────────────────────────────────────────────────────
# (display_name, type, path, input_dim)
EXPERIMENTS = [
    # Baselines
    ("Ridge base8",     "ridge",   MODELS / "baseline.pkl",              8),
    ("Ridge geom13",    "ridge",   MODELS / "baseline_geom13.pkl",       13),
    # Neural — base8
    ("MLP base8",       "mlp",     MODELS / "mlp_physics0.001_wcd1_seed0.pt", 8),
    # Neural — geom13 seeds
    ("MLP geom13 s0",   "mlp",     MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed0.pt", 13),
    ("MLP geom13 s1",   "mlp",     MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed1.pt", 13),
    ("MLP geom13 s2",   "mlp",     MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed2.pt", 13),
    ("MLP geom13 s3",   "mlp",     MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed3.pt", 13),
    # Neural — other feature sets
    ("MLP polar9",      "mlp",     MODELS / "mlp_polar9_physics0.001_wcd1_seed0.pt",  9),
    ("MLP re11",        "mlp",     MODELS / "mlp_re11_physics0.001_wcd1_seed0.pt",   11),
    ("MLP all16",       "mlp",     MODELS / "mlp_all16_physics0.001_wcd1_seed0.pt",  16),
]


def to_physical(Y_scaled, scaler):
    Y = scaler.inverse_transform(Y_scaled.copy())
    Y[:, 1] = np.exp(Y[:, 1])
    return Y


def metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
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
    X_test   = torch.from_numpy(np.load(PROCESSED / "X_test.npy"))
    Y_test_s = np.load(PROCESSED / "Y_test.npy")
    with open(PROCESSED / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    Y_true = to_physical(Y_test_s, scaler)
    X_np   = X_test.numpy()

    # Header
    w = 18
    print(f"\n{'Model':<{w}} {'Cl R²':>7} {'Cl RMSE':>9} {'Cl MAE':>8} "
          f"{'Cd R²':>7} {'Cd RMSE':>9} {'Cd MAE':>8} "
          f"{'Cm R²':>7} {'Cm RMSE':>9} {'Cm MAE':>8}")
    print("-" * (w + 3 * 25))

    geom13_preds = []   # collect for ensemble row

    for name, kind, path, dim in EXPERIMENTS:
        if not path.exists():
            print(f"{'  [missing] ' + name:<{w}}")
            continue

        # Slice X to the right feature dimension
        X_in = X_test[:, :dim] if kind == "mlp" else X_np[:, :dim]

        try:
            if kind == "mlp":
                pred_s = eval_mlp(path, X_test[:, :dim], dim)
            else:
                pred_s = eval_ridge(path, X_np[:, :dim])
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
    