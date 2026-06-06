"""
src/eval_regime.py

Cd alpha-regime breakdown for geom13 ensemble and seed0.
Reports R², RMSE, MAE by low/mid/high alpha.
Also prints per-regime Cd distribution stats.

Usage:
    PYTHONPATH=. python src/eval_regime.py
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from model import AirfoilMLP

PROCESSED = Path("data/processed")
MODELS    = Path("models")
INPUT_DIM = 13

GEOM13_CHECKPOINTS = [
    MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed0.pt",
    MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed1.pt",
    MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed2.pt",
    MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed3.pt",
]

REGIMES = [
    ("low   |α|<8",    lambda a: np.abs(a) < 8),
    ("mid   8≤|α|<12", lambda a: (np.abs(a) >= 8) & (np.abs(a) < 12)),
    ("high  |α|≥12",   lambda a: np.abs(a) >= 12),
]


def to_physical(Y_scaled, scaler):
    Y = scaler.inverse_transform(Y_scaled.copy())
    Y[:, 1] = np.exp(Y[:, 1])
    return Y


def regime_report(name: str, cd_true: np.ndarray, cd_pred: np.ndarray,
                  alpha: np.ndarray):
    print(f"\n{'='*60}")
    print(f"Cd regime breakdown — {name}")
    print(f"{'='*60}")
    print(f"{'Regime':<20} {'n':>6}  {'R²':>8}  {'RMSE':>10}  {'MAE':>10}")
    print("-" * 60)
    for label, mask_fn in REGIMES:
        mask = mask_fn(alpha)
        n = mask.sum()
        if n == 0:
            continue
        r2   = r2_score(cd_true[mask], cd_pred[mask])
        rmse = mean_squared_error(cd_true[mask], cd_pred[mask], squared=False)
        mae  = mean_absolute_error(cd_true[mask], cd_pred[mask])
        print(f"{label:<20} {n:>6}  {r2:>8.4f}  {rmse:>10.6f}  {mae:>10.6f}")


def cd_distribution(alpha: np.ndarray, cd_true: np.ndarray):
    print("\nCd distribution by regime (true values):")
    print(f"{'Regime':<20} {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    print("-" * 56)
    for label, mask_fn in REGIMES:
        mask = mask_fn(alpha)
        cd = cd_true[mask]
        print(f"{label:<20} {cd.mean():>8.5f}  {cd.std():>8.5f}  "
              f"{cd.min():>8.5f}  {cd.max():>8.5f}")


def run_model(ckpt_path, X):
    model = AirfoilMLP(input_dim=INPUT_DIM)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        return model(X).numpy()


def main():
    X_test   = torch.from_numpy(np.load(PROCESSED / "X_test.npy"))
    Y_test_s = np.load(PROCESSED / "Y_test.npy")
    alpha    = np.load(PROCESSED / "alpha_test.npy")

    with open(PROCESSED / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    Y_true_phys = to_physical(Y_test_s, scaler)
    cd_true = Y_true_phys[:, 1]

    # Distribution stats (true data)
    cd_distribution(alpha, cd_true)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    preds = []
    for ckpt in GEOM13_CHECKPOINTS:
        if ckpt.exists():
            preds.append(run_model(ckpt, X_test))
    
    if preds:
        ens_pred_phys = to_physical(np.mean(preds, axis=0), scaler)
        regime_report(f"Ensemble (n={len(preds)})", cd_true, 
                      ens_pred_phys[:, 1], alpha)

    # ── Seed0 ─────────────────────────────────────────────────────────────────
    if GEOM13_CHECKPOINTS[0].exists():
        seed0_phys = to_physical(preds[0], scaler)
        regime_report("geom13 seed0", cd_true, seed0_phys[:, 1], alpha)


if __name__ == "__main__":
    main()
