"""
src/eval_ensemble.py

Evaluate geom13 ensemble (seeds 0–3) on test set.
Averages predictions in standardized space, then inverse-transforms.
Compares against seed0 single model and geom13 ridge baseline.

Usage:
    PYTHONPATH=. python src/eval_ensemble.py
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from model import AirfoilMLP

PROCESSED = Path("data/processed")
MODELS    = Path("models")

GEOM13_CHECKPOINTS = [
    MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed0.pt",
    MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed1.pt",
    MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed2.pt",
    MODELS / "mlp_geom13_physics0.001_wcd1_lowcd1_seed3.pt",
]

INPUT_DIM = 13  # geom13


def load_test_data():
    X = torch.from_numpy(np.load(PROCESSED / "X_test.npy"))
    Y = np.load(PROCESSED / "Y_test.npy")
    with open(PROCESSED / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return X, Y, scaler


def to_physical(Y_scaled: np.ndarray, scaler) -> np.ndarray:
    """Inverse-transform standardized [Cl, log(Cd), Cm] → physical [Cl, Cd, Cm]."""
    Y = scaler.inverse_transform(Y_scaled)
    Y[:, 1] = np.exp(Y[:, 1])
    return Y


def report(name: str, Y_true_phys: np.ndarray, Y_pred_phys: np.ndarray):
    labels = ["Cl", "Cd", "Cm"]
    print(f"\n{name}")
    print("-" * 60)
    for i, lbl in enumerate(labels):
        r2   = r2_score(Y_true_phys[:, i], Y_pred_phys[:, i])
        rmse = root_mean_squared_error(Y_true_phys[:, i], Y_pred_phys[:, i])
        mae  = mean_absolute_error(Y_true_phys[:, i], Y_pred_phys[:, i])
        print(f"  {lbl}: R²={r2:.4f} | RMSE={rmse:.6f} | MAE={mae:.6f}")


def run_model(ckpt_path: Path, X: torch.Tensor) -> np.ndarray:
    model = AirfoilMLP(input_dim=INPUT_DIM)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        return model(X).numpy()


def main():
    X_test, Y_test_scaled, scaler = load_test_data()
    Y_true_phys = to_physical(Y_test_scaled.copy(), scaler)

    # ── Ensemble: average predictions in standardized space ──────────────────
    # Keyed by seed index (not list position) so a missing seed can't shift
    # which prediction later code treats as "seed0".
    preds_scaled = {}
    for seed_idx, ckpt in enumerate(GEOM13_CHECKPOINTS):
        if not ckpt.exists():
            print(f"WARNING: missing {ckpt.name}, skipping")
            continue
        preds_scaled[seed_idx] = run_model(ckpt, X_test)
        print(f"  Loaded {ckpt.name}")

    if len(preds_scaled) == 0:
        print("No checkpoints found — exiting.")
        return

    ensemble_pred_scaled = np.mean(list(preds_scaled.values()), axis=0)
    ensemble_pred_phys   = to_physical(ensemble_pred_scaled.copy(), scaler)
    report(f"Ensemble (n={len(preds_scaled)} seeds, geom13)",
           Y_true_phys, ensemble_pred_phys)

    # ── Seed0 single model for direct comparison ──────────────────────────────
    if 0 in preds_scaled:
        seed0_pred_phys = to_physical(preds_scaled[0].copy(), scaler)
        report("geom13 seed0 (single model)", Y_true_phys, seed0_pred_phys)
    else:
        print("\n(seed0 checkpoint unavailable — skipping seed0 comparison)")

    # ── Ridge baseline ────────────────────────────────────────────────────────
    baseline_path = MODELS / "baseline_geom13.pkl"
    if baseline_path.exists():
        with open(baseline_path, "rb") as f:
            ridge_models = pickle.load(f)
        X_np = X_test.numpy()
        ridge_pred = np.stack([m.predict(X_np) for m in ridge_models], axis=1)
        ridge_pred_phys = to_physical(ridge_pred.copy(), scaler)
        report("Polynomial Ridge geom13", Y_true_phys, ridge_pred_phys)
    else:
        print(f"\n(Ridge baseline not found at {baseline_path} — skipping)")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()