from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from model import AirfoilMLP
from preprocess import engineer_features, feature_names_for, shape_level_split
from train import chain_rule_dcl_scaled_dalpha


RAW_H5 = Path("data/raw/polar_dataset_relabeled.h5")
PROCESSED_DIR = Path("data/processed")
CKPT = Path("models/mlp_geom13_physics0.001_wcd1_lowcd1_seed0.pt")


def _load_h5_columns(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        return {key: h5[key][...] for key in h5.keys()}


def _load_model(input_dim: int) -> AirfoilMLP:
    assert CKPT.exists(), f"Missing checkpoint for derivative test: {CKPT}"

    model = AirfoilMLP(input_dim=input_dim)
    state = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model


def _predict_scaled_cl(model: AirfoilMLP, x_np: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.tensor(x_np, dtype=torch.float32)
        return model(x).detach().cpu().numpy()[:, 0]


def test_geom13_reconstructed_test_features_match_processed_x_test():
    """
    Sanity check before the finite-difference test: reconstruct X_test from the
    relabeled HDF5 and shape_level_split(), then verify it matches the saved
    processed X_test.npy. This proves the test is using the same rows/order as
    training and evaluation.
    """
    assert RAW_H5.exists(), f"Missing relabeled HDF5: {RAW_H5}"

    data = _load_h5_columns(RAW_H5)
    _, _, test_idx = shape_level_split(data)

    test_data = {key: value[test_idx] for key, value in data.items()}

    x_rebuilt, names = engineer_features(test_data, feature_set="geom13")
    x_saved = np.load(PROCESSED_DIR / "X_test.npy")

    assert names == feature_names_for("geom13")
    assert x_rebuilt.shape == x_saved.shape

    np.testing.assert_allclose(
        x_rebuilt,
        x_saved,
        rtol=1e-6,
        atol=1e-6,
    )


def test_geom13_chain_rule_derivative_matches_finite_difference():
    """
    Gold-standard check for the physics chain rule.

    For selected test rows, perturb alpha by ±eps radians, rebuild engineered
    features through preprocess.engineer_features, run the trained model, and
    compare finite-difference dCl_scaled/dalpha to the analytic chain-rule
    derivative used by the physics loss.
    """
    assert RAW_H5.exists(), f"Missing relabeled HDF5: {RAW_H5}"

    feature_set = "geom13"
    feature_names = feature_names_for(feature_set)

    data = _load_h5_columns(RAW_H5)
    _, _, test_idx = shape_level_split(data)

    test_data = {key: value[test_idx] for key, value in data.items()}
    x_test = np.load(PROCESSED_DIR / "X_test.npy")

    # Avoid alpha exactly near 0 because abs_sin_alpha has a kink there.
    alpha_deg = test_data["alpha"]
    candidates = np.where(np.abs(alpha_deg) > 0.25)[0]
    assert len(candidates) >= 8

    chosen = candidates[np.linspace(0, len(candidates) - 1, 8).astype(int)]

    model = _load_model(input_dim=len(feature_names))

    eps_rad = 1e-4
    eps_deg = np.rad2deg(eps_rad)

    base = {key: value[chosen] for key, value in test_data.items()}

    plus = {key: value.copy() for key, value in base.items()}
    minus = {key: value.copy() for key, value in base.items()}

    plus["alpha"] = plus["alpha"] + eps_deg
    minus["alpha"] = minus["alpha"] - eps_deg

    x_plus, plus_names = engineer_features(plus, feature_set=feature_set)
    x_minus, minus_names = engineer_features(minus, feature_set=feature_set)

    assert plus_names == feature_names
    assert minus_names == feature_names

    cl_plus = _predict_scaled_cl(model, x_plus)
    cl_minus = _predict_scaled_cl(model, x_minus)

    finite_diff = (cl_plus - cl_minus) / (2 * eps_rad)

    x_base = torch.tensor(x_test[chosen], dtype=torch.float32)
    analytic = chain_rule_dcl_scaled_dalpha(
        model=model,
        x=x_base,
        feature_names=feature_names,
    )
    analytic = analytic.detach().cpu().numpy()

    np.testing.assert_allclose(
        analytic,
        finite_diff,
        rtol=1e-3,
        atol=1e-3,
    )