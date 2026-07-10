from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from preprocess import FEATURE_SETS, feature_names_for, shape_level_split
from train import validate_alpha_derivative_coverage


RAW_H5 = Path("data/raw/polar_dataset_relabeled.h5")


def _load_h5_columns(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        return {key: h5[key][...] for key in h5.keys()}


def _shape_triples_from_arrays(m: np.ndarray, p: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.round(m * 100),
            np.round(p * 10),
            np.round(t * 100),
        ],
        axis=1,
    ).astype(int)


def test_relabelled_h5_has_discrete_geometry_labels():
    """
    Proves the HDF5 entering the honest pipeline has effective rounded NACA
    geometry labels, not pre-rounding continuous LHS labels.
    """
    assert RAW_H5.exists(), f"Missing relabeled dataset: {RAW_H5}"

    with h5py.File(RAW_H5, "r") as h5:
        m = h5["m"][...]
        p = h5["p"][...]
        t = h5["t"][...]

        features_are = h5.attrs.get("features_are", "")
        if isinstance(features_are, bytes):
            features_are = features_are.decode("utf-8")

    assert features_are == "rounded discrete NACA parameters"

    assert np.allclose(m * 100, np.round(m * 100), atol=1e-8)
    assert np.allclose(p * 10, np.round(p * 10), atol=1e-8)
    assert np.allclose(t * 100, np.round(t * 100), atol=1e-8)

    assert np.all(p[m == 0] == 0)
    assert set(np.round(m * 100).astype(int)) <= set(range(0, 7))
    assert np.all((t >= 0.08) & (t <= 0.18))


def test_shape_level_split_has_disjoint_airfoil_identities():
    """
    Proves the split logic has no discrete NACA shape leakage.

    This runs shape_level_split() on the same relabeled HDF5 that preprocessing
    consumes, then asserts that no integer NACA digit triple appears in more
    than one split.
    """
    assert RAW_H5.exists(), f"Missing relabeled dataset: {RAW_H5}"

    data = _load_h5_columns(RAW_H5)
    train_idx, val_idx, test_idx = shape_level_split(data)

    shapes = _shape_triples_from_arrays(data["m"], data["p"], data["t"])

    train_shapes = set(map(tuple, shapes[train_idx]))
    val_shapes = set(map(tuple, shapes[val_idx]))
    test_shapes = set(map(tuple, shapes[test_idx]))

    assert train_shapes.isdisjoint(val_shapes)
    assert train_shapes.isdisjoint(test_shapes)
    assert val_shapes.isdisjoint(test_shapes)

    assert len(train_shapes) + len(val_shapes) + len(test_shapes) == len(
        set(map(tuple, shapes))
    )


@pytest.mark.parametrize("feature_set", sorted(FEATURE_SETS))
def test_alpha_derivative_registry_covers_every_feature_set(feature_set):
    """
    Proves every engineered feature is explicitly classified as either
    alpha-dependent or alpha-independent.
    """
    assert len(feature_names_for(feature_set)) > 0
    validate_alpha_derivative_coverage(feature_set)
    