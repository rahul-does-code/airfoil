import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score, root_mean_squared_error

from model import AirfoilMLP, PolyRidgeBaseline


PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def infer_input_dim_from_state_dict(state_dict: dict) -> int:
    """Infer AirfoilMLP input_dim from the first Linear layer weight."""
    if "net.0.weight" not in state_dict:
        raise ValueError("Checkpoint does not contain net.0.weight; incompatible architecture.")
    return int(state_dict["net.0.weight"].shape[1])


def load_test_data():
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    Y_test = np.load(PROCESSED_DIR / "Y_test.npy").astype(np.float32)
    return X_test, Y_test


def evaluate_predictions(name, y_true_scaled, y_pred_scaled):
    print()
    print(name)
    print("-" * len(name))

    with open(PROCESSED_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Stored targets/predictions are standardized [Cl, log(Cd), Cm].
    # First undo StandardScaler, then exponentiate log(Cd) into physical Cd.
    y_true = scaler.inverse_transform(y_true_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled)

    y_true_phys = y_true.copy()
    y_pred_phys = y_pred.copy()
    y_true_phys[:, 1] = np.exp(y_true_phys[:, 1])
    y_pred_phys[:, 1] = np.exp(y_pred_phys[:, 1])

    for i, output_name in enumerate(["Cl", "Cd", "Cm"]):
        r2 = r2_score(y_true_phys[:, i], y_pred_phys[:, i])
        rmse = root_mean_squared_error(y_true_phys[:, i], y_pred_phys[:, i])
        mae = np.mean(np.abs(y_true_phys[:, i] - y_pred_phys[:, i]))

        print(
            f"{output_name}: "
            f"R²={r2:8.4f} | "
            f"RMSE={rmse:10.6f} | "
            f"MAE={mae:10.6f}"
        )


def eval_baseline(X_test, Y_test):
    baseline_paths = sorted(MODELS_DIR.glob("baseline_*.pkl"))

    legacy_path = MODELS_DIR / "baseline.pkl"
    if legacy_path.exists():
        baseline_paths.append(legacy_path)

    if not baseline_paths:
        print("Skipping baseline: no saved baseline model found.")
        return

    seen = set()
    for baseline_path in baseline_paths:
        if baseline_path in seen:
            continue
        seen.add(baseline_path)

        baseline = PolyRidgeBaseline.load(baseline_path)

        try:
            pred = baseline.predict(X_test)
        except ValueError as e:
            print(f"Skipping baseline ({baseline_path.name}): incompatible input dimension. {e}")
            continue

        evaluate_predictions(f"Polynomial Ridge baseline ({baseline_path.name})", Y_test, pred)


def eval_mlp(X_test, Y_test):
    model_paths = sorted(MODELS_DIR.glob("mlp_*_physics*_wcd*_lowcd*_seed0.pt"))

    # Include older naming patterns only so evaluation remains backward-compatible.
    model_paths += sorted(MODELS_DIR.glob("mlp_physics*_wcd*_lowcd*_seed0.pt"))
    model_paths += sorted(MODELS_DIR.glob("mlp_physics*_wcd*_seed0.pt"))

    legacy_path = MODELS_DIR / "mlp_seed0.pt"
    if legacy_path.exists():
        model_paths.append(legacy_path)

    if not model_paths:
        print("Skipping MLP: no saved MLP checkpoints found.")
        return

    seen = set()
    for model_path in model_paths:
        if model_path in seen:
            continue
        seen.add(model_path)

        state_dict = torch.load(model_path, map_location="cpu")

        try:
            input_dim = infer_input_dim_from_state_dict(state_dict)
        except ValueError as e:
            print(f"Skipping MLP ({model_path.name}): {e}")
            continue

        if input_dim != X_test.shape[1]:
            print(
                f"Skipping MLP ({model_path.name}): checkpoint input_dim={input_dim}, "
                f"current X_test input_dim={X_test.shape[1]}"
            )
            continue

        model = AirfoilMLP(input_dim=input_dim)
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            pred = model(torch.from_numpy(X_test)).numpy()

        evaluate_predictions(f"MLP surrogate ({model_path.name})", Y_test, pred)


def eval_ensemble(X_test, Y_test):
    ensemble_dir = MODELS_DIR / "ensemble"
    paths = sorted(ensemble_dir.glob("member_*.pt"))

    if not paths:
        print("Skipping ensemble: no models/ensemble/member_*.pt files found.")
        return

    preds = []
    used_paths = []

    for path in paths:
        state_dict = torch.load(path, map_location="cpu")

        try:
            input_dim = infer_input_dim_from_state_dict(state_dict)
        except ValueError as e:
            print(f"Skipping ensemble member ({path.name}): {e}")
            continue

        if input_dim != X_test.shape[1]:
            print(
                f"Skipping ensemble member ({path.name}): checkpoint input_dim={input_dim}, "
                f"current X_test input_dim={X_test.shape[1]}"
            )
            continue

        model = AirfoilMLP(input_dim=input_dim)
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            pred = model(torch.from_numpy(X_test)).numpy()

        preds.append(pred)
        used_paths.append(path)

    if not preds:
        print("Skipping ensemble: no compatible ensemble members found.")
        return

    preds = np.stack(preds, axis=0)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    evaluate_predictions(f"Deep ensemble mean prediction ({len(used_paths)} members)", Y_test, mean_pred)

    print()
    print("Average ensemble uncertainty")
    print("----------------------------")
    for i, output_name in enumerate(["Cl", "log(Cd)", "Cm"]):
        print(f"{output_name}: mean std in scaled units = {std_pred[:, i].mean():.6f}")


def main():
    X_test, Y_test = load_test_data()

    print("Test rows:", len(X_test))

    eval_baseline(X_test, Y_test)
    eval_mlp(X_test, Y_test)
    eval_ensemble(X_test, Y_test)


if __name__ == "__main__":
    main()
