import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb

from model import AirfoilMLP, PolyRidgeBaseline


def read_feature_set(processed_dir: Path) -> str:
    path = processed_dir / "feature_set.txt"
    if path.exists():
        return path.read_text().strip()
    return f"dim{np.load(processed_dir / 'X_train.npy').shape[1]}"


def physics_informed_loss(
    model: nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    alpha_raw: torch.Tensor,
    cl_std: float,
    w_cd: float = 1.0,
    w_cd_low_alpha: float = 1.0,
    w_physics: float = 0.01,
) -> torch.Tensor:
    """
    Combined supervised MSE + thin-airfoil gradient penalty.

    The model predicts standardized [Cl, log(Cd), Cm]. Therefore the
    thin-airfoil lift-slope target must also be standardized:
        dCl_scaled / d(alpha_rad) = 2π / Cl_std

    Because alpha appears in multiple engineered features, the physics loss
    uses the total derivative with respect to alpha, not only the partial
    derivative with respect to sin(alpha).
    """
    y_pred = model(x)

    cl_loss = F.mse_loss(y_pred[:, 0], y_true[:, 0])
    cm_loss = F.mse_loss(y_pred[:, 2], y_true[:, 2])

    # Cd is trained in standardized log(Cd) space. Low-alpha drag has tiny
    # physical variance, so optionally give those points extra weight.
    low_alpha_mask = alpha_raw.abs() < 8.0
    cd_weight = torch.where(
        low_alpha_mask,
        torch.full_like(y_true[:, 1], w_cd_low_alpha),
        torch.full_like(y_true[:, 1], w_cd),
    )
    cd_loss = (cd_weight * (y_pred[:, 1] - y_true[:, 1]) ** 2).mean()

    mse = cl_loss + cd_loss + cm_loss

    if w_physics <= 0:
        return mse

    linear_mask = alpha_raw.abs() < 8.0
    if linear_mask.sum() < 2:
        return mse

    x_lin = x[linear_mask].clone().detach().requires_grad_(True)
    y_lin = model(x_lin)
    cl_lin = y_lin[:, 0]

    grad_x = torch.autograd.grad(
        cl_lin.sum(),
        x_lin,
        create_graph=True,
    )[0]

    # Input feature columns:
    #   0 = sin(alpha)
    #   1 = cos(alpha)
    #   6 = cl_linear = 2*pi*sin(alpha)
    # Thin airfoil theory constrains dCl/dalpha_rad, so use the chain rule
    # through every alpha-dependent engineered input feature.
    sin_a = x_lin[:, 0]
    cos_a = x_lin[:, 1]

    dcl_scaled_dalpha = (
        grad_x[:, 0] * cos_a
        + grad_x[:, 1] * (-sin_a)
        + grad_x[:, 6] * (2 * torch.pi * cos_a)
    )

    target_slope = (2 * torch.pi) / cl_std
    physics_penalty = F.mse_loss(
        dcl_scaled_dalpha,
        torch.full_like(dcl_scaled_dalpha, target_slope),
    )

    return mse + w_physics * physics_penalty


def load_split(processed_dir: Path, split: str):
    X = torch.from_numpy(np.load(processed_dir / f"X_{split}.npy")).float()
    Y = torch.from_numpy(np.load(processed_dir / f"Y_{split}.npy")).float()
    alpha = torch.from_numpy(np.load(processed_dir / f"alpha_{split}.npy")).float()
    return X, Y, alpha


def train_baseline(processed_dir: Path, out_path: Path):
    X_train = np.load(processed_dir / "X_train.npy")
    Y_train = np.load(processed_dir / "Y_train.npy")
    feature_set = read_feature_set(processed_dir)
    print(f"Training baseline with feature_set={feature_set}, input_dim={X_train.shape[1]}")

    baseline = PolyRidgeBaseline(degree=3, alpha=10.0)
    baseline.fit(X_train, Y_train)
    baseline.save(out_path)

    print(f"Baseline saved → {out_path}")


def train_mlp(
    processed_dir: Path,
    out_path: Path,
    seed: int = 0,
    epochs: int = 150,
    batch_size: int = 2048,
    lr: float = 3e-4,
    w_cd: float = 1.0,
    w_cd_low_alpha: float = 1.0,
    w_physics: float = 0.01,
    use_wandb: bool = True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {seed}")

    X_train, Y_train, alpha_train = load_split(processed_dir, "train")
    X_val, Y_val, _ = load_split(processed_dir, "val")

    input_dim = X_train.shape[1]
    feature_set = read_feature_set(processed_dir)

    with open(processed_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    cl_std = float(scaler.scale_[0])
    print(f"Cl std from scaler: {cl_std:.6f}")
    print("Physics loss version: chain-rule dCl/dalpha through sin, cos, and cl_linear")
    print(
        f"Training config: feature_set={feature_set}, input_dim={input_dim}, "
        f"epochs={epochs}, batch_size={batch_size}, "
        f"lr={lr}, w_cd={w_cd}, w_cd_low_alpha={w_cd_low_alpha}, "
        f"w_physics={w_physics}"
    )

    train_ds = TensorDataset(X_train, Y_train, alpha_train)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = AirfoilMLP(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if use_wandb:
        wandb.init(
            project="airfoil-surrogate",
            config={
                "seed": seed,
                "feature_set": feature_set,
                "input_dim": input_dim,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "w_cd": w_cd,
                "w_cd_low_alpha": w_cd_low_alpha,
                "w_physics": w_physics,
            },
        )

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb, ab in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            ab = ab.to(device)

            optimizer.zero_grad()

            loss = physics_informed_loss(
                model,
                xb,
                yb,
                ab,
                cl_std=cl_std,
                w_cd=w_cd,
                w_cd_low_alpha=w_cd_low_alpha,
                w_physics=w_physics,
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        with torch.no_grad():
            xv = X_val.to(device)
            yv = Y_val.to(device)
            pred = model(xv)

            val_mse = F.mse_loss(pred, yv).item()
            val_cl = F.mse_loss(pred[:, 0], yv[:, 0]).item()
            val_cd = F.mse_loss(pred[:, 1], yv[:, 1]).item()
            val_cm = F.mse_loss(pred[:, 2], yv[:, 2]).item()

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if use_wandb:
            wandb.log(
                {
                    "train_loss": float(np.mean(train_losses)),
                    "val_mse": val_mse,
                    "val_cl_mse": val_cl,
                    "val_cd_mse": val_cd,
                    "val_cm_mse": val_cm,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"train={np.mean(train_losses):.6f} | "
                f"val={val_mse:.6f} | "
                f"cl={val_cl:.6f} cd={val_cd:.6f} cm={val_cm:.6f}"
            )

    if best_state is None:
        raise RuntimeError("Training failed: no best model state was recorded.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path)
    print(f"Saved best model → {out_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["baseline", "mlp", "ensemble"], default="mlp")
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out_dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--w_physics", type=float, default=0.01)
    parser.add_argument("--w_cd", type=float, default=1.0)
    parser.add_argument("--w_cd_low_alpha", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()

    if args.mode == "baseline":
        feature_set = read_feature_set(args.processed_dir)
        train_baseline(args.processed_dir, args.out_dir / f"baseline_{feature_set}.pkl")

    elif args.mode == "mlp":
        seed = args.seeds[0]
        feature_set = read_feature_set(args.processed_dir)
        train_mlp(
            processed_dir=args.processed_dir,
            out_path=args.out_dir
            / f"mlp_{feature_set}_physics{args.w_physics:g}_wcd{args.w_cd:g}_lowcd{args.w_cd_low_alpha:g}_seed{seed}.pt",
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            w_cd=args.w_cd,
            w_cd_low_alpha=args.w_cd_low_alpha,
            w_physics=args.w_physics,
            use_wandb=not args.no_wandb,
        )

    elif args.mode == "ensemble":
        for seed in args.seeds:
            feature_set = read_feature_set(args.processed_dir)
            print(f"\n── Ensemble member seed={seed} ──")
            train_mlp(
                processed_dir=args.processed_dir,
                out_path=args.out_dir / "ensemble" / f"member_{feature_set}_{seed}.pt",
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                w_cd=args.w_cd,
                w_cd_low_alpha=args.w_cd_low_alpha,
                w_physics=args.w_physics,
                use_wandb=not args.no_wandb,
            )