import pickle
import sys

sys.path.insert(0, "src")

import numpy as np
import torch

from model import AirfoilMLP
from preprocess import feature_names_for
from train import physics_informed_loss


P = "data/processed"

X = torch.from_numpy(np.load(f"{P}/X_train.npy")).float()[:512]
Y = torch.from_numpy(np.load(f"{P}/Y_train.npy")).float()[:512]
alpha = torch.from_numpy(np.load(f"{P}/alpha_train.npy")).float()[:512]

with open(f"{P}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

cl_std = float(scaler.scale_[0])
names = feature_names_for("geom13")

torch.manual_seed(0)
model = AirfoilMLP(input_dim=13)
params = list(model.parameters())


def grad_norm(loss):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = torch.zeros(())
    for grad in grads:
        if grad is not None:
            total = total + (grad ** 2).sum()
    return torch.sqrt(total).item()


mse = physics_informed_loss(
    model,
    X,
    Y,
    alpha,
    cl_std,
    w_physics=0.0,
    feature_names=names,
)

full = physics_informed_loss(
    model,
    X,
    Y,
    alpha,
    cl_std,
    w_physics=0.001,
    feature_names=names,
)

print("mse grad norm:        ", grad_norm(mse))
print("w*penalty grad norm:  ", grad_norm(full - mse))
print("loss difference:      ", float((full - mse).detach()))
