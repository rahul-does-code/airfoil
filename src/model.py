import pickle
from pathlib import Path

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.linear_model import Ridge

from sklearn.pipeline import Pipeline

class PolyRidgeBaseline:

    """

    Degree-3 polynomial features + Ridge regression.

    Fits one separate regressor for each output: Cl, Cd, Cm.

    """

    def __init__(self, degree: int = 3, alpha: float = 1.0):

        self.models = [

            Pipeline([

                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, solver = "lsqr")),

            ])

            for _ in range(3)

        ]

    def fit(self, X: np.ndarray, Y: np.ndarray):

        for i, model in enumerate(self.models):

            model.fit(X, Y[:, i])

    def predict(self, X: np.ndarray) -> np.ndarray:

        return np.stack([model.predict(X) for model in self.models], axis=1)

    def save(self, path: Path):

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:

            pickle.dump(self.models, f)

    @classmethod

    def load(cls, path: Path):

        obj = cls.__new__(cls)

        with open(path, "rb") as f:

            obj.models = pickle.load(f)

        return obj

class AirfoilMLP(nn.Module):

    def __init__(self, input_dim: int = 9, hidden: list[int] | None = None):
        
        super().__init__()

        if hidden is None:
            
            hidden = [256, 256, 256, 128, 64]

        layers = []
        
        in_dim = input_dim

        for h in hidden:
            
            layers.append(nn.Linear(in_dim, h))
            
            layers.append(nn.SiLU())
            
            in_dim = h

        layers.append(nn.Linear(in_dim, 3))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.net(x)

class DeepEnsemble:

    def __init__(self, n_members: int = 5, **mlp_kwargs):

        self.members = [AirfoilMLP(**mlp_kwargs) for _ in range(n_members)]

        self.n = n_members

    def train(self):

        for model in self.members:

            model.train()

    def eval(self):

        for model in self.members:

            model.eval()

    def predict(self, x: torch.Tensor):

        with torch.no_grad():

            preds = torch.stack([model(x) for model in self.members], dim=0)

        return preds.mean(dim=0), preds.std(dim=0)

    def save(self, dir_path: Path):

        dir_path.mkdir(parents=True, exist_ok=True)

        for i, model in enumerate(self.members):

            torch.save(model.state_dict(), dir_path / f"member_{i}.pt")

    @classmethod

    def load(cls, dir_path: Path, n_members: int = 5, **mlp_kwargs):

        obj = cls(n_members=0, **mlp_kwargs)

        obj.members = []

        for i in range(n_members):

            model = AirfoilMLP(**mlp_kwargs)

            model.load_state_dict(torch.load(dir_path / f"member_{i}.pt", map_location="cpu"))

            obj.members.append(model)

        obj.n = n_members

        return obj
