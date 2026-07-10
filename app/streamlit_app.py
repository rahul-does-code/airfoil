"""
app/streamlit_app.py

Airfoil surrogate model — interactive prediction app.
Uses the geom13 physics-informed MLP (seed0) as the backend.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from model import AirfoilMLP

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_PATH = Path("models/mlp_geom13_physics0.001_wcd1_lowcd1_seed0.pt")
SCALER_PATH = Path("data/processed/scaler.pkl")
INPUT_DIM   = 13
ALPHA_RANGE = (-5.0, 15.0)
ALPHA_STEP  = 0.5

# ── Load model (cached — only runs once) ─────────────────────────────────────
@st.cache_resource
def load_model():
    model = AirfoilMLP(input_dim=INPUT_DIM)
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model.eval()
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def engineer_features(m, p, t, log_re, alphas_deg):
    """Build geom13 feature matrix for a sweep of alpha values."""
    alpha_rad   = np.deg2rad(alphas_deg)
    sin_a       = np.sin(alpha_rad)
    cos_a       = np.cos(alpha_rad)
    log_re_arr  = np.full_like(sin_a, log_re)
    m_norm      = np.full_like(sin_a, m / 0.09)
    p_norm      = np.full_like(sin_a, p / 0.9)
    t_norm      = np.full_like(sin_a, t / 0.21)
    cl_linear   = 2 * np.pi * sin_a
    t_over_c    = np.full_like(sin_a, t)
    cl_lin_sq   = cl_linear ** 2
    abs_sin_a   = np.abs(sin_a)
    t_sq        = t_norm ** 2
    m_t         = m_norm * t_norm
    t_abs_alpha = t_norm * abs_sin_a

    return np.stack([
        sin_a, cos_a, log_re_arr,
        m_norm, p_norm, t_norm,
        cl_linear, t_over_c,
        cl_lin_sq, abs_sin_a, t_sq, m_t, t_abs_alpha
    ], axis=1).astype(np.float32)


def predict(model, scaler, m, p, t, re):
    alphas = np.arange(ALPHA_RANGE[0], ALPHA_RANGE[1] + ALPHA_STEP, ALPHA_STEP)
    log_re = np.log10(re)

    X = engineer_features(m, p, t, log_re, alphas)
    with torch.no_grad():
        Y_s = model(torch.from_numpy(X)).numpy()

    Y = scaler.inverse_transform(Y_s)
    cl = Y[:, 0]
    cd = np.exp(Y[:, 1])
    cm = Y[:, 2]

    return alphas, cl, cd, cm


def airfoil_coords(m, p, t, n=100):
    """
    Generate NACA 4-series airfoil (x, y_upper, y_lower) coordinates.
    m = max camber (e.g. 0.02), p = camber position (e.g. 0.4), t = thickness (e.g. 0.12)
    """
    x = np.linspace(0, 1, n)

    # Thickness distribution
    yt = (t / 0.2) * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    # Camber line
    yc = np.where(
        x < p,
        (m / p**2) * (2 * p * x - x**2) if p > 0 else np.zeros_like(x),
        (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x - x**2) if p < 1 else np.zeros_like(x),
    )

    y_upper = yc + yt
    y_lower = yc - yt
    return x, y_upper, y_lower


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Airfoil Surrogate", layout="wide")
st.title("Airfoil Aerodynamic Surrogate")
st.caption(
    "Physics-informed neural network surrogate for NACA 4-series airfoil aerodynamics. "
    "Predicts Cl, Cd, Cm without running XFOIL. Model note: This surrogate is trained on " \
    "NACA 4-series XFOIL-generated data. Predictions are most reliable within the training " \
    "range and before strong stall/separation effects."
)

model, scaler = load_model()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Airfoil Parameters")
    st.subheader("Geometry (NACA 4-digit)")

    m = st.slider(
        "Max camber m",
        0.00,
        0.06,
        0.02,
        step=0.01,
        help="1st digit / 100. NACA 0012 → m=0",
    )

    if m == 0:
        p = 0.0
        st.caption("symmetric NACA 00xx — p is 0 by definition")
    else:
        p = st.slider(
            "Camber position p",
            0.1,
            0.7,
            0.4,
            step=0.1,
            help="2nd digit / 10",
        )

    t = st.slider(
        "Thickness t",
        0.08,
        0.18,
        0.12,
        step=0.01,
        help="Digits 3–4 / 100. NACA 0012 → t=0.12",
    )

    st.caption(
        "Training envelope: m ∈ [0, 0.06], p ∈ {0} ∪ [0.1, 0.7], "
        "t ∈ [0.08, 0.18], Re ∈ [2×10⁵, 3×10⁶], α ∈ [−5°, 15°]."
    )

    st.subheader("Flow Conditions")
    re = st.select_slider(
    "Reynolds number",
    options=[2e5, 5e5, 1e6, 2e6, 3e6],
    value=1e6,
    format_func=lambda x: f"{x:.0e}",)

    naca_label = (
        f"NACA {int(round(m*100))}"
        f"{int(round(p*10))}"
        f"{int(round(t*100)):02d}"
    )
    st.info(f"**Airfoil:** {naca_label}")

# ── Prediction ────────────────────────────────────────────────────────────────
alphas, cl, cd, cm = predict(model, scaler, m, p, t, re)

# Summary metrics at AoA = 5°
idx_5 = np.argmin(np.abs(alphas - 5.0))
col1, col2, col3 = st.columns(3)
col1.metric("Cl  at α=5°", f"{cl[idx_5]:.4f}")
col2.metric("Cd  at α=5°", f"{cd[idx_5]:.5f}")
col3.metric("L/D at α=5°", f"{cl[idx_5]/cd[idx_5]:.1f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.patch.set_facecolor("#0e1117")
for ax in axes:
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

# Cl vs alpha
axes[0].plot(alphas, cl, color="#4c9be8", lw=2)
axes[0].axvline(0, color="#555", lw=0.8, ls="--")
axes[0].axhline(0, color="#555", lw=0.8, ls="--")
axes[0].set_xlabel("α (deg)")
axes[0].set_ylabel("Cl")
axes[0].set_title("Lift Curve")
axes[0].grid(True, alpha=0.2)

# Cd vs alpha (drag polar style)
axes[1].plot(cd, cl, color="#e8834c", lw=2)
axes[1].set_xlabel("Cd")
axes[1].set_ylabel("Cl")
axes[1].set_title("Drag Polar  (Cl vs Cd)")

# Cm vs alpha
axes[2].plot(alphas, cm, color="#6ce87a", lw=2)
axes[2].axhline(0, color="#555", lw=0.8, ls="--")
axes[2].set_xlabel("α (deg)")
axes[2].set_ylabel("Cm")
axes[2].set_title("Moment Curve")
axes[2].grid(True, alpha=0.2)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Airfoil shape plot ────────────────────────────────────────────────────────
st.subheader("Airfoil Shape")
x_coord, y_upper, y_lower = airfoil_coords(m, p, t)

fig2, ax2 = plt.subplots(figsize=(8, 2.5))
fig2.patch.set_facecolor("#0e1117")
ax2.set_facecolor("#0e1117")
ax2.plot(x_coord, y_upper, color="#4c9be8", lw=1.5)
ax2.plot(x_coord, y_lower, color="#4c9be8", lw=1.5)
ax2.fill_between(x_coord, y_upper, y_lower, alpha=0.15, color="#4c9be8")
ax2.set_aspect("equal")
ax2.set_xlabel("x/c", color="white")
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_edgecolor("#444")
ax2.set_title(naca_label, color="white")
ax2.grid(True, alpha=0.15)

st.pyplot(fig2)
plt.close()

# ── Raw data table ────────────────────────────────────────────────────────────
with st.expander("Raw prediction data"):
    import pandas as pd
    df = pd.DataFrame({
        "α (deg)": alphas,
        "Cl":      np.round(cl, 5),
        "Cd":      np.round(cd, 6),
        "Cm":      np.round(cm, 5),
        "L/D":     np.round(cl / np.maximum(cd, 1e-8), 2),
    })
    st.dataframe(df, use_container_width=True)

