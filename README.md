## NACA Airfoil Aerodynamic Surrogate Model

Physics-informed neural network surrogate for NACA 4-series airfoil aerodynamics,
predicting lift coefficient (Cl), drag coefficient (Cd), and pitching moment (Cm)
across a range of angles of attack and Reynolds numbers.

## Motivation
High-fidelity CFD is expensive. This surrogate replaces XFOIL queries with
~1ms neural network inference while maintaining physical consistency.

## Architecture
- 5-layer MLP with SiLU activation + BatchNorm
- Physics-informed input features: sin/cos(α), log(Re), thin airfoil theory Cl prediction, t/c ratio
- Physics-informed loss with gradient penalty enforcing dCl/d(sin α) ≈ 2π for |α| < 8°
- Cd head uses softplus + 1e-4 to enforce positivity

## Data
- XFOIL simulations across NACA 4-series airfoils
- Latin Hypercube Sampling across 5 input dimensions
- Shape-level train/val/test split (70/15/15) to prevent data leakage

## Usage
pip install -r requirements.txt
python train.py
streamlit run app.py
