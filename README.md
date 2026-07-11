# NACA Airfoil Aerodynamic Surrogate Model

Physics-informed surrogate modeling for NACA 4-series airfoil aerodynamics. The project trains fast regression models to predict lift coefficient `Cl`, drag coefficient `Cd`, and pitching moment coefficient `Cm` from engineered airfoil/Reynolds/angle-of-attack features generated from XFOIL polars.

The current rebuild focuses on credibility: geometry labels are stored as the effective rounded NACA geometry actually sent to XFOIL, train/validation/test splits are performed by discrete airfoil identity, and the physics loss is validated against a finite-difference derivative check.

## Data

The dataset contains XFOIL polars over discrete NACA 4-series airfoils. Geometry and Reynolds-number design points are sampled with Latin Hypercube Sampling over maximum camber `m`, camber location `p`, thickness `t`, and `log10(Re)`. Angle of attack is not LHS-sampled; it is swept from `-5°` to `15°` in `0.5°` increments.

Training envelope:

- `m ∈ [0, 0.06]`
- `p ∈ {0} ∪ [0.1, 0.7]`
- `t ∈ [0.08, 0.18]`
- `Re ∈ [2×10^5, 3×10^6]`
- `α ∈ [-5°, 15°]`

Unconverged XFOIL points are excluded. These failures tend to cluster near and after stall, so the dataset and reported metrics under-represent that regime.

## Split

Train, validation, and test splits are performed by discrete airfoil identity using rounded NACA `(m, p, t)` digits. All angle-of-attack and Reynolds-number rows for a given airfoil stay in the same split.

This measures generalization to unseen geometry, not generalization to unseen Reynolds numbers.

The rebuild includes a pytest invariant that reconstructs the discrete airfoil identities and asserts that train/validation/test shape sets are pairwise disjoint.

## Features

Feature ordering is defined in `src/preprocess.py` by `feature_names_for(feature_set)`. The deployed model uses `geom13`:

1. `sin_alpha`
2. `cos_alpha`
3. `log_re`
4. `m_norm`
5. `p_norm`
6. `t_norm`
7. `cl_linear`
8. `t_over_c`
9. `cl_linear_sq`
10. `abs_sin_alpha`
11. `t_norm_sq`
12. `m_t`
13. `t_abs_alpha`

where:

- `m_norm = m / 0.09`
- `p_norm = p / 0.9`
- `t_norm = t / 0.21`
- `cl_linear = 2π sin(α)`

## Model

The neural surrogate is an MLP with hidden widths `[256, 256, 256, 128, 64]` and SiLU activations.

Targets are standardized:

```text

[Cl, log(Cd), Cm]

```

Cd positivity is handled by predicting standardized `log(Cd`, inverse-transforming the model output, and exponentiating `log(Cd)` during evaluation and app inference. There is no softplus Cd head.

## Physics Loss

The physics term penalizes the total derivative `dCl/dα` in the low-angle linear region `|α| < 8°`.

Because the model predicts standardized `Cl`, the thin-airfoil target is also scaled:

```text

dCl_scaled/dα = 2π / σ_Cl

```

where `σ_Cl` is the `Cl` scale from the fitted target scaler.

The derivative is chained through every alpha-dependent engineered feature using a feature-name registry in `src/train.py`. The rebuild includes a finite-difference pytest check comparing the analytic chain-rule derivative against central finite differences through `src/preprocess.py` and the trained model.

The `2π` target is a soft leading-order thin-airfoil prior. XFOIL slopes include thickness and viscous effects, so the physics weight is kept small.

## Results

Evaluation command:

```bash

PYTHONPATH=. python src/evaluate.py

```

Test rows: `11,801`

| Model | Cl RMSE | Cl R² | Cd RMSE | Cd R² | Cm RMSE | Cm R² |
|---|---:|---:|---:|---:|---:|---:|
| PolyRidge baseline | 0.043486 | 0.9945 | 0.005557 | 0.7975 | 0.005130 | 0.9886 |
| MLP, `w_physics=0.001` | 0.046053 | 0.9938 | 0.006030 | 0.7616 | 0.005554 | 0.9866 |

The old pre-rebuild metrics should not be compared directly to these results because the previous split allowed discrete geometry leakage across train and test.

## Validation

The rebuild includes a merge-gate script:

```bash

./scripts/check_honest_rebuild.sh

```

It runs:

- HDF5 geometry-label invariant tests

- discrete airfoil split-disjointness tests

- alpha-derivative registry coverage tests

- finite-difference chain-rule derivative tests

- XFOIL NACA 0012 smoke verification

- model evaluation

Latest verified XFOIL smoke check:

```text

Cl slope per degree: 0.106711 accepted band [0.100000, 0.120000]

Cd at alpha≈0: 0.005400 accepted band [0.004000, 0.009000]

Cm at alpha≈0: 0.000000 accepted band abs(Cm) < 0.010000

verification passed

```

Latest validation status:

```text

HDF5/split/derivative registry invariant tests: 7 passed

Finite-difference chain-rule tests: 2 passed

Merge gate: passed

```

## Streamlit App

Run locally from the repository root:

```bash

streamlit run app/streamlit_app.py

```

The app is clamped to the training envelope. It does not allow `m > 0.06`, `t` outside `[0.08, 0.18]`, Reynolds numbers outside the trained list, or the physically invalid `m > 0, p = 0` combination.

## Usage

Install dependencies:

```bash

pip install -r requirements.txt

```

Generate data:

```bash

PYTHONPATH=. python src/generate_data.py --n_samples 2000 --output data/raw/polar_dataset.h5

```

Preprocess:

```bash

PYTHONPATH=. python src/preprocess.py \

  --h5_path data/raw/polar_dataset.h5 \

  --out_dir data/processed \

  --feature_set geom13

```

Train the MLP:

```bash

PYTHONPATH=. python src/train.py \

  --mode mlp \

  --processed_dir data/processed \

  --out_dir models \

  --w_physics 0.001 \

  --no_wandb

```

Evaluate:

```bash

PYTHONPATH=. python src/evaluate.py

```

Run the app:

```bash

streamlit run app/streamlit_app.py

```

## Data Provenance

The original local dataset was generated before rounded NACA geometry was stored in the HDF5 labels. `src/relabel_data.py` was used once to migrate that dataset from pre-rounding continuous labels to the effective simulated NACA geometry. Fresh datasets generated after the rebuild do not need this migration.

## Limitations

- XFOIL convergence censoring removes many near-stall and post-stall points.

- The physics target is a thin-airfoil leading-order prior, not the exact XFOIL slope.

- The geometry space is limited to NACA 4-series airfoils in the stated envelope.

- The split tests generalization to unseen geometry, not unseen Reynolds numbers.

- The Streamlit airfoil display uses an approximate NACA plotting routine.