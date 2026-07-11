#!/usr/bin/env bash
set -euo pipefail

echo "== HDF5 + split + derivative registry invariants =="
PYTHONPATH=src pytest tests/test_rebuild_invariants.py

echo
echo "== Chain-rule finite-difference test =="
PYTHONPATH=src pytest tests/test_physics_derivative.py

echo
echo "== XFOIL verification =="
PYTHONPATH=. python verify_xfoil.py

echo
echo "== Evaluation =="
PYTHONPATH=. python src/evaluate.py

echo
echo "Honest rebuild checks passed."