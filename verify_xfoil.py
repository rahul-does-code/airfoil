from xfoil import XFoil
from xfoil.model import Naca4
import numpy as np


xf = XFoil()
xf.airfoil = Naca4(0, 0, 1, 2)   # NACA 0012
xf.Re = 1e6
xf.max_iter = 40

alpha, cl, cd, cm, cp = xf.aseq(-5, 15, 0.5)

def assert_between(name: str, value: float, low: float, high: float) -> None:
    print(f"{name}: {value:.6f} accepted band [{low:.6f}, {high:.6f}]")
    assert low <= value <= high, f"{name}={value:.6f} outside [{low}, {high}]"

alpha = np.asarray(alpha)
cl = np.asarray(cl)
cd = np.asarray(cd)
cm = np.asarray(cm)

linear_mask = np.abs(alpha) < 5.0
slope_per_deg, _ = np.polyfit(alpha[linear_mask], cl[linear_mask], 1)

zero_idx = int(np.argmin(np.abs(alpha)))
cd0 = float(cd[zero_idx])
cm0 = float(cm[zero_idx])

# Accepted bands from standard NACA 0012 behavior; see Abbott & von Doenhoff,
# Theory of Wing Sections.
assert_between("Cl slope per degree", slope_per_deg, 0.10, 0.12)
assert_between("Cd at alpha≈0", cd0, 0.004, 0.009)

print(f"Cm at alpha≈0: {cm0:.6f} accepted band abs(Cm) < 0.010000")
assert abs(cm0) < 0.01, f"Cm0={cm0:.6f} too large for symmetric airfoil"

print("verification passed")

print("XFOIL verification passed.")
print("Number of converged points:", len(alpha))
print()

print("alpha, CL, CD, CM")
for i in range(len(alpha)):
    print(f"{alpha[i]:6.2f} {cl[i]:9.4f} {cd[i]:10.5f} {cm[i]:9.4f}")

print()
print("Max CL:", max(cl))

