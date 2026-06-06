from xfoil import XFoil
from xfoil.model import Naca4

xf = XFoil()
xf.airfoil = Naca4(0, 0, 1, 2)   # NACA 0012
xf.Re = 1e6
xf.max_iter = 40

a, cl, cd, cm, cp = xf.aseq(-5, 15, 0.5)

print("XFOIL verification passed.")
print("Number of converged points:", len(a))
print()

print("alpha, CL, CD, CM")
for i in range(len(a)):
    print(f"{a[i]:6.2f} {cl[i]:9.4f} {cd[i]:10.5f} {cm[i]:9.4f}")

print()
print("Max CL:", max(cl))

