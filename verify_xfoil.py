import numpy as np

from xfoil import XFoil
from xfoil.model import Naca4


def assert_between(name: str, value: float, low: float, high: float) -> None:
    print(f"{name}: {value:.6f} accepted band [{low:.6f}, {high:.6f}]")
    assert low <= value <= high, f"{name}={value:.6f} outside [{low}, {high}]"


def main():
    xf = XFoil()
    xf.airfoil = Naca4(0, 0, 1, 2)
    xf.Re = 1_000_000
    xf.M = 0.0
    xf.max_iter = 200

    try:
        alpha, cl, cd, cm, cp = xf.aseq(-4, 4, 0.5)
    except RuntimeError as exc:
        raise RuntimeError(
            "XFOIL verification failed before producing a polar file. "
            "Check that the Homebrew xfoil binary is installed, runnable from "
            "this conda environment, and that the local xfoil wrapper can write "
            "temporary polar files."
        ) from exc

    alpha = np.asarray(alpha)
    cl = np.asarray(cl)
    cd = np.asarray(cd)
    cm = np.asarray(cm)

    if len(alpha) < 9:
        raise RuntimeError(f"Expected at least 9 converged alpha points, got {len(alpha)}")

    linear_mask = np.abs(alpha) < 4.0
    if linear_mask.sum() < 5:
        raise RuntimeError(
            f"Not enough linear-region points for slope fit: {linear_mask.sum()}"
        )

    slope_per_deg, _ = np.polyfit(alpha[linear_mask], cl[linear_mask], 1)

    zero_idx = int(np.argmin(np.abs(alpha)))
    cd0 = float(cd[zero_idx])
    cm0 = float(cm[zero_idx])

    # Accepted bands from standard NACA 0012 behavior; see Abbott & von Doenhoff,
    # Theory of Wing Sections. These are smoke-test bands, not precision validation.
    assert_between("Cl slope per degree", slope_per_deg, 0.10, 0.12)
    assert_between("Cd at alpha≈0", cd0, 0.004, 0.009)

    print(f"Cm at alpha≈0: {cm0:.6f} accepted band abs(Cm) < 0.010000")
    assert abs(cm0) < 0.01, f"Cm0={cm0:.6f} too large for symmetric airfoil"

    print("verification passed")


if __name__ == "__main__":
    main()

