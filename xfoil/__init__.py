import subprocess
from pathlib import Path

from .model import Naca4


class XFoil:
    """Small Python wrapper around the command-line XFOIL program."""

    def __init__(self):
        self.airfoil = Naca4(0, 0, 1, 2)
        self.Re = 1_000_000
        self.M = 0.0
        self.max_iter = 100

    def aseq(self, alpha_start, alpha_end, alpha_step):
        """Run an angle-of-attack sweep.

        Returns:
            a, cl, cd, cm, cp

        cp is returned as None because this wrapper currently reads polar data,
        not pressure coefficient distributions.
        """
        output_file = f"polar_{self.airfoil}_{alpha_start}_{alpha_end}_{alpha_step}.txt"
        output_path = Path(output_file)

        if output_path.exists():
            output_path.unlink()

        commands = f"""PLOP
G

{self.airfoil.xfoil_command()}
PANE
OPER
VISC {self.Re}
MACH {self.M}
ITER {self.max_iter}
PACC
{output_file}

ASEQ {alpha_start} {alpha_end} {alpha_step}
PACC

QUIT
"""
        try:
            result = subprocess.run(
                ["xfoil"],
                input=commands,
                text=True,
                capture_output=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("XFOIL timed out.")
	
	    # XFOIL often writes noisy Fortran warnings to stderr.
        # Do not print them during large dataset generation.
        
        if not output_path.exists():
            raise RuntimeError("XFOIL did not create a polar file.")

        rows = read_polar_file(output_path)

        if not rows:
            raise RuntimeError("XFOIL created a polar file, but no data rows were found.")

        a = [row["alpha"] for row in rows]
        cl = [row["CL"] for row in rows]
        cd = [row["CD"] for row in rows]
        cm = [row["CM"] for row in rows]
        cp = None

        return a, cl, cd, cm, cp

    def a(self, alpha):
        """Run XFOIL at one angle of attack."""
        a, cl, cd, cm, cp = self.aseq(alpha, alpha, 1)
        return cl[0], cd[0], cm[0], cp


def read_polar_file(filename):
    """Return all polar data rows as a list of dictionaries."""
    path = Path(filename)
    rows = []

    with path.open("r") as file:
        for line in file:
            parts = line.split()

            if len(parts) >= 7:
                try:
                    alpha, cl, cd, cdp, cm, top_xtr, bot_xtr = [
                        float(value) for value in parts[:7]
                    ]
                    rows.append({
                        "alpha": alpha,
                        "CL": cl,
                        "CD": cd,
                        "CDp": cdp,
                        "CM": cm,
                        "Top_Xtr": top_xtr,
                        "Bot_Xtr": bot_xtr,
                    })
                except ValueError:
                    continue

    return rows
