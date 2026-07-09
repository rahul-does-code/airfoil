from __future__ import annotations

import numpy as np


def naca_digits(m: float, p: float, t: float) -> tuple[int, int, int]:
    """
    Convert continuous sampled NACA parameters to the discrete NACA 4-series
    digits actually sent to XFOIL.
    """
    d1 = int(np.clip(round(m * 100), 0, 9))
    d2 = int(np.clip(round(p * 10), 0, 9))

    if d1 == 0:
        d2 = 0

    t_digits = int(np.clip(round(t * 100), 6, 21))
    return d1, d2, t_digits


def digits_to_params(d1: int, d2: int, t_digits: int) -> tuple[float, float, float]:
    """
    Convert NACA digits back to effective physical parameters.
    """
    return d1 / 100.0, d2 / 10.0, t_digits / 100.0


def naca_code(d1: int, d2: int, t_digits: int) -> str:
    """
    Format NACA digits as a four-digit code string.
    """
    return f"{d1}{d2}{t_digits:02d}"
