class Naca4:
    """Represent a 4-digit NACA airfoil.

    Example:
        Naca4(0, 0, 1, 2) represents NACA 0012.
        Naca4(2, 4, 1, 2) represents NACA 2412.
    """

    def __init__(self, d1, d2, d3, d4):
        self.digits = f"{d1}{d2}{d3}{d4}"

    def xfoil_command(self):
        return f"NACA {self.digits}"

    def __str__(self):
        return f"naca{self.digits}"
    