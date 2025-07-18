import numpy as np
from scipy.interpolate import interp1d


def interpolate(x, y, x_new):
    """1D linear interpolation"""
    return np.interp(x_new, x, y)


class CellOCVReconstruction:
    """
    Reconstructs full-cell OCV from two PyBaMM OCV functions and their N:P ratio.
    """
    def __init__(self, cath_ocv_func, an_ocv_func, np_ratio: float, v_min=2.5, v_max=4.2):
        self.cath = cath_ocv_func  # PyBaMM OCV function: soc → V
        self.an = an_ocv_func      # PyBaMM OCV function: soc → V
        self.np_ratio = np_ratio
        self.np_offset = np_ratio * 0.0 - 0.0  # Assume no formation loss by default
        self.v_min = v_min
        self.v_max = v_max

    def align_anode_cathode(self, sol_cath, np_ratio=None, np_offset=None):
        if np_ratio is None:
            np_ratio = self.np_ratio
        if np_offset is None:
            np_offset = self.np_offset
        return (-sol_cath + 1 - np_offset) / np_ratio

    def reconstruct_voltage(self, an0, cath0, an1, cath1, direction="charge",
                            soc_vec=np.linspace(0, 1, 100)):
        soc_an = (an1 - an0) * soc_vec + an0
        soc_cath = (cath1 - cath0) * soc_vec + cath0

        volt_cath = self.cath(soc_cath)
        volt_an = self.an(soc_an)
        volt_cell = volt_cath - volt_an

        return volt_cell, volt_cath, volt_an

    def get_stoichiometries(self, np_ratio=None, np_offset=None,
                            v_min=None, v_max=None):
        if v_min is None:
            v_min = self.v_min
        if v_max is None:
            v_max = self.v_max

        if np_ratio is None:
            np_ratio = self.np_ratio
        if np_offset is None:
            np_offset = self.np_offset

        an0 = self.align_anode_cathode(0, np_ratio, np_offset)
        an1 = self.align_anode_cathode(1, np_ratio, np_offset)

        soc_vec = np.linspace(0, 1, 100)
        an_grid = self.align_anode_cathode(soc_vec, np_ratio, np_offset)

        # Compute full-cell voltage curves
        volt_cell_cha = self.reconstruct_voltage(an0, 0, an1, 1, "charge")[0]
        volt_cell_dis = self.reconstruct_voltage(an0, 0, an1, 1, "discharge")[0]

        # Interpolate to find stoichiometries at voltage limits
        an0 = interpolate(volt_cell_dis, an_grid, v_min)
        an1 = interpolate(volt_cell_cha, an_grid, v_max)
        cath0 = interpolate(volt_cell_dis, soc_vec, v_min)
        cath1 = interpolate(volt_cell_cha, soc_vec, v_max)

        return an0, cath0, an1, cath1

    def simulate_aging_modes(self, lampe, lamne, lli):
        np_aged = self.np_ratio * (1 - lamne) / (1 - lampe)
        np_offset_aged = 1 - (1 - self.np_offset) * (1 - lli) / (1 - lampe)
        return self.get_stoichiometries(np_ratio=np_aged, np_offset=np_offset_aged)
