import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from .utils import interpolate


class CellOCVReconstruction:
    """
    Reconstructs full-cell OCV from two PyBaMM OCV functions and their N:P ratio.
    Provides methods to compute stoichiometries and voltage curves.
    """
    def __init__(self, cath_ocv_func, an_ocv_func, np_ratio: float,
                 v_min=2.5, v_max=4.2, formation_loss_an=0.0, formation_loss_ca=0.0):
        self.cath = cath_ocv_func    # PyBaMM OCV: soc -> V
        self.an = an_ocv_func        # PyBaMM OCV: soc -> V
        self.np_ratio = np_ratio
        # formation losses (fraction) can be provided
        self.np_offset = np_ratio * formation_loss_an - formation_loss_ca
        self.v_min = v_min
        self.v_max = v_max

    def align_anode_cathode(self, sol_cath, np_ratio=None, np_offset=None):
        """
        Compute anode stoichiometry from cathode stoichiometry based on N:P ratio and offset.
        """
        if np_ratio is None:
            np_ratio = self.np_ratio
        if np_offset is None:
            np_offset = self.np_offset
        return (-sol_cath + 1 - np_offset) / np_ratio

    def reconstruct_voltage(self, an0, cath0, an1, cath1,
                            direction='charge',
                            soc_vec=None):
        """
        Reconstruct full-cell and half-cell voltages along a SOC vector.
        Returns:
          V_cell, V_cath, V_an, soc_an, soc_cath
        """
        if soc_vec is None:
            soc_vec = np.linspace(0, 1, 100)
        soc_an = an0 + (an1 - an0) * soc_vec
        soc_cath = cath0 + (cath1 - cath0) * soc_vec
        V_cath = self.cath(soc_cath)
        V_an = self.an(soc_an)
        V_cell = V_cath - V_an
        return V_cell, V_cath, V_an, soc_an, soc_cath

    def get_stoichiometries(self, np_ratio=None, np_offset=None,
                            v_min=None, v_max=None):
        """
        Compute the anode/cathode stoichiometries at cell cutoff voltages.
        Returns: an0, cath0, an1, cath1
        """
        if v_min is None:
            v_min = self.v_min
        if v_max is None:
            v_max = self.v_max
        if np_ratio is None:
            np_ratio = self.np_ratio
        if np_offset is None:
            np_offset = self.np_offset
        an0 = self.align_anode_cathode(1, np_ratio, np_offset)
        an1 = self.align_anode_cathode(0, np_ratio, np_offset)
        cath0 = 1.0
        cath1 = 0.0
        grid = np.linspace(0,1, 200)
        vc_dis, _, _, _, _ = self.reconstruct_voltage(an0, cath0, an1, cath1, 'charge', grid)
        vc_cha, _, _, _, _ = self.reconstruct_voltage(an0, cath0, an1, cath1, 'charge', grid)
        soc0_cut = interpolate(vc_dis, grid, v_min)
        soc1_cut = interpolate(vc_cha, grid, v_max)
        an0_cut = soc0_cut*(an1-an0)+an0
        an1_cut = soc1_cut*(an1-an0)+an0
        cath0_cut = soc0_cut*(cath1-cath0)+cath0
        cath1_cut = soc1_cut*(cath1-cath0)+cath0
        return an0_cut, cath0_cut, an1_cut, cath1_cut

    def simulate_aging_modes(self, lampe, lamne, lli):
        """
        Compute aged stoichiometries given aging fractions.
        Returns: an0, cath0, an1, cath1 for aged state.
        """
        np_aged = self.np_ratio * (1 - lamne) / (1 - lampe)
        np_offset_aged = 1 - (1 - self.np_offset) * (1 - lli) / (1 - lampe)
        return self.get_stoichiometries(np_ratio=np_aged, np_offset=np_offset_aged)