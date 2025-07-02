import numpy as np
import matplotlib.pyplot as plt
from .common import new_plot
from .active_material import ActiveMaterial

class CellOCVReconstruction:
    """
    Reconstructs full-cell OCV from two materials and their N:P ratio.
    """
    def __init__(self, pos_material: ActiveMaterial, neg_material: ActiveMaterial, np_ratio: float):
        self.pos = pos_material
        self.neg = neg_material
        self.np_ratio = np_ratio

    def reconstruct(self, mode: str = 'charge') -> (np.ndarray, np.ndarray):
        cap_pos = self.pos.effective_capacity()
        cap_neg = self.neg.effective_capacity() * self.np_ratio
        cap_cell = min(cap_pos, cap_neg)

        soc = np.linspace(0, 1, len(self.pos.ocv.soc))
        v_pos = np.interp(soc * cap_cell / cap_pos, self.pos.ocv.soc, self.pos.get_voltage(mode))
        v_neg = np.interp(soc * cap_cell / cap_neg, self.neg.ocv.soc, self.neg.ocv.get_voltage(mode))
        v_cell = v_pos - v_neg
        return soc, v_cell

    def plot(self, mode: str = 'charge'):
        soc, v_cell = self.reconstruct(mode)
        fig, ax = new_plot()
        ax.plot(soc, v_cell)
        ax.set_xlabel('Full-Cell SOC')
        ax.set_ylabel('Cell Voltage (V)')
        ax.set_title(f'Full-Cell OCV Reconstruction ({mode.capitalize()})')
        plt.show()