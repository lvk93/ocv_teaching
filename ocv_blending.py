import numpy as np
import matplotlib.pyplot as plt
from .common import new_plot
from .active_material import ActiveMaterial

class OCVBlending:
    """
    Blends two materials' OCV curves by capacity ratio.
    """
    def __init__(self, mat1: ActiveMaterial, mat2: ActiveMaterial, cap_ratio: float):
        self.mat1 = mat1
        self.mat2 = mat2
        self.cap_ratio = cap_ratio

    def blend(self, mode: str = 'charge') -> (np.ndarray, np.ndarray):
        soc = np.linspace(0, 1, len(self.mat1.ocv.soc))
        v1 = np.interp(soc, self.mat1.ocv.soc, self.mat1.ocv.get_voltage(mode))
        v2 = np.interp(soc, self.mat2.ocv.soc, self.mat2.ocv.get_voltage(mode))
        v_blend = self.cap_ratio * v1 + (1 - self.cap_ratio) * v2
        return soc, v_blend

    def plot(self, mode: str = 'charge'):
        soc, v_blend = self.blend(mode)
        fig, ax = new_plot()
        ax.plot(soc, v_blend)
        ax.set_xlabel('State of Charge (SOC)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Blended OCV Curve ({mode.capitalize()})')
        plt.show()