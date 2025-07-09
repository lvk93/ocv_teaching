import numpy as np
import matplotlib.pyplot as plt
from .common import new_plot
from .active_material import ActiveMaterial
from .utils import interpolate


class CellOCVReconstruction:
    """
    Reconstructs full-cell OCV from two materials and their N:P ratio.
    """
    def __init__(self, cath_material: ActiveMaterial, an_material: ActiveMaterial):
        self.cath = cath_material
        self.an = an_material

    def reconstruct(self, np_ratio:float, v_min:float, v_max:float) -> (float, float, float, float):

        np_offset = np_ratio*self.an.formation_loss - self.cath.formation_loss

        sol_an_fun = lambda x: (-x+1-np_offset)/ np_ratio
        
        soc_vec = np.linspace(0, 1, 100)
        volt_cath_cha = interpolate(self.cath.ocv.soc, self.cath.ocv.get_voltage('charge'), soc_vec)
        volt_cath_dis = interpolate(self.cath.ocv.soc, self.cath.ocv.get_voltage('discharge'), soc_vec)
        volt_an_cha = interpolate(self.an.ocv.soc, self.an.ocv.get_voltage('charge'), sol_an_fun(soc_vec))
        volt_an_dis = interpolate(self.an.ocv.soc, self.an.ocv.get_voltage('discharge'), sol_an_fun(soc_vec))

        # Determine cell voltage
        volt_cell_cha = volt_cath_cha - volt_an_cha
        volt_cell_dis = volt_cath_dis - volt_an_dis

        # Determine stoichiometries
        an0 = interpolate(volt_cell_dis,sol_an_fun(soc_vec), v_min)
        an1 = interpolate(volt_cell_cha,sol_an_fun(soc_vec), v_max)
        cath0 = interpolate(volt_cell_dis,soc_vec, v_min)
        cath1 = interpolate(volt_cell_cha,soc_vec, v_max)
        return an0, cath0, an1, cath1

    def simulate_aging_modes(self,LAMPE,LAMNE,LLI):
        return

    def plot(self):
        soc, v_cell = self.reconstruct()
        fig, ax = new_plot()
        ax.plot(soc, v_cell)
        ax.set_xlabel('Full-Cell SOC')
        ax.set_ylabel('Cell Voltage (V)')
        ax.set_title(f'Full-Cell OCV Reconstruction')
        plt.show()


