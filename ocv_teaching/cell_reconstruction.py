import numpy as np
import matplotlib.pyplot as plt
from .common import new_plot
from .active_material import ActiveMaterial

class CellOCVReconstruction:
    """
    Reconstructs full-cell OCV from two materials and their N:P ratio.
    """
    def __init__(self, cath_material: ActiveMaterial, an_material: ActiveMaterial):
        self.cath = cath_material
        self.an = an_material

    def reconstruct(self, np_ratio) -> (np.ndarray, np.ndarray):

        np_offset = np_ratio*self.an.formation_loss - self.cath.formation_loss

        sol_an_fun = lambda x: (-x+1-np_offset)/ np_ratio
        
        soc_vec = np.linspace(0, 1, 100)
        
        sol_an = sol_an_fun(soc_vec)
        sol_cath = np.linspace(0, 1, 100)
        sol_an = -1/np_ratio * sol_cath + (1-np_offset) 
        soc = np.linspace(0, 1, 100)

        v_cath = np.interp(sol_cath, self.cath.ocv.soc, self.cath.ocv.get_voltage())
        v_an = np.interp(sol_an, self.an.ocv.soc, self.an.ocv.get_voltage())
        v_cell = v_cath - v_an
        # Flip everything correctly
        v_cell = np.flip(v_cell)
        v_cath = np.flip(v_cath)
        v_an = np.flip(v_an)

        return soc,v_cath,v_an,v_cell

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