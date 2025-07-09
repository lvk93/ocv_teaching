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

    def get_stoichiometries(self, np_ratio:float, v_min:float, v_max:float) -> (float, float, float, float):

        np_offset = np_ratio*self.an.formation_loss - self.cath.formation_loss

        sol_an_fun = lambda x: (-x+1-np_offset)/ np_ratio
        an0 = sol_an_fun(0)
        an1 = sol_an_fun(1)

        # Determine cell voltage
        volt_cell_cha = self.reconstruct_voltage(an0,0,an1,1,direction="charge")[0]
        volt_cell_dis = self.reconstruct_voltage(an0,0,an1,1,direction="discharge")[0]

        # Determine stoichiometries
        soc_vec = np.linspace(0, 1, 100)
        an0 = interpolate(volt_cell_dis,sol_an_fun(soc_vec), v_min)
        an1 = interpolate(volt_cell_cha,sol_an_fun(soc_vec), v_max)
        cath0 = interpolate(volt_cell_dis,soc_vec, v_min)
        cath1 = interpolate(volt_cell_cha,soc_vec, v_max)
        return an0, cath0, an1, cath1

    def simulate_aging_modes(self,LAMPE,LAMNE,LLI):
        return
    
    def reconstruct_voltage(self,an0:float, cath0:float, an1:float, cath1:float,direction="charge") -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Reconstructs the full cell OCV based on the anode and cathode materials.
        """
        soc_vec = np.linspace(0, 1, 100)
        soc_an = (an1-an0) * soc_vec + an0
        soc_cath = (cath1-cath0) * soc_vec + cath0
        
        volt_cath = interpolate(self.cath.ocv.soc, self.cath.ocv.get_voltage(direction), soc_cath)
        volt_an = interpolate(self.an.ocv.soc, self.an.ocv.get_voltage(direction), soc_an)

        volt_cell = volt_cath - volt_an

        return volt_cell, volt_cath, volt_an





