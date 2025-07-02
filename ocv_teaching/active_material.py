from .ocv import OCV
import matplotlib.pyplot as plt
from .common import new_plot

class ActiveMaterial:
    """
    Active material with its own OCV, specific capacity, and formation loss.
    """
    def __init__(self, ocv: OCV, specific_capacity: float, formation_loss: float = 0.0):
        self.ocv = ocv
        self.specific_capacity = specific_capacity
        self.formation_loss = formation_loss

    def effective_capacity(self) -> float:
        return self.specific_capacity * (1 - self.formation_loss)

    def plot(self, mode: str = 'charge'):
        fig, ax = new_plot()
        ax.plot(self.ocv.soc, self.ocv.get_voltage(mode))
        ax.set_xlabel('State of Charge (SOC)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Material OCV ({mode.capitalize()})')
        plt.show()
