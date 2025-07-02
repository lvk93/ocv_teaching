import numpy as np
import matplotlib.pyplot as plt
from .common import new_plot

class OCV:
    """
    Open-circuit voltage (OCV) curve class for a single electrode.
    Stores SOC and corresponding charge/discharge voltage curves.
    """
    def __init__(self, soc: np.ndarray, voltage_charge: np.ndarray, voltage_discharge: np.ndarray):
        assert soc.ndim == 1, "SOC must be a 1D array"
        assert voltage_charge.shape == soc.shape, "Charge voltage must match SOC shape"
        assert voltage_discharge.shape == soc.shape, "Discharge voltage must match SOC shape"
        self.soc = soc
        self.voltage_charge = voltage_charge
        self.voltage_discharge = voltage_discharge

    def get_voltage(self, mode: str = 'charge') -> np.ndarray:
        if mode.lower() == 'charge':
            return self.voltage_charge
        elif mode.lower() == 'discharge':
            return self.voltage_discharge
        else:
            raise ValueError("Mode must be 'charge' or 'discharge'")

    def plot(self, mode: str = 'charge'):
        voltage = self.get_voltage(mode)
        fig, ax = new_plot()
        ax.plot(self.soc, voltage)
        ax.set_xlabel('State of Charge (SOC)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'OCV Curve ({mode.capitalize()})')
        plt.show()