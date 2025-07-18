from .cell_reconstruction import CellOCVReconstruction
import matplotlib.pyplot as plt
import numpy as np
from .utils import interpolate
class BatteryTankPlotter:
    """
    Uses CellOCVReconstruction to plot dQ/dV 'tanks' and OCV curves for BOL and aged cells.
    """
    def __init__(self, reconstructor: CellOCVReconstruction, resolution=500):
        self.rec = reconstructor
        self.resolution = resolution
        self.bc = {
            'an_bol': '#333333',
            'ca_bol': '#5DADE2',
            'an_aged': '#444444',
            'ca_aged': '#003B73',
            'fill_bol': '#44546A',
            'fill_aged': '#70AD47',
            'delta': '#F39C12'
        }

    def _compute_electrode_curves(self, an0, cath0, an1, cath1, direction):
        soc = np.linspace(1,0, self.resolution)
        V_cell, V_ca, V_an, sol_an, sol_ca = self.rec.reconstruct_voltage(
            an0, cath0, an1, cath1, direction, soc)
        dQdV_an = 1 / np.gradient(V_an, sol_an)
        dQdV_ca = 1 / np.gradient(V_ca, sol_ca)
        return soc, sol_an, V_an, dQdV_an, sol_ca, V_ca, dQdV_ca, V_cell

    def plot(self, soc=0.5, lampe=0.0, lamne=0.0, lli=0.0):
        ## Full curves
        soc_vec = np.linspace(0,1,self.resolution)
        V_an = self.rec.an(soc_vec)
        V_cath = self.rec.cath(soc_vec)
        dQdV_an = 1 / np.gradient(V_an, soc_vec)
        dQdV_ca = 1 / np.gradient(V_cath, soc_vec)
        
        # plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ## Plot the full curves BOL
        bc = self.bc
        ax1.plot(dQdV_an*self.rec.np_ratio, -V_an, color=bc['an_aged'],linestyle='--')
        ax1.plot(-dQdV_an*self.rec.np_ratio, -V_an, color=bc['an_aged'],linestyle='--')
        ax1.plot(dQdV_ca, -V_cath, color=bc['ca_bol'],linestyle='--')
        ax1.plot(-dQdV_ca, -V_cath, color=bc['ca_bol'],linestyle='--')

        ## Plot the full curves aged
        ax1.plot(dQdV_an*self.rec.np_ratio*(1-lamne), -V_an, color=bc['an_aged'], label='Anode')
        ax1.plot(-dQdV_an*self.rec.np_ratio*(1-lamne), -V_an, color=bc['an_aged'])
        ax1.plot(dQdV_ca*(1-lampe), -V_cath, color=bc['ca_bol'], label='Cathode')
        ax1.plot(-dQdV_ca*(1-lampe), -V_cath, color=bc['ca_bol'])

        # Compute aged filling
        # BOL stoichiometries
        an0_bol,cath0_bol,an1_bol,cath1_bol = self.rec.get_stoichiometries()
        # aged stoichiometries
        an0_a, ca0_a, an1_a, ca1_a = self.rec.simulate_aging_modes(lampe, lamne, lli)
        # get aged voltages
        v_an0 = self.rec.an(an0_a)
        v_an1 = self.rec.an(an1_a)
        v_ca1 = self.rec.cath(ca1_a)
        v_ca0 = self.rec.cath(ca0_a)
        # curves
        ax1.axhline(y=-v_an0, color=self.bc["fill_aged"], linestyle=":", linewidth=1)
        ax1.axhline(y=-v_an1, color=self.bc["fill_aged"], linestyle=":", linewidth=1)
        ax1.axhline(y=-v_ca0, color=self.bc["fill_aged"], linestyle=":", linewidth=1)
        ax1.axhline(y=-v_ca1, color=self.bc["fill_aged"], linestyle=":", linewidth=1)

        # Reconstruct cell voltage
        V_cell, V_ca, V_an, sol_an, sol_ca = self.rec.reconstruct_voltage(
            an0_a, ca0_a, an1_a, ca1_a, "charge",soc_vec)
        
        ax2.plot(soc_vec,V_cell)

        # (soc_full, sa_b, Va_b, dQa_b, sc_b, Vc_b, dQc_b, Vcell_b) = self._compute_electrode_curves(
        #     0, 1, 1, 0, 'charge')

        

        # aged tanks fill
        # mask_a = soc_full <= soc
        # Determine voltage for soc
        sol_an_cur = an0_a+(an1_a-an0_a)*(soc)
        sol_ca_cur = ca0_a+(ca1_a-ca0_a)*(soc)
        sol_an_vec = np.linspace(0,sol_an_cur,100)
        sol_ca_vec = np.linspace(0,sol_ca_cur,100)
        V_an_cur = self.rec.an(np.linspace(0,sol_an_cur,100))
        V_ca_cur = self.rec.cath(np.linspace(0,sol_ca_cur,100))
        dQ_an_cur = interpolate(soc_vec,dQdV_an,sol_an_vec)
        dQ_cath_cur = interpolate(soc_vec,dQdV_ca,sol_ca_vec)
        ax1.fill_betweenx(-V_an_cur, -dQ_an_cur*self.rec.np_ratio*(1-lamne), dQ_an_cur*self.rec.np_ratio*(1-lamne), color=bc['ca_aged'], alpha=0.5)
        ax1.fill_betweenx(-V_ca_cur, -dQ_cath_cur*(1-lampe), dQ_cath_cur*(1-lampe), color=bc['ca_aged'], alpha=0.5)
        
        # delta V arrow
        dV = -V_an_cur[-1].min() + V_ca_cur[-1].max()
        ax1.annotate('', xy=(0, -V_an_cur[-1].min()), xytext=(0,-V_ca_cur[-1].max()),
                     arrowprops=dict(arrowstyle='<->', color=bc['delta'], lw=2))
        ymid = -0.5*(V_an_cur[-1].min() + V_ca_cur[-1].max())
        ax1.text(0.1, ymid, f'ΔV={dV:.3f} V', color=bc['delta'])

        # Labels etc.
        ax1.set_xlabel('dQ/dV [Ah/V]')
        ax1.set_ylabel('Voltage [V]')
        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-5, 0)
        yticks = np.arange(-5, 1, 1)
        ax1.set_yticks(yticks)
        ax1.legend(loc='center left')
        ax1.grid(True)
        #Get BOL and aged OCV curves         
        (soc_full, sa_b, Va_b, dQa_b, sc_b, Vc_b, dQc_b, Vcell_b) = self._compute_electrode_curves(
           an0_bol, cath0_bol, an1_bol, cath1_bol, 'charge')
        (soc_full, sa_a, Va_a, dQa_a, sc_a, Vc_a, dQc_a, Vcell_a) = self._compute_electrode_curves(
            an0_a, ca0_a, an1_a, ca1_a, 'charge')
        # Plot OCV curve
        ax2.plot(soc_full, Vcell_b, color=bc['ca_aged'], label='Cell BOL', linestyle="--")
        ax2.plot(soc_full, Vcell_a, color=bc['ca_aged'], )
        # Plot Current SOC point
        ax2.plot(soc,dV,color=bc["ca_aged"],marker="o",markersize=5)
        ax2.set_xlabel('SOC')
        ax2.set_ylabel('Voltage [V]')
        ax2.legend()
        ax2.grid(True)
        ax2.set_xlim(0,1)
        plt.tight_layout()
        plt.show()