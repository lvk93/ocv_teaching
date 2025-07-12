import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

F = 96485  # Faraday constant [C/mol]

class BatteryMuPlot:
    def __init__(self, ocv_anode, ocv_cathode, np_ratio=1.0, v_min=2.5, v_max=4.2, resolution=1000):
        self.ocv_anode = ocv_anode
        self.ocv_cathode = ocv_cathode
        self.np_ratio = np_ratio
        self.v_min = v_min
        self.v_max = v_max
        self.resolution = resolution

        self.aging_mode = 'none'
        self.lli = 0.0
        self.lam_cath = 0.0
        self.lam_anode = 0.0

        # Setup state of lithiation grid (cathode driven)
        self.sol = np.linspace(0, 1, resolution)
        self._precompute_voltage_curve()
        self.set_soc(0.5)  # default

    def _precompute_voltage_curve(self):
        sol_cath_grid = self.sol #* (1 - self.lam_cath)
        sol_an_grid = (1 - self.lli - sol_cath_grid * (1 - self.lam_cath)) / (1 - self.lam_anode)


        valid = (sol_an_grid >= 0) & (sol_an_grid <= 1)

        self.sol_cath_valid = sol_cath_grid[valid]
        self.sol_an_valid = sol_an_grid[valid]

        ocv_cath = self.ocv_cathode(self.sol_cath_valid)
        ocv_an = self.ocv_anode(self.sol_an_valid)
        self.v_cell = ocv_cath - ocv_an

        # Ensure Vmin < Vmax
        mask = (self.v_cell >= self.v_min) & (self.v_cell <= self.v_max)
        self.v_cell_window = self.v_cell[mask]
        self.sol_cath_window = self.sol_cath_valid[mask]

        # Interpolation: SoC -> SoL_cath, SoL_cath -> SoC
        self.soc_to_sol_cath = interp1d(
            np.linspace(1,0, len(self.sol_cath_window)),
            self.sol_cath_window,
            kind='linear',
            bounds_error=False,
            fill_value=(self.sol_cath_window[0], self.sol_cath_window[-1])
        )
        self.sol_cath_to_soc = interp1d(
            self.sol_cath_window,
            np.linspace(1,0, len(self.sol_cath_window)),
            kind='linear',
            bounds_error=False,
            fill_value=(0.0, 1.0)
        )

    def set_soc(self, soc):
        self.soc = soc
        # Apply aging-adjusted values
        self.sol_cath = float(self.soc_to_sol_cath(self.soc)) 
        self.sol_an =  (1 - self.lli - self.sol_cath * (1 - self.lam_cath)) / (1 - self.lam_anode)



        self.ocv_an = self.ocv_anode(self.sol)
        self.ocv_cath = self.ocv_cathode(self.sol)
        self.mu_an = -F * self.ocv_an
        self.mu_cath = -F * self.ocv_cath

        self.ocv_cath_soc = self.ocv_cathode(self.sol_cath)
        self.ocv_an_soc = self.ocv_anode(self.sol_an)
        self.mu_cath_soc = -F * self.ocv_cath_soc
        self.mu_an_soc = -F * self.ocv_an_soc
        self.delta_U = self.ocv_cath_soc - self.ocv_an_soc

    def set_aging_mode(self, mode='none', lli=0.0, lam_cath=0.0, lam_anode=0.0):
        self.aging_mode = mode
        self.lli = lli
        self.lam_cath = lam_cath
        self.lam_anode = lam_anode
        self._precompute_voltage_curve()
        self.set_soc(self.soc)

    def plot_lines(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.sol, self.mu_cath, color='red', label='Cathode μₗᵢ')
        plt.plot(self.sol, self.mu_an, color='blue', label='Anode μₗᵢ')

        volt_ticks = np.linspace(0, 5, 11)
        mu_ticks = -F * volt_ticks
        plt.yticks(mu_ticks, [f"{v:.1f}" for v in volt_ticks])
        plt.ylim(mu_ticks[-1], mu_ticks[0])
        plt.ylabel("Voltage vs Li⁺/Li [V]")
        plt.xlabel("State of Lithiation (SoL)")
        plt.grid(True)
        plt.legend()

    def annotate_soc(self):
        fig, (ax_mu, ax_ocv) = plt.subplots(2, 1, figsize=(12,6), sharex=True, height_ratios=[3, 1])

        # === μ_Li curves ===
        ax_mu.plot(self.sol, self.mu_cath, color='red', label='Cathode μₗᵢ')
        ax_mu.plot(self.sol, self.mu_an, color='blue', label='Anode μₗᵢ')

        # Fill regions
        ax_mu.fill_between(self.sol[self.sol <= self.sol_cath],
                        self.mu_cath[self.sol <= self.sol_cath],
                        self.mu_cath_soc,
                        color='red', alpha=0.3)
        ax_mu.fill_between(self.sol[self.sol <= self.sol_an],
                        self.mu_an[self.sol <= self.sol_an],
                        self.mu_an_soc,
                        color='blue', alpha=0.3)

        # Markers
        ax_mu.plot(self.sol_cath, self.mu_cath_soc, 'o', color='red', markersize=10)
        ax_mu.plot(self.sol_an, self.mu_an_soc, 'o', color='blue', markersize=10)

        # Δμ arrow
        arrow_x = self.sol_cath
        ax_mu.annotate('', xy=(arrow_x, self.mu_an_soc), xytext=(arrow_x, self.mu_cath_soc),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        label_y = (self.mu_an_soc + self.mu_cath_soc) / 2
        ax_mu.text(arrow_x + 0.02, label_y, f'{self.delta_U:.2f} V',
                va='center', ha='left', fontsize=12, fontweight='bold')


        # μ_Li y-axis
        volt_ticks = np.linspace(0, 5, 11)
        mu_ticks = -F * volt_ticks
        ax_mu.set_yticks(mu_ticks)
        ax_mu.set_yticklabels([f"{v:.1f}" for v in volt_ticks])
        ax_mu.set_ylim(mu_ticks[-1], mu_ticks[0])
        ax_mu.set_ylabel("Voltage vs Li⁺/Li [V]")
        ax_mu.legend()
        ax_mu.grid(True)
        ax_mu.set_title(f"Chemical Potential at SoC = {self.soc:.2f}  |  V = {self.delta_U:.2f} V")

        # === Full-cell OCV ===
        ax_ocv.plot(self.sol_cath_window, self.v_cell_window, color='black', label='Full-Cell OCV')
        ax_ocv.plot(self.sol_cath, self.delta_U, 'ko', markersize=8)
        ax_ocv.axvline(self.sol_cath, color='gray', linestyle='--', alpha=0.6)

        # Shade cutoffs
        ax_ocv.axhspan(0, self.v_min, color='gray', alpha=0.15, label='Below Vmin')
        ax_ocv.axhspan(self.v_max, 5.0, color='gray', alpha=0.15, label='Above Vmax')

        # Labels
        ax_ocv.set_ylabel("Cell Voltage [V]")
        ax_ocv.set_xlabel("Cathode SoL")
        ax_ocv.set_ylim(2,4.5)
        ax_ocv.grid(True)
        ax_ocv.legend()
        plt.tight_layout()
        plt.show()


    def query_voltage(self, sol_query):
        U_cath = self.ocv_cathode(sol_query)
        U_an = self.ocv_anode((1 - sol_query) / self.np_ratio)
        voltage = U_cath - U_an
        print(f"At SoL_cath = {sol_query:.2f}:")
        print(f"  Cathode OCV  = {U_cath:.3f} V")
        print(f"  Anode OCV    = {U_an:.3f} V")
        print(f"  Cell voltage = {voltage:.3f} V")