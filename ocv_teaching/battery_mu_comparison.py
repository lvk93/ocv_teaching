from ocv_teaching.battery_mu_plot import BatteryMuPlot
import matplotlib.pyplot as plt
import numpy as np
F = 96485  # Faraday constant [C/mol]




class BatteryMuComparison:
    def __init__(self, bol_plot: BatteryMuPlot, aged_plot: BatteryMuPlot):
        self.bol = bol_plot
        self.aged = aged_plot

    def plot_comparison(self):
        fig, (ax_mu, ax_ocv) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, height_ratios=[3, 1])

        # === μ_Li subplot ===
        # Cathode and anode curves
        ax_mu.plot(self.bol.sol, self.bol.mu_cath, '--', color='red', label='Cathode μₗᵢ (BOL)')
        ax_mu.plot(self.bol.sol, self.bol.mu_an, '--', color='blue', label='Anode μₗᵢ (BOL)')
        ax_mu.plot(self.aged.sol, self.aged.mu_cath, '-', color='red', label='Cathode μₗᵢ (aged)')
        ax_mu.plot(self.aged.sol, self.aged.mu_an, '-', color='blue', label='Anode μₗᵢ (aged)')

        # BOL fill: dashed, faded
        ax_mu.fill_between(self.bol.sol[self.bol.sol <= self.bol.sol_cath],
                           self.bol.mu_cath[self.bol.sol <= self.bol.sol_cath],
                           self.bol.mu_cath_soc,
                           color='red', alpha=0.15, hatch='//', edgecolor='red', linewidth=0.0)
        ax_mu.fill_between(self.bol.sol[self.bol.sol <= self.bol.sol_an],
                           self.bol.mu_an[self.bol.sol <= self.bol.sol_an],
                           self.bol.mu_an_soc,
                           color='blue', alpha=0.15, hatch='//', edgecolor='blue', linewidth=0.0)

        # Aged fill: solid
        ax_mu.fill_between(self.aged.sol[self.aged.sol <= self.aged.sol_cath],
                           self.aged.mu_cath[self.aged.sol <= self.aged.sol_cath],
                           self.aged.mu_cath_soc,
                           color='red', alpha=0.3)
        ax_mu.fill_between(self.aged.sol[self.aged.sol <= self.aged.sol_an],
                           self.aged.mu_an[self.aged.sol <= self.aged.sol_an],
                           self.aged.mu_an_soc,
                           color='blue', alpha=0.3)

        # Markers
        ax_mu.plot(self.bol.sol_cath, self.bol.mu_cath_soc, 'o', color='red', markersize=6, alpha=0.5)
        ax_mu.plot(self.bol.sol_an, self.bol.mu_an_soc, 'o', color='blue', markersize=6, alpha=0.5)
        ax_mu.plot(self.aged.sol_cath, self.aged.mu_cath_soc, 'o', color='red', markersize=8)
        ax_mu.plot(self.aged.sol_an, self.aged.mu_an_soc, 'o', color='blue', markersize=8)

        # Δμ arrow (aged)
        arrow_x = self.aged.sol_cath
        ax_mu.annotate('', xy=(arrow_x, self.aged.mu_an_soc), xytext=(arrow_x, self.aged.mu_cath_soc),
                       arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        label_y = (self.aged.mu_an_soc + self.aged.mu_cath_soc) / 2
        ax_mu.text(arrow_x + 0.02, label_y, f'{self.aged.delta_U:.2f} V',
                   va='center', ha='left', fontsize=12, fontweight='bold')

        # Y-axis voltage
        volt_ticks = np.linspace(0, 5, 11)
        mu_ticks = -F * volt_ticks
        ax_mu.set_yticks(mu_ticks)
        ax_mu.set_yticklabels([f"{v:.1f}" for v in volt_ticks])
        ax_mu.set_ylim(mu_ticks[-1], mu_ticks[0])
        ax_mu.set_ylabel("Voltage vs Li⁺/Li [V]")
        ax_mu.set_title(f"Comparison at SoC = {self.aged.soc:.2f}")
        ax_mu.legend()
        ax_mu.grid(True)

        # === Full-cell OCV subplot ===
        ax_ocv.plot(self.bol.sol_cath_window, self.bol.v_cell_window, '--', color='gray', label='OCV (BOL)')
        ax_ocv.plot(self.aged.sol_cath_window, self.aged.v_cell_window, '-', color='black', label='OCV (aged)')

        # Points and arrow
        ax_ocv.plot(self.aged.sol_cath, self.aged.delta_U, 'ko', markersize=8)
        ax_ocv.axvline(self.aged.sol_cath, color='gray', linestyle='--', alpha=0.6)

        # Cutoff regions
        ax_ocv.axhspan(0, self.aged.v_min, color='gray', alpha=0.15, label='Below Vmin')
        ax_ocv.axhspan(self.aged.v_max, 5.0, color='gray', alpha=0.15, label='Above Vmax')

        ax_ocv.set_ylabel("Cell Voltage [V]")
        ax_ocv.set_xlabel("Cathode SoL")
        ax_ocv.grid(True)
        ax_ocv.legend()
        plt.tight_layout()
        plt.show()
