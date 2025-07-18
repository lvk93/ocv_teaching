import numpy as np
import matplotlib.pyplot as plt
import pybamm


class BatteryTankPlotter:
    def __init__(
        self,
        ocv_anode,
        ocv_cathode,
        soc_an_fun=lambda soc: soc,
        soc_ca_fun=lambda soc: 1 - soc,
        np_ratio=1.0,
        v_min=None,
        v_max=None,
        lamne=0.0,
        lampe=0.0,
        resolution=500,
    ):
        self.ocv_anode = ocv_anode
        self.ocv_cathode = ocv_cathode
        self.soc_an_fun = soc_an_fun
        self.soc_ca_fun = soc_ca_fun
        self.np_ratio = np_ratio
        self.v_min = v_min
        self.v_max = v_max
        self.lamne = lamne
        self.lampe = lampe
        self.resolution = resolution

        self.colors = {
            "gray": "#333333",
            "bright_gray": "#E7E6E6",
            "gray_blue": "#44546A",
            "navy_blue": "#003B73",
            "light_blue": "#5DADE2",
            "orange": "#F39C12",
            "green": "#70AD47",
        }

    def _compute_ocv_and_dqdv(self, ocv_func):
        soc_grid = np.linspace(0, 1, self.resolution)
        voltage = ocv_func(soc_grid)
        dVdQ = np.gradient(voltage, soc_grid)
        dQdV = 1.0 / dVdQ
        return soc_grid, voltage, dQdV

    def _plot_voltage_limit_lines(self, ax, soc_grid):
        lines = []

        if self.v_min is not None:
            # anode at SOC=1 (fully lithiated), cathode at SOC=1 (fully lithiated)
            V_an1 = self.ocv_anode(self.soc_an_fun(1))
            V_ca0 = self.ocv_cathode(self.soc_ca_fun(1))
            lines.append(("an1", -V_an1))
            lines.append(("cath0", -V_ca0))

        if self.v_max is not None:
            # anode at SOC=0 (delithiated), cathode at SOC=0 (delithiated)
            V_an0 = self.ocv_anode(self.soc_an_fun(0))
            V_ca1 = self.ocv_cathode(self.soc_ca_fun(0))
            lines.append(("an0", -V_an0))
            lines.append(("cath1", -V_ca1))

        for label, y in lines:
            ax.axhline(y=y, color=self.colors["green"], linestyle=":", linewidth=1)
            ax.text(
                x=4.6, y=y, s=label, va="center", ha="right", fontsize=9, color=self.colors["green"]
            )

    def _plot_dqdv_tanks(self, ax, soc, soc_grid, V_an, dQdV_an, V_ca, dQdV_ca):
        soc_an = np.clip(soc_grid, 0.0, soc)
        soc_ca = np.clip(soc_grid, 0.0, 1 - soc)

        V_an_cur = self.ocv_anode(soc_an)
        V_ca_cur = self.ocv_cathode(soc_ca)

        dQdV_an_cur = np.interp(soc_an, soc_grid, dQdV_an) * (1 - self.lamne) * self.np_ratio
        dQdV_ca_cur = np.interp(soc_ca, soc_grid, dQdV_ca) * (1 - self.lampe)

        # Plot tanks
        for sign in [1, -1]:
            ax.plot(sign * dQdV_an * (1 - self.lamne) * self.np_ratio, -V_an,
                    color=self.colors["gray"], label="Graphite tank" if sign == 1 else None)
            ax.plot(sign * dQdV_ca * (1 - self.lampe), -V_ca,
                    color=self.colors["light_blue"], label="NCM tank" if sign == 1 else None)

            ax.plot(sign * dQdV_an, -V_an * self.np_ratio,
                    color=self.colors["gray"], linestyle='--')
            ax.plot(sign * dQdV_ca, -V_ca,
                    color=self.colors["light_blue"], linestyle='--')

        # Fill tanks
        for dQdV, V in [(dQdV_an_cur, V_an_cur), (dQdV_ca_cur, V_ca_cur)]:
            for sign in [1, -1]:
                ax.fill_betweenx(-V, sign * dQdV, 0,
                                 color=self.colors["navy_blue"], alpha=0.8)

        # Voltage difference arrow
        delta_V = V_ca_cur.min() - V_an_cur.min()
        mid_y = -0.5 * (V_ca_cur.min() + V_an_cur.min())
        ax.annotate('', xy=(0, -V_ca_cur.min()), xytext=(0, -V_an_cur.min()),
                    arrowprops=dict(arrowstyle='<->', color=self.colors["orange"], linewidth=2))
        ax.text(0.1, mid_y, f'ΔV = {delta_V:.3f} V',
                color=self.colors["orange"], va='center')

        # Add voltage limit lines if set
        self._plot_voltage_limit_lines(ax, soc_grid)

        # Axes formatting
        ax.set_xlabel("dQ/dV [Ah/V]")
        ax.set_ylabel("Voltage [V]")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 0)
        yticks = np.arange(-5, 1, 1)
        ax.set_yticks(yticks)
        ax.set_yticklabels([abs(int(t)) for t in yticks])
        ax.legend(loc="center left")
        ax.grid(True)

    def _plot_ocv_curve(self, ax, soc, soc_grid):
        soc_an = self.soc_an_fun(soc_grid)
        soc_ca = self.soc_ca_fun(soc_grid)

        V_an = self.ocv_anode(soc_an)
        V_ca = self.ocv_cathode(soc_ca)
        V_cell = V_ca - V_an

        delta_V = self.ocv_cathode(self.soc_ca_fun(soc)) - self.ocv_anode(self.soc_an_fun(soc))

        if self.v_min is not None and self.v_max is not None and self.v_max > self.v_min:
            # Interpolate SoC values corresponding to Vmin and Vmax
            soc_vs_voltage = lambda V_target: np.interp(V_target, V_cell, soc_grid)

            soc_vmin = soc_vs_voltage(self.v_min)
            soc_vmax = soc_vs_voltage(self.v_max)

            if soc_vmax == soc_vmin:  # Avoid divide by zero
                soc_norm = soc_grid * 0
                soc_point = 0
            else:
                soc_norm = (soc_grid - soc_vmin) / (soc_vmax - soc_vmin)
                soc_point = (soc - soc_vmin) / (soc_vmax - soc_vmin)

            ax.set_xlim(0, 1)
            ax.set_xlabel("Normalized SoC (0 at Vmin, 1 at Vmax)")
        else:
            soc_norm = soc_grid
            soc_point = soc
            ax.set_xlim(0, 1)
            ax.set_xlabel("State of Charge (SOC)")

        ax.plot(soc_norm, V_cell, color=self.colors["navy_blue"], label="Cell OCV")
        ax.plot(soc_point, delta_V, color=self.colors["navy_blue"], marker='o', markersize=5, label="Current SOC")
        ax.set_ylim(0, 5)
        ax.set_ylabel("Voltage [V]")
        ax.grid(True)
        ax.legend()


    def plot(self, soc=0.5):
        soc_grid, V_an, dQdV_an = self._compute_ocv_and_dqdv(self.ocv_anode)
        _, V_ca, dQdV_ca = self._compute_ocv_and_dqdv(self.ocv_cathode)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self._plot_dqdv_tanks(ax1, soc, soc_grid, V_an, dQdV_an, V_ca, dQdV_ca)
        self._plot_ocv_curve(ax2, soc, soc_grid)

        plt.tight_layout()
        plt.show()
