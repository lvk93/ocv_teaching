from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def interpolate(xp, yp, x):
    """
    Interpolates the voltage values for given SOC values.
    """
    f = interp1d(
        xp, yp,
        kind='linear',
        fill_value="extrapolate",   # allow linear extrapolation
        assume_sorted=False         # will sort xp/yp internally
    )
    return f(x)


def new_plot(figsize=(8, 6), grid=True, grid_style="--", grid_alpha=0.5, dpi=100):
    """
    Create a new matplotlib figure and axis with default styling.

    Parameters:
    - figsize: tuple of (width, height) in inches.
    - grid: bool, whether to show a grid.
    - grid_style: string, matplotlib line style for grid lines.
    - grid_alpha: float, transparency for grid lines.
    - dpi: int, resolution of the figure.

    Returns:
    - fig: matplotlib.figure.Figure
    - ax: matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if grid:
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha)
    return fig, ax
