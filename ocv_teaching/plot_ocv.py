import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler

def plot_ocv(curves, 
             x_label='Capacity', 
             y_label='Voltage (V)', 
             title=None, 
             x_lim=[0,1], 
             y_lim=[0,5], 
             legend_loc='best', 
             figsize=(8, 6), 
             grid=True,
             colors=None):
    """
    Plot OCV curves with a default, extended PowerPoint color cycle.

    Parameters:
    - curves: list of dicts, each with:
        - 'x', 'y', optional 'label', optional 'style'
    - colors: list of hex color strings to set the cycle
      (defaults to ['#003B73', '#333333', '#5DADE2',
                    '#85C1E9', '#7F8C8D', '#48C9B0'])
    """
    default_colors = [
        '#003B73',  # Deep Navy
        '#5DADE2',  # Sky Blue
        '#138D75',  # Teal
        '#27AE60',  # Green
        '#F39C12',  # Mustard
        '#E74C3C',  # Brick Red
        '#8E44AD',  # Purple
        '#7F8C8D',  # Slate Gray
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_prop_cycle(cycler('color', colors or default_colors))
    ax.set_frame_on(True)

    rcParams['axes.linewidth'] = 2
    for curve in curves:
        x = curve['x']
        y = curve['y']
        label = curve.get('label', None)
        style = curve.get('style', {})
        ax.plot(x, y, label=label, linewidth=3, **style)

    if grid:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if title:
        ax.set_title(title, fontsize=18)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=12, colors='#333333')
    for spine in ax.spines.values():
        spine.set_color('#333333')
    if any(curve.get('label') for curve in curves):
        ax.legend(loc=legend_loc, fontsize=14)
    fig.tight_layout()
    return fig, ax

