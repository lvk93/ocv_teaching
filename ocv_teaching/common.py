import matplotlib.pyplot as plt

# Standard plotting configuration
plt.rcParams.update({
    'font.size': 14,
    'figure.figsize': (8, 6),
    'lines.linewidth': 2,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
})

def new_plot():
    """
    Create a new standardized Matplotlib figure and axis.
    """
    fig, ax = plt.subplots()
    ax.grid(True)
    return fig, ax