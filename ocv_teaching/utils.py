from scipy.interpolate import interp1d

def interpolate(xp, yp, x):
    """
    Interpolates the voltage values for given SOC values.
    """
    f = interp1d(
        xp, yp,
        kind='linear',
        fill_value='extrapolate',   # allow linear extrapolation
        assume_sorted=False         # will sort xp/yp internally
    )
    return f(x)