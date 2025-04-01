"""For analyzing the frequency vs gamma signal"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import find_peaks, peak_widths


def get_resonance(ydata, show_plot: bool = True, **kwargs) -> pd.Series:
    """Given frequency vs reflection dB what's the center f and amp.
    kwargs are passed to scipy.signal.find_peaks
    """
    ydata = pd.Series(ydata)
    i, _ = find_peaks(-ydata, **kwargs)
    if show_plot:
        fig, ax = plt.subplots()
        ax.plot(ydata)
        ax.plot(ydata.iloc[i], 'x')
        plt.show()
    return ydata.iloc[i]


def get_bandwidth(gamma: pd.DataFrame, thresholds: float | list[float], npoints: int = 6, show_plot: bool = True) -> pd.DataFrame:
    if type(thresholds) is float or type(thresholds) is int:  # Type checking is just a suggestion
        thresholds = [thresholds]
    if show_plot:
        fig, ax = plt.subplots()
        ax.plot(gamma)
        ax.grid()

    bandwidths = []
    for t in thresholds:
        points = gamma.iloc[ (gamma+t).abs().argsort() ].iloc[:npoints]
        points = points.reset_index()
        midpoint = (points['index'].max() + points['index'].min()) / 2
        low = points.loc[ points['index'] < midpoint ].mean()
        high = points.loc[ points['index'] > midpoint ].mean()
        bw = high['index'] - low['index']
        bandwidths.append(bw)
        if show_plot:
            ax.scatter(points['index'], points['b'])
            ax.hlines(-t, low['index'], high['index'], color='black')
            ax.annotate(text=f"-{t} dB\n{bw:.3f} GHz", xy=high, xytext=(high['index']+bw*1.5, high['b']), arrowprops=dict(shrink=0.2, facecolor='black'))
    if show_plot:
        plt.show()
    df = pd.DataFrame(dict(dB=thresholds, GHz=bandwidths))
    return df