
import numpy as np
import pandas as pd

def hampel_filter(series, window_size=5, n_sigmas=3):
    """
    Applies the Hampel filter to detect and replace outliers in a series.
    """
    new_series = series.copy()
    k = 1.4826  # scale factor for Gaussian distribution
    rolling_median = series.rolling(window=2*window_size, center=True).median()
    MAD = k * (series.rolling(window=2*window_size, center=True)
                      .apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True))
    
    difference = np.abs(series - rolling_median)
    outlier_idx = difference > n_sigmas * MAD
    new_series[outlier_idx] = rolling_median[outlier_idx]
    return new_series
