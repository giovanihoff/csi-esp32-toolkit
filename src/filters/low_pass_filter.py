
from scipy.signal import butter, filtfilt

def low_pass_filter(signal, cutoff=0.2, order=3):
    """
    Applies a Butterworth low-pass filter to the input signal.
    """
    b, a = butter(order, cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)
