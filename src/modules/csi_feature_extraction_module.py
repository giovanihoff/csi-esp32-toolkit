# csi_feature_extraction_module.py

import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import rfft, rfftfreq

def extract_statistical_features(magnitudes):
    magnitudes = np.asarray(magnitudes)
    features = {
        "mean": np.mean(magnitudes),
        "std": np.std(magnitudes),
        "max": np.max(magnitudes),
        "min": np.min(magnitudes),
        "energy": np.sum(magnitudes ** 2)
    }

    if features["std"] == 0:
        features["skewness"] = 0.0
        features["kurtosis"] = 0.0
        features["entropy"] = 0.0
        features["zero_crossing_rate"] = 0.0
        features["fft_peak_freq"] = 0.0
        features["spectral_centroid"] = 0.0
        features["spectral_spread"] = 0.0
    else:
        features["skewness"] = skew(magnitudes, nan_policy='omit')
        features["kurtosis"] = kurtosis(magnitudes, nan_policy='omit')

        # Entropia
        hist, _ = np.histogram(magnitudes, bins=10, density=True)
        features["entropy"] = entropy(hist + 1e-10)  # para evitar log(0)

        # Zero-crossing rate
        zero_crossings = np.where(np.diff(np.sign(magnitudes - np.mean(magnitudes))))[0]
        features["zero_crossing_rate"] = len(zero_crossings) / len(magnitudes)

        # FFT e features espectrais
        fft_vals = np.abs(rfft(magnitudes))
        fft_freqs = rfftfreq(len(magnitudes), d=1)

        if len(fft_vals) > 0:
            peak_idx = np.argmax(fft_vals)
            features["fft_peak_freq"] = fft_freqs[peak_idx]

            spectral_centroid = np.sum(fft_freqs * fft_vals) / np.sum(fft_vals)
            features["spectral_centroid"] = spectral_centroid

            spectral_spread = np.sqrt(np.sum(((fft_freqs - spectral_centroid) ** 2) * fft_vals) / np.sum(fft_vals))
            features["spectral_spread"] = spectral_spread
        else:
            features["fft_peak_freq"] = 0.0
            features["spectral_centroid"] = 0.0
            features["spectral_spread"] = 0.0

    return features

