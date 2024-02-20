import numpy as np


def DF(signal, tau):  # difference function
    return np.sum((signal[0: len(signal)//2] - signal[tau: tau+len(signal)//2])**2)


def CMDF(signal, tau):  # normalisation
    if tau == 0:
        return 1
    return DF(signal, tau) / np.sum([DF(signal, j+1) for j in range(tau)])*tau


def detect_pitch_yin(signal, sample_rate=50000, bounds=[5, 500], tresh=0.1, pas=1):
    CMDF_vals = [CMDF(0.001*signal, tau)
                 for tau in range(bounds[0], bounds[-1], pas)]
    sample = None

    for i, val in enumerate(CMDF_vals):
        if val < tresh:
            sample = pas*i+bounds[0]
            break

    if sample is None:
        sample = pas*np.argmin(CMDF_vals)+bounds[0]
    return sample_rate/sample
