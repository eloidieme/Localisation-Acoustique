import numpy as np
from scipy import signal


def detect_pitch_stft(data, sample_rate=50000):
    f, t, stft = signal.stft(data, nperseg=1000, fs=sample_rate)
    print(data)
    mmin = np.min(np.argmax(stft[0], axis=0))
    mmax = np.max(np.argmax(stft[0], axis=0))
    mean = (f[mmin] + f[mmax])/2
    return mean


def detect_pitch_stft_v2(data, sample_rate=50000): #on prend la fréquence qui a le plus souvent été maximale
    f, t, stft = signal.stft(data, nperseg=1000, fs=sample_rate)
    return f[np.argmax(np.sum(stft[0], axis=1))]