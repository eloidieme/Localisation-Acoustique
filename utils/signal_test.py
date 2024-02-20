import numpy as np
import sounddevice as sd

"""
Fonctions permettant d'émettre des signaux sonores depuis le haut-parleur de l'ordinateur. 
"""

def start_signal(length, fs, f0):
    T = np.arange(length) / fs
    amps = np.array([0, 1])
    sig = np.sin(2 * np.pi * f0 * T)[:, np.newaxis] @ amps[:, np.newaxis].T
    sd.play(sig, fs)

def make_signal(func, length, fs, f0):
    T = np.arange(length) / fs
    amps = np.array([0, 1])
    if func == 'sinus':
        return np.sin(2 * np.pi * f0 * T)[:, np.newaxis] @ amps[:, np.newaxis].T
    elif func == 'cosinus':
        return np.cos(2 * np.pi * f0 * T)[:, np.newaxis] @ amps[:,
                                                                np.newaxis].T
    elif func == 'tangente':
        return np.tan(2 * np.pi * f0 * T)[:, np.newaxis] @ amps[:,
                                                                np.newaxis].T
    elif func == 'sinus cardinal':
        return np.sinc(2 * np.pi * f0 * T)[:, np.newaxis] @ amps[:,
                                                                np.newaxis].T

def make_signaux(signaux, length, fs, f_0):
    signal_final = 0
    for signal in signaux:
        signal_final += make_signal(signal, length, fs, f_0[signal])
    return signal_final

def launch_signal(signaux, length, fs, f_0):
    signal = make_signaux(signaux, length, fs, f_0)
    sd.play(signal, fs)
