import os
import acoular
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random as rd
from scipy import signal


MICGEOMETRYFILE = 'utils/xml/mu32.xml'  # microphone geometry
DATASAVEFILE = 'utils/temp/generationSinus.h5'  # file to save the data


def sine_generator(source_x, source_y, source_distance, sine_freq, sample_freq, nsamples):
    """
    Génére un signal sinusoïdal à l'aide d'Acoular.

    Parameters:
    source_x (float): position x de la source à générer.
    source_y (float): position y de la source à générer.
    source_distance (float): distance de la source à générer aux micros.
    sine_freq (float): fréquence de la sinusoïde à générer.
    nsamples (int): nombre d'échantillons du signal généré.

    Returns:
    data (np.ndarray): matrice d'acquisition du signal généré.
    """
    mics = acoular.MicGeom(from_file=MICGEOMETRYFILE)  # we load the geometry
    sine = acoular.SineGenerator(
        sample_freq=sample_freq, numsamples=nsamples, freq=sine_freq)  # first signal
    source = acoular.PointSource(signal=sine, mics=mics, loc=(
        source_x, source_y, source_distance))
    p = acoular.Mixer(source=source)
    write_h5 = acoular.WriteH5(source=p, name=DATASAVEFILE)
    write_h5.save()

    h5_file = h5py.File(DATASAVEFILE, 'r')
    data = h5_file[('time_data')]
    data = data[:]
    h5_file.close()
    os.remove(DATASAVEFILE)
    return data


def sine_w_noise_generator(source_x, source_y, source_distance, sine_freq, sample_freq, nsamples):
    mics = acoular.MicGeom(from_file=MICGEOMETRYFILE)  # we load the geometry
    sine = acoular.SineGenerator(
        sample_freq=sample_freq, numsamples=nsamples, freq=sine_freq, rms=1.0)  # first signal
    additional_sines = [acoular.SineGenerator(
        sample_freq=sample_freq, numsamples=nsamples, freq=freq, rms=rd.normalvariate(0, 1)) for freq in range(2000, 3000, 50)]
    noise = acoular.FiltWNoiseGenerator(
        sample_freq=sample_freq, numsamples=nsamples, rms=0.4)  # first signal
    source = acoular.PointSource(signal=sine, mics=mics, loc=(
        source_x, source_y, source_distance))
    additional_sources = [acoular.PointSource(signal=add_sine, mics=mics, loc=(
        source_x, source_y, source_distance)) for add_sine in additional_sines]
    noise_source = acoular.PointSource(signal=noise, mics=mics, loc=(
        source_x, source_y, source_distance))
    p = acoular.Mixer(source=source, sources=additional_sources)
    write_h5 = acoular.WriteH5(source=p, name=DATASAVEFILE)
    write_h5.save()

    h5_file = h5py.File(DATASAVEFILE, 'r')
    data = h5_file[('time_data')]
    data = data[:]
    h5_file.close()
    os.remove(DATASAVEFILE)
    return data


def white_noise_generator(source_x, source_y, source_distance, sample_freq, nsamples):
    mics = acoular.MicGeom(from_file=MICGEOMETRYFILE)  # we load the geometry
    noise = acoular.FiltWNoiseGenerator(
        sample_freq=sample_freq, numsamples=nsamples)  # first signal
    noise_source = acoular.PointSource(signal=noise, mics=mics, loc=(
        source_x, source_y, source_distance))
    p = acoular.Mixer(source=noise_source)
    write_h5 = acoular.WriteH5(source=p, name=DATASAVEFILE)
    write_h5.save()

    h5_file = h5py.File(DATASAVEFILE, 'r')
    data = h5_file[('time_data')]
    data = data[:]
    h5_file.close()
    os.remove(DATASAVEFILE)
    return data


def sine_w_hnoise_generator(source_x, source_y, source_distance, sine_freq, sample_freq, nsamples):
    mics = acoular.MicGeom(from_file=MICGEOMETRYFILE)  # we load the geometry
    sine = acoular.SineGenerator(
        sample_freq=sample_freq, numsamples=nsamples, freq=sine_freq, rms=1.0)  # first signal
    additional_sines = [acoular.SineGenerator(
        sample_freq=sample_freq, numsamples=nsamples, freq=freq, rms=0.1) for freq in range(20000, 30000, 1000)]
    source = acoular.PointSource(signal=sine, mics=mics, loc=(
        source_x, source_y, source_distance))
    additional_sources = [acoular.PointSource(signal=add_sine, mics=mics, loc=(
        source_x, source_y, source_distance)) for add_sine in additional_sines]
    p = acoular.Mixer(source=source, sources=additional_sources)
    write_h5 = acoular.WriteH5(source=p, name=DATASAVEFILE)
    write_h5.save()

    h5_file = h5py.File(DATASAVEFILE, 'r')
    data = h5_file[('time_data')]
    data = data[:]
    h5_file.close()
    os.remove(DATASAVEFILE)
    return data


def plot_signal(data, sample_freq):
    """
    Réalise le tracé d'un signal et de sa STFT.

    Parameters:
    data (np.ndarray): matrice contenant le signal à tracer.

    Returns: None
    """
    fig, (ax1, ax2) = plt.subplots(2)
    freq, time, stft = signal.stft(data.T, nperseg=1000, fs=sample_freq)

    x_axis = 1/100*np.arange(data[0:time.shape[0], 0].size)
    ax1.plot(x_axis, data[0:time.shape[0], 0]/np.max(data[0:time.shape[0], 0]))
    ax1.set(xlabel="Temps (s)", ylabel="Amplitude")

    ax2.pcolormesh(time, freq, np.abs(stft[0]))
    ax2.set(xlabel="Temps (s)", ylabel="Fréquence (Hz)")

    fig.suptitle("Tracé et TFTD du signal du premier micro")

    plt.show()
