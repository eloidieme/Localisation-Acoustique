import math
import numpy as np
from scipy import signal
from utils.geometrie import *

sfreq = 50000  # sampling frequency (in kHz)
source_freq = 2000  # Hz

c = 340  # m.s^(-1)
k = 2*np.pi*source_freq/340  # m^(-1)

grille_sources = define_grille_source(
    HAUTEUR_SOURCES, LONGUEUR_SOURCES, PAS_SOURCES)
grille_micros = define_grille_micros(
    HAUTEUR_MICROS, LONGUEUR_MICROS, PAS_MICROS, LIGNE_MICROS_HORIZONTAUX, COLONNE_MICROS_VERTICAUX)

coord_micros = coordonnees_micros(
    PAS_MICROS, LIGNE_MICROS_HORIZONTAUX, COLONNE_MICROS_VERTICAUX)

A = []
for i in range(len(grille_sources)):
    for j in range(len(grille_sources[0])):
        A.append(array_manifold(j * PAS_SOURCES, i * PAS_SOURCES,
                 coord_micros, DISTANCE_SOURCES_MICROS, source_freq, c))
A = np.transpose(A)
fact_norm = sum(abs(A) ** 2)
fact_norm = np.sqrt(fact_norm)
An = A/fact_norm/math.sqrt(NOMBRE_MICROS)


def legacy_beamforming(data, sample_freq, signal_freq):
    data = data.T
    t, f, stft = signal.stft(data, fs=sample_freq, nperseg=1000)

    nb_Fr = int(signal_freq//(sample_freq/100))
    useful_data = stft[:, nb_Fr]

    # On remet dans l'ordre les lignes
    useful_data = ordre_micros(useful_data.T).T

    Bc = np.abs(np.dot(np.conj(np.transpose(An)), useful_data))
    Bc = np.multiply(Bc, Bc)

    Z = np.reshape(Bc[:, 0], (math.floor(
        HAUTEUR_SOURCES/PAS_SOURCES) + 1, math.floor(HAUTEUR_SOURCES/PAS_SOURCES) + 1))
    return Z
