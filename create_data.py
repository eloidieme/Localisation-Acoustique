import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QPushButton, QDoubleSpinBox, QLabel, QGroupBox, QComboBox, QSplitter, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

import numpy as np
import pandas as pd

from utils.geometrie import *
from algos.beamforming import beamforming
from algos.mvdr_beamforming import mvdr
from algos.music_beamforming import music
from utils.signal_gen_acoular import sine_generator, sine_w_noise_generator

CONFIG = pd.read_csv('config/config_input.csv', sep=";").set_index('params')
NSAMPLES = 400
SAMPLING_FREQ = float(CONFIG.iloc[1, 0])
SIGNAL_FREQ = float(CONFIG.iloc[3, 0])

CELERITY = 340
k = 2 * np.pi * SIGNAL_FREQ / CELERITY

GRILLE_SOURCES = define_grille_source(
    HAUTEUR_SOURCES, LONGUEUR_SOURCES, PAS_SOURCES)

GRILLE_MICROS = define_grille_micros(
    HAUTEUR_MICROS, LONGUEUR_MICROS, PAS_MICROS, LIGNE_MICROS_HORIZONTAUX, COLONNE_MICROS_VERTICAUX)

COORD_MICROS = coordonnees_micros(
    PAS_MICROS, LIGNE_MICROS_HORIZONTAUX, COLONNE_MICROS_VERTICAUX)

coords = [[(i, j) for i in np.random.uniform(0.0, 2.0, 100)]
          for j in np.random.uniform(0.0, 2.0, 100)]
file_number = 1
resX = pd.DataFrame()
resY = pd.DataFrame()

for i in range(len(coords)):
    for j in range(len(coords[0])):
        print(file_number)
        data = sine_generator(coords[i][j][0], coords[i][j][1], DISTANCE_SOURCES_MICROS,
                              SIGNAL_FREQ, SAMPLING_FREQ, NSAMPLES)
        data = data.T
        R = 1/(NSAMPLES)*np.dot(data, data.conj().T)

        resX[f"{file_number}"] = pd.DataFrame(data=R, columns=[
            f"mic {i}" for i in range(1, 33)]).stack()

        Z = np.zeros((GRILLE_SOURCES.shape[0], GRILLE_SOURCES.shape[1]))
        Z[round(coords[i][j][0] / PAS_SOURCES),
          round(coords[i][j][1] / PAS_SOURCES)] = 1
        print(round(coords[i][j][0] / PAS_SOURCES))
        print(round(coords[i][j][1] / PAS_SOURCES))
        df2 = pd.DataFrame(data=Z).stack()
        resY[f"{file_number}"] = df2
        file_number += 1


resX.to_csv(f'data/X/records1.csv', sep=',')
resY.to_csv(f'data/y/records1.csv', sep=',')
