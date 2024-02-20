import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QPushButton, QDoubleSpinBox, QLabel, QGroupBox, QComboBox, QSplitter, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

import numpy as np
import pandas as pd
from scipy import signal
import scipy.io
import time

import usb.core
import usb.util
import usb.backend.libusb1

from utils.geometrie import *
from algos.beamforming import beamforming
from algos.beamforming import beamforming
from algos.mvdr_beamforming import mvdr
from algos.music_beamforming import music
from algos.yin import detect_pitch_yin
from algos.stft_fundamental_frequency import detect_pitch_stft_v2
from utils.signal_gen_acoular import sine_generator, sine_w_noise_generator
from utils.enregistrement import recording

### Création du backend USB ###

# backend = usb.backend.libusb1.get_backend()

### Communication avec le boitier ###

# DEV = usb.core.find(idVendor=0xfe27, idProduct=0xac03)

CONFIG = pd.read_csv('config/config_input.csv', sep=";").set_index('params')
NSAMPLES = int(CONFIG.iloc[0, 0])
SAMPLING_FREQ = float(CONFIG.iloc[1, 0])
SRINV = int(CONFIG.iloc[2, 0])
SIGNAL_FREQ = float(CONFIG.iloc[3, 0])
silence_transitoire = float(CONFIG.iloc[4, 0])
# COORD_MICROS = coordonnees_micros_xml_sorted('utils/xml/micgeom2.xml')
COORD_MICROS = coordonnees_micros(
    PAS_MICROS, LIGNE_MICROS_HORIZONTAUX, COLONNE_MICROS_VERTICAUX)
NOMBRE_MICROS = len(COORD_MICROS)

NSEGMENTS = (4 * NSAMPLES * NOMBRE_MICROS) // 1024

CELERITY = 340
k = 2 * np.pi * SIGNAL_FREQ / CELERITY

GRILLE_SOURCES = define_grille_source(
    HAUTEUR_SOURCES, LONGUEUR_SOURCES, PAS_SOURCES)

GRILLE_MICROS = define_grille_micros(
    HAUTEUR_MICROS, LONGUEUR_MICROS, PAS_MICROS, LIGNE_MICROS_HORIZONTAUX, COLONNE_MICROS_VERTICAUX)

# GRILLE_MICROS = define_grille_micros_xml(
#    HAUTEUR_MICROS, LONGUEUR_MICROS, PAS_MICROS, 'utils/xml/micgeom2.xml')


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initData()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Choose Algorithm
        self.choose_algorithm_label = QLabel('Choisissez un algorithme :')
        self.choose_algorithm_combo = QComboBox()
        self.choose_algorithm_combo.addItem('Beamforming conventionnel')
        self.choose_algorithm_combo.addItem('MVDR')
        self.choose_algorithm_combo.addItem('MUSIC')
        self.layout.addWidget(self.choose_algorithm_label)
        self.layout.addWidget(self.choose_algorithm_combo)

        # Ajout des champs pour le nombre d'échantillons, la fréquence d'échantillonnage et la célérité
        self.settings_group = QGroupBox("Paramètres")
        self.settings_layout = QVBoxLayout()

        self.auto_freq_checkbox = QCheckBox(
            'Détection de fréquence fondamentale')
        self.choose_freq_detection = QComboBox()
        self.choose_freq_detection.addItem('Yin')
        self.choose_freq_detection.addItem('STFT')
        self.auto_freq_checkbox.stateChanged.connect(self.autoFreqStateChanged)
        self.settings_layout.addWidget(self.auto_freq_checkbox)
        self.settings_layout.addWidget(self.choose_freq_detection)

        self.signal_label = QLabel('Choix du signal:')
        self.choose_signal = QComboBox()
        self.choose_signal.addItem('Sinus')
        self.choose_signal.addItem('Sinus bruité')
        self.choose_signal.addItem(
            'Enregistrement 19/12')
        self.choose_signal.addItem('Live')
        self.settings_layout.addWidget(self.signal_label)
        self.settings_layout.addWidget(self.choose_signal)

        self.signal_freq_label = QLabel('Fréquence du signal:')
        self.signal_freq_spinbox = QDoubleSpinBox()
        # Optional, to specify number of decimal places
        self.signal_freq_spinbox.setDecimals(2)
        self.signal_freq_spinbox.setMinimum(0)   # Set minimum value
        # Set maximum value to 10000 or any other desired value
        self.signal_freq_spinbox.setMaximum(10000)
        self.signal_freq_spinbox.setValue(SIGNAL_FREQ)
        self.settings_layout.addWidget(self.signal_freq_label)
        self.settings_layout.addWidget(self.signal_freq_spinbox)

        self.nsamples_label = QLabel("Nombre d'échantillons:")
        self.nsamples_spinbox = QDoubleSpinBox()
        self.nsamples_spinbox.setRange(1, 100000)
        self.nsamples_spinbox.setValue(NSAMPLES)
        self.nsamples_spinbox.valueChanged.connect(self.updateNSamples)
        self.settings_layout.addWidget(self.nsamples_label)
        self.settings_layout.addWidget(self.nsamples_spinbox)

        self.sampling_freq_label = QLabel("Fréquence d'échantillonnage:")
        self.sampling_freq_spinbox = QDoubleSpinBox()
        self.sampling_freq_spinbox.setRange(1, 100000)
        self.sampling_freq_spinbox.setValue(SAMPLING_FREQ)
        self.sampling_freq_spinbox.valueChanged.connect(
            self.updateSamplingFreq)
        self.settings_layout.addWidget(self.sampling_freq_label)
        self.settings_layout.addWidget(self.sampling_freq_spinbox)

        self.celerity_label = QLabel("Célérité:")
        self.celerity_spinbox = QDoubleSpinBox()
        self.celerity_spinbox.setRange(1, 2000)
        self.celerity_spinbox.setValue(CELERITY)
        self.celerity_spinbox.valueChanged.connect(self.updateCelerity)
        self.settings_layout.addWidget(self.celerity_label)
        self.settings_layout.addWidget(self.celerity_spinbox)

        self.settings_group.setLayout(self.settings_layout)
        self.layout.addWidget(self.settings_group)

        # Launch Button
        self.launch_button = QPushButton('Lancer')
        self.launch_button.clicked.connect(self.launchVisualization)
        self.layout.addWidget(self.launch_button)

        # Ajout du bouton pour stopper la localisation
        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stopLocalization)
        self.layout.addWidget(self.stop_button)

        self.graphWidget = pg.ImageView(view=pg.PlotItem(
            title="Sortie du Beamformer (normalisée)"))
        self.layout.addWidget(self.graphWidget)

        self.heatmap = pg.colormap.get('viridis')
        self.graphWidget.setColorMap(self.heatmap)

        # Créer un QSplitter pour avoir deux vues côte à côte
        self.splitter = QSplitter(Qt.Horizontal)
        self.layout.addWidget(self.splitter)

        # Layout pour le signal
        self.signal_layout = QVBoxLayout()
        self.signal_container = QWidget()
        self.signal_container.setLayout(self.signal_layout)
        self.splitter.addWidget(self.signal_container)

        # Widget pour le tracé du signal
        self.signalPlot = pg.PlotWidget(title="Signal mesuré au premier micro")
        self.signal_curve = self.signalPlot.plot(pen='y')
        self.signal_layout.addWidget(self.signalPlot)

        # Ajouter le graphWidget de la localisation à droite du splitter
        self.splitter.addWidget(self.graphWidget)

        self.setWindowTitle('Localisation Acoustique')
        self.setGeometry(10, 10, 1920, 1080)

    def launchVisualization(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateData)
        self.timer.start(50)
        self.initData()

    def stopLocalization(self):
        self.timer.stop()

    def updateNSamples(self):
        global NSAMPLES
        NSAMPLES = int(self.nsamples_spinbox.value())

    def updateSamplingFreq(self):
        global SAMPLING_FREQ
        SAMPLING_FREQ = self.sampling_freq_spinbox.value()

    def updateCelerity(self):
        global CELERITY
        CELERITY = self.celerity_spinbox.value()

    def initData(self):
        self.x, self.y = 0.5, 0.5
        self.dx, self.dy = 0.1, 0.1
        self.mat_index = 0
        self.mat = self.mat = scipy.io.loadmat(
            'data/enregistrements/4_2m_x_0_mvt_y_0_1000hz_15s.mat')
        p_sources = 0.01  # m
        N = 16
        H_sources = 2  # m
        l_sources = 2  # m

        def grillesource(H_sources, l_sources, p_sources):
            return np.ones((math.floor(H_sources/p_sources) + 1, math.floor(l_sources/p_sources) + 1))
        Grille_sources = grillesource(H_sources, l_sources, p_sources)
        Coord_micros = np.array([[1.28, 0.16],
                                 [1.12, 0.16],
                                 [0.96, 0.16],
                                 [0.8, 0.16],
                                 [0.64, 0.16],
                                 [0.48, 0.16],
                                 [0.32, 0.16],
                                 [0.16, 0.16],
                                 [0., 0.],
                                 [0., 0.16],
                                 [0., 0.32],
                                 [0., 0.48],
                                 [0., 0.64],
                                 [0., 0.8],
                                 [0., 0.96],
                                 [0., 1.12]])
        A = []
        for i in range(len(Grille_sources)):
            for j in range(len(Grille_sources[0])):
                A.append(array_manifold(j * p_sources, i *
                         p_sources, Coord_micros, 2, 2000, 343))
        A = np.transpose(A)
        fact_norm = sum(abs(A) ** 2)
        fact_norm = np.sqrt(fact_norm)
        self.An = A/fact_norm/math.sqrt(N)

    def updateData(self):
        if self.x >= 1.9:
            self.x = 0.1
        if self.y >= 1.9:
            self.y = 0.1

        if self.choose_signal.currentText() == 'Sinus':
            data = sine_generator(self.x, self.y, DISTANCE_SOURCES_MICROS,
                                  self.signal_freq_spinbox.value(), SAMPLING_FREQ, NSAMPLES)
        elif self.choose_signal.currentText() == 'Sinus bruité':
            data = sine_w_noise_generator(self.x, self.y, DISTANCE_SOURCES_MICROS,
                                          self.signal_freq_spinbox.value(), SAMPLING_FREQ, NSAMPLES)
        elif self.choose_signal.currentText() == 'Enregistrement 19/12':
            data = self.mat['mat'].T[int(self.mat_index % (self.mat['mat'].shape[1])): int((
                NSAMPLES + self.mat_index) % (self.mat['mat'].shape[1])), :]
            self.mat_index += 0.1 * SAMPLING_FREQ

        # else:
            # data = recording(DEV, NSAMPLES, NOMBRE_MICROS,
            #                 NSEGMENTS, silence_transitoire)

        def butter_lowpass(cutoff, fs, order=2):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff,
                                 btype='low', analog=False)
            return b, a

        auto_freq_samples = min(1000, NSAMPLES)

        if self.auto_freq_checkbox.isChecked():
            if self.choose_freq_detection.currentText() == "Yin":
                detected_freq = detect_pitch_yin(data[0:auto_freq_samples, 0], bounds=[
                    5, 500], tresh=0.01)
            else:
                detected_freq = detect_pitch_stft_v2(data.T)
            self.signal_freq_spinbox.setValue(detected_freq)

        if self.choose_signal.currentText() == 'Enregistrement 19/12':
            sfreq = 50000  # sampling frequency (in kHz)
            nsamples = 32768

            def grillesource(H_sources, l_sources, p_sources):
                return np.ones((math.floor(H_sources/p_sources) + 1, math.floor(l_sources/p_sources) + 1))

            # Définissons les paramètres :
            # Les distances sont en centimètres
            H_sources = 2  # m
            l_sources = 2  # m
            p_sources = 0.01  # m

            H_micros = 2  # m
            l_micros = 1.28  # m
            p_micros = 0.16  # m
            ligne_micros = 8
            colonne_micros = 0  # math.floor(l_micros/p_micros)

            # m  # il s'agit de la distance entre le 0 de la grille source
            dist_source_to_micros = 2
            # et celui de la grille des micros

            N = 16  # Le nombre total de micros

            Coord_micros = np.array([[1.28, 0.16],
                                     [1.12, 0.16],
                                     [0.96, 0.16],
                                     [0.8, 0.16],
                                     [0.64, 0.16],
                                     [0.48, 0.16],
                                     [0.32, 0.16],
                                     [0.16, 0.16],
                                     [0., 0.],
                                     [0., 0.16],
                                     [0., 0.32],
                                     [0., 0.48],
                                     [0., 0.64],
                                     [0., 0.8],
                                     [0., 0.96],
                                     [0., 1.12]])

            c = 340  # m.s^(-1)
            k = 2*np.pi*2000/340  # m^(-1)

            # Définissons les variables :

            Grille_sources = grillesource(H_sources, l_sources, p_sources)
            Grille_micros = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 1.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 1.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 1.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 1.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 1.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 1.],
                                      [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 1.]])

            n = 16

            # Stockons dans la matrice A les array_manifold pour chaque point source. La matrice A contient 32 lignes
            # et (math.floor(l_sources/p_sources) + 1)**2 colonnes, ce qui correspond au nombre de positions sources possibles.

            f, t, stft = signal.stft(data.T, fs=50000, nperseg=1000)

            nb_Fr = 40
            useful_data = stft[:, nb_Fr]

            # useful_data = ordre_micros(useful_data.T).T

            Bc = np.abs(np.dot(np.conj(np.transpose(self.An)), useful_data))
            Bc0 = np.multiply(Bc, Bc)
            Bm = np.mean(Bc0, axis=1)
            Z = np.reshape(Bm, (math.floor(H_sources/p_sources) +
                           1, math.floor(H_sources/p_sources) + 1))
            Z /= np.max(Z)
            Z = np.flip(Z)
        elif (self.choose_algorithm_combo.currentText() == 'MVDR'):
            Z = mvdr(data, array_manifold, self.signal_freq_spinbox.value(),
                     COORD_MICROS, ordre_micros, DISTANCE_SOURCES_MICROS,
                     GRILLE_SOURCES, NSAMPLES, PAS_SOURCES, CELERITY)
        elif (self.choose_algorithm_combo.currentText()) == 'MUSIC':
            Z = music(data, array_manifold, self.signal_freq_spinbox.value(),
                      COORD_MICROS, ordre_micros, DISTANCE_SOURCES_MICROS,
                      GRILLE_SOURCES, NSAMPLES, PAS_SOURCES, CELERITY)
        else:
            Z = beamforming(data, array_manifold, self.signal_freq_spinbox.value(),
                            COORD_MICROS, ordre_micros, DISTANCE_SOURCES_MICROS,
                            GRILLE_SOURCES, NSAMPLES, PAS_SOURCES, CELERITY)

        self.graphWidget.setImage(Z.T)

        # Mise à jour du tracé du signal
        time_array = np.arange(min(250, NSAMPLES)) / SAMPLING_FREQ
        self.signal_curve.setData(time_array, data[0:min(250, NSAMPLES), 0])

        self.x += self.dx
        self.y += self.dy

    def autoFreqStateChanged(self):
        if self.auto_freq_checkbox.isChecked():
            self.signal_freq_spinbox.setDisabled(True)
        else:
            self.signal_freq_spinbox.setEnabled(True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
