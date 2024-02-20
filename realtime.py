import usb.core
import usb.util
import usb.backend.libusb1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.activation_functions import configuration, initialisation, activation
from utils.geometrie import *
from algos.beamforming import *

### Création du backend USB ###

backend = usb.backend.libusb1.get_backend()

### Communication avec le boitier ###

DEV = usb.core.find(idVendor=0xfe27, idProduct=0xac03)

### Récupération des paramètres depuis un fichier csv ###

CONFIG = pd.read_csv('config/config_input.csv', sep=";").set_index('params')
NSAMPLES = int(CONFIG.iloc[0, 0])
SAMPLING_FREQ = float(CONFIG.iloc[1, 0])
SRINV = int(CONFIG.iloc[2, 0])
SIGNAL_FREQ = float(CONFIG.iloc[3, 0])
silence_transitoire = float(CONFIG.iloc[4, 0])

### Établissement de la communication ###

configuration(dev=DEV)
initialisation(dev=DEV)
activation(micros=None, dev=DEV, srinv=SRINV, length=NSAMPLES)

### Variables globales ###

NSEGMENTS = (4 * NSAMPLES * NOMBRE_MICROS) // 1024

CELERITY = 340  # m.s^(-1)
k = 2*np.pi*SIGNAL_FREQ/CELERITY  # m^(-1)

### Géométrie ###

GRILLE_SOURCES = define_grille_source(
    HAUTEUR_SOURCES, LONGUEUR_SOURCES, PAS_SOURCES)

GRILLE_MICROS = define_grille_micros(
    HAUTEUR_MICROS, LONGUEUR_MICROS, PAS_MICROS, LIGNE_MICROS_HORIZONTAUX, COLONNE_MICROS_VERTICAUX)

COORD_MICROS = coordonnees_micros(
    PAS_MICROS, LIGNE_MICROS_HORIZONTAUX, COLONNE_MICROS_VERTICAUX)

### Initialisation ###

Z = np.zeros((201, 201))
plt.ion()

mat = plt.imshow(Z, origin='lower', interpolation='bicubic')

### MAIN LOOP ###
while True:
    ### Acquisition ###
    record = np.zeros([1024, NSEGMENTS])

    for n in range(NSEGMENTS):
        # lire la doc megamicro
        # envoie la requete pour pouvoir ecouter sur le port usb
        DEV.ctrl_transfer(0x40, 0xB1, 0, 0, [0x02])
        # lecture des données
        record[:, n] = DEV.read(0x81, 1024, 1000)

    U = record.T.ravel()
    # reconstruction des int32
    V = U.reshape([NSAMPLES * NOMBRE_MICROS, 4])
    W = V[:, 3] * 256**3 + V[:, 2] * 256**2 + V[:, 1] * 256 + V[:, 0]
    W[W > 2**31] = W[W > 2**31] - 2**32
    # restructuration en canaux pour pouvoir être envoyé à la partie traitement
    R = W.reshape([W.shape[0] // NOMBRE_MICROS, NOMBRE_MICROS])
    # on enlève le début (silence et transitoire)
    data = R[int(silence_transitoire):, :]

    ### Calcul du Beamformer ###

    Z = beamforming(data, array_manifold, SIGNAL_FREQ, coordonnees_micros(PAS_MICROS, LIGNE_MICROS_HORIZONTAUX,
                                                                          COLONNE_MICROS_VERTICAUX), ordre_micros, DISTANCE_SOURCES_MICROS, GRILLE_SOURCES, NSAMPLES, PAS_SOURCES, CELERITY)

    ### Affichage ###

    mat = plt.imshow(Z)

    plt.draw()
    plt.gcf().canvas.flush_events()
    plt.gcf().canvas.blit()

    plt.pause(0.05)
