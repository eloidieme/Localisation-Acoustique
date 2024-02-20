import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def faisceau(mic_value):
    """
    Écrit en binaire les micros à activer par faisceau.
    """
    # lire la doc megamicros

    mic = "0b"
    mic_value.sort()
    for i in range(0, 8):
        if i in mic_value:
            mic += "1"
        else:
            mic += "0"
    return mic


def make_micros(path='config/micros.csv'):
    """
    Renvoie les micros à activer par faisceau de 0 à 3 à partir d'un fichier
    de configuration.

    Parameters:
    path (str) = 'config/micros.csv': chemin du fichier de configuration.

    Returns:
    mic0 (str): nombre binaire décrivant le 1er faisceau.
    mic1 (str): nombre binaire décrivant le 2nd faisceau.
    mic2 (str): nombre binaire décrivant le 3eme faisceau.
    mic3 (str): nombre binaire décrivant le 4eme faisceau.
    """
    config_dataframe = pd.read_csv(path, sep=";").set_index('faisceau')

    return str(config_dataframe.iloc[0, 0]), str(config_dataframe.iloc[1, 0]), str(config_dataframe.iloc[2, 0]), str(config_dataframe.iloc[3, 0])


def nombre_micros(mic0, mic1, mic2, mic3):
    """
    Compte les micros sur les faisceaux fournis.

    Parameters:
    mic0 (str): nombre binaire décrivant le 1er faisceau.
    mic1 (str): nombre binaire décrivant le 2nd faisceau.
    mic2 (str): nombre binaire décrivant le 3eme faisceau.
    mic3 (str): nombre binaire décrivant le 4eme faisceau.

    Returns:
    nombre_mics (int): nombre de micros à activer.
    """

    return mic0.count("1") + mic1.count("1") + mic2.count(
        "1") + mic3.count("1")


def echantillons(length):
    """
    Génère des longueurs d'échantillons. 

    Parameters:
    length (int): longueur de base.

    Returns:
    length_0, length_1, length_2, length_3 (int tuple): 4 longueurs d'échantillons
    """
    length_0 = length // 256**3
    length_1 = (length - length_0 * 256**3) // 256**2
    length_2 = (length - length_0 * 256**3 - length_1 * 256**2) // 256**1
    length_3 = length - length_0 * 256**3 - length_1 * 256**2 - length_2 * 256
    return length_0, length_1, length_2, length_3


def plot_mics(mic_coords, highlight_index=None):
    """
    Trace la position des microphones dans l'espace selon
    les coordonnées fournies. Peut mettre en relief un micro
    correspondant à l'indice fourni.

    Parameters:
    mic_coords (np.ndarray): coordonnées des microphones dans la grille.
    highlight_index (int) = None: indice du micro à mettre en valeur

    Returns: None
    """
    if highlight_index is not None:
        for i in range(mic_coords.shape[0]):
            if i != highlight_index:
                plt.plot(mic_coords[i, 0], mic_coords[i, 1], 'ko')
            else:
                plt.plot(mic_coords[i, 0], mic_coords[i, 1], 'ro')
    else:
        plt.plot(mic_coords[:, 0], mic_coords[:, 1], 'ko')
    plt.xlabel('Position X (m)')
    plt.ylabel('Position Y (m)')
    plt.title('Positions des microphones')
    plt.axis('equal')


def verification_position(original_test_recording, arranged_test_recording, mic_coords):
    """
    Vérifie l'ordre des micros après réarrangement et met en relief le 
    micro au signal le plus élevé (en rouge). Si cela est correct,
    le réarrangement est correct vis-à-vis des coordonnées entrées.

    Parameters:
    original_test_recording (np.ndarray): enregistrement brut.
    arranged_test_recording (np.ndarray): enregistrement ordonné.
    mic_coords (np.ndarray): coordonnées des microphones dans la grille.

    Returns: None
    """
    original_array = np.mean(np.abs(original_test_recording), axis=1)
    arranged_array = np.mean(np.abs(arranged_test_recording), axis=1)
    max_index_original = np.argmax(original_array)
    max_index_arranged = np.argmax(arranged_array)
    plot_mics(mic_coords, max_index_arranged)
    print(f"Detected mic index: {max_index_original} -> {max_index_arranged}")
    plt.show()
