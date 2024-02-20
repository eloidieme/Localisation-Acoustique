import math
import numpy as np
from xml.etree import ElementTree as ET

NOMBRE_MICROS = 32

HAUTEUR_SOURCES = 2  # m
LONGUEUR_SOURCES = 2  # m
PAS_SOURCES = 0.03  # m

HAUTEUR_MICROS = 2  # m
LONGUEUR_MICROS = 1.36  # m
PAS_MICROS = 8.5e-2  # m
LIGNE_MICROS_HORIZONTAUX = 8
COLONNE_MICROS_VERTICAUX = 0

DISTANCE_SOURCES_MICROS = 2  # m


def array_manifold(x_source, y_source, mic_coordinates, source_to_mic_distance, signal_frequency, celerity):
    # Regarder l'article de Gilles Chardon; normaliser chaque vecteur colonne indépendamment avant d'assembler la matrice.

    L = []
    c = celerity
    k = 2*np.pi*signal_frequency/c

    for coordmicro in mic_coordinates:
        x_micro = coordmicro[0]
        y_micro = coordmicro[1]
        rn = math.sqrt((x_source - x_micro)**2 + (y_source -
                       y_micro)**2 + source_to_mic_distance**2)
        L.append(1/rn * np.exp(- k * rn * 1j))
    return np.array(L)


def define_grille_source(hauteur, longueur, pas):
    """
    Crée la grille contenant les sources sonores.

    Parameters:
    hauteur (float): hauteur de la grille
    longueur (float): longueur de la grille
    pas (float): pas de la grille / définit la résolution de localisation

    Returns:
    np.ndarray: matrice remplie de 1 et aux dimensions correspondant aux entrées.
    """
    return np.ones((math.floor(hauteur/pas) + 1, math.floor(longueur/pas) + 1))


def define_grille_micros(hauteur, longueur, pas, ligne_micros_horizontaux, colonne_micros_verticaux):
    """
    Crée la grille contenant les microphones, selon notre géométrie.

    Parameters:
    hauteur (float): hauteur de la grille
    longueur (float): longueur de la grille
    pas (float): pas de la grille / définit la résolution de localisation
    ligne_micros_horizontaux (int): indice de la ligne contenant les micros disposés horizontalement
    colonne_micros_verticaux (int): indice de la colonne contenant les micros disposés verticalement

    Returns:
    np.ndarray: matrice contenant des 1 où les micros se trouvent
    et aux dimensions correspondant aux entrées.
    """
    grille = np.zeros((math.floor(hauteur/pas) + 1,
                      math.floor(longueur/pas) + 1))
    for j in range(17):
        grille[ligne_micros_horizontaux][j] = 1
    for i in range(16):
        grille[i][colonne_micros_verticaux] = 1
    return grille


def define_grille_micros_xml(hauteur, longueur, pas, path):
    """
    Crée la grille contenant les microphones, selon notre géométrie, à partir d'un fichier XML.

    Parameters:
    hauteur (float): hauteur de la grille
    longueur (float): longueur de la grille
    pas (float): pas de la grille / définit la résolution de localisation
    path (str): chemin vers le fichier XML

    Returns:
    np.ndarray: matrice contenant des 1 où les micros se trouvent 
    et aux dimensions correspondant aux entrées.
    """
    grille = np.zeros((math.floor(hauteur/pas) + 1,
                      math.floor(longueur/pas) + 1))
    coords = coordonnees_micros_xml(path)
    for x, y in coords:
        grille[round(y/pas)][round(x/pas)] = 1
    return grille


def coordonnees_micros(pas, ligne_micros_horizontaux, colonne_micros_verticaux):
    """
    Retourne les coordonnées des microphones dans l'espace.

    Parameters:
    pas (float): pas de la grille / définit la résolution de localisation
    ligne_micros_horizontaux (int): indice de la ligne contenant les micros disposés horizontalement
    colonne_micros_verticaux (int): indice de la colonne contenant les micros disposés verticalement

    Returns:
    np.ndarray: vecteur de couples contenant les coordonnées des micros.
    """
    coordonnees = []
    for j in range(1, 17):
        coordonnees.append((j * pas, ligne_micros_horizontaux*pas))
    for i in range(16):
        coordonnees.append((colonne_micros_verticaux*pas, i*pas))
    return np.array(coordonnees)


def coordonnees_micros_xml(path):
    """
    Retourne les coordonnées des microphones dans l'espace, à partir d'un fichier XML.

    Parameters:
    path (str): chemin vers le fichier XML

    Returns:
    np.ndarray: vecteur de couples contenant les coordonnées des micros.
    """
    with open(path, 'r', encoding='utf-8') as file:
        tree = ET.parse(file)
        root = tree.getroot()

    coords = [(float(pos.attrib['x']), float(pos.attrib['y']))
              for pos in root.findall('.//pos')]

    return np.array(coords)


def coordonnees_micros_xml_sorted(path):
    """
    Retourne les coordonnées des microphones dans l'espace, à partir d'un fichier XML et dans l'ordre des points.

    Parameters:
    path (str): chemin vers le fichier XML

    Returns:
    np.ndarray: vecteur de couples contenant les coordonnées des micros.
    """
    with open(path, 'r', encoding='utf-8') as file:
        tree = ET.parse(file)
        root = tree.getroot()

    points = [(int(pos.attrib['Name'].split()[1]), (float(pos.attrib['x']), float(
        pos.attrib['y']))) for pos in root.findall('.//pos')]

    points.sort(key=lambda x: x[0])

    return np.array([coord for _, coord in points])


def coordonnees_sources(grille_source, pas):
    """
    Retourne les coordonnées des sources dans l'espace.

    Parameters:
    pas (float): pas de la grille / définit la résolution de localisation
    grille_source (np.ndarray): matrice remplie de 1 et aux dimensions correspondant aux entrées.

    Returns:
    np.ndarray: vecteur de couples contenant les coordonnées des sources.
    """
    coordonnees = []
    for i in range(len(grille_source)):
        for j in range(len(grille_source[0])):
            coordonnees.append((j * pas, i * pas))
    return np.array(coordonnees)


# def ordre_micros(matrice_acquisition):
#     """
#     Remets dans l'ordre les microphones en échangeant les colonnes
#     de la matrice d'acquisition.

#     Parameters:
#     matrice_acquisition (np.ndarray): matrice d'acquisition.

#     Returns:
#     np.ndarray: matrice réarrangée.
#     """
#     matrice_arangee = np.zeros(matrice_acquisition.shape, dtype="complex_")
#     # De 0 à 7
#     matrice_arangee[:, 8] = matrice_acquisition[:, 0]
#     matrice_arangee[:, 9] = matrice_acquisition[:, 1]
#     matrice_arangee[:, 10] = matrice_acquisition[:, 2]
#     matrice_arangee[:, 11] = matrice_acquisition[:, 3]
#     matrice_arangee[:, 12] = matrice_acquisition[:, 4]
#     matrice_arangee[:, 13] = matrice_acquisition[:, 5]
#     matrice_arangee[:, 14] = matrice_acquisition[:, 6]
#     matrice_arangee[:, 15] = matrice_acquisition[:, 7]

#     # De 8 à 15
#     matrice_arangee[:, 7] = matrice_acquisition[:, 8]
#     matrice_arangee[:, 6] = matrice_acquisition[:, 9]
#     matrice_arangee[:, 5] = matrice_acquisition[:, 10]
#     matrice_arangee[:, 4] = matrice_acquisition[:, 11]
#     matrice_arangee[:, 3] = matrice_acquisition[:, 12]
#     matrice_arangee[:, 2] = matrice_acquisition[:, 13]
#     matrice_arangee[:, 1] = matrice_acquisition[:, 14]
#     matrice_arangee[:, 0] = matrice_acquisition[:, 15]
#     return matrice_arangee

def ordre_micros(matrice_acquisition):
    """
    Remets dans l'ordre les microphones en échangeant les colonnes 
    de la matrice d'acquisition (moins élégant mais fonctionnel)

    Parameters:
    matrice_acquisition (np.ndarray): pas de la grille / définit la résolution de localisation

    Returns:
    np.ndarray: vecteur de couples contenant les coordonnées des micros.
    """
    matrice_arangee = np.zeros(matrice_acquisition.shape, dtype="complex_")
# De 0 à 7
    matrice_arangee[:, 0] = matrice_acquisition[:, 0]
    matrice_arangee[:, 2] = matrice_acquisition[:, 1]
    matrice_arangee[:, 4] = matrice_acquisition[:, 2]
    matrice_arangee[:, 6] = matrice_acquisition[:, 3]
    matrice_arangee[:, 8] = matrice_acquisition[:, 4]
    matrice_arangee[:, 10] = matrice_acquisition[:, 5]
    matrice_arangee[:, 12] = matrice_acquisition[:, 6]
    matrice_arangee[:, 14] = matrice_acquisition[:, 7]

    # De 8 à 15
    matrice_arangee[:, 1] = matrice_acquisition[:, 8]
    matrice_arangee[:, 3] = matrice_acquisition[:, 9]
    matrice_arangee[:, 5] = matrice_acquisition[:, 10]
    matrice_arangee[:, 7] = matrice_acquisition[:, 11]
    matrice_arangee[:, 9] = matrice_acquisition[:, 12]
    matrice_arangee[:, 11] = matrice_acquisition[:, 13]
    matrice_arangee[:, 13] = matrice_acquisition[:, 14]
    matrice_arangee[:, 15] = matrice_acquisition[:, 15]

    # De 16 à 23
    matrice_arangee[:, 16] = matrice_acquisition[:, 16]
    matrice_arangee[:, 18] = matrice_acquisition[:, 17]
    matrice_arangee[:, 20] = matrice_acquisition[:, 18]
    matrice_arangee[:, 22] = matrice_acquisition[:, 19]
    matrice_arangee[:, 24] = matrice_acquisition[:, 20]
    matrice_arangee[:, 26] = matrice_acquisition[:, 21]
    matrice_arangee[:, 28] = matrice_acquisition[:, 22]
    matrice_arangee[:, 30] = matrice_acquisition[:, 23]

    # De 24 à 31
    matrice_arangee[:, 17] = matrice_acquisition[:, 24]
    matrice_arangee[:, 19] = matrice_acquisition[:, 25]
    matrice_arangee[:, 21] = matrice_acquisition[:, 26]
    matrice_arangee[:, 23] = matrice_acquisition[:, 27]
    matrice_arangee[:, 25] = matrice_acquisition[:, 28]
    matrice_arangee[:, 27] = matrice_acquisition[:, 29]
    matrice_arangee[:, 29] = matrice_acquisition[:, 30]
    matrice_arangee[:, 31] = matrice_acquisition[:, 31]

    return matrice_arangee
