from utils.fonctions_utils import nombre_micros
import numpy as np


def tableau(dev, length):
    """
    Fonction réalisant la lecture de l'acquisition des microphones.

    Parameters:
    dev: appareil avec lequel s'effectue l'acquisition
    length (int): longueur de l'échantillon à acquérir

    Returns:
    Z (np.ndarray): matrice contenant l'acquisition brute

    """
    Nmics = nombre_micros()
    print("Nombres de micros active", Nmics)
    Nseg = (4 * length * Nmics) // 1024
    Z = np.zeros([1024, Nseg])

    # transferts
    for n in range(Nseg):
        # lire la doc megamicro
        # envoie la requete pour pouvoir ecouter sur le port usb
        dev.ctrl_transfer(0x40, 0xB1, 0, 0, [0x02])
        # lire le donnees et le mettre dans un tableau numpy
        Z[:, n] = dev.read(0x81, 1024, 1000)
    return Z


def shaping(dev, length, transitoire):
    """
    Une donnée recoltée par le boitier est une valeur binaire entre 0 et 255. Pour pouvoir reconstituer un
    int32 il faut convertir les données binaires en en fusionnant 4 pour obtenir une valeur entière.
    L'utilisation des float32 à la place des int32 n'a pas fonctionné.

    Parameters:
    dev: appareil avec lequel s'effectue l'acquisition
    length (int): longueur de l'échantillon à acquérir
    transitoire (int): nombre d'échantillons à retirer au début de l'acquisition (silence transitoire)

    Returns:
    X (np.ndarray): matrice d'acquisition du signal
    """
    Nmics = nombre_micros()
    Z = tableau(dev, length)
    U = Z.T.ravel()
    # reconstruction des int32 (naif)
    V = U.reshape([length * Nmics, 4])
    W = V[:, 3] * 256**3 + V[:, 2] * 256**2 + V[:, 1] * 256 + V[:, 0]
    W[W > 2**31] = W[W > 2**31] - 2**32
    # normalisation
    W = W / np.max(np.abs(W))
    # restructuration en canaux pour pouvoir etre envoye a la partie traitement
    X = W.reshape([W.shape[0] // Nmics, Nmics])
    # on enlève le début (silence et transitoire)
    X = X[int(transitoire):, :]
    return X


def recording(dev, nsamples, nmics, nsegments, silence_transitoire=20000):
    record = np.zeros([1024, nsegments])

    for n in range(nsegments):
        # lire la doc megamicro
        # envoie la requete pour pouvoir ecouter sur le port usb
        dev.ctrl_transfer(0x40, 0xB1, 0, 0, [0x02])
        # lecture des données
        record[:, n] = dev.read(0x81, 1024, 1000)

    U = record.T.ravel()
    # reconstruction des int32
    V = U.reshape([nsamples * nmics, 4])
    W = V[:, 3] * 256**3 + V[:, 2] * 256**2 + V[:, 1] * 256 + V[:, 0]
    W[W > 2**31] = W[W > 2**31] - 2**32
    # restructuration en canaux pour pouvoir être envoyé à la partie traitement
    R = W.reshape([W.shape[0] // nmics, nmics])
    # on enlève le début (silence et transitoire)
    data = R[int(silence_transitoire):, :]
    return data
