import numpy as np


def music(data, array_manifold, signal_frequency, mic_coordinates, ordre_micros, source_to_mic_distance, grille_sources, nsamples, pas_sources, celerity):
    data = ordre_micros(data).T
    l1 = grille_sources.shape[0]
    l2 = grille_sources.shape[1]

    R = 1/(nsamples)*np.dot(data, data.conj().T)
    Csm = np.zeros((l1, l2))

    # Décomposition en valeurs propres de R
    eigenvalues, eigenvectors = np.linalg.eigh(R)

    q = 5  # Nombre de sources (à ajuster)

    # Séparation des sous-espaces propres de bruit
    noise_eigenvectors = eigenvectors[:, :-q]

    for x in range(l1):
        for y in range(l2):
            a = array_manifold(y * pas_sources, x * pas_sources, mic_coordinates,
                               source_to_mic_distance, signal_frequency, celerity)

            # Calcul du pseudo-spectre
            numerator = np.dot(a.conj().T, a)
            denominator = np.dot(a.conj().T, np.dot(
                noise_eigenvectors, noise_eigenvectors.conj().T).dot(a))

            # Cas d'un dénominateur proche de zéro
            if np.abs(denominator) > 1e-10:
                Csm[x, y] = np.abs(numerator / denominator)
            else:
                Csm[x, y] = np.inf

    pp = np.zeros(l2)
    for k in range(1, l2):
        pp[k] = np.max(Csm[k, :])

    Csm = Csm/np.max(pp)  # normalisation
    return Csm
