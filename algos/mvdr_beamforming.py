import numpy as np


def mvdr(data, array_manifold, signal_frequency, mic_coordinates, ordre_micros, source_to_mic_distance, grille_sources, nsamples, pas_sources, celerity):
    data = ordre_micros(data).T
    l1 = grille_sources.shape[0]
    l2 = grille_sources.shape[1]
    alpha = 1e-9

    R = np.cov(data) + alpha*np.eye(data.shape[0], data.shape[0])
    Csm = np.zeros((l1, l2))

    Rinv = np.linalg.inv(R)
    for x in range(l1):
        for y in range(l2):
            a = array_manifold(y * pas_sources, x * pas_sources, mic_coordinates,
                               source_to_mic_distance, signal_frequency, celerity)
            Csm[x, y] = 1/(np.abs(np.dot(np.dot(a.conj().T, Rinv), a)))

    pp = np.zeros(l2)
    for k in range(1, l2):
        pp[k] = np.max(Csm[k, :])  # le plus grand element de la ligne k

    Csm = Csm/np.max(pp)  # normalisation
    return Csm
