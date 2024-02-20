import numpy as np


def beamforming(data, array_manifold, signal_frequency, mic_coordinates, ordre_micros, source_to_mic_distance, grille_sources, nsamples, pas_sources, celerity):
    data = ordre_micros(data).T
    l1 = grille_sources.shape[0]
    l2 = grille_sources.shape[1]

    R = 1/(nsamples)*np.dot(data, data.conj().T)
    Csm = np.zeros((l1, l2))

    for x in range(l1):
        for y in range(l2):
            a = array_manifold(y * pas_sources, x * pas_sources, mic_coordinates,
                               source_to_mic_distance, signal_frequency, celerity)
            Csm[x, y] = (np.abs(np.dot(np.dot(a.conj().T, R), a)))

    pp = np.zeros(l2)
    for k in range(1, l2):
        pp[k] = np.max(Csm[k, :])  # le plus grand element de la ligne k

    Csm = Csm/np.max(pp)  # normalisation
    return Csm


def broadband_beamforming(data, array_manifold, mic_coordinates, ordre_micros, source_to_mic_distance, grille_sources, nsamples, pas_sources, celerity, freq_range):
    data = ordre_micros(data).T
    l1 = grille_sources.shape[0]
    l2 = grille_sources.shape[1]

    R = 1/(nsamples)*np.dot(data, data.conj().T)
    Csm = np.zeros((l1, l2))

    for freq in freq_range:
        for x in range(l1):
            for y in range(l2):
                a = array_manifold(y * pas_sources, x * pas_sources, mic_coordinates,
                                   source_to_mic_distance, freq, celerity)
                Csm[x, y] += (np.abs(np.dot(np.dot(a.conj().T, R), a)))

    # Normalize for the number of frequencies
    Csm /= len(freq_range)

    pp = np.zeros(l2)
    for k in range(1, l2):
        pp[k] = np.max(Csm[k, :])  # le plus grand element de la ligne k

    Csm = Csm/np.max(pp)  # normalisation
    return Csm
