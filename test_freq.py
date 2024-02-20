import numpy as np
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt

from algos.yin import detect_pitch_yin, CMDF
from algos.stft_fundamental_frequency import detect_pitch_stft

SAMPLING_FREQ = 50000

t = np.arange(0, 0.5, 1/SAMPLING_FREQ)
x = np.sin(2*np.pi*500*t)  # + np.random.uniform(-0.2, 0.2, size=len(t))
data = np.array([x]*32)

freq = detect_pitch_yin(x, bounds=[0, 500], tresh=0.01)
freq2 = detect_pitch_stft(data)

fig, axs = plt.subplots(1, 2)

axs[0].plot(t[:400], x[:400])
axs[1].plot([i for i in range(0, 200)], [CMDF(x, i)
                                         for i in range(0, 200)], 'o')

plt.suptitle('Yin')
axs[0].legend([str(freq2)])
plt.show()
