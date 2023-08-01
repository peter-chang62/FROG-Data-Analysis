"""This one was too ambitious, the retrieval is not good!"""

# %% -----
import matplotlib.pyplot as plt
import numpy as np
import shg_frog
import clipboard
from scipy.constants import c

# %% --------------------------------------------------------------------------
path = r"G:\\Research_Projects\\FROG\\Data\\06-16-2023_PC_UBFS/"

ret = shg_frog.python_phase_retrieval.Retrieval()
ret.load_data(path + "HNLF_input.txt")
ret.spectrogram[:] += ret.spectrogram[::-1]
ret.spectrogram[:] -= ret.spectrogram[0]
threshold = ret.spectrogram[1].mean() + ret.spectrogram[1].std()
ret.spectrogram[:] = np.where(ret.spectrogram < threshold, 0, ret.spectrogram)

# %% --------------------------------------------------------------------------
# fig, ax = plt.subplots(1, 1)
# ax.pcolormesh(ret.T_fs, ret.wl_nm, ret.spectrogram.T, cmap="gnuplot2_r")

# fig, ax = plt.subplots(1, 1)
# ax.plot(ret.spectrogram[ret.spectrogram.shape[0] // 2] ** 0.5)

# fig, ax = plt.subplots(1, 1)
# ax.plot(ret.F_THz, ret.spectrogram[ret.spectrogram.shape[0] // 2] ** 0.5)

# %% --------------------------------------------------------------------------
ll, ul = 1096, 1237
ret.spectrogram[:, :ll] = 0
ret.spectrogram[:, ul:] = 0

ll_thz, ul_thz = 362, 408
ret.set_signal_freq(ll_thz, ul_thz)
ret.correct_for_phase_matching()

# %% --------------------------------------------------------------------------
ret.set_initial_guess(
    wl_min_nm=c * 1e9 * 2 / (ul_thz * 1e12),
    wl_max_nm=c * 1e9 * 2 / (ll_thz * 1e12),
    center_wavelength_nm=1560,
    time_window_ps=20,
    NPTS=2**8,
)

# %% --------------------------------------------------------------------------
t = 100
ret.retrieve(
    -t,
    t,
    50,
    iter_set=None,
    plot_update=1,
)
fig, ax = ret.plot_results()
