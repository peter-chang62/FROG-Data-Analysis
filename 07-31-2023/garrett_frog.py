# %% -----
import numpy as np
import shg_frog
import matplotlib.pyplot as plt
import clipboard

# %% --------------------------------------------------------------------------
ret = shg_frog.python_phase_retrieval.Retrieval()
ret.load_data("data/run15.txt")
ret.spectrogram[:] -= ret.spectrogram[0]
ret.spectrogram[:] = np.where(ret.spectrogram < 0, 0, ret.spectrogram)

ret.spectrogram[:] += ret.spectrogram[::-1]

# %% --------------------------------------------------------------------------
spec = np.genfromtxt(
    "data/20230726_50cm_5.5mW_nm.csv", skip_header=76, skip_footer=1, delimiter=";"
)
spec[:, 1] = 10 ** (spec[:, 1] / 10)
spec[:, 0] *= 1e-3

# %% --------------------------------------------------------------------------
# fig, ax = plt.subplots(1, 1)
# ax.pcolormesh(ret.T_fs, ret.wl_nm, ret.spectrogram.T, cmap="gnuplot2_r")

# fig, ax = plt.subplots(1, 1)
# ax.plot(ret.spectrogram[ret.spectrogram.shape[0] // 2] ** 0.5)

# fig, ax = plt.subplots(1, 1)
# ax.plot(ret.F_THz, ret.spectrogram[ret.spectrogram.shape[0] // 2] ** 0.5)

# %% --------------------------------------------------------------------------
ll, ul = 471, 1630
ret.spectrogram[:, :ll] = 0
ret.spectrogram[:, ul:] = 0

threshold = ret.spectrogram[1].mean() - ret.spectrogram[1].std()
ret.spectrogram[:] = np.where(ret.spectrogram < threshold, 0, ret.spectrogram)

ret.set_signal_freq(281, 616)
ret.correct_for_phase_matching()

# %% --------------------------------------------------------------------------
ret.set_initial_guess(
    wl_min_nm=1000.0,
    wl_max_nm=2000.0,
    center_wavelength_nm=1550,
    time_window_ps=5.5,
    NPTS=2**10,
)

# %% --------------------------------------------------------------------------
ret.load_spectrum_data(spec[:, 0], spec[:, 1])
t = 925
ret.retrieve(
    -t,
    t,
    50,
    iter_set=None,
    plot_update=True,
)

# %% --------------------------------------------------------------------------
fig, ax = ret.plot_results()
# ax[2].set_xlim(-150, 150)
# ax[3].set_xlim(-150, 150)

# %% --------------------------------------------------------------------------
s = shg_frog.python_phase_retrieval.calculate_spectrogram(
    ret.pulse_data, ret.T_fs * 1e-15
)
fig, ax = plt.subplots(1, 1)
ax.pcolormesh(ret.T_fs, ret.pulse_data.v_grid * 1e-12, s.T, cmap="gnuplot2_r")
