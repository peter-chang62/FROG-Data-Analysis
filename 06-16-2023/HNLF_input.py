# %% -----
import matplotlib.pyplot as plt
import numpy as np
import pynlo_extras as pe
import clipboard_and_style_sheet as cr

# %% -----
path = r"/Volumes/Peter SSD/Research_Projects/FROG/Data/06-16-2023_PC_UBFS/"

ret = pe.python_phase_retrieval.Retrieval()
ret.load_data(path + "HNLF_input.txt")
ret.spectrogram[:] -= ret.spectrogram[0]
threshold = ret.spectrogram[1].mean() + ret.spectrogram[1].std()
ret.spectrogram[:] = np.where(ret.spectrogram < threshold, 0, ret.spectrogram)

# %% -----
ll, ul = 1130, 1190
ret.spectrogram[:, :ll] = 0
ret.spectrogram[:, ul:] = 0

ret.set_signal_freq(375, 395)
ret.correct_for_phase_matching()

# %% -----
ret.set_initial_guess(
    wl_min_nm=1300,
    wl_max_nm=1800,
    center_wavelength_nm=1560,
    time_window_ps=20,
    NPTS=2**10,
)

# %% -----
# it looks like 350 is the limit, past which the
# iterations can cause it to diverge

ret.retrieve(
    0,
    300,
    200,
    iter_set=None,
    plot_update=0,
)
fig, ax = ret.plot_results()
[i.set_xlim(-500, 500) for i in ax[[2, 3]]]
[i.set_ylim(190, 195) for i in ax[[2, 3]]]
print(ret.error.min())
