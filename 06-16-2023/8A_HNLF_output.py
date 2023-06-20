# %% -----
import matplotlib.pyplot as plt
import numpy as np
import pynlo_extras as pe
import clipboard_and_style_sheet as cr

# %% -----
# path = r"/Volumes/Peter SSD/Research_Projects/FROG/Data/06-16-2023_PC_UBFS/"
path = r"/media/peterchang/Peter SSD/Research_Projects/FROG/Data/06-16-2023_PC_UBFS/"

ret = pe.python_phase_retrieval.Retrieval()
ret.load_data(path + "8A_HNLF_output.txt")
ret.spectrogram[:] -= ret.spectrogram[0]
# threshold = ret.spectrogram[1].mean() + ret.spectrogram[1].std()
threshold = ret.spectrogram[1].max()
ret.spectrogram[:] = np.where(ret.spectrogram < threshold, 0, ret.spectrogram)

# %% -----
# plt.figure()
# plt.plot(ret.spectrogram[ret.spectrogram.shape[0] // 2] ** 0.5)

# %% -----
ll, ul = 678, 1427
ret.spectrogram[:, :ll] = 0
ret.spectrogram[:, ul:] = 0

ret.set_signal_freq(300, 520)
ret.correct_for_phase_matching()

# %% -----
ret.set_initial_guess(
    wl_min_nm=1000,
    wl_max_nm=2000,
    center_wavelength_nm=1560,
    time_window_ps=3,
    NPTS=2**8,
)

# %% -----
# for i in range(25, 1000, 25):
#     ret.set_initial_guess(
#         wl_min_nm=1000,
#         wl_max_nm=2000,
#         center_wavelength_nm=1560,
#         time_window_ps=3,
#         NPTS=2**8,
#     )

#     ret.retrieve(
#         0,
#         i,
#         100,
#         iter_set=None,
#         plot_update=0,
#     )
#     fig, ax = ret.plot_results()
#     fig.suptitle(i)
#     fig.savefig(f"fig_7A_HNLF_output/{i}.png")
#     plt.close(fig)

# %% -----
ret.retrieve(
    0,
    60,
    150,
    iter_set=None,
    plot_update=1,
)
fig, ax = ret.plot_results()
ax[2].set_xlim(-150, 150)
ax[3].set_xlim(-150, 150)
ax[2].set_ylim(176, 214)
ax[3].set_ylim(176, 214)
print(ret.error.min())

# %% -----
fig_wl, ax_wl = plt.subplots(1, 1)
ax_wl.plot(ret.pulse.wl_grid * 1e6, ret.pulse.p_v)
ax_wl.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_wl.set_xlim(1.46, 1.67)
fig_wl.tight_layout()
