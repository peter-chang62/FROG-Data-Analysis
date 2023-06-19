# %% -----
import matplotlib.pyplot as plt
import numpy as np
import pynlo_extras as pe
import clipboard_and_style_sheet as cr

# %% -----
path = r"/Volumes/Peter SSD/Research_Projects/FROG/Data/06-16-2023_PC_UBFS/"

ret = pe.python_phase_retrieval.Retrieval()

xlims = -500, 500
ylims = 692, 859
fig, axs = plt.subplots(1, 4, figsize=np.array([11.32, 4.8]))

# %% -----
ret.load_data(path + "HNLF_input.txt")
axs[0].pcolormesh(ret.T_fs, ret.wl_nm, ret.spectrogram.T, cmap="gnuplot2_r")

# %% -----
ret.load_data(path + "6A_HNLF_output.txt")
axs[1].pcolormesh(ret.T_fs, ret.wl_nm, ret.spectrogram.T, cmap="gnuplot2_r")

# %% -----
ret.load_data(path + "7A_HNLF_output.txt")
axs[2].pcolormesh(ret.T_fs, ret.wl_nm, ret.spectrogram.T, cmap="gnuplot2_r")

# %% -----
ret.load_data(path + "8A_HNLF_output.txt")
axs[3].pcolormesh(ret.T_fs, ret.wl_nm, ret.spectrogram.T, cmap="gnuplot2_r")

# %% -----
[i.set_ylim(*ylims) for i in axs]
[i.set_xlim(*xlims) for i in axs]
[i.set_xlabel("T (fs)") for i in axs]
axs[0].set_ylabel("wavelength (nm)")

fig.tight_layout()
