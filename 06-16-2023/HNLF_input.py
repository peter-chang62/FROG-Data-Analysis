# %% -----
import matplotlib.pyplot as plt
import numpy as np
import pynlo_extras as pe
import clipboard as cr
import scipy.constants as sc
from pynlo_extras import materials
from pynlo_extras import utilities as util
import copy
from tqdm import tqdm

# %% --------------------------------------------------------------------------
# path = r"/Volumes/Peter SSD/Research_Projects/FROG/Data/06-16-2023_PC_UBFS/"
path = r"/media/peterchang/Peter SSD/Research_Projects/FROG/Data/06-16-2023_PC_UBFS/"

ret = pe.python_phase_retrieval.Retrieval()
ret.load_data(path + "HNLF_input.txt")
ret.spectrogram[:] -= ret.spectrogram[0]
# threshold = ret.spectrogram[1].mean() + ret.spectrogram[1].std()
threshold = ret.spectrogram[1].max()
ret.spectrogram[:] = np.where(ret.spectrogram < threshold, 0, ret.spectrogram)

# %% --------------------------------------------------------------------------
# plt.figure()
# plt.plot(ret.spectrogram[ret.spectrogram.shape[0] // 2] ** 0.5)

# %% --------------------------------------------------------------------------
ll, ul = 1125, 1196
ret.spectrogram[:, :ll] = 0
ret.spectrogram[:, ul:] = 0

ret.set_signal_freq(360, 410)
ret.correct_for_phase_matching()

# %% --------------------------------------------------------------------------
# for i in range(25, 1000, 25):
#     ret.set_initial_guess(
#         wl_min_nm=1300,
#         wl_max_nm=1800,
#         center_wavelength_nm=1560,
#         time_window_ps=20,
#         NPTS=2**10,
#     )
#     ret.retrieve(
#         -i,
#         i,
#         100,
#         iter_set=None,
#         plot_update=0,
#     )
#     fig, ax = ret.plot_results()
#     fig.suptitle(i)
#     fig.savefig(f"fig_HNLF_input/{i}.png")
#     plt.close(fig)

# %% --------------------------------------------------------------------------
# 350 converges to 300 with less stability
# 400 is unstable
# -> so 300
ret.set_initial_guess(
    wl_min_nm=1300,
    wl_max_nm=1800,
    center_wavelength_nm=1560,
    time_window_ps=20,
    NPTS=2**8,
)

ret.retrieve(
    -630,
    630,
    25,
    iter_set=None,
    plot_update=1,
)
fig, ax = ret.plot_results()
ax[2].set_xlim(-1000, 1000)
ax[3].set_xlim(-1000, 1000)
ax[2].set_ylim(190, 195)
ax[3].set_ylim(190, 195)

# %% --------------------------------------------------------------------------
fig_wl, ax_wl = plt.subplots(1, 1)
ax_wl.plot(ret.pulse.wl_grid * 1e6, ret.pulse.p_v)
ax_wl.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_wl.set_xlim(1.46, 1.67)
fig.tight_layout()

# %% --------------------------------------------------------------------------
# propagate through HNLF
pulse = pe.light.Pulse.Sech(
    2**10,
    sc.c / 3000e-9,
    sc.c / 700e-9,
    sc.c / 1560e-9,
    3.5e-9,
    50e-15,
    10e-12,
)
phi_v = ret.pulse.phi_v
threshold = 1e-3 * ret.pulse.p_v.max()
pulse.import_p_v(
    ret.pulse.v_grid,
    ret.pulse.p_v,
    phi_v=None,
)

pm1550 = materials.Fiber()
pm1550.load_fiber_from_dict(materials.pm1550, "slow")
model_pm1550 = pm1550.generate_model(pulse)
dz = util.estimate_step_size(model_pm1550, local_error=1e-6)
result_pm1550 = model_pm1550.simulate(
    22.0e-2, dz=dz, local_error=1e-6, n_records=100, plot=None
)

# %% -------- propagate through a little bit of ad-hnlf -----------------------
# result_pm1550.pulse_out.e_p = 3.5e-9
hnlf = materials.Fiber()
hnlf.load_fiber_from_dict(materials.hnlf_5p7, "slow")
model_hnlf = hnlf.generate_model(result_pm1550.pulse_out)
dz = util.estimate_step_size(model_hnlf, local_error=1e-6)
result_hnlf = model_hnlf.simulate(
    15.0e-2, dz=dz, local_error=1e-6, n_records=100, plot=None
)

# %% --------------------------------------------------------------------------
result_hnlf.plot("wvl")

# %% --------------------------------------------------------------------------
s_hnlf = np.genfromtxt(path + "Spectrum_Stitched_Together_wl_nm.txt")
pulse_data = copy.deepcopy(pulse)
pulse_data.import_p_v(sc.c / (s_hnlf[:, 0] * 1e-9), s_hnlf[:, 1])

# %% --------------------------------------------------------------------------
# the "start from dispersionless power spectrum" approach doesn't really do
# better than the FROG, the dispersionless one is "too good" the FROG here
# fails to capture the dispersive wave at 1 um
(ind,) = np.logical_and(pulse.wl_grid > 0.8e-6, pulse.wl_grid < 3e-6).nonzero()
fig, ax = plt.subplots(1, 1)
save = True
for n, s in enumerate(tqdm(result_hnlf.p_v)):
    ax.clear()
    ax.semilogy(pulse.wl_grid[ind] * 1e6, s[ind], label="simulated")
    ax.semilogy(pulse.wl_grid[ind] * 1e6, pulse_data.p_v[ind], label="experimental")
    ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
    ax.legend(loc="best")
    ax.set_xlim(0.74, 2.33)
    ax.set_ylim(2.93e-27, 3.21e-21)
    fig.tight_layout()
    if save:
        plt.savefig(f"../fig/{n}.png", transparent=True)
    else:
        plt.pause(0.05)
