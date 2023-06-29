# %% -----
import matplotlib.pyplot as plt
import numpy as np
import clipboard as cr
import scipy.constants as sc
import shg_frog as sf
import copy
from tqdm import tqdm
import pynlo

# %% --------------------------------------------------------------------------
path = r"/Volumes/Peter SSD/Research_Projects/FROG/Data/06-16-2023_PC_UBFS/"
# path = r"/media/peterchang/Peter SSD/Research_Projects/FROG/Data/06-16-2023_PC_UBFS/"

ret = sf.python_phase_retrieval.Retrieval()
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
pulse = pynlo.light.Pulse.Sech(
    2**10,
    sc.c / 3000e-9,
    sc.c / 700e-9,
    sc.c / 1560e-9,
    3.5e-9,
    50e-15,
    10e-12,
)
phi_v = ret.pulse.phi_v
pulse.import_p_v(
    ret.pulse.v_grid,
    ret.pulse.p_v,
    phi_v=None,
)

pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550, "slow")
model_pm1550 = pm1550.generate_model(pulse)
dz = model_pm1550.estimate_step_size()
result_pm1550 = model_pm1550.simulate(
    22.0e-2, dz=dz, local_error=1e-6, n_records=100, plot=None
)

# %% -------- propagate through a little bit of ad-hnlf -----------------------
# result_pm1550.pulse_out.e_p = 3.5e-9
hnlf = pynlo.materials.SilicaFiber()
hnlf.load_fiber_from_dict(pynlo.materials.hnlf_5p7, "slow")
model_hnlf = hnlf.generate_model(result_pm1550.pulse_out)
dz = model_hnlf.estimate_step_size()
result_hnlf = model_hnlf.simulate(
    15.0e-2, dz=dz, local_error=1e-6, n_records=100, plot=None
)

# %% --------------------------------------------------------------------------
s_hnlf = np.genfromtxt(path + "Spectrum_Stitched_Together_wl_nm.txt")
pulse_data = pulse.copy()
pulse_data.import_p_v(sc.c / (s_hnlf[:, 0] * 1e-9), s_hnlf[:, 1])

# %% --------------------------------------------------------------------------
result_hnlf.animate("wvl", save=False, p_ref=pulse_data)

# %% --------------------------------------------------------------------------
p = pulse.copy()
T_WIDTH = np.zeros(result_hnlf.z.shape)
for n, a_v in enumerate(tqdm(result_hnlf.a_v)):
    p.a_v = a_v
    t_width = p.t_width()
    T_WIDTH[n] = t_width.eqv
idx = T_WIDTH.argmin()

p.a_v = result_hnlf.a_v[idx]

fig, ax = plt.subplots(1, 1)
ax.plot(p.wl_grid * 1e6, p.p_v * model_hnlf.dv_dl * 1e3)
ax.plot(p.wl_grid * 1e6, pulse_data.p_v * model_hnlf.dv_dl * 1e3)
ax.set_ylabel("mW / nm")
ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")

# %% --------------------------------------------------------------------------
fig, ax = result_hnlf.plot("wvl")
ax[1, 0].axhline(result_hnlf.z[idx] * 1e3, color="k", linestyle="--")
ax[1, 1].axhline(result_hnlf.z[idx] * 1e3, color="k", linestyle="--")
