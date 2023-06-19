# %% -----
import matplotlib.pyplot as plt
import numpy as np
import pynlo_extras as pe
import clipboard_and_style_sheet as cr
import scipy.constants as sc
from pynlo_extras import materials
from pynlo_extras import utilities as util

# %% -----
path = r"/Volumes/Peter SSD/Research_Projects/FROG/Data/06-16-2023_PC_UBFS/"

ret = pe.python_phase_retrieval.Retrieval()
ret.load_data(path + "HNLF_input.txt")
ret.spectrogram[:] -= ret.spectrogram[0]
# threshold = ret.spectrogram[1].mean() + ret.spectrogram[1].std()
threshold = ret.spectrogram[1].max()
ret.spectrogram[:] = np.where(ret.spectrogram < threshold, 0, ret.spectrogram)

# %% -----
# plt.figure()
# plt.plot(ret.spectrogram[ret.spectrogram.shape[0] // 2])

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
# for i in range(25, 1000, 25):
#     ret.set_initial_guess(
#         wl_min_nm=1300,
#         wl_max_nm=1800,
#         center_wavelength_nm=1560,
#         time_window_ps=20,
#         NPTS=2**10,
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
#     fig.savefig(f"fig_HNLF_input/{i}.png")
#     plt.close(fig)

# %% -----
# 350 converges to 300 with less stability
# 400 is unstable
# -> so 300
ret.retrieve(
    0,
    325,
    100,
    iter_set=None,
    plot_update=1,
)
fig, ax = ret.plot_results()
ax[2].set_xlim(-500, 500)
ax[3].set_xlim(-500, 500)
ax[2].set_ylim(190, 195)
ax[3].set_ylim(190, 195)

# %% -----
fig_wl, ax_wl = plt.subplots(1, 1)
ax_wl.plot(ret.pulse.wl_grid * 1e6, ret.pulse.p_v)
ax_wl.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_wl.set_xlim(1.46, 1.67)
fig.tight_layout()

# %% -----
# propagate through HNLF
pulse = pe.light.Pulse.Sech(
    2**10,
    sc.c / 3000e-9,
    sc.c / 700e-9,
    sc.c / 1560e-9,
    5e-9,
    50e-15,
    10e-12,
)
phi_v = ret.pulse.phi_v
threshold = 1e-3 * ret.pulse.p_v.max()
phi_v = np.where(ret.pulse.p_v < threshold, 0, phi_v)
pulse.import_p_v(ret.pulse.v_grid, ret.pulse.p_v, phi_v=phi_v)

pm1550 = materials.Fiber()
pm1550.load_fiber_from_dict(materials.pm1550, "slow")
model_pm1550 = pm1550.generate_model(pulse)
dz = util.estimate_step_size(model_pm1550, local_error=1e-6)
result_pm1550 = model_pm1550.simulate(
    10e-2, dz=dz, local_error=1e-6, n_records=100, plot=None
)

# %% -------- propagate through a little bit of ad-hnlf -----------------------
hnlf = materials.Fiber()
hnlf.load_fiber_from_dict(materials.hnlf_5p7, "slow")
model_hnlf = hnlf.generate_model(result_pm1550.pulse_out)
dz = util.estimate_step_size(model_hnlf, local_error=1e-6)
result_hnlf = model_hnlf.simulate(
    7.0e-2, dz=dz, local_error=1e-6, n_records=100, plot=None
)

# %% -----
result_hnlf.plot("wvl")

# %% ----- set pulse a_v to ad-hnlf output ------------------------------------
# pulse_ppln = result_hnlf.pulse_out.clone_pulse(result_hnlf.pulse_out)
# ind_z = np.argmin(abs(result_hnlf.z - 8.88e-3))
# a_v = result_hnlf.a_v[ind_z]
# pulse_ppln.import_p_v(result_hnlf.pulse_out.v_grid, abs(a_v) ** 2, phi_v=np.angle(a_v))

# # %% ----- propagate through ppln ---------------------------------------------
# a_eff = np.pi * 15.0e-6**2
# length = 1e-3
# polling_period = 30.0e-6
# ppln = materials.PPLN()
# model = ppln.generate_model(
#     pulse_ppln,
#     a_eff,
#     length,
#     polling_period=polling_period,
#     is_gaussian_beam=True,
# )

# dz = util.estimate_step_size(model, local_error=1e-6)
# z_grid = util.z_grid_from_polling_period(polling_period, length)
# result_ppln = model.simulate(z_grid, dz=dz, local_error=1e-6, n_records=100)
