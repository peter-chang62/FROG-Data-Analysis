# -*- coding: utf-8 -*-

# %% Imports
import numpy as np
from scipy.constants import c
from matplotlib import pyplot as plt
import pynlo
import clipboard
from numpy.fft import fftshift, ifftshift
import shg_frog
import blit
import jax


def fft(x, axis=-1, fsc=1.0):
    return fftshift(np.fft.fft(ifftshift(x, axes=axis), axis=axis), axes=axis) * fsc


def ifft(x, axis=-1, fsc=1.0):
    return fftshift(np.fft.ifft(ifftshift(x, axes=axis), axis=axis), axes=axis) / fsc


v_min = c / 2200e-9
v_max = c / 1090e-9
v0 = c / 1550e-9
e_p = 1e-9
t_fwhm = 200e-15
time_window = 10e-12
pulse = pynlo.light.Pulse.Sech(
    n=256,
    v_min=v_min,
    v_max=v_max,
    v0=v0,
    e_p=e_p,
    t_fwhm=t_fwhm,
    min_time_window=time_window,
)
pulse.chirp_pulse_W(0, 2e-39)

hnlf_info = pynlo.materials.hnlf_5p7
hnlf = pynlo.materials.SilicaFiber()
hnlf.load_fiber_from_dict(hnlf_info)

model = hnlf.generate_model(pulse=pulse)
dz = model.estimate_step_size()
sim = model.simulate(0.18, dz=dz, n_records=100, plot=None)
sim.pulse_out.e_p = e_p

# %% ----- calculate spectrogram
pulse_pr = shg_frog.light.Pulse.Sech(
    n_points=256,
    v_min=142e12,
    v_max=242e12,
    v0=pulse.v_grid[sim.pulse_out.p_v.argmax()],
    # v0=v0,
    e_p=e_p,
    t_fwhm=t_fwhm,
    time_window=1.5e-12,
)
pulse_guess = pulse_pr.clone_pulse(pulse_pr)
phase = np.random.uniform(low=0, high=1, size=pulse_guess.n) * np.pi / 8
pulse_guess.a_t[:] = pulse_guess.a_t * np.exp(1j * phase)

pulse_pr.import_p_v(
    pulse.v_grid,
    sim.pulse_out.p_v,
    phi_v=np.unwrap(sim.pulse_out.phi_v),
)
# pulse_guess.import_p_v(
#     pulse.v_grid,
#     sim.pulse_out.p_v,
#     phi_v=np.unwrap(sim.pulse_out.phi_v),
# )

o = pulse_pr.a_t * np.c_[pulse_pr.a_t]
o_rs = np.zeros(o.shape, dtype=complex)
for r in range(o.shape[0]):
    o_rs[r] = np.roll(o[r], -r)
s_t = fftshift(o_rs, axes=1)
s_v = fft(s_t, axis=0, fsc=pulse_pr.dt)

spectrogram = abs(s_v) ** 2

# %% ----- setup figures
fig, ax = plt.subplots(1, 3, figsize=np.array([9.08, 4.82]))
ax[0].plot(
    pulse_pr.v_grid * 1e-12,
    pulse_pr.p_v / pulse_pr.p_v.max(),
    "--",
    linewidth=2,
)
ax[1].plot(
    pulse_pr.t_grid * 1e12,
    pulse_pr.p_t / pulse_pr.p_t.max(),
    "--",
    linewidth=2,
)
(l_v,) = ax[0].plot(
    pulse_guess.v_grid * 1e-12,
    pulse_guess.p_v / pulse_guess.p_v.max(),
    animated=True,
)
(l_t,) = ax[1].plot(
    pulse_guess.t_grid * 1e12,
    pulse_guess.p_t / pulse_guess.p_t.max(),
    animated=True,
)
img = ax[2].imshow(
    np.ones((pulse_guess.n,) * 2),
    cmap="RdBu_r",
    vmin=0,
    vmax=1,
    animated=True,
)
ax[2].axis(False)
ax[0].set_xlabel("frequency (THz)")
ax[0].set_ylabel("Power (a.u.)")
ax[1].set_xlabel("time (ps)")
ax[1].set_ylabel("Power (a.u.)")
fig.tight_layout()
bm = blit.BlitManager(fig.canvas, [l_v, l_t, img])
bm.update()

# $$ ----- phase retrieval!!
loop_count = 0
iter_limit = 300
o_rs = np.zeros((pulse_guess.n,) * 2, dtype=complex)
error = np.zeros(iter_limit)
_denom_error = np.sum(spectrogram)

while loop_count < iter_limit:
    o = np.c_[pulse_guess.a_t] * pulse_guess.a_t
    for r in range(o.shape[0]):
        o_rs[r] = np.roll(o[r], -r)
    s_t = fftshift(o_rs, axes=1)
    s_v = fft(s_t, axis=0, fsc=pulse_guess.dt)

    error[loop_count] = (
        (abs(s_v) ** 2 - spectrogram) ** 2
    ).mean() ** 0.5 / _denom_error

    img.set_data((abs(s_v) / abs(s_v).max()) ** 2)

    s_v[:] = np.sqrt(spectrogram) * np.exp(1j * np.angle(s_v))

    s_t = ifft(s_v, axis=0, fsc=pulse_guess.dt)
    o_rs = ifftshift(s_t, axes=1)
    for r in range(o_rs.shape[0]):
        o[r] = np.roll(o_rs[r], r)

    u, s, vh = jax.scipy.linalg.svd(o)
    pulse_guess.a_t[:] = u[:, 0]

    # pulse_guess.a_t[:] = o @ o.T @ pulse_guess.a_t
    # pulse_guess.e_p = e_p

    pulse_guess.a_t[:] = np.roll(
        pulse_guess.a_t,
        pulse_guess.n // 2 - pulse_guess.p_t.argmax(),
    )

    l_v.set_ydata(pulse_guess.p_v / pulse_guess.p_v.max())
    l_t.set_ydata(pulse_guess.p_t / pulse_guess.p_t.max())
    bm.update()

    loop_count += 1
    print(loop_count)
