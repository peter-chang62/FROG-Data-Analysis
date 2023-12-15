"""
Pricipal Component Generalized Projection Algorithm

just scripting here, using same data as HNLG_input.py

At the end of the day, this script should be implemented by modifying the
retrieve function in python_phase_retrieva.py
"""

# %% -----
import numpy as np
import matplotlib.pyplot as plt
import clipboard
from scipy.constants import c
import collections
from shg_frog import light, BBO
from scipy.interpolate import RectBivariateSpline
import blit
import jax

DataCollection = collections.namedtuple("DataCollection", ["t_grid", "v_grid", "data"])


def fft(x, axis=-1, fsc=1.0):
    return (
        np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis
        )
        * fsc
    )


def ifft(x, axis=-1, fsc=1.0):
    return (
        np.fft.fftshift(
            np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis
        )
        / fsc
    )


# %% ----- load the spectrogram data
path = r"/Volumes/Peter SSD/Research_Projects/FROG/Data/06-16-2023_PC_UBFS/"
data = np.genfromtxt("12-14-2023/Menlo Comb A FROG.txt")

# time and frequency axis of grating spectrometer + translation stage
# shift time axis to center
t_fs = data[:, 0][1:]
wl_nm = data[0][1:]
data = data[:, 1:][1:]
data += data[::-1]  # symmetrize the FROG
I_corr = np.sum(data, axis=1)
t_fs -= t_fs[np.unravel_index(data.argmax(), data.shape)[0]]

# remove background
bckgnd = data[0].copy()
data -= bckgnd
data = np.where(data < data[1].max(), 0, data)

# set no signal region to zero
filt_wl_nm = (753.1344395581024, 808.7937034269221)
filt_t_fs = (-4000, 4000)

filt_v = c / (np.asarray(filt_wl_nm[::-1]) * 1e-9)
filt_t = np.asarray(filt_t_fs) * 1e-15

(idx_filt,) = np.logical_or(wl_nm < filt_wl_nm[0], wl_nm > filt_wl_nm[1]).nonzero()
data[:, idx_filt] = 0

# convert to frequency axis
nu = c / (wl_nm * 1e-9)
data *= c / nu**2

# divide out phase matching
bbo = BBO.BBOSHG()
r = bbo.R(wl_nm * 1e-3 * 2, 50, bbo.phase_match_angle_rad(1.55), 3.576 * np.pi / 180)
data /= r
data /= data.max()

data = DataCollection(t_grid=t_fs * 1e-15, v_grid=nu, data=data)

# %% ----- instantiate pulse instance
pulse = light.Pulse.Sech(
    n_points=256,
    v_min=183e12,
    v_max=196e12,
    v0=189167376325088.34,
    e_p=1e-9,
    t_fwhm=200e-15,
    time_window=10e-12,
)

# %% ----- interpolate data to grid
spl = RectBivariateSpline(
    x=data.v_grid[::-1],
    y=data.t_grid,
    z=data.data[:, ::-1].T,
)

# time and frequency window to interpolate
(idx_filt_t,) = np.logical_and(
    filt_t[0] < pulse.t_grid, pulse.t_grid < filt_t[-1]
).nonzero()
(idx_filt_v,) = np.logical_and(
    filt_v[0] / 2 < pulse.v_grid, pulse.v_grid < filt_v[1] / 2
).nonzero()
x, y = pulse.v_grid[idx_filt_v] * 2, pulse.t_grid[idx_filt_t]
spectrogram = spl(x, y)
spectrogram = np.where(spectrogram < 0, 0, spectrogram)

# zero pad spectrogram to match pulse's grid size
spectrogram = np.pad(
    spectrogram,
    (
        (idx_filt_v[0], pulse.n - idx_filt_v[-1] - 1),
        (idx_filt_t[0], pulse.n - idx_filt_t[-1] - 1),
    ),
    constant_values=(0, 0),
)

# scale spectrogram to match pulse energy
num = np.convolve(pulse.p_t, pulse.p_t[::-1]) * pulse.dt
denom = np.sum(spectrogram * pulse.dv, axis=0)
factor = np.sum(num) / np.sum(denom)
spectrogram *= factor

# %% ----- experimental spectrum, if available
spectrum = np.genfromtxt(
    "12-14-2023/1050mA_Preamp_Current.CSV",
    delimiter=",",
    skip_header=39,
)
spectrum[:, 0] = c / (spectrum[:, 0] * 1e-9)
spectrum[:, 1] *= c / spectrum[:, 0] ** 2
pulse_data = pulse.clone_pulse(pulse)
pulse_data.import_p_v(spectrum[:, 0], spectrum[:, 1])

# pulse.import_p_v(spectrum[:, 0], spectrum[:, 1])

phase = np.random.uniform(low=0, high=1, size=pulse.n) * np.pi / 8
pulse.a_t[:] = pulse.a_t * np.exp(1j * phase)

# %% ----- phase retrieval algorithm! All the previous stuff was just setting up :)
loop_count = 0
iter_limit = 300

o_rs = np.zeros((pulse.n, pulse.n), dtype=complex)
error = np.zeros(iter_limit)
_denom_error = np.sum(spectrogram)
AT = np.zeros((iter_limit, pulse.n), dtype=complex)
e_p = pulse.e_p

fig, ax = plt.subplots(1, 3, figsize=np.array([9.08, 4.82]))
ax[0].plot(
    pulse_data.v_grid * 1e-12, pulse_data.p_v / pulse_data.p_v.max(), "--", linewidth=2
)

(l_v,) = ax[0].plot(pulse.v_grid * 1e-12, pulse.p_v / pulse.p_v.max(), animated=True)
(l_t,) = ax[1].plot(pulse.t_grid * 1e15, pulse.p_t / pulse.p_t.max(), animated=True)
img = ax[2].imshow(
    np.ones((pulse.n,) * 2), cmap="RdBu_r", vmin=0, vmax=1, animated=True
)
ax[0].set_xlabel("frequency (THz)")
ax[0].set_ylabel("Power (a.u.)")
ax[1].set_xlabel("time (fs)")
ax[1].set_ylabel("Power (a.u.)")
fig.tight_layout()
bm = blit.BlitManager(fig.canvas, [l_v, l_t, img])
bm.update()
while loop_count < iter_limit:
    # calculate outer product o, and from o calculate the spectrogram
    # for the calculated spectrogram, each column is the spectrum at a given
    # time delay
    o = pulse.a_t * np.c_[pulse.a_t]
    for r in range(o.shape[0]):
        o_rs[r] = np.roll(o[r], -r)
    s_t = np.fft.fftshift(o_rs, axes=1)
    s_v = fft(s_t, axis=0, fsc=pulse.dt)

    error[loop_count] = (
        (abs(s_v) ** 2 - spectrogram) ** 2
    ).mean() ** 0.5 / _denom_error

    img.set_data((abs(s_v) / abs(s_v).max()) ** 2)

    # replacement all of s_v's amplitude
    s_v[:] = np.sqrt(spectrogram) * np.exp(1j * np.angle(s_v))

    # replace a subset of s_v's amplitude
    # s_v_repl = s_v[idx_filt_v[0] : idx_filt_v[-1], idx_filt_t[0] : idx_filt_t[-1]]
    # spectrogram_rpl = spectrogram[
    #     idx_filt_v[0] : idx_filt_v[-1], idx_filt_t[0] : idx_filt_t[-1]
    # ]
    # s_v_repl[:] = np.sqrt(spectrogram_rpl) * np.exp(1j * np.angle(s_v_repl))

    # back track from the spectrogram to get back to o
    s_t = ifft(s_v, axis=0, fsc=pulse.dt)
    o_rs = np.fft.ifftshift(s_t, axes=1)
    for r in range(o.shape[0]):
        o[r] = np.roll(o_rs[r], r)

    # update the pulse field
    u, s, vh = jax.scipy.linalg.svd(o)
    pulse.a_t[:] = u[:, 0]
    pulse.e_p = e_p

    pulse.a_t[:] = np.roll(pulse.a_t, pulse.n // 2 - pulse.p_t.argmax())

    AT[loop_count] = pulse.a_t

    loop_count += 1

    l_v.set_ydata(pulse.p_v / pulse.p_v.max())
    l_t.set_ydata(pulse.p_t / pulse.p_t.max() * 1)
    bm.update()

    print(loop_count)

# %% ----- plot results
pulse.a_t[:] = AT[error.argmin()]
o = pulse.a_t * np.c_[pulse.a_t]
for r in range(o.shape[0]):
    o_rs[r] = np.roll(o[r], -r)
s_t = np.fft.fftshift(o_rs, axes=1)
s_v = fft(s_t, axis=0, fsc=pulse.dt)

l_v.set_ydata(pulse.p_v / pulse.p_v.max())
l_t.set_ydata(pulse.p_t / pulse.p_t.max() * 1)
bm.update()

fig, ax = plt.subplots(1, 1)
ax.plot(error, linewidth=2)
ax.set_xlabel("iteration #")
ax.set_ylabel("error")
fig.tight_layout()

fig, ax = plt.subplots(1, 2)
ax[0].pcolormesh(
    pulse.t_grid * 1e15, pulse.v_grid * 1e-12, abs(s_v) ** 2, cmap="RdBu_r"
)
ax[1].pcolormesh(pulse.t_grid * 1e15, pulse.v_grid * 1e-12, spectrogram, cmap="RdBu_r")
ax[0].set_xlabel("time (fs)")
ax[0].set_ylabel("frequency (THz)")
ax[1].set_xlabel("time (fs)")
ax[1].set_ylabel("frequency (THz)")
fig.tight_layout()

tg_v = np.where(pulse.p_v / pulse.p_v.max() > 1e-3, pulse.tg_v, np.nan)
vg_t = np.where(pulse.p_t / pulse.p_t.max() > 1e-3, pulse.vg_t, np.nan)
fig, ax = plt.subplots(1, 2, figsize=np.array([8.85, 4.8]))
ax[0].plot(pulse_data.v_grid * 1e-12, pulse_data.p_v, "--", linewidth=2)
ax[0].plot(pulse.v_grid * 1e-12, pulse.p_v)
ax_2 = ax[0].twinx()
ax_2.plot(pulse.v_grid * 1e-12, tg_v * 1e15, "C2")
ax[0].set_xlabel("frequency (THz)")
ax[0].set_ylabel("J / Hz")
ax_2.set_ylabel("delay (fs)")
ax[1].plot(pulse.t_grid * 1e15, pulse.p_t)
ax_3 = ax[1].twinx()
ax_3.plot(pulse.t_grid * 1e15, vg_t * 1e-12, "C1")
ax_3.set_ylabel("Frequency (THz)")
ax[1].set_xlabel("time (fs)")
ax[1].set_ylabel("J / s")
fig.tight_layout()
