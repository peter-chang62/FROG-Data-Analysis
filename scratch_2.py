"""
This is the exact same as PCGPA.py, but for garrett's FROG results instead of
Matt's.
"""

# %% -----
import numpy as np
from numpy.fft import fftshift, ifftshift
import matplotlib.pyplot as plt
import clipboard
from scipy.constants import c
import collections
from shg_frog import light, BBO
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
import blit
import jax
import pandas as pd
from scipy.signal import butter, filtfilt

DataCollection = collections.namedtuple("DataCollection", ["t_grid", "v_grid", "data"])


def fft(x, axis=-1, fsc=1.0):
    return fftshift(np.fft.fft(ifftshift(x, axes=axis), axis=axis), axes=axis) * fsc


def ifft(x, axis=-1, fsc=1.0):
    return fftshift(np.fft.ifft(ifftshift(x, axes=axis), axis=axis), axes=axis) / fsc


def rfft(x, axis=-1, fsc=1.0):
    return np.fft.rfft(ifftshift(x, axes=axis), axis=axis) * fsc


def irfft(x, axis=-1, fsc=1.0):
    return fftshift(np.fft.irfft(x, axis=axis), axes=axis) / fsc


# %% ----- load the spectrogram data
data = np.genfromtxt("07-31-2023/data/run15.txt")

# time and frequency axis of grating spectrometer + translation stage
# shift time axis to center
t_fs = data[:, 0][1:]
wl_nm = data[0][1:]
data = data[:, 1:][1:]

idx_t0 = np.unravel_index(data.argmax(), data.shape)[0]
idx_keep = min([idx_t0, t_fs.size // 2])
data = data[idx_t0 - idx_keep : idx_t0 + idx_keep]
t_fs = t_fs[idx_t0 - idx_keep : idx_t0 + idx_keep]

b, a = butter(N=2, Wn=0.25, btype="low")
data = filtfilt(b, a, data, axis=0)
data = filtfilt(b, a, data, axis=1)

data += data[::-1]  # symmetrize the FROG

# remove background
bckgnd = data[0].copy()
data -= bckgnd
data = np.where(data < data[1].max(), 0, data)

# convert to frequency axis
nu = c / (wl_nm * 1e-9)
data *= c / nu**2

data = DataCollection(t_grid=t_fs * 1e-15, v_grid=nu, data=data)

spl = RectBivariateSpline(
    x=data.v_grid[::-1],
    y=data.t_grid,
    z=data.data[:, ::-1].T,
)

v_grid = np.linspace(data.v_grid.min(), data.v_grid.max(), 512)
t_grid = np.linspace(data.t_grid.min(), data.t_grid.max(), 512)
autoconv = np.sum(spl(v_grid, t_grid), axis=1)
autoconv /= autoconv.max()
roots = InterpolatedUnivariateSpline(v_grid, autoconv - 1e-2).roots()
v0 = v_grid[autoconv.argmax()]
roots = sorted(list(roots), key=lambda v: abs(v0 - v))
roots = sorted(roots[:2])
v_width = np.diff(roots[:2])

# divide out phase matching
bbo = BBO.BBOSHG()
r = bbo.R(wl_nm * 1e-3 * 2, 50, bbo.phase_match_angle_rad(1.55), 3.576 * np.pi / 180)

(idx,) = np.logical_and(roots[0] < data.v_grid, data.v_grid < roots[1]).nonzero()
data.data[:, : idx[0]] = 0
data.data[:, idx[-1] :] = 0
data.data[:] /= r
data.data[:] /= data.data.max()

spl = RectBivariateSpline(
    x=data.v_grid[::-1],
    y=data.t_grid,
    z=data.data[:, ::-1].T,
)

# %% ----- instantiate pulse instance
pulse = light.Pulse.Sech(
    n_points=256,
    v_min=v0 / 2 - v_width / 2,
    v_max=v0 / 2 + v_width / 2,
    v0=v0 / 2,
    e_p=1e-9,
    t_fwhm=200e-15,
    time_window=data.t_grid.max() - data.t_grid.min(),
)

# %% ----- interpolate data to grid

# time and frequency window to interpolate
filt_v = data.v_grid.min(), data.v_grid.max()
filt_t = data.t_grid.min(), data.t_grid.max()

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
spectrum = pd.read_excel("07-31-2023/data/20230726_50cm_5.5mW_reformatted.xlsx").values
spectrum[:, 0] *= 1e12
spectrum[:, 1] = 10 ** (spectrum[:, 1] / 10)
pulse_data = pulse.clone_pulse(pulse)
pulse_data.import_p_v(spectrum[:, 0], spectrum[:, 1])

# pulse.import_p_v(spectrum[:, 0], spectrum[:, 1])

phase = np.random.uniform(low=0, high=1, size=pulse.n) * np.pi / 8
pulse.a_t[:] = pulse.a_t * np.exp(1j * phase)

# %% ----- set up figures for live update
fig, ax = plt.subplots(1, 3, figsize=np.array([9.08, 4.82]))
ax[0].plot(
    pulse_data.v_grid * 1e-12, pulse_data.p_v / pulse_data.p_v.max(), "--", linewidth=2
)

(l_v,) = ax[0].plot(pulse.v_grid * 1e-12, pulse.p_v / pulse.p_v.max(), animated=True)
(l_t,) = ax[1].plot(pulse.t_grid * 1e15, pulse.p_t / pulse.p_t.max(), animated=True)
img = ax[2].imshow(
    np.ones((pulse.n,) * 2), cmap="RdBu_r", vmin=0, vmax=1, animated=True
)
ax[2].axis(False)
ax[0].set_xlabel("frequency (THz)")
ax[0].set_ylabel("Power (a.u.)")
ax[1].set_xlabel("time (fs)")
ax[1].set_ylabel("Power (a.u.)")
fig.tight_layout()
bm = blit.BlitManager(fig.canvas, [l_v, l_t, img])
bm.update()

# %% ----- phase retrieval algorithm! All the previous stuff was just setting up :)
loop_count = 0
iter_limit = 300

o_rs = np.zeros((pulse.n, pulse.n), dtype=complex)
error = np.zeros(iter_limit)
_denom_error = np.sum(spectrogram)
AT = np.zeros((iter_limit, pulse.n), dtype=complex)
e_p = pulse.e_p

while loop_count < iter_limit:
    # calculate outer product o, and from o calculate the spectrogram
    # for the calculated spectrogram, each column is the spectrum at a given
    # time delay
    o = pulse.a_t * np.c_[pulse.a_t]
    for r in range(o.shape[0]):
        o_rs[r] = np.roll(o[r], -r)
    s_t = fftshift(o_rs, axes=1)
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
    o_rs = ifftshift(s_t, axes=1)
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
o_rs = np.zeros(o.shape, dtype=complex)
for r in range(o.shape[0]):
    o_rs[r] = np.roll(o[r], -r)
s_t = fftshift(o_rs, axes=1)
s_v = fft(s_t, axis=0, fsc=pulse.dt)

l_v.set_ydata(pulse.p_v / pulse.p_v.max())
l_t.set_ydata(pulse.p_t / pulse.p_t.max() * 1)
bm.update()

fig, ax = plt.subplots(1, 1)
ax.plot(error, linewidth=2)
ax.set_xlabel("iteration #")
ax.set_title("error")
fig.tight_layout()

fig, ax = plt.subplots(1, 2)
ax[0].pcolormesh(
    pulse.t_grid * 1e15,
    pulse.v_grid * 1e-12,
    abs(s_v) ** 2,
    cmap="RdBu_r",
)
ax[0].set_title("retrieved")
ax[1].pcolormesh(
    pulse.t_grid * 1e15,
    pulse.v_grid * 1e-12,
    spectrogram,
    cmap="RdBu_r",
)
ax[1].set_title("experimental")
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

fig, ax = plt.subplots(1, 1)
ax.plot(pulse.v_grid * 1e-12, np.sum(spectrogram, axis=1), label="experiment")
ax.plot(pulse.v_grid * 1e-12, np.sum(abs(s_v) ** 2, axis=1), label="retrieved")
ax.set_xlabel("frequency (THz)")
ax.set_title("Frequency Marginal")
ax.legend(loc="best")

# %% ----- just curious what's the interferometric FROG?
# the signal fields are column vectors! -> the gate fields are row vectors
pulse_if = light.Pulse.Sech(
    256,
    pulse.v0 - 2 * pulse.v0,
    pulse.v0 + 2 * pulse.v0,
    pulse.v0,
    pulse.e_p,
    200e-15,
    np.diff(pulse.t_grid[[0, -1]]),
)
pulse_if.import_p_v(pulse.v_grid, pulse.p_v, np.unwrap(pulse.phi_v))

w0 = 2 * np.pi * pulse_if.v0
n = pulse_if.n

signal = np.c_[pulse_if.a_t] * np.ones(n)

gate = pulse_if.a_t * np.c_[np.ones(n)]
for r in range(n):
    gate[r] = np.roll(gate[r], -r)
gate = fftshift(gate, axes=1)
a
s_t_if = (signal + gate * np.exp(-1j * w0 * pulse_if.t_grid)) ** 2
s_v_if = fft(s_t_if, axis=0, fsc=pulse_if.dt)

spectrogram_if = abs(s_v_if) ** 2
spectrogram_v_if = rfft(spectrogram_if, axis=1, fsc=pulse_if.dt)

x = spectrogram_v_if.copy()
x[:, x.shape[1] // 4 :] = 0
y = irfft(x, axis=1, fsc=pulse_if.dt)
shg = abs(fft(pulse_if.a_t**2, fsc=pulse_if.dt)) ** 2
z = (y.T - shg * 2).T

# %% --- plot numerical collinear FROG results
fig, ax = plt.subplots(1, 1)
v_grid_if = np.fft.rfftfreq(spectrogram_if.shape[1], d=pulse_if.dt) + pulse_if.v0
ax.plot(v_grid_if * 1e-12, np.sum(abs(spectrogram_v_if), axis=0))
ax.axvline(v_grid_if[0] * 1e-12 * 2, color="k", linestyle="--")
ax.axvline(v_grid_if[0] * 1e-12 * 3, color="k", linestyle="--")
ax.axvline(v_grid_if[v_grid_if.size // 4] * 1e-12, color="r", linestyle="--")
ax.set_xlabel("frequency (THz)")

fig, ax = plt.subplots(1, 3, figsize=np.array([16.65, 5.3]))
(idx,) = np.logical_and(
    pulse.v_grid.min() < pulse_if.v_grid, pulse_if.v_grid < pulse.v_grid.max()
).nonzero()
ax[0].pcolormesh(
    pulse_if.t_grid * 1e15,
    pulse_if.v_grid[idx] * 1e-12,
    spectrogram_if[idx],
    cmap="RdBu_r",
)
ax[0].set_xlim(-100, 100)
ax[1].pcolormesh(
    pulse_if.t_grid * 1e15, pulse_if.v_grid[idx] * 1e-12, y[idx], cmap="RdBu_r"
)
ax[2].pcolormesh(
    pulse_if.t_grid * 1e15, pulse_if.v_grid[idx] * 1e-12, z[idx], cmap="RdBu_r"
)
I_corr = np.sum(spectrogram, axis=0) * pulse.dv
I_corr_if = np.sum(spectrogram_if, axis=0) * pulse_if.dv
I_corr_if -= I_corr_if.mean()
# ax[3].plot(pulse_if.t_grid * 1e15, I_corr_if / I_corr_if.max())
# ax[3].plot(pulse.t_grid * 1e15, I_corr / I_corr.max())
[i.set_xlabel("time (fs)") for i in ax]
[i.set_ylabel("frequency (THz)") for i in ax[:-1]]
ax[0].set_title("CFROG")
ax[1].set_title("CFROG DC")
ax[2].set_title("CFROG DC - SHG background")
fig.tight_layout()
