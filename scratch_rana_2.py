"""
This is the exact same as PCGPA.py, but for garrett's FROG results instead of
Matt's
"""

# %% -----
import numpy as np
import matplotlib.pyplot as plt
import clipboard
from scipy.constants import c
import collections
from shg_frog import light, BBO
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
import blit
import jax
import pandas as pd
from numpy.fft import fftshift, ifftshift
from scipy.optimize import minimize, Bounds
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
data = np.genfromtxt("12-14-2023/Menlo Comb A FROG.txt")

# time and frequency axis of grating spectrometer + translation stage
# shift time axis to center
t_fs = data[:, 0][1:]
wl_nm = data[0][1:]
data = data[:, 1:][1:]

idx_t0 = np.unravel_index(data.argmax(), data.shape)[0]
idx_keep = min([idx_t0, t_fs.size // 2])
data = data[idx_t0 - idx_keep : idx_t0 + idx_keep]
t_fs = t_fs[idx_t0 - idx_keep : idx_t0 + idx_keep]

b, a = butter(N=4, Wn=0.25, btype="low")
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


# %% ----- RANA
def func(X, d0, d1, d2):
    a, b, c = X
    return a * abs(d0) ** 2 + b * abs(d1) ** 2 + c * abs(d2) ** 2


autoconv = np.sum(spectrogram, axis=1)
autoconv_t = rfft(autoconv)
s_t_p = np.emath.sqrt(autoconv_t)
s_t_m = -s_t_p.real + 1j * s_t_p.imag

s_t = np.zeros(s_t_p.size, dtype=complex)
s_t[:2] = s_t_p[:2]
for n in range(1, s_t.size - 1):
    d0_p = s_t_p[n + 1] - s_t[n]
    d1_p = s_t_p[n + 1] - s_t[n] - (s_t[n] - s_t[n - 1])
    d2_p = ((s_t_p[n + 1] - s_t[n]) - (s_t[n] - s_t[n - 1])) - (
        (s_t[n] - s_t[n - 1]) - (s_t[n - 1] - s_t[n - 2])
    )

    d0_m = s_t_m[n + 1] - s_t[n]
    d1_m = s_t_m[n + 1] - s_t[n] - (s_t[n] - s_t[n - 1])
    d2_m = ((s_t_m[n + 1] - s_t[n]) - (s_t[n] - s_t[n - 1])) - (
        (s_t[n] - s_t[n - 1]) - (s_t[n - 1] - s_t[n - 2])
    )

    a, b, c = 0.09, 0.425, 1
    res_p = a * abs(d0_p) ** 2 + b * abs(d1_p) ** 2 + c * abs(d2_p) ** 2
    res_m = a * abs(d0_m) ** 2 + b * abs(d1_m) ** 2 + c * abs(d2_m) ** 2
    idx = np.array([res_p, res_m]).argmin()

    s_t[n + 1] = np.array([s_t_p[n + 1], s_t_m[n + 1]][idx])

SW = irfft(s_t)
