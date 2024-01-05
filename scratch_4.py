# %% -----
import numpy as np
from numpy.fft import fftshift, ifftshift
import matplotlib.pyplot as plt
import clipboard
from scipy.constants import c
import collections
from shg_frog import light, BBO
from scipy.interpolate import RectBivariateSpline
import blit
import jax
import scipy

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
path = r"12-22-2023/"

data = np.genfromtxt(path + "7A_HNLF_output.txt")
t_fs = data[:, 0][1:]
wl_nm = data[0][1:]
data = data[:, 1:][1:]

# %% ----- center t0
idx_t0 = np.unravel_index(data.argmax(), data.shape)[0]
idx_keep = min([idx_t0, t_fs.size // 2])
data = data[idx_t0 - idx_keep : idx_t0 + idx_keep]
t_fs = t_fs[idx_t0 - idx_keep : idx_t0 + idx_keep]

# %% ----- set array outside of signal location to 0
coll_autoconv = DataCollection(
    t_grid=t_fs * 1e-15,
    v_grid=c / (wl_nm * 1e-9),
    data=data - data[0],
)
spl_autoconv = RectBivariateSpline(
    x=coll_autoconv.v_grid[::-1],
    y=coll_autoconv.t_grid,
    z=coll_autoconv.data[:, ::-1].T,
)

pulse_autoconv = light.Pulse.Sech(
    256,
    coll_autoconv.v_grid.min(),
    coll_autoconv.v_grid.max(),
    (coll_autoconv.v_grid.max() - coll_autoconv.v_grid.min()) / 2
    + coll_autoconv.v_grid.min(),
    1e-9,
    200e-15,
    coll_autoconv.t_grid.max() - coll_autoconv.t_grid.min(),
)

autoconv = np.sum(spl_autoconv(pulse_autoconv.v_grid, pulse_autoconv.t_grid), axis=1)
autoconv /= autoconv.max()
pulse_autoconv.import_p_v(pulse_autoconv.v_grid, autoconv)

v_width = pulse_autoconv.v_width(200).eqv.real
v0 = pulse_autoconv.v_grid[pulse_autoconv.p_v.argmax()]

(idx,) = np.logical_and(
    (v0 - (v_width * 1.5)) < coll_autoconv.v_grid,
    coll_autoconv.v_grid < (v0 + (v_width * 1.5)),
).nonzero()
data[:, : idx[0]] = 0
data[:, idx[-1] :] = 0

# %% ----- filter out interference of shg and non-collinear FROG
coll_filt = DataCollection(
    t_grid=t_fs * 1e-15,
    v_grid=c / (wl_nm * 1e-9),
    data=data,
)
spl_filt = RectBivariateSpline(
    x=coll_filt.v_grid[::-1],
    y=coll_filt.t_grid,
    z=coll_filt.data[:, ::-1].T,
)

pulse_filt = light.Pulse.Sech(
    n_points=256,
    v_min=v0 - 2 * v0,
    v_max=v0 + 2 * v0,
    v0=v0,
    e_p=1e-9,
    t_fwhm=200e-15,
    time_window=coll_filt.t_grid.max() - coll_filt.t_grid.min(),
)

# data is now interpolated onto a pulse grid!
data = spl_filt(pulse_filt.v_grid, pulse_filt.t_grid)

b, a = scipy.signal.butter(N=10, Wn=1 / 8, btype="low")
ft_marg_t = abs(np.sum(rfft(data, axis=1), axis=0))
data_filt = scipy.signal.filtfilt(b, a, data, axis=1)

b, a = scipy.signal.butter(N=10, Wn=1 / 4, btype="low")
ft_marg_frq = abs(np.sum(rfft(data_filt, axis=0), axis=1))
data_filt = scipy.signal.filtfilt(b, a, data_filt, axis=0)

# %% ----- background subtraction
shg = np.genfromtxt(path + "7A_HNLF_output_shg_bckgnd.txt")
t_fs = shg[:, 0][1:]
wl_nm = shg[0][1:]
shg = shg[:, 1:][1:]
shg[:, : idx[0]] = 0
shg[:, idx[-1] :] = 0

shg_coll = DataCollection(
    t_grid=t_fs * 1e-15,
    v_grid=c / (wl_nm * 1e-9),
    data=shg,
)
spl_shg = RectBivariateSpline(
    x=shg_coll.v_grid[::-1],
    y=shg_coll.t_grid,
    z=shg_coll.data[:, ::-1].T,
)
shg = spl_shg(pulse_filt.v_grid, pulse_filt.t_grid)
shg = np.mean(shg, axis=1)

data_filt = (data_filt.T - shg).T
data_filt = np.where(data_filt < 0, 0, data_filt)

# %% ----- scaling conversion to go from wl -> frequency
data_filt = (data_filt.T * c / pulse_filt.v_grid**2).T
data_filt /= data_filt.max()

coll_retrvl = DataCollection(
    t_grid=pulse_filt.t_grid,
    v_grid=pulse_filt.v_grid,
    data=data_filt,
)
spl_retrvl = RectBivariateSpline(
    x=coll_retrvl.v_grid,
    y=coll_retrvl.t_grid,
    z=coll_retrvl.data,
)

# %% ----- instantiate pulse instance
pulse = light.Pulse.Sech(
    n_points=256,
    v_min=v0 / 2 - v_width / 2,
    v_max=v0 / 2 + v_width / 2,
    v0=v0 / 2,
    e_p=1e-9,
    t_fwhm=200e-15,
    time_window=(coll_retrvl.t_grid.max() - coll_retrvl.t_grid.min()) * 4,
)

phase = np.random.uniform(low=0, high=1, size=pulse.n) * np.pi / 8
pulse.a_t[:] = pulse.a_t * np.exp(1j * phase)

# %% ----- interpolate data to grid
filt_v = coll_retrvl.v_grid.min(), coll_retrvl.v_grid.max()
filt_t = coll_retrvl.t_grid.min(), coll_retrvl.t_grid.max()

(idx_filt_t,) = np.logical_and(
    filt_t[0] < pulse.t_grid, pulse.t_grid < filt_t[-1]
).nonzero()
(idx_filt_v,) = np.logical_and(
    filt_v[0] / 2 < pulse.v_grid, pulse.v_grid < filt_v[1] / 2
).nonzero()
x, y = pulse.v_grid[idx_filt_v] * 2, pulse.t_grid[idx_filt_t]
spectrogram = spl_retrvl(x, y)
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

# %% ----- set up figures for live update
fig, ax = plt.subplots(1, 3, figsize=np.array([9.08, 4.82]))
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
