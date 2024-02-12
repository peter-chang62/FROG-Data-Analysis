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
from scipy.signal import butter, filtfilt
import pynlo

DataCollection = collections.namedtuple("DataCollection", ["t_grid", "v_grid", "data"])

output = collections.namedtuple("output", ["model", "sim"])


def fft(x, axis=-1, fsc=1.0):
    return fftshift(np.fft.fft(ifftshift(x, axes=axis), axis=axis), axes=axis) * fsc


def ifft(x, axis=-1, fsc=1.0):
    return fftshift(np.fft.ifft(ifftshift(x, axes=axis), axis=axis), axes=axis) / fsc


def rfft(x, axis=-1, fsc=1.0):
    return np.fft.rfft(ifftshift(x, axes=axis), axis=axis) * fsc


def irfft(x, axis=-1, fsc=1.0):
    return fftshift(np.fft.irfft(x, axis=axis), axes=axis) / fsc


def propagate(fiber, pulse, length, n_records=None, plot=None):
    """
    propagates a given pulse through fiber of given length

    Args:
        fiber (instance of SilicaFiber): Fiber
        pulse (instance of Pulse): Pulse
        length (float): fiber elngth

    Returns:
        output: model, sim
    """
    fiber: pynlo.materials.SilicaFiber
    model = fiber.generate_model(pulse)
    dz = model.estimate_step_size()
    sim = model.simulate(length, dz=dz, n_records=n_records, plot=plot)
    return output(model=model, sim=sim)


# %% ----- load the spectrogram data
data = np.genfromtxt("01-09-2023/HNLF_input.txt")

# time and frequency axis of grating spectrometer + translation stage
# shift time axis to center
t_fs = data[:, 0][1:]
wl_nm = data[0][1:]
data = data[:, 1:][1:]

idx_t0 = np.unravel_index(data.argmax(), data.shape)[0]
idx_keep = min([idx_t0, t_fs.size // 2])
data = data[idx_t0 - idx_keep : idx_t0 + idx_keep]
t_fs = t_fs[idx_t0 - idx_keep : idx_t0 + idx_keep]

b, a = butter(N=2, Wn=0.1, btype="low")
data = filtfilt(b, a, data, axis=0)
b, a = butter(N=2, Wn=0.1, btype="low")
data = filtfilt(b, a, data, axis=1)

# symmetrize the FROG
data += data[::-1]  # symmetrize the FROG

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
roots = InterpolatedUnivariateSpline(v_grid, autoconv - 1e-3).roots()
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
    t_fwhm=15e-15,
    time_window=(data.t_grid.max() - data.t_grid.min()) * 12,
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

# %% ----- RANA, kind of
autoconv = np.sum(spectrogram, axis=1) * pulse.dv
autoconv_t = ifft(autoconv, fsc=pulse.dt) ** 0.5
sw = abs(fft(autoconv_t, fsc=pulse.dt))

pulse.import_p_v(pulse.v_grid, sw)

# %% ----- experimental spectrum, if available
# spectrum = pd.read_excel("07-31-2023/data/20230726_50cm_5.5mW_reformatted.xlsx").values
# spectrum[:, 0] *= 1e12
# spectrum[:, 1] = 10 ** (spectrum[:, 1] / 10)
# pulse_data = pulse.clone_pulse(pulse)
# pulse_data.import_p_v(spectrum[:, 0], spectrum[:, 1])

# pulse.import_p_v(spectrum[:, 0], spectrum[:, 1])

phase = np.random.uniform(low=0, high=1, size=pulse.n) * np.pi / 8
pulse.a_t[:] = pulse.a_t * np.exp(1j * phase)

# %% ----- set up figures for live update
fig, ax = plt.subplots(1, 3, figsize=np.array([9.08, 4.82]))
# ax[0].plot(
#     pulse_data.v_grid * 1e-12, pulse_data.p_v / pulse_data.p_v.max(), "--", linewidth=2
# )

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
iter_limit = 100

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

    # randomize the pulse a little every now and then
    # doesn't really affect it :(
    # if loop_count % 25 == 0:
    #     pulse.a_t[:] = pulse.a_t * np.exp(1j * phase)

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
# ax[0].plot(pulse_data.v_grid * 1e-12, pulse_data.p_v, "--", linewidth=2)
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

# %% ----- hnlf propagation :)
hnlf = pynlo.materials.SilicaFiber()
hnlf.load_fiber_from_dict(pynlo.materials.hnlf_5p7)

pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550)

p = pynlo.light.Pulse.Sech(
    n=256,
    v_min=c / 3e-6,
    v_max=c / 800e-9,
    v0=pulse.v0,
    e_p=3.5e-9,
    t_fwhm=15e-15,
    min_time_window=pulse.t_grid.max() - pulse.t_grid.min(),
)

p.import_p_v(pulse.v_grid, pulse.p_v, phi_v=np.unwrap(pulse.phi_v))

model_pm1550, sim_pm1550 = propagate(pm1550, p, 0.18, n_records=100, plot="wvl")
model_hnlf, sim_hnlf = propagate(
    hnlf, sim_pm1550.pulse_out, 0.065, n_records=100, plot="wvl"
)

fig, ax = plt.subplots(1, 1)
p_out = sim_hnlf.pulse_out
dv_dl = p_out.v_grid**2 / c
ax.plot(p_out.wl_grid * 1e9, p_out.p_v * dv_dl)

sim_hnlf.plot("wvl")
