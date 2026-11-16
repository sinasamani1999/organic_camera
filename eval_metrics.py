"""Objective metrics of chapter 3 (table 3-6), numpy only so they can be tested without a GPU.

All signals are residual rotations in degrees per frame at 30 Hz, shape [N, 2] = (yaw, pitch).
"""
import numpy as np
from scipy.signal import welch

FS = 30.0
BANDS = [(1.0, 3.0), (3.0, 6.0), (6.0, 15.0)]  # Hz: step/gait, corrections, tremor (below Nyquist)
AXES = ["yaw", "pitch"]
_trapz = getattr(np, "trapezoid", None) or np.trapz


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def per_axis(fn, pred, true):
    return {a: fn(pred[:, i], true[:, i]) for i, a in enumerate(AXES)}


def corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def std_ratio(a, b):
    return float(np.std(a) / max(np.std(b), 1e-12))


def band_powers(x, fs=FS):
    """Power of x in each band (from Welch PSD)."""
    f, p = welch(x, fs=fs, nperseg=min(len(x), 256))
    return np.array([_trapz(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]) for lo, hi in BANDS]), f, p


def log_spectral_distance(a, b, fs=FS, fmin=1.0):
    """RMS difference of the log PSDs above fmin (dB). 0 = same spectral shape and level."""
    f, pa = welch(a, fs=fs, nperseg=min(len(a), 256))
    _, pb = welch(b, fs=fs, nperseg=min(len(b), 256))
    m = f >= fmin
    return float(np.sqrt(np.mean((10 * np.log10(pa[m] + 1e-12) - 10 * np.log10(pb[m] + 1e-12)) ** 2)))


def all_metrics(pred, true):
    """Dict of metrics for one method against the real residual."""
    out = {"mae_deg": mae(pred, true), "mae_zero_deg": mae(np.zeros_like(true), true)}
    out["mae_ratio_to_zero"] = out["mae_deg"] / max(out["mae_zero_deg"], 1e-12)
    for i, a in enumerate(AXES):
        out[f"std_ratio_{a}"] = std_ratio(pred[:, i], true[:, i])
        out[f"corr_{a}"] = corr(pred[:, i], true[:, i])
        out[f"lsd_db_{a}"] = log_spectral_distance(pred[:, i], true[:, i])
        bp, _, _ = band_powers(pred[:, i])
        bt, _, _ = band_powers(true[:, i])
        for (lo, hi), x, y in zip(BANDS, bp, bt):
            out[f"band_{lo:g}-{hi:g}Hz_ratio_{a}"] = float(x / max(y, 1e-12))
    return out


# ------------------------------------------------------------------ Perlin baseline (same as Unreal's mode 2)
def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin1d(x, seed=0):
    """1-D gradient noise in [-1, 1] (same family as FMath::PerlinNoise1D)."""
    rng = np.random.default_rng(seed)
    grads = rng.uniform(-1, 1, 512)
    i0 = np.floor(x).astype(int)
    t = x - i0
    g0, g1 = grads[i0 % 512], grads[(i0 + 1) % 512]
    return 2.0 * ((1 - _fade(t)) * g0 * t + _fade(t) * g1 * (t - 1))


PERLIN_FREQ_HZ = 2.0   # base frequency = typical walking step rate
PERLIN_OCTAVES = 3     # 2, 4, 8 Hz: covers the 1-3, 3-6 and 6-15 Hz bands
PERLIN_PERSISTENCE = 0.5
PERLIN_LACUNARITY = 2.0


def perlin_offset(t, freq_hz=PERLIN_FREQ_HZ, octaves=PERLIN_OCTAVES, seed=0, phase=0.0):
    """Fractal (multi-octave) Perlin noise in about [-1, 1], same formula as Unreal's mode 2.

    Octave k has frequency freq_hz * 2^k and amplitude 0.5^k; the sum is divided by the sum of
    the amplitudes. With one octave this is the plain single-frequency noise used before 15 Nov 2026.
    """
    out, norm = np.zeros_like(t, dtype=float), 0.0
    for k in range(octaves):
        a = PERLIN_PERSISTENCE ** k
        out += a * perlin1d(t * freq_hz * PERLIN_LACUNARITY ** k + phase + 13.1 * k, seed + 2 * k)
        norm += a
    return out / norm


def perlin_residual(n, amp_deg, freq_hz=PERLIN_FREQ_HZ, fs=FS, seed=0, octaves=PERLIN_OCTAVES):
    """Per-frame rotation produced by the Perlin camera offset (offset is the noise; residual is its change)."""
    t = np.arange(n + 1) / fs
    off = np.stack([amp_deg * perlin_offset(t, freq_hz, octaves, seed),
                    amp_deg * perlin_offset(t, freq_hz, octaves, seed + 1, phase=57.3)], 1)
    return np.diff(off, axis=0)


def perlin_matched(true, freq_hz=PERLIN_FREQ_HZ, seed=0, octaves=PERLIN_OCTAVES):
    """Perlin residual scaled to the same overall RMS as the real residual (the fair comparison of chapter 3)."""
    r = perlin_residual(len(true), 1.0, freq_hz, seed=seed, octaves=octaves)
    return r * (np.sqrt(np.mean(true ** 2)) / max(np.sqrt(np.mean(r ** 2)), 1e-12))
