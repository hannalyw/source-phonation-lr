import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, fftconvolve
import librosa


def add_noise_at_snr(x, snr_db, rng):
    n = rng.normal(size=len(x)).astype(np.float32)

    px = np.mean(x**2) + 1e-12
    pn = np.mean(n**2) + 1e-12

    n *= np.sqrt(
        px / (pn * 10**(snr_db / 10))
    )

    return (x + n).astype(np.float32)


def telephone(x, sr=16000):
    sos = butter(
        6,
        [300, 3400],
        btype="bandpass",
        fs=sr,
        output="sos"
    )

    return sosfiltfilt(
        sos,
        x
    ).astype(np.float32)


def resample_to_16k(path):
    x, sr = librosa.load(
        path,
        sr=16000,
        mono=True
    )

    return x.astype(np.float32), 16000

def duration_2s_center(x, sr=16000):
    target_samples = int(2 * sr)

    if len(x) < target_samples:
        raise ValueError(
            f"Audio terlalu pendek untuk duration2s: "
            f"{len(x) / sr:.3f} detik"
        )

    start = (len(x) - target_samples) // 2
    end = start + target_samples

    return x[start:end].astype(np.float32)
