import numpy as np, parselmouth, librosa
from parselmouth.praat import call


def safe_stat(x, fn=np.nanmedian):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(fn(x)) if len(x) else np.nan


def extract_source_features(path):
    snd = parselmouth.Sound(path)

    pitch = snd.to_pitch(
        time_step=0.01,
        pitch_floor=70,
        pitch_ceiling=400
    )

    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]

    point = call(
        snd,
        "To PointProcess (periodic, cc)",
        70,
        400
    )

    jitter = call(
        point,
        "Get jitter (local)",
        0, 0, 1e-4, 0.02, 1.3
    )

    shimmer = call(
        [snd, point],
        "Get shimmer (local)",
        0, 0, 1e-4, 0.02, 1.3, 1.6
    )

    harm = call(
        snd,
        "To Harmonicity (cc)",
        0.01, 70, 0.1, 1.0
    )

    hnr = call(
        harm,
        "Get mean",
        0,
        0
    )

    y, sr = librosa.load(
        path,
        sr=16000,
        mono=True
    )

    S = np.abs(
        librosa.stft(
            y,
            n_fft=512,
            hop_length=160,
            win_length=400
        )
    ) + 1e-8

    freqs = librosa.fft_frequencies(
        sr=sr,
        n_fft=512
    )

    band = (freqs >= 300) & (freqs <= 4000)

    x = np.log(freqs[band] + 1.0)

    slopes = []

    for frame in S[band].T:
        slopes.append(
            np.polyfit(
                x,
                np.log(frame + 1e-8),
                1
            )[0]
        )

    return {
        "f0_median": safe_stat(f0),
        "f0_std": safe_stat(f0, np.nanstd),
        "f0_p90_p10": safe_stat(
            f0,
            lambda v:
            np.nanpercentile(v, 90)
            - np.nanpercentile(v, 10)
        ),
        "jitter_local": float(jitter),
        "shimmer_local": float(shimmer),
        "hnr_db": float(hnr),
        "spectral_tilt": safe_stat(slopes),
        "voiced_frames": int(len(f0)),
    }
