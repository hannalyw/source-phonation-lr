import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import parselmouth
from pathlib import Path


MANIFEST = Path("manifests/utterances.csv")
OUTPUT = Path("results/source_feature_audit_20.csv")

SR = 16000
PITCH_FLOOR = 70
PITCH_CEILING = 400
MINIMUM_VOICED_FRAMES = 20


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=SR, mono=True)

    duration = len(y) / sr

    sound = parselmouth.Sound(y, sampling_frequency=sr)

    pitch = sound.to_pitch(
        time_step=0.01,
        pitch_floor=PITCH_FLOOR,
        pitch_ceiling=PITCH_CEILING
    )

    f0_values = pitch.selected_array["frequency"]
    voiced_f0 = f0_values[f0_values > 0]

    voiced_frames = len(voiced_f0)

    if voiced_frames >= MINIMUM_VOICED_FRAMES:
        f0_median = np.median(voiced_f0)
        f0_std = np.std(voiced_f0)
        f0_p90_p10 = (
            np.percentile(voiced_f0, 90)
            - np.percentile(voiced_f0, 10)
        )
    else:
        f0_median = np.nan
        f0_std = np.nan
        f0_p90_p10 = np.nan

    point_process = parselmouth.praat.call(
        sound,
        "To PointProcess (periodic, cc)",
        PITCH_FLOOR,
        PITCH_CEILING
    )

    jitter = parselmouth.praat.call(
        point_process,
        "Get jitter (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3
    )

    shimmer = parselmouth.praat.call(
        [sound, point_process],
        "Get shimmer (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6
    )

    harmonicity = sound.to_harmonicity_cc(
        time_step=0.01,
        minimum_pitch=PITCH_FLOOR
    )

    hnr_values = harmonicity.values[
        np.isfinite(harmonicity.values)
    ]

    hnr_db = (
        float(np.mean(hnr_values))
        if len(hnr_values) > 0
        else np.nan
    )

    spectrum = sound.to_spectrum()
    frequencies = spectrum.xs()
    power = np.mean(spectrum.values ** 2, axis=0)

    valid = (
        (frequencies > 300)
        & (frequencies < 4000)
        & (power > 0)
    )

    if np.sum(valid) >= 2:
        x = np.log10(frequencies[valid])
        y_power = 10 * np.log10(power[valid])

        slope = np.polyfit(x, y_power, 1)[0]
        spectral_tilt = slope
    else:
        spectral_tilt = np.nan

    return {
        "duration_sec": duration,
        "sr": sr,
        "voiced_frames": voiced_frames,
        "f0_median": f0_median,
        "f0_std": f0_std,
        "f0_p90_p10": f0_p90_p10,
        "jitter_local": jitter,
        "shimmer_local": shimmer,
        "hnr_db": hnr_db,
        "spectral_tilt": spectral_tilt,
    }


def main():
    manifest = pd.read_csv(MANIFEST)

    sample = manifest.sample(
        n=min(20, len(manifest)),
        random_state=2026
    )

    rows = []

    for _, row in sample.iterrows():
        audio_path = row["file_path"]

        try:
            features = extract_features(audio_path)

            features["path"] = audio_path
            features["speaker_id"] = row["speaker_id"]
            features["qc_flag"] = "PASS"

            if features["voiced_frames"] < MINIMUM_VOICED_FRAMES:
                features["qc_flag"] = "FAIL_LOW_VOICED_FRAMES"

        except Exception as e:
            features = {
                "path": audio_path,
                "speaker_id": row["speaker_id"],
                "qc_flag": f"FAIL_{type(e).__name__}",
                "error": str(e),
            }

        rows.append(features)

    output = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)

    print("\nAudit selesai.")
    print(f"Jumlah file: {len(output)}")
    print(f"Output: {OUTPUT}")
    print("\nQC summary:")
    print(output["qc_flag"].value_counts())


if __name__ == "__main__":
    main()
