import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

from mismatch import add_noise_at_snr, telephone, resample_to_16k


SEED = 2026
OUTPUT_DIR = Path("data/mismatch_examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Baca manifest
utterances = pd.read_csv("manifests/utterances.csv")
trials = pd.read_csv("manifests/trials.csv")

# Ambil satu questioned utterance clean sebagai contoh
questioned_ids = set(trials["questioned_id"])

candidates = utterances[
    (utterances["condition"] == "clean") &
    (utterances["utterance_id"].isin(questioned_ids))
].copy()

if len(candidates) == 0:
    raise RuntimeError("Tidak ada questioned utterance clean ditemukan.")

example = candidates.iloc[0]

utterance_id = example["utterance_id"]
input_path = example["file_path"]

print("Example utterance :", utterance_id)
print("Input file        :", input_path)

# Load/resample ke 16 kHz sesuai fungsi panduan
x, sr = resample_to_16k(input_path)

rng20 = np.random.default_rng(SEED)
rng10 = np.random.default_rng(SEED)

# Noise mismatch
x_noise20 = add_noise_at_snr(x, 20, rng20)
x_noise10 = add_noise_at_snr(x, 10, rng10)

# Telephone mismatch
x_telephone = telephone(x, sr)

# Output
noise20_path = OUTPUT_DIR / f"{utterance_id}_noise20.wav"
noise10_path = OUTPUT_DIR / f"{utterance_id}_noise10.wav"
telephone_path = OUTPUT_DIR / f"{utterance_id}_telephone.wav"

sf.write(noise20_path, x_noise20, sr)
sf.write(noise10_path, x_noise10, sr)
sf.write(telephone_path, x_telephone, sr)

print("\nGenerated:")
print(noise20_path)
print(noise10_path)
print(telephone_path)

print("\nSample rate :", sr)
print("Samples     :", len(x))
print("Duration    :", len(x) / sr)

print("\n=== MISMATCH EXAMPLES GENERATED ===")
