import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, "src")

from mismatch import (
    add_noise_at_snr,
    telephone,
    resample_to_16k,
    duration_2s_center,
)
from source_features import extract_source_features


SEED = 2026
MINIMUM_VOICED_FRAMES = 20

CONDITIONS = [
    "noise20",
    "noise10",
    "telephone",
    "duration2s",
]


# =========================================================
# LOAD QUESTIONED UTTERANCES
# =========================================================

utterances = pd.read_csv("manifests/utterances.csv")
trials = pd.read_csv("manifests/trials.csv")

questioned_ids = set(trials["questioned_id"])

q = utterances[
    utterances["utterance_id"].isin(questioned_ids)
].copy()

# Pilot deterministik: 10 utterance pertama
q = q.sort_values("utterance_id").head(10)

print("=== WEEK 8 PILOT ===")
print("Questioned utterances:", len(q))


# =========================================================
# EXTRACTION
# =========================================================

rows = []
failures = []

for _, row in q.iterrows():

    uid = row["utterance_id"]

    try:
        x, sr = resample_to_16k(row["file_path"])

        for condition in CONDITIONS:

            # ---------------------------------------------
            # Transform questioned audio
            # ---------------------------------------------

            if condition == "noise20":
                rng = np.random.default_rng(
                    SEED + sum(uid.encode("utf-8"))
                )
                y = add_noise_at_snr(x, 20, rng)

            elif condition == "noise10":
                rng = np.random.default_rng(
                    SEED + sum(uid.encode("utf-8"))
                )
                y = add_noise_at_snr(x, 10, rng)

            elif condition == "telephone":
                y = telephone(x, sr)

            elif condition == "duration2s":

                if len(x) < 2 * sr:
                    failures.append({
                        "utterance_id": uid,
                        "condition": condition,
                        "reason": "original_duration_below_2s",
                    })
                    continue

                y = duration_2s_center(x, sr)

            # ---------------------------------------------
            # Temporary WAV
            # ---------------------------------------------

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as tmp:

                tmp_path = Path(tmp.name)

            try:
                sf.write(tmp_path, y, sr)

                features = extract_source_features(
                    str(tmp_path)
                )

            finally:
                tmp_path.unlink(missing_ok=True)

            # ---------------------------------------------
            # QC sama seperti Week 3
            # ---------------------------------------------

            qc_flag = (
                "PASS"
                if features["voiced_frames"]
                >= MINIMUM_VOICED_FRAMES
                else "FAIL"
            )

            rows.append({
                "utterance_id": f"{uid}_{condition}",
                "speaker_id": row["speaker_id"],
                "split": row["split"],
                "condition": condition,
                "source_file_id": row["source_file_id"],
                **features,
                "qc_flag": qc_flag,
            })

    except Exception as e:

        failures.append({
            "utterance_id": uid,
            "condition": "unknown",
            "reason": str(e),
        })


# =========================================================
# SAVE PILOT
# =========================================================

Path("features").mkdir(exist_ok=True)

out = pd.DataFrame(rows)
fail = pd.DataFrame(failures)

out.to_csv(
    "features/mismatch_features_pilot.csv",
    index=False
)

fail.to_csv(
    "features/mismatch_features_pilot_failures.csv",
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n=== PILOT RESULTS ===")

if len(out):
    print(
        out.groupby(
            ["condition", "qc_flag"]
        ).size()
    )

print("\nExtracted rows :", len(out))
print("Failures       :", len(fail))

print("\nSaved:")
print("features/mismatch_features_pilot.csv")
print("features/mismatch_features_pilot_failures.csv")

print("\n=== WEEK 8 PILOT SELESAI ===")
