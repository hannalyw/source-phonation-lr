import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

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
CHECKPOINT_EVERY = 500

CONDITIONS = [
    "noise20",
    "noise10",
    "telephone",
    "duration2s",
]

OUTPUT_DIR = Path("features/mismatch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_OUTPUT = OUTPUT_DIR / "mismatch_features.csv"
FAILURE_OUTPUT = OUTPUT_DIR / "mismatch_failures.csv"


# =========================================================
# LOAD QUESTIONED UTTERANCES
# =========================================================

utterances = pd.read_csv("manifests/utterances.csv")
trials = pd.read_csv("manifests/trials.csv")

questioned_ids = set(
    trials["questioned_id"].drop_duplicates()
)

q = utterances[
    utterances["utterance_id"].isin(questioned_ids)
].copy()

q = q.sort_values("utterance_id").reset_index(drop=True)

print("=== WEEK 8 FULL MISMATCH FEATURE EXTRACTION ===")
print("Questioned utterances :", len(q))
print("Conditions            :", CONDITIONS)


# =========================================================
# RESUME CHECKPOINT
# =========================================================

if FEATURE_OUTPUT.exists():
    old_features = pd.read_csv(FEATURE_OUTPUT)
else:
    old_features = pd.DataFrame()

if FAILURE_OUTPUT.exists():
    old_failures = pd.read_csv(FAILURE_OUTPUT)
else:
    old_failures = pd.DataFrame()

completed = set()

if len(old_features):
    completed.update(
        zip(
            old_features["source_file_id"],
            old_features["condition"],
        )
    )

if len(old_failures):
    if {"source_file_id", "condition"}.issubset(
        old_failures.columns
    ):
        completed.update(
            zip(
                old_failures["source_file_id"],
                old_failures["condition"],
            )
        )

print("Already completed pairs :", len(completed))


# =========================================================
# BUFFER
# =========================================================

feature_rows = []
failure_rows = []


def save_checkpoint():
    global old_features, old_failures
    global feature_rows, failure_rows

    if feature_rows:
        new_features = pd.DataFrame(feature_rows)

        if len(old_features):
            old_features = pd.concat(
                [old_features, new_features],
                ignore_index=True,
            )
        else:
            old_features = new_features

        old_features = old_features.drop_duplicates(
            subset=["source_file_id", "condition"],
            keep="last",
        )

        old_features.to_csv(
            FEATURE_OUTPUT,
            index=False,
        )

        feature_rows = []

    if failure_rows:
        new_failures = pd.DataFrame(failure_rows)

        if len(old_failures):
            old_failures = pd.concat(
                [old_failures, new_failures],
                ignore_index=True,
            )
        else:
            old_failures = new_failures

        old_failures = old_failures.drop_duplicates(
            subset=["source_file_id", "condition"],
            keep="last",
        )

        old_failures.to_csv(
            FAILURE_OUTPUT,
            index=False,
        )

        failure_rows = []


# =========================================================
# FULL EXTRACTION
# =========================================================

for i, row in tqdm(
    q.iterrows(),
    total=len(q),
    desc="Week 8 mismatch extraction",
):

    uid = row["utterance_id"]

    try:
        x, sr = resample_to_16k(row["file_path"])

    except Exception as e:

        for condition in CONDITIONS:

            key = (row["source_file_id"], condition)

            if key in completed:
                continue

            failure_rows.append({
                "utterance_id": uid,
                "source_file_id": row["source_file_id"],
                "speaker_id": row["speaker_id"],
                "split": row["split"],
                "condition": condition,
                "reason": f"audio_load_error: {e}",
            })

            completed.add(key)

        continue

    for condition in CONDITIONS:

        key = (row["source_file_id"], condition)

        if key in completed:
            continue

        # =================================================
        # TRANSFORM
        # =================================================

        try:

            if condition == "noise20":

                rng = np.random.default_rng(
                    SEED + sum(uid.encode("utf-8"))
                )

                y = add_noise_at_snr(
                    x,
                    20,
                    rng,
                )

            elif condition == "noise10":

                rng = np.random.default_rng(
                    SEED + sum(uid.encode("utf-8"))
                )

                y = add_noise_at_snr(
                    x,
                    10,
                    rng,
                )

            elif condition == "telephone":

                y = telephone(
                    x,
                    sr,
                )

            elif condition == "duration2s":

                if len(x) < 2 * sr:

                    failure_rows.append({
                        "utterance_id": uid,
                        "source_file_id": row["source_file_id"],
                        "speaker_id": row["speaker_id"],
                        "split": row["split"],
                        "condition": condition,
                        "reason":
                            "original_duration_below_2s",
                    })

                    completed.add(key)
                    continue

                y = duration_2s_center(
                    x,
                    sr,
                )

            # =================================================
            # TEMPORARY WAV
            # =================================================

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as tmp:

                tmp_path = Path(tmp.name)

            try:

                sf.write(
                    tmp_path,
                    y,
                    sr,
                )

                features = extract_source_features(
                    str(tmp_path)
                )

            finally:

                tmp_path.unlink(
                    missing_ok=True
                )

            # =================================================
            # QC — SAMA DENGAN WEEK 3
            # =================================================

            qc_flag = (
                "PASS"
                if features["voiced_frames"]
                >= MINIMUM_VOICED_FRAMES
                else "FAIL"
            )

            feature_rows.append({
                "utterance_id":
                    f"{uid}_{condition}",
                "speaker_id":
                    row["speaker_id"],
                "split":
                    row["split"],
                "condition":
                    condition,
                "source_file_id":
                    row["source_file_id"],
                **features,
                "qc_flag":
                    qc_flag,
            })

            completed.add(key)

        except Exception as e:

            failure_rows.append({
                "utterance_id": uid,
                "source_file_id":
                    row["source_file_id"],
                "speaker_id":
                    row["speaker_id"],
                "split":
                    row["split"],
                "condition":
                    condition,
                "reason":
                    str(e),
            })

            completed.add(key)

    # =====================================================
    # CHECKPOINT
    # =====================================================

    if (i + 1) % CHECKPOINT_EVERY == 0:

        save_checkpoint()

        print(
            f"\nCheckpoint: "
            f"{i + 1}/{len(q)} questioned utterances"
        )


# =========================================================
# FINAL SAVE
# =========================================================

save_checkpoint()


# =========================================================
# SUMMARY
# =========================================================

features_df = pd.read_csv(FEATURE_OUTPUT)

if FAILURE_OUTPUT.exists():
    failures_df = pd.read_csv(FAILURE_OUTPUT)
else:
    failures_df = pd.DataFrame()


print("\n=== WEEK 8 MISMATCH FEATURE SUMMARY ===")

print("\nExtracted by condition:")
print(
    features_df[
        "condition"
    ].value_counts()
)

print("\nQC by condition:")
print(
    features_df.groupby(
        ["condition", "qc_flag"]
    ).size()
)

print("\nFailure/exclusion reasons:")

if len(failures_df):
    print(
        failures_df.groupby(
            ["condition", "reason"]
        ).size()
    )
else:
    print("None")

print("\nTotal extracted :", len(features_df))
print("Total failures  :", len(failures_df))

print("\nSaved:")
print(FEATURE_OUTPUT)
print(FAILURE_OUTPUT)

print("\n=== MISMATCH FEATURE EXTRACTION SELESAI ===")
