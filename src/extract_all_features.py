from pathlib import Path

import pandas as pd
from tqdm import tqdm

from source_features import extract_source_features


# =========================================================
# PATH
# =========================================================

MANIFEST_PATH = Path("manifests/utterances.csv")
OUTPUT_DIR = Path("features")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_OUTPUT = OUTPUT_DIR / "utterance_features.csv"
FAILURE_OUTPUT = OUTPUT_DIR / "extraction_failures.csv"


# =========================================================
# QC PARAMETER
# Parameter ini sudah dibekukan pada Week 2
# =========================================================

MINIMUM_VOICED_FRAMES = 20


# =========================================================
# LOAD MANIFEST
# =========================================================

df = pd.read_csv(MANIFEST_PATH)

feature_rows = []
failure_rows = []


# =========================================================
# FULL FEATURE EXTRACTION
# =========================================================

for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Extracting source features"
):

    utterance_id = row["utterance_id"]
    audio_path = row["file_path"]

    try:
        features = extract_source_features(audio_path)

        # QC berdasarkan minimum voiced frames
        if features["voiced_frames"] >= MINIMUM_VOICED_FRAMES:
            qc_flag = "PASS"
        else:
            qc_flag = "FAIL"

        feature_rows.append({
            "utterance_id": utterance_id,
            "speaker_id": row["speaker_id"],
            "split": row["split"],
            "condition": row["condition"],
            "source_file_id": row["source_file_id"],
            **features,
            "qc_flag": qc_flag,
        })

        # File yang tidak memenuhi QC tetap dicatat
        if qc_flag == "FAIL":
            failure_rows.append({
                "utterance_id": utterance_id,
                "speaker_id": row["speaker_id"],
                "file_path": audio_path,
                "reason": "voiced_frames_below_20",
            })

    except Exception as e:

        # Kegagalan ekstraksi tidak dihapus diam-diam
        failure_rows.append({
            "utterance_id": utterance_id,
            "speaker_id": row["speaker_id"],
            "file_path": audio_path,
            "reason": str(e),
        })


# =========================================================
# SAVE OUTPUT
# =========================================================

features_df = pd.DataFrame(feature_rows)
failures_df = pd.DataFrame(
    failure_rows,
    columns=[
        "utterance_id",
        "speaker_id",
        "file_path",
        "reason"
    ]
)

features_df.to_csv(
    FEATURE_OUTPUT,
    index=False
)

failures_df.to_csv(
    FAILURE_OUTPUT,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n=== FULL FEATURE EXTRACTION SUMMARY ===")
print(f"Total utterances : {len(df)}")
print(f"Extracted        : {len(features_df)}")
print(f"PASS QC          : {(features_df['qc_flag'] == 'PASS').sum()}")
print(f"FAIL QC          : {(features_df['qc_flag'] == 'FAIL').sum()}")
print(f"Failure records  : {len(failures_df)}")
print()
print(f"Saved: {FEATURE_OUTPUT}")
print(f"Saved: {FAILURE_OUTPUT}")
