import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, "src")


FEATURES = [
    "f0_median",
    "f0_std",
    "f0_p90_p10",
    "jitter_local",
    "shimmer_local",
    "hnr_db",
    "spectral_tilt",
]

CONDITIONS = [
    "noise20",
    "noise10",
    "telephone",
    "duration2s",
]

OUTDIR = Path("scores/cross_condition")
OUTDIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

print("=== WEEK 8 CROSS-CONDITION SCORING ===")

utterances = pd.read_csv("manifests/utterances.csv")
trials = pd.read_csv("manifests/trials.csv")
references = pd.read_csv("manifests/reference_utterances.csv")
clean_features = pd.read_csv("features/utterance_features.csv")
mismatch_features = pd.read_csv(
    "features/mismatch/mismatch_features.csv"
)

print("Trials            :", len(trials))
print("Clean features    :", len(clean_features))
print("Mismatch features :", len(mismatch_features))


# =========================================================
# CLEAN PASS FEATURES
# =========================================================

clean = clean_features[
    clean_features["qc_flag"] == "PASS"
].copy()

mismatch = mismatch_features[
    mismatch_features["qc_flag"] == "PASS"
].copy()


# =========================================================
# FIT ROBUST SCALER
# DEVELOPMENT CLEAN ONLY
# =========================================================

dev_clean = clean[
    clean["split"] == "development"
].copy()

scaler = RobustScaler()

scaler.fit(
    dev_clean[FEATURES].to_numpy(dtype=float)
)

print(
    "RobustScaler fitted on development-clean:",
    len(dev_clean),
)


# =========================================================
# CLEAN REFERENCE CENTROIDS
# =========================================================

ref = references.merge(
    clean[
        ["utterance_id", "speaker_id"] + FEATURES
    ],
    on=["utterance_id", "speaker_id"],
    how="inner",
)

print("Valid clean reference utterances:", len(ref))

X_ref = scaler.transform(
    ref[FEATURES].to_numpy(dtype=float)
)

for j, feature in enumerate(FEATURES):
    ref[f"_scaled_{feature}"] = X_ref[:, j]

scaled_cols = [
    f"_scaled_{feature}"
    for feature in FEATURES
]

centroids = (
    ref.groupby("speaker_id")[scaled_cols]
    .mean()
)

print("Reference centroids:", len(centroids))


# =========================================================
# SCORE EACH CONDITION
# =========================================================

summary = []

for condition in CONDITIONS:

    print(f"\n--- {condition} ---")

    q = mismatch[
        mismatch["condition"] == condition
    ].copy()

    print("PASS questioned features:", len(q))

    # -----------------------------------------------------
    # SCALE QUESTIONED FEATURES USING SAME CLEAN SCALER
    # -----------------------------------------------------

    X_q = scaler.transform(
        q[FEATURES].to_numpy(dtype=float)
    )

    for j, feature in enumerate(FEATURES):
        q[f"_scaled_{feature}"] = X_q[:, j]

    # source_file_id is the original clean questioned ID
    q_index = q.set_index("source_file_id")

    valid_questioned = set(q_index.index)

    condition_trials = trials[
        trials["questioned_id"].isin(
            valid_questioned
        )
    ].copy()

    print(
        "Trials with valid questioned feature:",
        len(condition_trials),
    )

    # -----------------------------------------------------
    # KEEP ONLY TRIALS WITH VALID REFERENCE CENTROID
    # -----------------------------------------------------

    valid_reference_speakers = set(
        centroids.index
    )

    condition_trials = condition_trials[
        condition_trials["reference_speaker"].isin(
            valid_reference_speakers
        )
    ].copy()

    print(
        "Trials after reference check:",
        len(condition_trials),
    )

    # -----------------------------------------------------
    # VECTORISED SCORING
    # score = -Euclidean / sqrt(number of features)
    # -----------------------------------------------------

    q_matrix = q_index.loc[
        condition_trials["questioned_id"],
        scaled_cols,
    ].to_numpy(dtype=float)

    r_matrix = centroids.loc[
        condition_trials["reference_speaker"],
        scaled_cols,
    ].to_numpy(dtype=float)

    scores = (
        -np.linalg.norm(
            q_matrix - r_matrix,
            axis=1,
        )
        / np.sqrt(len(FEATURES))
    )

    condition_trials["score"] = scores

    # mismatch condition replaces original clean condition
    condition_trials["condition"] = condition

    # -----------------------------------------------------
    # SANITY CHECKS
    # -----------------------------------------------------

    self_comparison = (
        (
            condition_trials["questioned_speaker"]
            == condition_trials["reference_speaker"]
        )
        & (condition_trials["target"] == 0)
    ).sum()

    target_scores = condition_trials.loc[
        condition_trials["target"] == 1,
        "score",
    ]

    non_target_scores = condition_trials.loc[
        condition_trials["target"] == 0,
        "score",
    ]

    print("Scored trials :", len(condition_trials))
    print(
        "Target median :",
        target_scores.median(),
    )
    print(
        "Non-target median :",
        non_target_scores.median(),
    )
    print(
        "Invalid self-comparison :",
        self_comparison,
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    output_path = (
        OUTDIR / f"{condition}_scores.csv"
    )

    condition_trials.to_csv(
        output_path,
        index=False,
    )

    print("Saved:", output_path)

    summary.append({
        "condition": condition,
        "pass_questioned": len(q),
        "scored_trials": len(condition_trials),
        "target_trials": int(
            (condition_trials["target"] == 1).sum()
        ),
        "non_target_trials": int(
            (condition_trials["target"] == 0).sum()
        ),
        "target_median_score":
            target_scores.median(),
        "non_target_median_score":
            non_target_scores.median(),
        "invalid_self_comparison":
            int(self_comparison),
    })


# =========================================================
# SUMMARY
# =========================================================

summary_df = pd.DataFrame(summary)

summary_path = (
    OUTDIR / "cross_condition_score_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)

print("\n=== CROSS-CONDITION SUMMARY ===")
print(summary_df.to_string(index=False))

print("\nSaved summary:")
print(summary_path)

print("\n=== WEEK 8 CROSS-CONDITION SCORING SELESAI ===")
