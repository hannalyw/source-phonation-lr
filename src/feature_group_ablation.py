import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler


FEATURE_GROUPS = {
    "f0": [
        "f0_median",
        "f0_std",
        "f0_p90_p10",
    ],
    "voice_quality": [
        "jitter_local",
        "shimmer_local",
        "hnr_db",
    ],
    "tilt": [
        "spectral_tilt",
    ],
    "all": [
        "f0_median",
        "f0_std",
        "f0_p90_p10",
        "jitter_local",
        "shimmer_local",
        "hnr_db",
        "spectral_tilt",
    ],
}


feat = pd.read_csv(
    "features/utterance_features.csv"
).set_index("utterance_id")

trials = pd.read_csv("manifests/trials.csv")
refs = pd.read_csv("manifests/reference_utterances.csv")

Path("scores/ablation").mkdir(parents=True, exist_ok=True)


# Hanya utterance yang lolos QC
feat_pass = feat[feat["qc_flag"] == "PASS"].copy()


for group_name, FEATURES in FEATURE_GROUPS.items():

    print(f"\n=== FEATURE GROUP: {group_name} ===")

    # Fit scaler hanya pada development utterances
    idx_dev = feat_pass["split"].eq("development")

    dev_median = feat_pass.loc[idx_dev, FEATURES].median()

    scaler = RobustScaler().fit(
        feat_pass.loc[idx_dev, FEATURES].fillna(dev_median)
    )

    X = scaler.transform(
        feat_pass[FEATURES].fillna(dev_median)
    )

    Z = pd.DataFrame(
        X,
        index=feat_pass.index,
        columns=FEATURES,
    )

    # Reference centroid
    centroids = {}

    for spk, g in refs.groupby("speaker_id"):

        valid_ids = [
            u for u in g["utterance_id"]
            if u in Z.index
        ]

        if len(valid_ids) > 0:
            centroids[spk] = (
                Z.loc[valid_ids]
                .mean(axis=0)
                .to_numpy()
            )

    rows = []
    excluded = []

    for _, t in trials.iterrows():

        if t.questioned_id not in Z.index:
            excluded.append({
                "trial_id": t.trial_id,
                "reason": "questioned_failed_qc",
            })
            continue

        if t.reference_speaker not in centroids:
            excluded.append({
                "trial_id": t.trial_id,
                "reason": "reference_centroid_unavailable",
            })
            continue

        q = Z.loc[t.questioned_id].to_numpy()
        r = centroids[t.reference_speaker]

        score = -float(
            np.linalg.norm(q - r) /
            np.sqrt(len(FEATURES))
        )

        rows.append({
            **t.to_dict(),
            "raw_score": score,
        })

    out = pd.DataFrame(rows)

    out_file = f"scores/ablation/{group_name}_scores.csv"
    out.to_csv(out_file, index=False)

    excluded_file = (
        f"scores/ablation/{group_name}_excluded_qc.csv"
    )

    pd.DataFrame(excluded).to_csv(
        excluded_file,
        index=False,
    )

    print("Scored trials :", len(out))
    print("Excluded trials:", len(excluded))

    print("\nMedian raw score:")
    print(out.groupby("target")["raw_score"].median())

    print("Saved:", out_file)


print("\n=== WEEK 6 RAW SCORING SELESAI ===")
