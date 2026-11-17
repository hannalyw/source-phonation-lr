import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

FEATURES = [
    "f0_median", "f0_std", "f0_p90_p10",
    "jitter_local", "shimmer_local", "hnr_db", "spectral_tilt"
]

feat = pd.read_csv("features/utterance_features.csv").set_index("utterance_id")
trials = pd.read_csv("manifests/trials.csv")
refs = pd.read_csv("manifests/reference_utterances.csv")

# Minggu 4: clean matched; gunakan fitur yang lolos QC
feat = feat[(feat["condition"] == "clean") & (feat["qc_flag"] == "PASS")].copy()
trials = trials[trials["condition"] == "clean"].copy()

# Fit scaler hanya pada development utterances (sesuai panduan)
idx_dev = feat["split"].eq("development")
dev_median = feat.loc[idx_dev, FEATURES].median()

scaler = RobustScaler().fit(
    feat.loc[idx_dev, FEATURES].fillna(dev_median)
)

X = scaler.transform(feat[FEATURES].fillna(dev_median))
Z = pd.DataFrame(X, index=feat.index, columns=FEATURES)

# Gabungkan reference utterances menjadi centroid per speaker
centroids = {}
for spk, g in refs.groupby("speaker_id"):
    valid_ids = g.loc[g["utterance_id"].isin(Z.index), "utterance_id"]

    if len(valid_ids) > 0:
        centroids[spk] = Z.loc[valid_ids].mean(axis=0).to_numpy()

# Hitung raw score: negative distance questioned-centroid
rows = []
excluded = []

for _, t in trials.iterrows():
    if t["questioned_id"] not in Z.index:
        excluded.append({
            "trial_id": t["trial_id"],
            "reason": "questioned_failed_qc"
        })
        continue

    if t["reference_speaker"] not in centroids:
        excluded.append({
            "trial_id": t["trial_id"],
            "reason": "reference_centroid_unavailable"
        })
        continue

    q = Z.loc[t["questioned_id"]].to_numpy()
    r = centroids[t["reference_speaker"]]

    score = -float(np.linalg.norm(q - r) / np.sqrt(len(FEATURES)))
    rows.append({**t.to_dict(), "raw_score": score})

# Simpan output Minggu 4
pd.DataFrame(rows).to_csv("scores/source_scores.csv", index=False)

pd.DataFrame(
    excluded, columns=["trial_id", "reason"]
).to_csv("scores/excluded_trials_qc.csv", index=False)

# Verifikasi median target lebih tinggi daripada non-target
scores = pd.DataFrame(rows)

print("\n=== WEEK 4: REFERENCE SCORING BASELINE ===")
print("Total clean trials :", len(trials))
print("Scored trials      :", len(scores))
print("Excluded trials    :", len(excluded))

print("\nMedian raw score:")
print(scores.groupby("target")["raw_score"].median())

print("\nSaved: scores/source_scores.csv")
print("Saved: scores/excluded_trials_qc.csv")
