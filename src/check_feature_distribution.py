import pandas as pd

# Load hasil full feature extraction
features = pd.read_csv("features/utterance_features.csv")

# Hanya data yang lolos QC
features = features[features["qc_flag"] == "PASS"].copy()

# Load metadata speaker VCTK
speaker_info = pd.read_csv(
    "data/raw/VCTK-Corpus-0.92/speaker-info.txt",
    sep=r"\s+",
    usecols=["ID", "AGE", "GENDER"]
)

speaker_info = speaker_info.rename(
    columns={
        "ID": "speaker_id",
        "GENDER": "sex"
    }
)

# Gabungkan metadata dengan feature extraction
df = features.merge(
    speaker_info[["speaker_id", "sex"]],
    on="speaker_id",
    how="left"
)

feature_cols = [
    "f0_median",
    "f0_std",
    "f0_p90_p10",
    "jitter_local",
    "shimmer_local",
    "hnr_db",
    "spectral_tilt"
]

print("=== FEATURE DISTRIBUTION CHECK ===")

print("\nJumlah PASS QC:")
print(len(df))

print("\nJumlah speaker:")
print(df["speaker_id"].nunique())

print("\nDistribusi utterance per sex:")
print(df["sex"].value_counts(dropna=False))

print("\nSpeaker per sex:")
print(
    df[["speaker_id", "sex"]]
    .drop_duplicates()["sex"]
    .value_counts(dropna=False)
)

print("\nMissing metadata sex:")
print(df["sex"].isna().sum())

print("\n=== DISTRIBUSI FITUR PER SEX ===")
print(
    df.groupby("sex")[feature_cols]
    .agg(["count", "mean", "std", "median"])
    .round(4)
)

print("\n=== DISTRIBUSI PER SPEAKER ===")
speaker_summary = (
    df.groupby(["speaker_id", "sex"])[feature_cols]
    .agg(["count", "mean", "std", "median"])
    .round(4)
)

speaker_summary.to_csv(
    "results/feature_distribution_by_speaker.csv"
)

sex_summary = (
    df.groupby("sex")[feature_cols]
    .agg(["count", "mean", "std", "median"])
    .round(4)
)

sex_summary.to_csv(
    "results/feature_distribution_by_sex.csv"
)

print("\nSaved:")
print("results/feature_distribution_by_speaker.csv")
print("results/feature_distribution_by_sex.csv")
