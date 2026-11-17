
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, "src")

from calibration import LogLRCalibrator
from lr_metrics import cllr, empirical_min_cllr, eer, misleading_rates


GROUPS = {
    "f0": ["f0_median", "f0_std", "f0_p90_p10"],
    "voice_quality": ["jitter_local", "shimmer_local", "hnr_db"],
    "tilt": ["spectral_tilt"],
}

ALL_FEATURES = sum(GROUPS.values(), [])

CONFIGS = {
    "all": ALL_FEATURES,
    "all_without_f0": GROUPS["voice_quality"] + GROUPS["tilt"],
    "all_without_voice_quality": GROUPS["f0"] + GROUPS["tilt"],
    "all_without_tilt": GROUPS["f0"] + GROUPS["voice_quality"],
}

OUT = Path("scores/ablation_loto")
MODELS = Path("models/ablation_loto")
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True)


def main():
    trials = pd.read_csv("manifests/trials.csv")
    refs = pd.read_csv("manifests/reference_utterances.csv")
    clean = pd.read_csv("features/utterance_features.csv")
    mismatch = pd.read_csv("features/mismatch/mismatch_features.csv")

    clean = clean[
        (clean.qc_flag == "PASS") & (clean.condition == "clean")
    ].copy()
    mismatch = mismatch[
        (mismatch.qc_flag == "PASS") & (mismatch.condition == "noise10")
    ].copy()

    results = []

    for name, features in CONFIGS.items():
        print(f"\n=== {name} ===")

        dev = clean[clean.split == "development"]
        medians = dev[features].median()

        if medians.isna().any():
            raise ValueError(f"Missing development feature values: {name}")

        scaler = RobustScaler().fit(dev[features].fillna(medians))

        ref = refs.merge(
            clean[["utterance_id", "speaker_id"] + features],
            on=["utterance_id", "speaker_id"],
            how="inner",
        )

        scaled_cols = [f"_z_{f}" for f in features]
        zref = scaler.transform(ref[features].fillna(medians))
        for i, col in enumerate(scaled_cols):
            ref[col] = zref[:, i]

        centroids = ref.groupby("speaker_id")[scaled_cols].mean()

        def score(condition):
            q = clean if condition == "clean" else mismatch
            key = "utterance_id" if condition == "clean" else "source_file_id"

            q = q.dropna(subset=features, how="all").copy()
            zq = scaler.transform(q[features].fillna(medians))
            for i, col in enumerate(scaled_cols):
                q[col] = zq[:, i]

            q = q.drop_duplicates(subset=[key]).set_index(key)
            t = trials[trials.questioned_id.isin(q.index)].copy()
            t = t[t.reference_speaker.isin(centroids.index)].copy()

            t["raw_score"] = (
                -np.linalg.norm(
                    q.loc[t.questioned_id, scaled_cols].to_numpy()
                    - centroids.loc[t.reference_speaker, scaled_cols].to_numpy(),
                    axis=1,
                ) / np.sqrt(len(features))
            )
            t["condition"] = condition
            return t

        clean_scores = score("clean")
        noise_scores = score("noise10")

        cal = clean_scores[
            (clean_scores.split == "calibration")
            & np.isfinite(clean_scores.raw_score)
        ]

        calibrator = LogLRCalibrator().fit(
            cal.raw_score.to_numpy(), cal.target.to_numpy()
        )
        joblib.dump(calibrator, MODELS / f"{name}_calibrator.joblib")

        for condition, scored in [
            ("clean", clean_scores),
            ("noise10", noise_scores),
        ]:
            val = scored[
                (scored.split == "validation")
                & np.isfinite(scored.raw_score)
            ].copy()

            val["log_lr"] = calibrator.transform(val.raw_score.to_numpy())
            raw = val.raw_score.to_numpy()
            y = val.target.to_numpy(dtype=int)
            llr = val.log_lr.to_numpy()

            min_c = empirical_min_cllr(raw, y)
            mis = misleading_rates(llr, y)
            c = cllr(llr, y)

            results.append({
                "config": name,
                "condition": condition,
                "n_features": len(features),
                "n_validation": len(val),
                "target_trials": int((y == 1).sum()),
                "non_target_trials": int((y == 0).sum()),
                "cllr": c,
                "min_cllr": min_c,
                "calibration_loss": c - min_c,
                "eer": eer(raw, y),
                "misleading": mis["misleading"],
                "strong_misleading": mis["strong_misleading"],
            })

            val.to_csv(OUT / f"{name}_{condition}_validation.csv", index=False)
            print(
                f"{condition}: N={len(val)}, Cllr={c:.6f}, "
                f"EER={eer(raw, y):.6f}"
            )

    df = pd.DataFrame(results)
    df.to_csv("results/leave_one_group_out_metrics.csv", index=False)
    print("\n=== A2 RESULTS ===")
    print(df.to_string(index=False))
    print("\nSaved: results/leave_one_group_out_metrics.csv")


if __name__ == "__main__":
    main()
