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

# Keep noise10 as a comparator, and add the requested conditions.
CONDITIONS = ["clean", "noise20", "noise10", "telephone", "duration2s"]

OUT = Path("scores/ablation_loto")
MODELS = Path("models/ablation_loto")
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True)


def score_condition(trials, clean, mismatch, refs, features, medians, scaler,
                    centroids, condition):
    """Score one condition using the clean-development scaler and clean centroids."""
    if condition == "clean":
        q = clean.copy()
        key = "utterance_id"
    else:
        q = mismatch[mismatch["condition"] == condition].copy()
        key = "source_file_id"

    if q.empty:
        print(f"WARNING: no PASS feature rows for condition={condition}")
        return pd.DataFrame()

    q = q.dropna(subset=features, how="all").copy()
    if q.empty:
        return pd.DataFrame()

    zq = scaler.transform(q[features].fillna(medians))
    scaled_cols = [f"_z_{f}" for f in features]
    for i, col in enumerate(scaled_cols):
        q[col] = zq[:, i]

    # One questioned feature vector per original utterance.
    q = q.drop_duplicates(subset=[key]).set_index(key)
    t = trials[trials["questioned_id"].isin(q.index)].copy()
    t = t[t["reference_speaker"].isin(centroids.index)].copy()
    if t.empty:
        return t

    t["raw_score"] = (
        -np.linalg.norm(
            q.loc[t["questioned_id"], scaled_cols].to_numpy(dtype=float)
            - centroids.loc[t["reference_speaker"], scaled_cols].to_numpy(dtype=float),
            axis=1,
        ) / np.sqrt(len(features))
    )
    t["condition"] = condition
    return t


def main():
    trials = pd.read_csv("manifests/trials.csv")
    refs = pd.read_csv("manifests/reference_utterances.csv")
    clean_all = pd.read_csv("features/utterance_features.csv")
    mismatch_all = pd.read_csv("features/mismatch/mismatch_features.csv")

    clean = clean_all[
        (clean_all["qc_flag"] == "PASS")
        & (clean_all["condition"] == "clean")
    ].copy()

    mismatch = mismatch_all[mismatch_all["qc_flag"] == "PASS"].copy()

    results = []

    for name, features in CONFIGS.items():
        print(f"\n=== CONFIG: {name} ===")

        dev = clean[clean["split"] == "development"]
        medians = dev[features].median()
        if medians.isna().any():
            missing = medians[medians.isna()].index.tolist()
            raise ValueError(f"Features entirely missing in development for {name}: {missing}")

        # Fit preprocessing only on clean development utterances.
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
        if centroids.empty:
            raise ValueError("No clean reference centroids could be formed.")

        scored_by_condition = {}
        for condition in CONDITIONS:
            scored = score_condition(
                trials, clean, mismatch, refs, features, medians,
                scaler, centroids, condition
            )
            scored_by_condition[condition] = scored
            print(f"{condition}: scored trials={len(scored)}")

        # Calibrator is fitted only on clean calibration trials for this feature
        # configuration, then held fixed for all mismatch conditions.
        clean_scores = scored_by_condition["clean"]
        cal = clean_scores[
            (clean_scores["split"] == "calibration")
            & np.isfinite(clean_scores["raw_score"])
        ].copy()
        if cal.empty or cal["target"].nunique() < 2:
            raise ValueError(f"Insufficient clean calibration trials for {name}")

        calibrator = LogLRCalibrator().fit(
            cal["raw_score"].to_numpy(), cal["target"].to_numpy()
        )
        joblib.dump(calibrator, MODELS / f"{name}_clean_calibrator.joblib")

        for condition in CONDITIONS:
            scored = scored_by_condition[condition]
            if scored.empty:
                continue

            val = scored[
                (scored["split"] == "validation")
                & np.isfinite(scored["raw_score"])
            ].copy()
            if val.empty or val["target"].nunique() < 2:
                print(f"WARNING: skip metrics for {name}/{condition}; validation incomplete")
                continue

            val["log_lr"] = calibrator.transform(val["raw_score"].to_numpy())
            raw = val["raw_score"].to_numpy(dtype=float)
            y = val["target"].to_numpy(dtype=int)
            llr = val["log_lr"].to_numpy(dtype=float)

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

            val.to_csv(
                OUT / f"{name}_{condition}_validation.csv",
                index=False,
            )
            print(
                f"{condition}: N={len(val)}, Cllr={c:.6f}, "
                f"minCllr={min_c:.6f}, EER={eer(raw, y):.6f}"
            )

    df = pd.DataFrame(results)
    output = Path("results/leave_one_group_out_cross_condition_metrics.csv")
    df.to_csv(output, index=False)
    print("\n=== CROSS-CONDITION LOGO RESULTS ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
