import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from calibration import LogLRCalibrator
from lr_metrics import cllr, empirical_min_cllr, eer, misleading_rates


GROUPS = [
    "f0",
    "voice_quality",
    "tilt",
    "all",
]

Path("models/ablation").mkdir(parents=True, exist_ok=True)
Path("scores/ablation").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True)

results = []
validation_trial_ids = {}


for group in GROUPS:

    print(f"\n=== ABLATION: {group} ===")

    # Raw score hasil feature-group ablation
    scores = pd.read_csv(
        f"scores/ablation/{group}_scores.csv"
    )

    # Pisahkan calibration dan validation
    calibration = scores[
        scores["split"] == "calibration"
    ].copy()

    validation = scores[
        scores["split"] == "validation"
    ].copy()

    # Gunakan hanya raw score finite
    calibration = calibration[
        np.isfinite(calibration["raw_score"])
    ].copy()

    validation = validation[
        np.isfinite(validation["raw_score"])
    ].copy()

    print("Calibration trials:", len(calibration))
    print("Validation trials :", len(validation))

    # Simpan trial_id untuk pengecekan identik
    validation_trial_ids[group] = validation[
        "trial_id"
    ].tolist()

    # Fit kalibrator HANYA pada calibration
    cal = LogLRCalibrator()

    cal.fit(
        calibration["raw_score"].to_numpy(),
        calibration["target"].to_numpy(),
    )

    joblib.dump(
        cal,
        f"models/ablation/{group}_calibrator.joblib"
    )

    # Terapkan tanpa fit ulang pada validation
    validation["loglr"] = cal.transform(
        validation["raw_score"].to_numpy()
    )

    validation.to_csv(
        f"scores/ablation/{group}_validation_loglr.csv",
        index=False,
    )

    # Metrics validation
    raw = validation["raw_score"].to_numpy()
    llr = validation["loglr"].to_numpy()
    y = validation["target"].to_numpy()

    result_cllr = cllr(llr, y)
    result_min_cllr = empirical_min_cllr(raw, y)
    result_eer = eer(raw, y)

    calibration_loss = (
        result_cllr - result_min_cllr
    )

    misleading = misleading_rates(llr, y)

    results.append({
        "feature_group": group,
        "n_calibration": len(calibration),
        "n_validation": len(validation),
        "cllr": result_cllr,
        "min_cllr": result_min_cllr,
        "calibration_loss": calibration_loss,
        "eer": result_eer,
        "misleading": misleading["misleading"],
        "strong_misleading":
            misleading["strong_misleading"],
    })


# Pastikan validation trial_id identik
base = validation_trial_ids["all"]

for group in GROUPS:
    if validation_trial_ids[group] != base:
        raise RuntimeError(
            f"trial_id validation tidak identik: {group}"
        )

print("\nValidation trial_id identik untuk semua feature group: YES")


# Simpan tabel hasil Week 6
results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/feature_group_ablation_metrics.csv",
    index=False,
)

print("\n=== WEEK 6: FEATURE-GROUP ABLATION RESULTS ===")
print(results_df.to_string(index=False))

print("\nSaved:")
print("results/feature_group_ablation_metrics.csv")
print("\n=== WEEK 6 SELESAI ===")

