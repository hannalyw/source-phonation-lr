import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, "src")

from lr_metrics import (
    cllr,
    empirical_min_cllr,
    eer,
    misleading_rates,
)


# =========================================================
# CONFIG
# =========================================================

CONDITIONS = [
    "clean",
    "noise20",
    "noise10",
    "telephone",
    "duration2s",
]

SCORE_DIR = Path("scores/cross_condition")
OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATOR_PATH = Path("models/calibrator.joblib")

VALIDATION_LOGLR_DIR = Path(
    "scores/cross_condition/validation_loglr"
)
VALIDATION_LOGLR_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# SCORE FILES
# =========================================================

score_files = {
    "clean": Path("scores/source_scores.csv"),
    "noise20": SCORE_DIR / "noise20_scores.csv",
    "noise10": SCORE_DIR / "noise10_scores.csv",
    "telephone": SCORE_DIR / "telephone_scores.csv",
    "duration2s": SCORE_DIR / "duration2s_scores.csv",
}


# =========================================================
# LOAD CLEAN CALIBRATOR FROM WEEK 5
# =========================================================

print("=== WEEK 9 CALIBRATION ROBUSTNESS ===")

if not CALIBRATOR_PATH.exists():
    raise FileNotFoundError(
        f"Calibrator not found: {CALIBRATOR_PATH}"
    )

calibrator = joblib.load(CALIBRATOR_PATH)

print("Loaded clean calibrator:")
print(CALIBRATOR_PATH)

print(
    "IMPORTANT: calibrator is NOT refitted "
    "for mismatch conditions."
)


# =========================================================
# EVALUATE CONDITIONS
# =========================================================

results = []

for condition in CONDITIONS:

    print(f"\n--- {condition} ---")

    path = score_files[condition]

    if not path.exists():
        raise FileNotFoundError(
            f"Missing score file: {path}"
        )

    df = pd.read_csv(path)

    # -----------------------------------------------------
    # VALIDATION ONLY
    # -----------------------------------------------------

    val = df[
        df["split"] == "validation"
    ].copy()

    print("Validation trials:", len(val))

    if len(val) == 0:
        raise RuntimeError(
            f"No validation trials for {condition}"
        )

    # -----------------------------------------------------
    # SCORE COLUMN
    # clean Week 4    = raw_score
    # mismatch Week 8 = score
    # -----------------------------------------------------

    if "raw_score" in val.columns:
        score_column = "raw_score"

    elif "score" in val.columns:
        score_column = "score"

    else:
        raise KeyError(
            f"No score column found for {condition}. "
            f"Columns: {val.columns.tolist()}"
        )

    print("Score column     :", score_column)

    raw_scores = val[
        score_column
    ].to_numpy(dtype=float)

    labels = val[
        "target"
    ].to_numpy(dtype=int)

    # -----------------------------------------------------
    # APPLY SAME CLEAN CALIBRATOR
    # NO REFITTING PER CONDITION
    # -----------------------------------------------------

    loglr = calibrator.transform(
        raw_scores
    )

    val["log_lr"] = loglr

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    condition_cllr = cllr(
        loglr,
        labels,
    )

    condition_min_cllr = empirical_min_cllr(
        raw_scores,
        labels,
    )

    calibration_loss = (
        condition_cllr
        - condition_min_cllr
    )

    condition_eer = eer(
        raw_scores,
        labels,
    )

    # misleading_rates() returns a DICTIONARY
    misleading_result = misleading_rates(
        loglr,
        labels,
    )

    misleading = misleading_result[
        "misleading"
    ]

    strong_misleading = misleading_result[
        "strong_misleading"
    ]

    target_trials = int(
        (labels == 1).sum()
    )

    non_target_trials = int(
        (labels == 0).sum()
    )

    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print("Target trials     :", target_trials)
    print("Non-target trials :", non_target_trials)

    print("Cllr              :", condition_cllr)
    print("minCllr           :", condition_min_cllr)
    print("Calibration loss  :", calibration_loss)
    print("EER               :", condition_eer)
    print("Misleading        :", misleading)
    print("Strong misleading :", strong_misleading)

    # -----------------------------------------------------
    # SAVE VALIDATION LOG LR
    # -----------------------------------------------------

    loglr_path = (
        VALIDATION_LOGLR_DIR
        / f"{condition}_validation_loglr.csv"
    )

    val.to_csv(
        loglr_path,
        index=False,
    )

    print("Saved:", loglr_path)

    # -----------------------------------------------------
    # STORE SUMMARY
    # -----------------------------------------------------

    results.append({
        "condition": condition,
        "validation_trials": len(val),
        "target_trials": target_trials,
        "non_target_trials": non_target_trials,
        "cllr": condition_cllr,
        "min_cllr": condition_min_cllr,
        "calibration_loss": calibration_loss,
        "eer": condition_eer,
        "misleading_rate": misleading,
        "strong_misleading_rate": strong_misleading,
    })


# =========================================================
# SAVE SUMMARY
# =========================================================

results_df = pd.DataFrame(results)

output_path = (
    OUT_DIR
    / "calibration_robustness_metrics.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)


# =========================================================
# WORST MISMATCH CONDITIONS
# =========================================================

mismatch_only = results_df[
    results_df["condition"] != "clean"
].copy()

worst_cllr = mismatch_only.loc[
    mismatch_only["cllr"].idxmax()
]

worst_misleading = mismatch_only.loc[
    mismatch_only["misleading_rate"].idxmax()
]

worst_strong_misleading = mismatch_only.loc[
    mismatch_only["strong_misleading_rate"].idxmax()
]


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n=== WEEK 9 ROBUSTNESS SUMMARY ===")

print(
    results_df.to_string(
        index=False
    )
)

print(
    "\nWorst mismatch by Cllr:",
    worst_cllr["condition"],
    "| Cllr =",
    worst_cllr["cllr"],
)

print(
    "Worst mismatch by misleading rate:",
    worst_misleading["condition"],
    "| misleading =",
    worst_misleading["misleading_rate"],
)

print(
    "Worst mismatch by strong misleading rate:",
    worst_strong_misleading["condition"],
    "| strong misleading =",
    worst_strong_misleading[
        "strong_misleading_rate"
    ],
)

print("\nSaved:")
print(output_path)

print(
    "\n=== WEEK 9 CALIBRATION ROBUSTNESS SELESAI ==="
)
