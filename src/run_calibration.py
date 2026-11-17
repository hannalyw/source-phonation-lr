import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from calibration import LogLRCalibrator
from lr_metrics import cllr, empirical_min_cllr, eer, misleading_rates


# Folder output
Path("models").mkdir(exist_ok=True)
Path("scores").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# 1. Baca raw score Minggu 4
scores = pd.read_csv("scores/source_scores.csv")

required = ["trial_id", "split", "target", "raw_score"]
missing = [col for col in required if col not in scores.columns]
if missing:
    sys.exit(f"Kolom tidak ditemukan: {missing}")

# 2. Pisahkan data calibration dan validation
calibration = scores[scores["split"] == "calibration"].copy()
validation = scores[scores["split"] == "validation"].copy()

# Jangan gunakan trial dengan raw_score tidak valid
calibration = calibration[
    np.isfinite(calibration["raw_score"])
].copy()

validation = validation[
    np.isfinite(validation["raw_score"])
].copy()

if calibration.empty or validation.empty:
    sys.exit("Data calibration atau validation kosong. Periksa kolom split.")

print("Jumlah trial calibration:", len(calibration))
print("Jumlah trial validation :", len(validation))

# 3. Fit kalibrator HANYA pada calibration
cal = LogLRCalibrator()
cal.fit(
    calibration["raw_score"].to_numpy(),
    calibration["target"].to_numpy()
)

joblib.dump(cal, "models/calibrator.joblib")

# 4. Terapkan model yang sama pada validation, tanpa fit ulang
validation["loglr"] = cal.transform(
    validation["raw_score"].to_numpy()
)

validation.to_csv(
    "scores/validation_loglr.csv",
    index=False
)

# 5. Hitung metrik pada validation
raw = validation["raw_score"].to_numpy()
llr = validation["loglr"].to_numpy()
y = validation["target"].to_numpy()

result_cllr = cllr(llr, y)
result_min_cllr = empirical_min_cllr(raw, y)
result_eer = eer(raw, y)
result_calibration_loss = result_cllr - result_min_cllr
result_misleading = misleading_rates(llr, y)

metrics = pd.DataFrame([{
    "split": "validation",
    "cllr": result_cllr,
    "min_cllr": result_min_cllr,
    "calibration_loss": result_calibration_loss,
    "eer": result_eer,
    "misleading": result_misleading["misleading"],
    "strong_misleading": result_misleading["strong_misleading"]
}])

metrics.to_csv(
    "results/validation_lr_metrics.csv",
    index=False
)

print("\n=== WEEK 5: VALIDATION RESULTS ===")
print(metrics.to_string(index=False))

print("\nFile tersimpan:")
print("models/calibrator.joblib")
print("scores/validation_loglr.csv")
print("results/validation_lr_metrics.csv")
