import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

B = 1000
SEED = 2026

CONDITIONS = [
    "clean",
    "noise20",
    "noise10",
    "telephone",
    "duration2s",
]

LOGLR_DIR = Path(
    "scores/cross_condition/validation_loglr"
)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("results/figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

def get_score_column(df):
    if "raw_score" in df.columns:
        return "raw_score"

    if "score" in df.columns:
        return "score"

    raise KeyError(
        f"No raw score column. Columns: {df.columns.tolist()}"
    )


def get_loglr_column(df):
    # Antisipasi nama kolom Week 9
    if "loglr" in df.columns:
        return "loglr"

    if "log_lr" in df.columns:
        return "log_lr"

    raise KeyError(
        f"No logLR column. Columns: {df.columns.tolist()}"
    )


def calculate_metrics(df):

    score_col = get_score_column(df)
    loglr_col = get_loglr_column(df)

    scores = df[
        score_col
    ].to_numpy(dtype=float)

    loglr = df[
        loglr_col
    ].to_numpy(dtype=float)

    labels = df[
        "target"
    ].to_numpy(dtype=int)

    c = cllr(
        loglr,
        labels,
    )

    mc = empirical_min_cllr(
        scores,
        labels,
    )

    e = eer(
        scores,
        labels,
    )

    misleading = misleading_rates(
        loglr,
        labels,
    )

    return {
        "cllr": c,
        "min_cllr": mc,
        "calibration_loss": c - mc,
        "eer": e,
        "misleading_rate":
            misleading["misleading"],
        "strong_misleading_rate":
            misleading["strong_misleading"],
    }


# =========================================================
# SPEAKER BOOTSTRAP
# =========================================================

def speaker_bootstrap(
    df,
    B=1000,
    seed=2026,
):

    rng = np.random.default_rng(seed)

    speakers = (
        df["questioned_speaker"]
        .drop_duplicates()
        .to_numpy()
    )

    # Pre-group supaya lebih cepat
    groups = {
        speaker: df[
            df["questioned_speaker"] == speaker
        ]
        for speaker in speakers
    }

    metric_names = [
        "cllr",
        "min_cllr",
        "calibration_loss",
        "eer",
        "misleading_rate",
        "strong_misleading_rate",
    ]

    values = {
        name: []
        for name in metric_names
    }

    for b in range(B):

        chosen = rng.choice(
            speakers,
            size=len(speakers),
            replace=True,
        )

        sample = pd.concat(
            [
                groups[speaker]
                for speaker in chosen
            ],
            ignore_index=True,
        )

        metrics = calculate_metrics(
            sample
        )

        for name in metric_names:
            values[name].append(
                metrics[name]
            )

        if (b + 1) % 100 == 0:
            print(
                f"    bootstrap {b + 1}/{B}"
            )

    output = {}

    for name in metric_names:

        q = np.quantile(
            values[name],
            [0.025, 0.5, 0.975],
        )

        output[name] = {
            "lower_95": q[0],
            "median": q[1],
            "upper_95": q[2],
        }

    return output


# =========================================================
# LOAD WEEK 9 VALIDATION LOGLR
# =========================================================

data = {}

print(
    "=== WEEK 10 BOOTSTRAP + PLOTS ==="
)

for condition in CONDITIONS:

    path = (
        LOGLR_DIR
        / f"{condition}_validation_loglr.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Week 9 file: {path}"
        )

    df = pd.read_csv(path)

    data[condition] = df

    print(
        condition,
        "| trials =",
        len(df),
        "| questioned speakers =",
        df["questioned_speaker"].nunique(),
    )


# =========================================================
# SPEAKER BOOTSTRAP
# =========================================================

bootstrap_rows = []

for condition in CONDITIONS:

    print(
        f"\n--- Bootstrap: {condition} ---"
    )

    result = speaker_bootstrap(
        data[condition],
        B=B,
        seed=SEED,
    )

    for metric, q in result.items():

        bootstrap_rows.append({
            "condition": condition,
            "metric": metric,
            "lower_95": q["lower_95"],
            "median": q["median"],
            "upper_95": q["upper_95"],
            "B": B,
            "seed": SEED,
        })


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)

bootstrap_path = (
    RESULTS_DIR
    / "bootstrap_ci.csv"
)

bootstrap_df.to_csv(
    bootstrap_path,
    index=False,
)

print(
    "\nSaved bootstrap CI:",
    bootstrap_path,
)


# =========================================================
# LOAD WEEK 9 POINT ESTIMATES
# =========================================================

metrics_path = (
    RESULTS_DIR
    / "calibration_robustness_metrics.csv"
)

metrics_df = pd.read_csv(
    metrics_path
)


# =========================================================
# WORST MISMATCH BERDASARKAN Cllr
# =========================================================

mismatch = metrics_df[
    metrics_df["condition"] != "clean"
]

worst_condition = mismatch.loc[
    mismatch["cllr"].idxmax(),
    "condition",
]

print(
    "\nWorst mismatch by Cllr:",
    worst_condition,
)


# =========================================================
# TIPPETT PLOT
# =========================================================

def tippett_plot(
    df,
    title,
    out_png,
):

    loglr_col = get_loglr_column(df)

    plt.figure(
        figsize=(6, 4)
    )

    for target, label in [
        (1, "same-speaker"),
        (0, "different-speaker"),
    ]:

        v = np.sort(
            df.loc[
                df["target"] == target,
                loglr_col,
            ].to_numpy()
        )

        cdf = (
            np.arange(
                1,
                len(v) + 1,
            )
            / len(v)
        )

        plt.plot(
            v,
            cdf,
            label=label,
        )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "log likelihood ratio"
    )

    plt.ylabel(
        "empirical cumulative proportion"
    )

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        out_png,
        dpi=200,
    )

    plt.close()


# =========================================================
# TIPPETT CLEAN
# =========================================================

clean_tippett = (
    FIGURES_DIR
    / "tippett_clean.png"
)

tippett_plot(
    data["clean"],
    "Tippett Plot - Clean Matched",
    clean_tippett,
)

print(
    "Saved:",
    clean_tippett,
)


# =========================================================
# TIPPETT WORST MISMATCH
# =========================================================

worst_tippett = (
    FIGURES_DIR
    / f"tippett_{worst_condition}.png"
)

tippett_plot(
    data[worst_condition],
    f"Tippett Plot - {worst_condition}",
    worst_tippett,
)

print(
    "Saved:",
    worst_tippett,
)


# =========================================================
# Cllr DEGRADATION PLOT
# =========================================================

plot_df = (
    metrics_df
    .set_index("condition")
    .loc[CONDITIONS]
    .reset_index()
)

plt.figure(
    figsize=(7, 4)
)

plt.plot(
    plot_df["condition"],
    plot_df["cllr"],
    marker="o",
)

plt.xlabel("condition")
plt.ylabel("Cllr")

plt.title(
    "Cllr Degradation Across Conditions"
)

plt.tight_layout()

cllr_plot = (
    FIGURES_DIR
    / "cllr_degradation.png"
)

plt.savefig(
    cllr_plot,
    dpi=200,
)

plt.close()

print(
    "Saved:",
    cllr_plot,
)


# =========================================================
# STRONG MISLEADING DEGRADATION
# =========================================================

plt.figure(
    figsize=(7, 4)
)

plt.plot(
    plot_df["condition"],
    plot_df[
        "strong_misleading_rate"
    ],
    marker="o",
)

plt.xlabel("condition")

plt.ylabel(
    "strong misleading rate"
)

plt.title(
    "Strong Misleading Evidence Across Conditions"
)

plt.tight_layout()

strong_plot = (
    FIGURES_DIR
    / "strong_misleading_degradation.png"
)

plt.savefig(
    strong_plot,
    dpi=200,
)

plt.close()

print(
    "Saved:",
    strong_plot,
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print(
    "\n=== WEEK 10 SUMMARY ==="
)

print(
    bootstrap_df.to_string(
        index=False
    )
)

print(
    "\n=== WEEK 10 BOOTSTRAP + PLOTS SELESAI ==="
)
