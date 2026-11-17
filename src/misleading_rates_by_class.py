from pathlib import Path
import pandas as pd
import numpy as np

# Root proyek = folder tempat script ini berada
ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "scores" / "cross_condition" / "validation_loglr"
OUTPUT_DIR = ROOT / "results"
OUTPUT_FILE = OUTPUT_DIR / "misleading_rates_by_class.csv"

# Sesuaikan nama file jika nama output di proyekmu berbeda.
CONDITIONS = {
    "clean": "clean_validation_loglr.csv",
    "noise20": "noise20_validation_loglr.csv",
    "noise10": "noise10_validation_loglr.csv",
    "telephone": "telephone_validation_loglr.csv",
    "duration2s": "duration2s_validation_loglr.csv",
}

# Sesuaikan jika nama kolom pada CSV berbeda.
LABEL_COL = "label"
LOGLR_COL = "logLR"

THRESHOLD = np.log(10)


def find_column(df, candidates):
    """Cari kolom berdasarkan beberapa kemungkinan nama."""
    lower_map = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"Kolom tidak ditemukan. Dicari: {candidates}. "
        f"Kolom tersedia: {list(df.columns)}"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for condition, filename in CONDITIONS.items():
        path = INPUT_DIR / filename

        if not path.exists():
            print(f"[SKIP] File tidak ditemukan: {path}")
            continue

        df = pd.read_csv(path)

        label_col = find_column(
            df, [LABEL_COL, "target", "is_target", "same_speaker", "y"]
        )
        loglr_col = find_column(
            df, [LOGLR_COL, "loglr", "log_lr", "log_likelihood_ratio"]
        )

        df = df[[label_col, loglr_col]].copy()
        df[loglr_col] = pd.to_numeric(df[loglr_col], errors="coerce")
        df = df.dropna(subset=[label_col, loglr_col])

        # Konversi label menjadi 1 = target, 0 = non-target.
        labels = df[label_col]

        if labels.dtype == object:
            normalized = labels.astype(str).str.strip().str.lower()
            mapping = {
                "1": 1, "target": 1, "same": 1,
                "same-speaker": 1, "same_speaker": 1,
                "0": 0, "non-target": 0, "nontarget": 0,
                "different": 0, "different-speaker": 0,
                "different_speaker": 0,
            }
            labels = normalized.map(mapping)
        else:
            labels = pd.to_numeric(labels, errors="coerce")

        df["binary_label"] = labels
        df = df[df["binary_label"].isin([0, 1])]

        loglr = df[loglr_col].to_numpy()
        binary_label = df["binary_label"].astype(int).to_numpy()

        for class_name, class_value in [
            ("target", 1),
            ("non_target", 0),
        ]:
            mask = binary_label == class_value
            class_loglr = loglr[mask]
            n = len(class_loglr)

            if n == 0:
                print(
                    f"[WARNING] Tidak ada trial {class_name} "
                    f"pada kondisi {condition}"
                )
                continue

            if class_value == 1:
                # Target salah arah jika logLR < 0.
                misleading_mask = class_loglr < 0
                strong_mask = class_loglr <= -THRESHOLD
            else:
                # Non-target salah arah jika logLR > 0.
                misleading_mask = class_loglr > 0
                strong_mask = class_loglr >= THRESHOLD

            misleading_count = int(misleading_mask.sum())
            strong_count = int(strong_mask.sum())

            results.append({
                "condition": condition,
                "class": class_name,
                "n_trials": n,
                "misleading_count": misleading_count,
                "misleading_rate": misleading_count / n,
                "strong_misleading_count": strong_count,
                "strong_misleading_rate": strong_count / n,
            })

        print(f"[OK] {condition}: {len(df)} trial valid")

    if not results:
        raise RuntimeError(
            "Tidak ada hasil yang dihitung. Periksa INPUT_DIR "
            "dan nama file pada CONDITIONS."
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\n=== MISLEADING RATE PER KELAS ===")
    print(result_df.to_string(index=False))
    print(f"\nHasil disimpan di: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
