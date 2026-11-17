import pandas as pd
from pathlib import Path

UTTERANCE_ID = "p225_003_mic2"

utterances = pd.read_csv("manifests/utterances.csv")

original = utterances[
    utterances["utterance_id"] == UTTERANCE_ID
].iloc[0]

conditions = {
    "noise20": f"data/mismatch_examples/{UTTERANCE_ID}_noise20.wav",
    "noise10": f"data/mismatch_examples/{UTTERANCE_ID}_noise10.wav",
    "telephone": f"data/mismatch_examples/{UTTERANCE_ID}_telephone.wav",
    "duration2s": f"data/mismatch_examples/{UTTERANCE_ID}_duration2s.wav",
}

rows = []

for condition, path in conditions.items():

    if not Path(path).exists():
        raise FileNotFoundError(path)

    rows.append({
        "utterance_id": f"{UTTERANCE_ID}_{condition}",
        "speaker_id": original["speaker_id"],
        "file_path": path,
        "source_file_id": original["source_file_id"],
        "duration_s": 2.0 if condition == "duration2s"
                      else original["duration_s"],
        "sr": 16000,
        "condition": condition,
        "split": original["split"],
    })

out = pd.DataFrame(rows)

Path("manifests").mkdir(exist_ok=True)

out.to_csv(
    "manifests/mismatch_examples.csv",
    index=False
)

print("\n=== WEEK 7 MISMATCH EXAMPLE MANIFEST ===")
print(out.to_string(index=False))

print("\nMetadata inheritance:")
print("speaker_id unique    :", out["speaker_id"].nunique() == 1)
print("source_file_id unique:", out["source_file_id"].nunique() == 1)
print("split unique         :", out["split"].nunique() == 1)

print("\nSaved: manifests/mismatch_examples.csv")
