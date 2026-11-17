from pathlib import Path
import pandas as pd, soundfile as sf
from sklearn.model_selection import train_test_split

root = Path("data/raw/VCTK-Corpus-0.92/wav48_silence_trimmed")
rows = []

for spk_dir in sorted(root.iterdir()):
    if not spk_dir.is_dir():
        continue

    for wav in sorted(list(spk_dir.glob("*.flac")) + list(spk_dir.glob("*.wav"))):
        info = sf.info(str(wav))

        if info.duration < 1.0:
            continue

        rows.append({
            "utterance_id": wav.stem,
            "speaker_id": spk_dir.name,
            "file_path": str(wav),
            "source_file_id": wav.stem,
            "duration_s": info.duration,
            "sr": info.samplerate,
            "condition": "clean",
        })

df = pd.DataFrame(rows)
speakers = sorted(df.speaker_id.unique())

dev, rest = train_test_split(
    speakers,
    test_size=0.40,
    random_state=2026
)

cal, val = train_test_split(
    rest,
    test_size=0.55,
    random_state=2026
)

role = {s: "development" for s in dev}
role.update({s: "calibration" for s in cal})
role.update({s: "validation" for s in val})

df["split"] = df.speaker_id.map(role)

df.to_csv("manifests/utterances.csv", index=False)

assert df.groupby("speaker_id").split.nunique().max() == 1

print(df.groupby("split").speaker_id.nunique())