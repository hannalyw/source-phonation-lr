import pandas as pd, numpy as np

rng = np.random.default_rng(2026)
df = pd.read_csv("manifests/utterances.csv")

rows, ref_rows = [], []

for split, part in df.groupby("split"):
    by_spk = {s:g.sort_values("utterance_id") for s,g in part.groupby("speaker_id")}
    speakers = sorted(by_spk)

    for spk, g in by_spk.items():
        ref_ids = g.head(min(5, max(2, len(g)//3))).utterance_id.tolist()

        ref_rows.extend(
            {"speaker_id":spk, "utterance_id":u, "split":split}
            for u in ref_ids
        )

        q = g[~g.utterance_id.isin(ref_ids)]

        for _, u in q.iterrows():
            rows.append({
                "trial_id": f"{split}_{u.utterance_id}_{spk}_T",
                "split": split,
                "questioned_id": u.utterance_id,
                "questioned_speaker": spk,
                "reference_speaker": spk,
                "target": 1,
                "condition": u.condition,
            })

            impostors = [x for x in speakers if x != spk]

            for imp in rng.choice(
                impostors,
                size=min(10, len(impostors)),
                replace=False
            ):
                rows.append({
                    "trial_id": f"{split}_{u.utterance_id}_{imp}_N",
                    "split": split,
                    "questioned_id": u.utterance_id,
                    "questioned_speaker": spk,
                    "reference_speaker": imp,
                    "target": 0,
                    "condition": u.condition,
                })

pd.DataFrame(ref_rows).to_csv(
    "manifests/reference_utterances.csv",
    index=False
)

pd.DataFrame(rows).to_csv(
    "manifests/trials.csv",
    index=False
)