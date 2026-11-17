import json
import os
import random

from datasets import load_dataset

SEED = 42
DATASET_ID = "isaacus/open-australian-legal-qa"


def doc_key(row):
    src = row["source"]
    return src.get("citation") or src["version_id"]


def main():
    ds = load_dataset(DATASET_ID, split="train")
    print("Columns:", ds.column_names)
    rows = [dict(r) for r in ds]
    print("Total pairs:", len(rows))

    # Group pairs by citation so no instrument appears in two splits
    groups = {}
    for r in rows:
        groups.setdefault(doc_key(r), []).append(r)
    print("Unique citations:", len(groups))
    print("Unique version_ids:", len({r["source"]["version_id"] for r in rows}))

    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)

    n = len(rows)
    splits = {"train": [], "val": [], "test": []}
    for k in keys:
        if len(splits["train"]) < 0.8 * n:
            splits["train"].extend(groups[k])
        elif len(splits["val"]) < 0.1 * n:
            splits["val"].extend(groups[k])
        else:
            splits["test"].extend(groups[k])

    # Safety check: no citation in more than one split
    cites = {name: {doc_key(r) for r in rs} for name, rs in splits.items()}
    assert not (cites["train"] & cites["val"])
    assert not (cites["train"] & cites["test"])
    assert not (cites["val"] & cites["test"])
    print("Overlap check passed: no citation appears in two splits.")

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    stats = {}
    for name, rs in splits.items():
        with open(f"data/processed/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                slim = {k: v for k, v in r.items() if k not in ("prompt", "text")}
                f.write(json.dumps(slim, ensure_ascii=False) + "\n")
        stats[name] = {
            "questions": len(rs),
            "documents": len({r["source"]["version_id"] for r in rs}),
            "mean_answer_words": round(sum(len(r["answer"].split()) for r in rs) / len(rs), 1),
        }
    with open("results/split_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
