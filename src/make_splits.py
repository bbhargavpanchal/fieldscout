"""Stage 1b: leakage-safe train/val/test splits."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from data_utils import find_images
from dedup import phash_ints, group_near_duplicates


def grouped_split(df, test_frac, val_frac, seed):
    df = df.copy()
    df["split"] = "train"

    n_test = max(2, round(1 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, df["label"], df["group"]))
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"

    rest = df[df.split == "train"]
    n_val = max(2, round((1 - test_frac) / val_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed + 1)
    _, val_idx = next(sgkf.split(rest, rest["label"], rest["group"]))
    df.loc[rest.index[val_idx], "split"] = "val"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="data/splits.csv")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--dup-distance", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = list(find_images(args.data))
    df = pd.DataFrame({"path": [str(p) for p, _ in items], "label": [l for _, l in items]})
    df["group"] = group_near_duplicates(phash_ints(df["path"].tolist()), args.dup_distance)

    largest = df["group"].value_counts().iloc[0]
    if largest > max(10, 0.02 * len(df)):
        print(f"WARNING: largest near-duplicate group has {largest} images. Try a smaller --dup-distance.")

    df = grouped_split(df, args.test_frac, args.val_frac, args.seed)

    leaks = df.groupby("group")["split"].nunique()
    assert (leaks == 1).all(), "a near-duplicate group was split across sets"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(pd.crosstab(df["label"], df["split"], margins=True))
    print(f"\nWrote {args.out}. No near-duplicate group crosses splits.")


if __name__ == "__main__":
    main()
