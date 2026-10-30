"""Stage 1a: look at the data before touching a model."""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from data_utils import find_images
from dedup import phash_ints, group_near_duplicates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--dup-distance", type=int, default=6)
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = list(find_images(args.data))
    if not items:
        raise SystemExit(f"No images found under {args.data}")
    paths = [p for p, _ in items]
    labels = [l for _, l in items]

    sizes = []
    for p in paths:
        with Image.open(p) as im:
            sizes.append(im.size)
    w = np.array([s[0] for s in sizes])
    h = np.array([s[1] for s in sizes])

    hashes = phash_ints(paths)
    groups = group_near_duplicates(hashes, args.dup_distance)
    group_sizes = Counter(groups)
    dup_images = sum(c for c in group_sizes.values() if c > 1)

    df = pd.DataFrame({"path": [str(p) for p in paths], "label": labels, "group": groups})
    mixed = df.groupby("group")["label"].nunique()
    mixed_groups = int((mixed > 1).sum())

    counts = Counter(labels)
    report = {
        "n_images": len(paths),
        "class_counts": dict(sorted(counts.items())),
        "imbalance_ratio_max_over_min": round(max(counts.values()) / min(counts.values()), 2),
        "width": {"min": int(w.min()), "median": int(np.median(w)), "max": int(w.max())},
        "height": {"min": int(h.min()), "median": int(np.median(h)), "max": int(h.max())},
        "near_duplicate_groups": int(sum(1 for c in group_sizes.values() if c > 1)),
        "images_in_near_duplicate_groups": int(dup_images),
        "groups_with_mixed_labels": mixed_groups,
        "largest_group_size": int(max(group_sizes.values())),
    }
    (out / "data_report.json").write_text(json.dumps(report, indent=2))
    df.to_csv(out / "image_index.csv", index=False)

    classes = sorted(counts)
    k = args.samples
    fig, axes = plt.subplots(len(classes), k, figsize=(2.2 * k, 2.2 * len(classes)), squeeze=False)
    rng = np.random.default_rng(0)
    for r, c in enumerate(classes):
        cpaths = df.loc[df.label == c, "path"].tolist()
        pick = rng.choice(len(cpaths), size=min(k, len(cpaths)), replace=False)
        for j in range(k):
            ax = axes[r][j]
            ax.axis("off")
            if j < len(pick):
                with Image.open(cpaths[pick[j]]) as im:
                    im = im.convert("RGB")
                    im.thumbnail((256, 256))
                    ax.imshow(im)
            if j == 0:
                ax.set_title(c, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=110)

    print(json.dumps(report, indent=2))
    if mixed_groups:
        print(f"WARNING: {mixed_groups} near-duplicate groups contain more than one label.")


if __name__ == "__main__":
    main()
