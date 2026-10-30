"""Stage 3: error analysis and Grad-CAM (where is the model looking?).

Outputs, in --out:
  predictions.csv      one record per test image (true, predicted, confidence)
  metrics_ci.json      test macro-F1 with a 95% bootstrap confidence interval
  errors_gradcam.png   every misclassified test image with a Grad-CAM heatmap
  correct_gradcam.png  a few correctly classified images per class, for comparison
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from torchvision import transforms as T

from train_classifier import make_transforms


def gradcam(model, x, class_idx):
    """Grad-CAM on the final convolutional feature map (works for timm CNNs)."""
    feats = model.forward_features(x)          # B x C x H x W
    feats.retain_grad()
    logits = model.forward_head(feats)
    model.zero_grad()
    logits[0, class_idx].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), logits.softmax(1)[0].detach().cpu().numpy()


def bootstrap_macro_f1(y, p, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        scores.append(f1_score(y[idx], p[idx], average="macro", zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def plot_grid(rows, path, title, cols=4):
    """rows: list of (rgb_image, cam, caption)."""
    if not rows:
        return
    n = len(rows)
    r = int(np.ceil(n / cols))
    fig, axes = plt.subplots(r, cols, figsize=(3.2 * cols, 3.4 * r), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (img, cam, cap) in zip(axes.flat, rows):
        ax.imshow(img)
        ax.imshow(cam, cmap="jet", alpha=0.4)
        ax.set_title(cap, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run folder containing best.pt")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--correct-per-class", type=int, default=3)
    ap.add_argument("--max-errors", type=int, default=32)
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out) if args.out else run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(run / "best.pt", map_location=device)
    classes, size = ckpt["classes"], ckpt["img_size"]
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    _, eval_tfm = make_transforms(size)
    show_tfm = T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size)])  # same crop, no normalising

    test = pd.read_csv(args.splits).query("split == 'test'").reset_index(drop=True)
    records, errors, correct = [], [], {c: [] for c in classes}
    for _, row in test.iterrows():
        with Image.open(row["path"]) as im:
            im = im.convert("RGB")
            x = eval_tfm(im).unsqueeze(0).to(device)
            shown = np.asarray(show_tfm(im))
        with torch.no_grad():
            probs = model(x).softmax(1)[0].cpu().numpy()
        pred = int(probs.argmax())
        true = classes.index(row["label"])
        records.append({"path": row["path"], "true": row["label"], "pred": classes[pred],
                        "confidence": round(float(probs[pred]), 4),
                        "prob_true_class": round(float(probs[true]), 4)})
        if pred != true and len(errors) < args.max_errors:
            cam, _ = gradcam(model, x, pred)
            errors.append((shown, cam, f"true {row['label']}\npred {classes[pred]} ({probs[pred]:.2f})"))
        elif pred == true and len(correct[row["label"]]) < args.correct_per_class:
            cam, _ = gradcam(model, x, pred)
            correct[row["label"]].append((shown, cam, f"{row['label']} ({probs[pred]:.2f})"))

    df = pd.DataFrame(records)
    df.to_csv(out / "predictions.csv", index=False)

    y = df["true"].map(classes.index).to_numpy()
    p = df["pred"].map(classes.index).to_numpy()
    lo, hi = bootstrap_macro_f1(y, p)
    wrong = df[df.true != df.pred]
    summary = {
        "n_test": int(len(df)),
        "test_macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
        "macro_f1_95ci": [round(lo, 3), round(hi, 3)],
        "n_errors": int(len(wrong)),
        "mean_confidence_correct": round(float(df.loc[df.true == df.pred, "confidence"].mean()), 3),
        "mean_confidence_wrong": round(float(wrong["confidence"].mean()), 3) if len(wrong) else None,
        "most_common_confusions": {f"{t} -> {p_}": int(n) for (t, p_), n in
                                   wrong.groupby(["true", "pred"]).size()
                                        .sort_values(ascending=False).head(5).items()},
    }
    (out / "metrics_ci.json").write_text(json.dumps(summary, indent=2))

    plot_grid(errors, out / "errors_gradcam.png",
              "Misclassified test images (heatmap = evidence for the wrong prediction)")
    plot_grid([r for c in classes for r in correct[c]], out / "correct_gradcam.png",
              "Correct predictions (heatmap = evidence used)", cols=args.correct_per_class)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote predictions.csv, metrics_ci.json, errors_gradcam.png, correct_gradcam.png to {out}/")


if __name__ == "__main__":
    main()
