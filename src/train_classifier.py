"""Stage 1c: baseline wheat disease classifier (transfer learning)."""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class LeafDataset(Dataset):
    def __init__(self, df, class_to_idx, tfm):
        self.paths = df["path"].tolist()
        self.targets = [class_to_idx[l] for l in df["label"]]
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.tfm(im.convert("RGB"))
        return x, self.targets[i]


def make_transforms(size):
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(0.3, 0.3, 0.2, 0.02),
        T.RandomApply([T.GaussianBlur(5)], p=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    evaluate = T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    return train, evaluate


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        ps.append(logits.argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/baseline")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.splits)
    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_tfm, eval_tfm = make_transforms(args.img_size)
    loaders = {}
    for split, tfm in [("train", train_tfm), ("val", eval_tfm), ("test", eval_tfm)]:
        ds = LeafDataset(df[df.split == split], class_to_idx, tfm)
        loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"),
                                    num_workers=args.workers, pin_memory=(device == "cuda"))

    model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=len(classes))
    model.to(device)

    counts = np.bincount(loaders["train"].dataset.targets, minlength=len(classes)).astype(float)
    weights = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_f1 = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            running += loss.item() * len(y)
            n += len(y)
        scheduler.step()

        yv, pv = predict(model, loaders["val"], device)
        val_f1 = f1_score(yv, pv, average="macro", zero_division=0)
        val_acc = float((yv == pv).mean())
        history.append({"epoch": epoch, "train_loss": running / n, "val_macro_f1": val_f1, "val_acc": val_acc})
        flag = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model": model.state_dict(), "classes": classes, "arch": args.model,
                        "img_size": args.img_size}, out / "best.pt")
            flag = "  <- best"
        print(f"epoch {epoch:3d} | loss {running/n:.4f} | val macro-F1 {val_f1:.3f} | val acc {val_acc:.3f} "
              f"| {time.time()-t0:.0f}s{flag}")

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    yt, pt = predict(model, loaders["test"], device)
    labels_idx = list(range(len(classes)))
    report = classification_report(yt, pt, labels=labels_idx, target_names=classes,
                                   zero_division=0, output_dict=True)
    metrics = {
        "best_val_macro_f1": best_f1,
        "test_macro_f1": f1_score(yt, pt, average="macro", zero_division=0),
        "test_accuracy": float((yt == pt).mean()),
        "per_class_recall": {c: report[c]["recall"] for c in classes},
        "n_test": int(len(yt)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(classification_report(yt, pt, labels=labels_idx, target_names=classes, zero_division=0))

    cm = confusion_matrix(yt, pt, labels=labels_idx)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False,
                                                            xticks_rotation=45)
    ax.set_title("Test set confusion matrix")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)
    print(f"Saved checkpoint, metrics and confusion matrix to {out}/")


if __name__ == "__main__":
    main()
