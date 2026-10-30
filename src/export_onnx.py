"""Stage 4a: export to ONNX, quantise to INT8, and compare size, speed and accuracy."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import timm
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image
from sklearn.metrics import f1_score

from train_classifier import make_transforms


def export(model, size, path):
    dummy = torch.randn(1, 3, size, size)
    kwargs = dict(input_names=["image"], output_names=["logits"], opset_version=17,
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except Exception as e:
        print(f"Classic exporter failed ({type(e).__name__}); trying the newer exporter.")
        torch.onnx.export(model, dummy, str(path), dynamo=True, external_data=False, **kwargs)


def load_test(splits, size, classes):
    _, eval_tfm = make_transforms(size)
    test = pd.read_csv(splits).query("split == 'test'")
    xs, ys = [], []
    for _, row in test.iterrows():
        with Image.open(row["path"]) as im:
            xs.append(eval_tfm(im.convert("RGB")).numpy())
        ys.append(classes.index(row["label"]))
    return np.stack(xs).astype(np.float32), np.array(ys)


def ort_session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1          # one CPU thread: closer to a small edge device
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def latency_ms(fn, x, warmup=5, runs=30):
    for _ in range(warmup):
        fn(x)
    t = time.perf_counter()
    for _ in range(runs):
        fn(x)
    return (time.perf_counter() - t) / runs * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="export")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    ckpt = torch.load(Path(args.run) / "best.pt", map_location="cpu")
    classes, size = ckpt["classes"], ckpt["img_size"]
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()

    fp32, int8 = out / "model_fp32.onnx", out / "model_int8.onnx"
    export(model, size, fp32)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    (out / "classes.txt").write_text("\n".join(classes) + "\n")

    x, y = load_test(args.splits, size, classes)
    sessions = {"onnx_fp32": ort_session(fp32), "onnx_int8": ort_session(int8)}

    def torch_logits(batch):
        with torch.no_grad():
            return model(torch.from_numpy(batch)).numpy()

    def ort_logits(sess):
        return lambda batch: sess.run(["logits"], {"image": batch})[0]

    runners = {"pytorch": torch_logits, **{k: ort_logits(s) for k, s in sessions.items()}}
    ref = np.concatenate([torch_logits(x[i:i + 8]) for i in range(0, len(x), 8)])

    report = {"img_size": size, "n_test": int(len(y)), "classes": classes}
    for name, fn in runners.items():
        logits = np.concatenate([fn(x[i:i + 8]) for i in range(0, len(x), 8)])
        pred = logits.argmax(1)
        size_mb = None if name == "pytorch" else round((fp32 if name == "onnx_fp32" else int8).stat().st_size / 1e6, 1)
        report[name] = {
            "size_mb": size_mb,
            "latency_ms_1_image_1_thread": round(latency_ms(fn, x[:1]), 1),
            "test_macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
            "agreement_with_pytorch": round(float((pred == ref.argmax(1)).mean()), 4),
            "max_abs_logit_diff_vs_pytorch": round(float(np.abs(logits - ref).max()), 5),
        }
    (out / "edge_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
