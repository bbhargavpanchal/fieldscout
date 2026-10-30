"""Stage 4b check: does the C++ program give the same answers as Python?"""
import argparse

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score

from train_classifier import make_transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--cpp", required=True, help="CSV written by fieldscout_infer")
    ap.add_argument("--splits", required=True, help="splits.csv, for the true labels")
    args = ap.parse_args()

    classes = [l.strip() for l in open(args.classes) if l.strip()]
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    size = sess.get_inputs()[0].shape[2]
    _, eval_tfm = make_transforms(size)

    cpp = pd.read_csv(args.cpp)
    truth = pd.read_csv(args.splits).set_index("path")["label"]

    py_pred, py_conf = [], []
    for path in cpp["path"]:
        with Image.open(path) as im:
            x = eval_tfm(im.convert("RGB")).unsqueeze(0).numpy()
        logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        py_pred.append(classes[int(p.argmax())])
        py_conf.append(float(p.max()))

    cpp["python_pred"] = py_pred
    cpp["python_conf"] = py_conf
    cpp["true"] = cpp["path"].map(truth)
    agree = (cpp["predicted_class"] == cpp["python_pred"]).mean()
    conf_diff = (cpp["confidence"] - cpp["python_conf"]).abs()

    print(f"images compared:               {len(cpp)}")
    print(f"C++ vs Python same prediction: {agree:.1%}")
    print(f"confidence difference:         mean {conf_diff.mean():.4f}, max {conf_diff.max():.4f}")
    print(f"C++ test macro-F1:             {f1_score(cpp['true'], cpp['predicted_class'], average='macro', zero_division=0):.4f}")
    print(f"C++ median inference time:     {cpp['inference_ms'].median():.1f} ms (1 thread)")
    diff = cpp[cpp["predicted_class"] != cpp["python_pred"]]
    if len(diff):
        print("\nImages where C++ and Python disagree:")
        print(diff[["path", "true", "predicted_class", "confidence", "python_pred", "python_conf"]].to_string(index=False))


if __name__ == "__main__":
    main()
