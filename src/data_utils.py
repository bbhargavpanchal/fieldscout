"""Shared helpers: find images and work out their class label."""
from pathlib import Path
import re

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

LABEL_PATTERNS = [
    ("yellow_rust", r"yellow|stripe|yr\b"),
    ("brown_rust", r"brown|leaf[\s_-]?rust|br\b"),
    ("septoria", r"septoria|stb"),
    ("mildew", r"mildew|powdery|pm\b"),
    ("healthy", r"healthy|control|clean"),
]


def canonical_label(text: str):
    t = text.lower()
    for label, pattern in LABEL_PATTERNS:
        if re.search(pattern, t):
            return label
    return None


def find_images(root):
    """Yield (path, label) for every image under root."""
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS or p.name.startswith("."):
            continue
        label = None
        for parent in p.relative_to(root).parents:
            if str(parent) == ".":
                break
            label = canonical_label(parent.name)
            if label:
                break
        if label is None:
            label = canonical_label(p.stem)
        if label is None:
            label = p.parent.name.lower()
        yield p, label
