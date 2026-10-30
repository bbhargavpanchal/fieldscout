"""Near-duplicate grouping with perceptual hashes, so twins never cross train/test."""
import numpy as np
from PIL import Image
import imagehash


def phash_ints(paths, hash_size=8):
    out = np.zeros(len(paths), dtype=np.uint64)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            h = imagehash.phash(im.convert("RGB"), hash_size=hash_size)
        out[i] = np.uint64(int(str(h), 16))
    return out


def _popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def group_near_duplicates(hashes, max_distance=6, chunk=1024):
    n = len(hashes)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, n, chunk):
        block = hashes[start:start + chunk]
        dist = _popcount64(block[:, None] ^ hashes[None, :])
        rows, cols = np.nonzero(dist <= max_distance)
        for r, c in zip(rows + start, cols):
            if c > r:
                ra, rb = find(r), find(c)
                if ra != rb:
                    parent[rb] = ra
    return np.array([find(i) for i in range(n)])
