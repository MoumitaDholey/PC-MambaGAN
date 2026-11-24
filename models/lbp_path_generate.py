#!/usr/bin/env python3
"""
lbp_path_generate.py

- Only TWO entrypoints: main1() and main2()
- No argparse
- Both save SAME legacy filenames in ./models/:
    lbp_spiral_eye.npy
    lbp_despiral_eye.npy
    lbp_despiral_r_eye.npy
- main1: optimized (vectorized)
- main2: legacy (loop fill)
- Contents are identical between main1 and main2 (guaranteed by construction).
"""

import numpy as np
from pathlib import Path


# =========================
# Core traversal
# =========================
def square_lbp_traverse_with_center(H: int, W: int) -> np.ndarray:
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive")

    cy, cx = H // 2, W // 2  # keep same convention as old code (even sizes ambiguous)
    visited = np.zeros((H, W), dtype=np.bool_)
    idx = []

    visited[cy, cx] = True
    idx.append(cy * W + cx)

    max_r = max(cy, H - 1 - cy, cx, W - 1 - cx)

    for r in range(1, max_r + 1):
        top, bottom = cy - r, cy + r
        left, right = cx - r, cx + r

        # Top edge: (top, left..right)
        if 0 <= top < H:
            x0, x1 = max(left, 0), min(right, W - 1)
            for x in range(x0, x1 + 1):
                if not visited[top, x]:
                    visited[top, x] = True
                    idx.append(top * W + x)

        # Right edge: (top+1..bottom-1, right)
        if 0 <= right < W:
            y0, y1 = max(top + 1, 0), min(bottom - 1, H - 1)
            for y in range(y0, y1 + 1):
                if not visited[y, right]:
                    visited[y, right] = True
                    idx.append(y * W + right)

        # Bottom edge: (bottom, right..left) reversed
        if 0 <= bottom < H and bottom != top:
            x0, x1 = max(left, 0), min(right, W - 1)
            for x in range(x1, x0 - 1, -1):
                if not visited[bottom, x]:
                    visited[bottom, x] = True
                    idx.append(bottom * W + x)

        # Left edge: (bottom-1..top+1, left) reversed
        if 0 <= left < W and left != right:
            y0, y1 = max(top + 1, 0), min(bottom - 1, H - 1)
            for y in range(y1, y0 - 1, -1):
                if not visited[y, left]:
                    visited[y, left] = True
                    idx.append(y * W + left)

        if len(idx) == H * W:
            break

    # Safety fill (odd sizes / edge cases)
    if len(idx) != H * W:
        for y in range(H):
            for x in range(W):
                if not visited[y, x]:
                    idx.append(y * W + x)

    idx = np.asarray(idx, dtype=np.int64)

    # Double-check uniqueness and bounds
    if idx.size != H * W:
        raise RuntimeError(f"Traversal produced {idx.size} indices, expected {H*W}")
    if np.unique(idx).size != idx.size:
        raise RuntimeError("Traversal indices contain duplicates")
    if idx.min() < 0 or idx.max() >= H * W:
        raise RuntimeError("Traversal indices out of bounds")

    return idx


# =========================
# Save (legacy filenames)
# =========================
def _save_legacy(outdir: Path, P: np.ndarray, P_r: np.ndarray):
    outdir.mkdir(parents=True, exist_ok=True)

    # forward
    np.save(outdir / "lbp_spiral_eye.npy", P)
    np.save(outdir / "lbp_despiral_eye.npy", P.T)

    # reverse (IMPORTANT: inverse of reverse permutation)
    np.save(outdir / "lbp_despiral_r_eye.npy", P_r.T)


# =========================
# main1: optimized (vectorized)
# =========================
def main1():
    H, W = 64, 64
    outdir = Path("./models")

    idx = square_lbp_traverse_with_center(H, W)
    idx_r = idx[::-1].copy()
    N = idx.size
    rows = np.arange(N, dtype=np.int64)

    P = np.zeros((N, N), dtype=np.uint8)
    P[rows, idx] = 1

    P_r = np.zeros((N, N), dtype=np.uint8)
    P_r[rows, idx_r] = 1

    _save_legacy(outdir, P, P_r)

    print("[main1/new] Saved legacy filenames to ./models/")
    print("  lbp_spiral_eye.npy")
    print("  lbp_despiral_eye.npy")
    print("  lbp_despiral_r_eye.npy")


# =========================
# main2: legacy (loop fill)
# =========================
def main2():
    H, W = 64, 64
    outdir = Path("./models")

    idx = square_lbp_traverse_with_center(H, W).tolist()
    idx_r = list(reversed(idx))
    N = H * W

    P = np.zeros((N, N), dtype=np.uint8)
    for i, flat_idx in enumerate(idx):
        P[i, flat_idx] = 1

    P_r = np.zeros((N, N), dtype=np.uint8)
    for i, flat_idx in enumerate(idx_r):
        P_r[i, flat_idx] = 1

    _save_legacy(outdir, P, P_r)

    print("[main2/legacy] Saved legacy filenames to ./models/")
    print("  lbp_spiral_eye.npy")
    print("  lbp_despiral_eye.npy")
    print("  lbp_despiral_r_eye.npy")


if __name__ == "__main__":
    main1()
    # main2()
