"""Find pieces in a photo of loose pieces on a plain table.

extract_pieces(photo_rgb) -> (pieces, report)
  1. Segment: background = median Lab color of the image border; distance threshold (Otsu);
     morphological cleanup; connected components. Blobs < 0.5x or > 1.6x the median area are flagged
     'broken' / 'merged'. Tiny blobs (< 0.15x median) are dropped as noise.
  2. Corners: candidate corner points on the mask whose neighborhood is ~25% filled (convex 90-degree
     corners; tab necks are ~75%, straight edges ~50%); pick the 4 forming the most square-like quad
     whose area matches the blob area.
  3. Straighten: rotate so the quad is axis-aligned; store that rotation as straighten_cw_deg.
  4. Features: core.features.compute_side_features (same code the synthetic oracle pieces use).
Pieces are numbered top-to-bottom, left-to-right (row bands one piece tall).
"""
import itertools
import math

import cv2
import numpy as np

from features import order_corners
from generate import make_piece


# ---------------------------------------------------------------- segmentation
def segment(photo: np.ndarray, border: int = 12):
    lab = cv2.cvtColor(photo.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    frame = np.concatenate([lab[:border].reshape(-1, 3), lab[-border:].reshape(-1, 3),
                            lab[:, :border].reshape(-1, 3), lab[:, -border:].reshape(-1, 3)])
    bg = np.median(frame, axis=0)
    diff = lab - bg
    diff[:, :, 0] *= 0.5                       # lightness changes (shadows, lighting) count less
    dist = np.linalg.norm(diff, axis=2)
    d8 = np.clip(dist / max(dist.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    d8 = cv2.GaussianBlur(d8, (5, 5), 0)
    _, fg = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # fill holes
    inv = cv2.bitwise_not(fg)
    n, lbl = cv2.connectedComponents(inv)
    for k in range(1, n):
        comp = lbl == k
        ys, xs = np.nonzero(comp)
        if ys.min() > 0 and xs.min() > 0 and ys.max() < fg.shape[0] - 1 and xs.max() < fg.shape[1] - 1:
            fg[comp] = 255
    n, lbl, stats, cents = cv2.connectedComponentsWithStats((fg > 0).astype(np.uint8))
    return lbl, stats[1:], cents[1:]           # drop background component 0 (labels start at 1)


# ---------------------------------------------------------------- corners
def _occupancy(mask: np.ndarray, pt, rad: int) -> float:
    x, y = int(round(pt[0])), int(round(pt[1]))
    H, W = mask.shape
    y0, y1, x0, x1 = max(0, y - rad), min(H, y + rad + 1), max(0, x - rad), min(W, x + rad + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = (yy - y) ** 2 + (xx - x) ** 2 <= rad ** 2
    return float(mask[y0:y1, x0:x1][disc].mean()) if disc.any() else 0.0


def find_corners(mask: np.ndarray):
    """Return 4x2 corner points (cyclic order) or None.

    Candidates = local minima of disc occupancy along the outer contour (a convex 90-degree corner
    is ~25-30% filled). Then choose the 4 that form the most square-like quad with the right area.
    """
    area = float(mask.sum())
    side = math.sqrt(area)
    rad = max(3, int(round(0.10 * side)))
    disc = np.zeros((2 * rad + 1, 2 * rad + 1), np.float32)
    cv2.circle(disc, (rad, rad), rad, 1.0, -1)
    occ_map = cv2.filter2D(mask.astype(np.float32), -1, disc / disc.sum(), borderType=cv2.BORDER_CONSTANT)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cont = max(cnts, key=cv2.contourArea)[:, 0, :]
    occ = occ_map[cont[:, 1], cont[:, 0]]
    n = len(cont)
    win = max(3, int(0.12 * side))
    cand = []
    for i in range(n):
        if occ[i] > 0.45:
            continue
        nb = occ[np.arange(i - win, i + win + 1) % n]
        if occ[i] <= nb.min() + 1e-6:
            if not cand or np.linalg.norm(cont[i] - cont[cand[-1]]) > win:
                cand.append(i)
    cand = cont[cand].astype(np.float64)
    if len(cand) < 4:
        return None
    if len(cand) > 20:                        # keep the 20 sharpest
        o = occ_map[cand[:, 1].astype(int), cand[:, 0].astype(int)]
        cand = cand[np.argsort(o)[:20]]
    best, best_s = None, math.inf
    for quad in itertools.combinations(range(len(cand)), 4):
        q = order_corners(cand[list(quad)])
        sides = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)
        if sides.min() < 0.4 * side:
            continue
        v1 = np.roll(q, 1, axis=0) - q
        v2 = np.roll(q, -1, axis=0) - q
        cosang = np.abs((v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9))
        qa = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))
        s = cosang.sum() + sides.std() / sides.mean() + abs(math.log(max(qa, 1) / area))
        if s < best_s:
            best, best_s = q, s
    return best


# ---------------------------------------------------------------- straightening
def straighten(img: np.ndarray, mask: np.ndarray, quad: np.ndarray):
    """Rotate so the quad is axis-aligned. Returns (img, mask, corners TL..BR..BL, straighten_cw_deg)."""
    q = order_corners(quad)
    dx, dy = q[1] - q[0]
    theta = math.degrees(math.atan2(dy, dx))        # y down: positive = piece currently turned clockwise
    theta = (theta + 45) % 90 - 45                   # nearest axis
    cw = -theta                                      # rotate counter-clockwise by theta to straighten
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -cw, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(math.ceil(h * sin + w * cos)), int(math.ceil(h * cos + w * sin))
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    rimg = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR)
    rmask = cv2.warpAffine(mask.astype(np.uint8) * 255, M, (nw, nh), flags=cv2.INTER_LINEAR) > 127
    rq = (np.hstack([q, np.ones((4, 1))]) @ M.T)
    rimg[~rmask] = 0
    return rimg, rmask, order_corners(rq), cw % 360


# ---------------------------------------------------------------- main entry
def extract_pieces(photo: np.ndarray):
    lbl, stats, cents = segment(photo)
    areas = stats[:, cv2.CC_STAT_AREA].astype(float)
    if len(areas) == 0:
        return [], {"n_blobs": 0, "flags": {}}
    med = float(np.median(areas))
    keep = [k for k in range(len(areas)) if areas[k] >= 0.15 * med]
    band = math.sqrt(med)
    keep.sort(key=lambda k: (int(cents[k][1] // band), cents[k][0]))

    pieces, report = [], {"n_blobs": len(keep), "flags": {}, "failed": []}
    for k in keep:
        x, y, w, h, a = stats[k]
        pad = int(0.3 * max(w, h)) + 4
        y0, y1 = max(0, y - pad), min(photo.shape[0], y + h + pad)
        x0, x1 = max(0, x - pad), min(photo.shape[1], x + w + pad)
        mask = lbl[y0:y1, x0:x1] == (k + 1)
        img = photo[y0:y1, x0:x1].copy()
        img[~mask] = 0
        flags = []
        if a > 1.6 * med:
            flags.append("merged")
        elif a < 0.5 * med:
            flags.append("broken")
        quad = find_corners(mask)
        if quad is None:
            report["failed"].append(int(k))
            continue
        simg, smask, scorners, cw = straighten(img, mask, quad)
        pid = len(pieces)
        try:
            pieces.append(make_piece(pid, simg, smask, scorners, straighten_cw_deg=cw,
                                     centroid_xy=tuple(cents[k]), flags=flags))
        except ValueError:
            report["failed"].append(int(k))
            continue
        if flags:
            report["flags"][pid] = flags
    return pieces, report


def estimate_grid_size(pieces: list):
    """Guess (R, C) from the number of pieces and border pieces (those with a flat side). R <= C."""
    N = len(pieces)
    B = sum(1 for p in pieces if 0 in p.side_type)
    s = (B + 4) / 2.0                     # R + C
    disc = s * s - 4 * N
    if disc < 0:
        return None
    R = int(round((s - math.sqrt(disc)) / 2))
    if R <= 0:
        return None
    C = int(round(N / R))
    return (R, C) if R * C == N else None
