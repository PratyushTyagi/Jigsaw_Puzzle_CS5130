"""Synthetic dataset: cut images into puzzles, render photos, save/load puzzles.

Build the dataset:
    python generate.py --images source_images/ --out dataset/ --sizes 3x3,4x4,6x6,10x10,15x20

Each puzzle folder holds:
    scene.png   photo of scattered pieces (rotated, on a plain background)
    oracle.pkl  the same pieces already cut out and straightened (each turned a random multiple of 90 deg)
    gt.json     piece id -> correct cell, correct turn (oracle and scene), centroid in scene.png
    meta.json   R, C, source image, split (train/test by source image), variant
Real photos use the same folder layout: scene.png + meta.json (R, C) + labels.csv (piece,row,col, 1-based).

Piece ids are shuffled, so an id never tells you the correct cell.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import pickle

import cv2
import numpy as np

from common import Piece, Puzzle
from features import compute_side_features, order_corners

BG_PALETTE = [(42, 46, 56), (232, 230, 222), (58, 104, 70), (120, 82, 58), (70, 90, 140)]


# ---------------------------------------------------------------- cutting
def _fit_image(img: np.ndarray, R: int, C: int, P: int) -> np.ndarray:
    h, w = img.shape[:2]
    target = C / R
    if w / h > target:
        nw = int(round(h * target))
        x0 = (w - nw) // 2
        img = img[:, x0:x0 + nw]
    else:
        nh = int(round(w / target))
        y0 = (h - nh) // 2
        img = img[y0:y0 + nh]
    interp = cv2.INTER_AREA if img.shape[1] > C * P else cv2.INTER_CUBIC
    return cv2.resize(img, (C * P, R * P), interpolation=interp)


def _paint_circle(label: np.ndarray, cx: float, cy: float, rad: float, owner: int) -> None:
    H, W = label.shape
    x0, x1 = max(0, int(cx - rad) - 1), min(W, int(cx + rad) + 2)
    y0, y1 = max(0, int(cy - rad) - 1), min(H, int(cy + rad) + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
    label[y0:y1, x0:x1][inside] = owner


def cut_label_map(R: int, C: int, P: int, rng: np.random.Generator) -> np.ndarray:
    """Integer map (R*P x C*P): each pixel's owning cell index row*C + col. Tabs are circles on interior edges."""
    yy, xx = np.mgrid[0:R * P, 0:C * P]
    label = (yy // P) * C + (xx // P)

    def tab_params():
        rad = rng.uniform(0.15, 0.19) * P
        off = rng.uniform(0.10, min(0.13, rad / P - 0.02)) * P     # center beyond the edge, < radius
        jit = rng.uniform(-0.06, 0.06) * P
        return rad, off, jit

    for row in range(R):                    # vertical cut lines between (row, col) and (row, col+1)
        for col in range(C - 1):
            rad, off, jit = tab_params()
            s = rng.choice([-1, 1])         # +1: left piece owns the tab, bulging right
            cx = (col + 1) * P + s * off
            cy = row * P + P / 2 + jit
            owner = row * C + col if s > 0 else row * C + col + 1
            _paint_circle(label, cx, cy, rad, owner)
    for row in range(R - 1):                # horizontal cut lines between (row, col) and (row+1, col)
        for col in range(C):
            rad, off, jit = tab_params()
            s = rng.choice([-1, 1])         # +1: top piece owns the tab, bulging down
            cx = col * P + P / 2 + jit
            cy = (row + 1) * P + s * off
            owner = row * C + col if s > 0 else (row + 1) * C + col
            _paint_circle(label, cx, cy, rad, owner)
    return label


def _rot90_points(pts: np.ndarray, H: int, W: int, k: int):
    """Point coordinates after rotating an HxW image k quarter turns CLOCKWISE."""
    pts = pts.copy()
    for _ in range(k % 4):
        pts = np.stack([H - 1 - pts[:, 1], pts[:, 0]], axis=1)
        H, W = W, H
    return pts


def make_piece(pid: int, img: np.ndarray, mask: np.ndarray, corners: np.ndarray,
               straighten_cw_deg: float = 0.0, centroid_xy=(0.0, 0.0), flags=None) -> Piece:
    feats = compute_side_features(img, mask, corners)
    return Piece(id=pid, image=img, mask=mask, corners=np.asarray(corners, np.float32),
                 side_type=feats["side_type"], strip=feats["strip"], contour=feats["contour"],
                 side_len=feats["side_len"], straighten_cw_deg=float(straighten_cw_deg),
                 centroid_xy=tuple(map(float, centroid_xy)), flags=list(flags or []))


# ---------------------------------------------------------------- scene rendering
def _pick_background(img: np.ndarray, rng) -> tuple:
    """Plain table color that the fewest image pixels resemble (maximizes the 2nd-percentile Lab distance)."""
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    lab = lab[rng.choice(len(lab), size=min(20000, len(lab)), replace=False)]
    best, bestd = None, -1.0
    for col in BG_PALETTE:
        c = cv2.cvtColor(np.array([[col]], np.float32) / 255, cv2.COLOR_RGB2LAB)[0, 0]
        d = float(np.percentile(np.linalg.norm(lab - c, axis=1), 2))
        if d > bestd:
            best, bestd = col, d
    return best


def _rotate_rgba(rgba: np.ndarray, cw_deg: float):
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -cw_deg, 1.0)   # OpenCV: positive = counter-clockwise
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(math.ceil(h * sin + w * cos)), int(math.ceil(h * cos + w * sin))
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    return out


def generate_puzzle(image: np.ndarray, R: int, C: int, seed: int = 0, piece_px: int = 64,
                    blur: float = 0.0, shadow: bool = False, lighting: bool = False,
                    noise: float = 2.0, bg_color=None, make_scene: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    P = int(piece_px)
    img = _fit_image(image, R, C, P)
    label = cut_label_map(R, C, P, rng)
    M = int(math.ceil(0.35 * P)) + 2
    imgp = np.pad(img, ((M, M), (M, M), (0, 0)))
    labp = np.pad(label, M, constant_values=-1)

    N = R * C
    perm = rng.permutation(N)                    # cell index -> piece id
    oracle = [None] * N
    crops = [None] * N
    gt = {}
    for cell in range(N):
        row, col = divmod(cell, C)
        pid = int(perm[cell])
        y0, x0 = row * P, col * P                # in padded coords the cell starts at (y0+M, x0+M)
        crop_img = imgp[y0:y0 + P + 2 * M, x0:x0 + P + 2 * M].copy()
        crop_mask = labp[y0:y0 + P + 2 * M, x0:x0 + P + 2 * M] == cell
        crop_img[~crop_mask] = 0
        corners = np.array([[M, M], [M + P - 1, M], [M + P - 1, M + P - 1], [M, M + P - 1]], float)
        crops[pid] = (crop_img, crop_mask)

        r0 = int(rng.integers(4))
        H, W = crop_mask.shape
        o_img = np.ascontiguousarray(np.rot90(crop_img, k=-r0))
        o_mask = np.ascontiguousarray(np.rot90(crop_mask, k=-r0))
        o_corners = order_corners(_rot90_points(corners, H, W, r0))
        oracle[pid] = make_piece(pid, o_img, o_mask, o_corners)
        gt[pid] = {"cell": [row, col], "turn_oracle": float((360 - 90 * r0) % 360)}

    meta = {"R": R, "C": C, "piece_px": P, "seed": seed, "n_total": N,
            "blur": blur, "shadow": shadow, "lighting": lighting, "noise": noise}
    out = {"oracle": oracle, "gt": gt, "meta": meta, "scene": None}
    if not make_scene:
        return out

    # ---- scatter rotated pieces on a plain background (jittered slots, so pieces never overlap)
    bg = tuple(bg_color) if bg_color is not None else _pick_background(img, rng)
    S = int(math.ceil(math.sqrt(2) * (P + 2 * M))) + 4
    ncols = int(math.ceil(math.sqrt(N * 1.3 * 4 / 3)))
    nrows = int(math.ceil(N * 1.3 / ncols))
    slots = rng.permutation(nrows * ncols)[:N]
    Hs, Ws = nrows * S + 2 * S // 4, ncols * S + 2 * S // 4
    scene = np.empty((Hs, Ws, 3), np.float32)
    scene[:] = bg
    pad0 = S // 4
    for pid in range(N):
        crop_img, crop_mask = crops[pid]
        alpha = rng.uniform(0, 360)
        rgba = np.dstack([crop_img, crop_mask.astype(np.uint8) * 255])
        rot = _rotate_rgba(rgba, alpha)
        h, w = rot.shape[:2]
        sr, sc = divmod(int(slots[pid]), ncols)
        oy = pad0 + sr * S + int(rng.integers(0, max(1, S - h + 1)))
        ox = pad0 + sc * S + int(rng.integers(0, max(1, S - w + 1)))
        a = rot[:, :, 3:4].astype(np.float32) / 255
        if shadow:
            sh = cv2.GaussianBlur(a[:, :, 0], (0, 0), 2.5)[:, :, None] * 0.45
            sy, sx = oy + 4, ox + 4
            region = scene[sy:sy + h, sx:sx + w]
            region *= 1 - sh[:region.shape[0], :region.shape[1]]
        region = scene[oy:oy + h, ox:ox + w]
        region[:] = region * (1 - a) + rot[:, :, :3].astype(np.float32) * a
        ys, xs = np.nonzero(rot[:, :, 3] > 127)
        gt[pid]["turn_scene"] = float((360 - alpha) % 360)
        gt[pid]["centroid"] = [float(ox + xs.mean()), float(oy + ys.mean())]

    if lighting:
        th = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:Hs, 0:Ws]
        g = (np.cos(th) * xx / Ws + np.sin(th) * yy / Hs)
        g = (g - g.min()) / (np.ptp(g) + 1e-9)
        scene *= (0.75 + 0.4 * g)[:, :, None]
    if blur > 0:
        scene = cv2.GaussianBlur(scene, (0, 0), blur)
    if noise > 0:
        scene += rng.normal(0, noise, scene.shape).astype(np.float32)
    out["scene"] = np.clip(scene, 0, 255).astype(np.uint8)
    meta["background"] = list(bg)
    return out


# ------------------------------------------------------------------ save / load
def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img[:, :, ::-1].copy()


def save_puzzle(folder, gen, extra_meta=None):
    os.makedirs(folder, exist_ok=True)
    meta = dict(gen["meta"], **(extra_meta or {}))
    cv2.imwrite(os.path.join(folder, "scene.png"), gen["scene"][:, :, ::-1])
    json.dump({str(k): v for k, v in gen["gt"].items()}, open(os.path.join(folder, "gt.json"), "w"))
    json.dump(meta, open(os.path.join(folder, "meta.json"), "w"), indent=1)
    pickle.dump(gen["oracle"], open(os.path.join(folder, "oracle.pkl"), "wb"))


def load_puzzle(folder, mode="oracle"):
    """mode "oracle": pre-cut pieces from oracle.pkl.  mode "full": pieces found in scene.png by extract.py."""
    meta = json.load(open(os.path.join(folder, "meta.json")))
    meta.setdefault("puzzle_id", os.path.basename(os.path.normpath(folder)))
    meta["mode"] = mode
    R, C = int(meta["R"]), int(meta["C"])
    meta["n_total"] = R * C
    gt_path = os.path.join(folder, "gt.json")
    gt_raw = json.load(open(gt_path)) if os.path.exists(gt_path) else None

    if mode == "oracle":
        pieces = pickle.load(open(os.path.join(folder, "oracle.pkl"), "rb"))
        gt = {int(k): {"cell": tuple(v["cell"]), "turn": v["turn_oracle"]} for k, v in gt_raw.items()}
        return Puzzle(pieces, R, C, gt, meta)

    from extract import extract_pieces
    pieces, report = extract_pieces(read_rgb(os.path.join(folder, "scene.png")))
    meta["n_found"] = len(pieces)
    labels = os.path.join(folder, "labels.csv")
    gt = {}
    if os.path.exists(labels):                       # real photo labeled by hand (1-based numbers)
        for row in csv.DictReader(open(labels)):
            if row.get("row"):
                turn = row.get("turn") or None
                gt[int(row["piece"]) - 1] = {"cell": (int(row["row"]) - 1, int(row["col"]) - 1),
                                             "turn": float(turn) if turn else None}
    elif gt_raw:                                     # synthetic photo: match found pieces to truth by position
        G = np.array([gt_raw[str(k)]["centroid"] for k in range(len(gt_raw))])
        size = math.sqrt(np.median([p.mask.sum() for p in pieces])) if pieces else 1
        for p in pieces:
            dist = np.linalg.norm(G - np.array(p.centroid_xy), axis=1)
            k = int(np.argmin(dist))
            if dist[k] < 0.5 * size:
                gt[p.id] = {"cell": tuple(gt_raw[str(k)]["cell"]), "turn": gt_raw[str(k)]["turn_scene"]}
    return Puzzle(pieces, R, C, gt, meta)


def list_puzzles(data_dir, sizes=None, split=None):
    """Puzzle folders in data_dir, optionally filtered by sizes like ["3x3", "4x4"] and split."""
    out = []
    for name in sorted(os.listdir(data_dir)):
        mp = os.path.join(data_dir, name, "meta.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        if sizes and f"{m['R']}x{m['C']}" not in sizes:
            continue
        if split and m.get("split") != split:
            continue
        out.append(os.path.join(data_dir, name))
    return out


# ------------------------------------------------------------------ dataset builder
VARIANTS = {"clean": dict(blur=0.0, shadow=False, lighting=False, noise=1.0),
            "effects": dict(blur=0.8, shadow=True, lighting=True, noise=3.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of source images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="3x3,4x4,6x6,10x10,15x20")
    ap.add_argument("--variants", default="clean,effects")
    ap.add_argument("--piece-px", type=int, default=64)
    ap.add_argument("--test-frac", type=float, default=0.3)
    a = ap.parse_args()

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = sorted(os.path.join(d, f) for d, _, fs in os.walk(a.images) for f in fs if f.lower().endswith(exts))
    sizes = [tuple(map(int, s.split("x"))) for s in a.sizes.split(",")]
    seed = 0
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        h = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        split = "test" if h < a.test_frac else "train"
        img = read_rgb(path)
        gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (256, 256)).astype(np.float32)
        texture = float(np.mean(np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))))
        for R, C in sizes:
            for v in a.variants.split(","):
                seed += 1
                folder = os.path.join(a.out, f"{name}_{R}x{C}_{v}")
                if os.path.exists(os.path.join(folder, "meta.json")):
                    continue
                gen = generate_puzzle(img, R, C, seed=seed, piece_px=a.piece_px, **VARIANTS[v])
                save_puzzle(folder, gen, {"source": os.path.basename(path), "split": split, "variant": v,
                                           "texture": texture})   # low = sky/water-like
        print("done", name)


if __name__ == "__main__":
    main()
