"""PERSON 4 - Simulated annealing (search) + Learned classifier (edge scorer).

Train the classifier once (GPU recommended), then it works like any other scorer:
    python person4_anneal_learned.py --train --data dataset/
"""
import math
import os
import time

import numpy as np

from common import (BLANK, BOTTOM, DIRS, LEFT, PENALTY, RIGHT, TAB, Timer, energy, frame_ok,
                    opposite, pair_cost, side_facing)
from person2_greedy_gradient import greedy_reading


# ------------------------------------------------------------------ search
def anneal(puzzle, C, time_limit=60, seed=0, iters_per_piece=2000, start_accept=0.8):
    """Start from the greedy solution; repeatedly swap or rotate pieces. Keep changes that lower the
    energy; accept worse ones with probability exp(-delta / T) so we can climb out of local minima.
    T starts where ~80% of worsening moves are accepted and cools geometrically to 0.1% of that
    (on an iteration budget or the time limit, whichever ends first).
    Swaps stay within corner / edge / interior cells. Node expansion = one cell changed by a move.
    """
    rng = np.random.default_rng(seed)
    R, W, N = puzzle.R, puzzle.C, puzzle.n
    types = puzzle.types()
    timer = Timer(time_limit)
    start = greedy_reading(puzzle, C)
    grid = [row[:] for row in start["grid"]]
    nodes = start["nodes"]

    def cls(y, x):
        return (y in (0, R - 1)) + (x in (0, W - 1))      # 2 corner, 1 edge, 0 interior
    groups = [[(y, x) for y in range(R) for x in range(W) if cls(y, x) == k] for k in (0, 1, 2)]
    groups = [g for g in groups if len(g) >= 2]
    interior = [(y, x) for y in range(R) for x in range(W) if cls(y, x) == 0]
    if not groups:
        return {"grid": grid, "nodes": nodes, "timed_out": False}

    def local(cells):
        """Energy of everything touching these cells (each edge counted once)."""
        e, seen = 0.0, set()
        for (y, x) in cells:
            p = grid[y][x]
            if p is None:
                e += PENALTY
                continue
            if not frame_ok(types, p[0], p[1], R, W, y, x):
                e += PENALTY
            for d, (dy, dx) in DIRS.items():
                yy, xx = y + dy, x + dx
                if 0 <= yy < R and 0 <= xx < W and grid[yy][xx] is not None:
                    edge = frozenset([(y, x), (yy, xx)])
                    if edge not in seen:
                        seen.add(edge)
                        c = pair_cost(C, p, d, grid[yy][xx])
                        e += c if c < math.inf else PENALTY
        return e

    def best_turn(y, x):
        pid = grid[y][x][0]
        scores = []
        for r in range(4):
            grid[y][x] = (pid, r)
            scores.append(local([(y, x)]))
        grid[y][x] = (pid, int(np.argmin(scores)))

    def propose():
        """Make a random move. Returns (delta, cells, old placements) so it can be undone."""
        if interior and rng.random() < 0.2:                       # rotate one interior piece
            cells = [interior[rng.integers(len(interior))]]
            old = [grid[y][x] for y, x in cells]
            before = local(cells)
            y, x = cells[0]
            grid[y][x] = (old[0][0], (old[0][1] + int(rng.integers(1, 4))) % 4)
        else:                                                     # swap two pieces, re-turn both
            sizes = np.array([len(g) for g in groups], float)
            g = groups[rng.choice(len(groups), p=sizes / sizes.sum())]
            i, j = rng.choice(len(g), 2, replace=False)
            cells = [g[i], g[j]]
            old = [grid[y][x] for y, x in cells]
            before = local(cells)
            (ya, xa), (yb, xb) = cells
            grid[ya][xa], grid[yb][xb] = old[1], old[0]
            best_turn(ya, xa)
            best_turn(yb, xb)
        return local(cells) - before, cells, old

    def undo(cells, old):
        for (y, x), p in zip(cells, old):
            grid[y][x] = p

    # starting temperature: accept ~80% of worsening moves at first
    ups = []
    for _ in range(200):
        delta, cells, old = propose()
        undo(cells, old)
        if delta > 0:
            ups.append(delta)
    T0 = -np.mean(ups) / math.log(start_accept) if ups else 1.0
    iters = iters_per_piece * N

    e = energy(grid, C, types)
    best_e, best_grid = e, [row[:] for row in grid]
    timed_out = False
    T = T0
    for it in range(iters):
        if it % 200 == 0:
            if timer.up():
                timed_out = True
                break
            # cool by iterations or by time, whichever runs out first, so T always gets low
            progress = max(it / iters, (time.perf_counter() - timer.t0) / time_limit)
            T = T0 * (1e-3) ** progress
        delta, cells, old = propose()
        nodes += len(cells)
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            e += delta
            if e < best_e - 1e-9:
                best_e, best_grid = e, [row[:] for row in grid]
        else:
            undo(cells, old)
    return {"grid": best_grid, "nodes": nodes, "timed_out": timed_out}


# ------------------------------------------------------------------ learned edge scorer
HEAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_head.pt")
_cache = {}


def _side_band(piece, a, face):
    """Crop the strip of image along side a, after turning the piece so that side faces `face`
    (RIGHT or LEFT). Returns a 64 x 48 RGB float image. The band reaches 24 px into the piece
    and 20 px past the corner line (so tabs are included)."""
    import cv2
    from features import order_corners
    from generate import _rot90_points
    r = (face - a) % 4                                   # turning r clockwise makes side a face `face`
    img = np.ascontiguousarray(np.rot90(piece.image, k=-r))
    H, W = piece.image.shape[:2]
    TL, TR, BR, BL = order_corners(_rot90_points(np.asarray(piece.corners, float), H, W, r))
    pad = 40
    img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)))
    if face == RIGHT:
        xe, y0, y1 = (TR[0] + BR[0]) / 2, TR[1], BR[1]
        x0, x1 = xe - 24, xe + 20
    else:
        xe, y0, y1 = (TL[0] + BL[0]) / 2, TL[1], BL[1]
        x0, x1 = xe - 20, xe + 24
    crop = img[int(y0) + pad:int(y1) + pad + 1, int(x0) + pad:int(x1) + pad + 1]
    return cv2.resize(crop, (48, 64)).astype(np.float32) / 255.0


def _backbone():
    import torch
    import torchvision
    if "net" not in _cache:
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _cache["net"], _cache["dev"] = net.eval().to(dev), dev
    return _cache["net"], _cache["dev"]


def _embed(pieces):
    """ER[i, a] = features of side a facing right; EL[i, a] = same side facing left. Each N x 4 x 512."""
    import torch
    net, dev = _backbone()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    for face in (RIGHT, LEFT):
        bands = np.stack([_side_band(p, a, face) for p in pieces for a in range(4)])
        x = torch.tensor(bands, device=dev).permute(0, 3, 1, 2)
        feats = []
        with torch.no_grad():
            for k in range(0, len(x), 256):
                feats.append(net((x[k:k + 256] - mean) / std))
        out.append(torch.cat(feats).view(len(pieces), 4, -1))
    return out[0], out[1]


def _head():
    import torch
    return torch.nn.Sequential(torch.nn.Linear(1536, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))


def _pair_input(fr, fl):
    import torch
    return torch.cat([fr, fl, (fr - fl).abs()], dim=-1)


def learned_scorer(pieces):
    """Cost = -log P(match) from the trained head, clipped at 10."""
    import torch
    if "head" not in _cache:
        head = _head()
        head.load_state_dict(torch.load(HEAD_PATH, map_location="cpu"))
        _cache["head"] = head.eval().to(_backbone()[1])
    ER, EL = _embed(pieces)
    N = len(pieces)
    ER, EL = ER.reshape(N * 4, -1), EL.reshape(N * 4, -1)
    out = np.empty((N * 4, N * 4))
    with torch.no_grad():
        for q in range(N * 4):
            logit = _cache["head"](_pair_input(ER[q].expand_as(EL), EL)).squeeze(1)
            out[q] = torch.nn.functional.softplus(-logit).clamp(max=10).cpu().numpy()
    return out.reshape(N, 4, N, 4)


def train_learned(data_dir, epochs=5, neg_per_pos=3, seed=0):
    """Train the head on matching / non-matching side pairs from TRAIN-split synthetic puzzles."""
    import torch
    from evaluate import gt_grid
    from generate import list_puzzles, load_puzzle
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for folder in list_puzzles(data_dir, split="train"):
        pz = load_puzzle(folder, "oracle")
        ER, EL = _embed(pz.pieces)
        types = pz.types()
        g = gt_grid(pz)
        for y in range(pz.R):
            for x in range(pz.C):
                for d in (RIGHT, BOTTOM):
                    dy, dx = DIRS[d]
                    if y + dy >= pz.R or x + dx >= pz.C:
                        continue
                    (i, ri), (j, rj) = g[y][x], g[y + dy][x + dx]
                    a, b = side_facing(ri, d), side_facing(rj, opposite(d))
                    X.append(_pair_input(ER[i, a], EL[j, b]).cpu())
                    Y.append(1.0)
                    want = BLANK if types[i, a] == TAB else TAB   # negatives that still fit by shape
                    options = [(jj, bb) for jj in range(pz.n) for bb in range(4)
                               if types[jj, bb] == want and (jj, bb) != (j, b) and jj != i]
                    for k in rng.choice(len(options), min(neg_per_pos, len(options)), replace=False):
                        jj, bb = options[k]
                        X.append(_pair_input(ER[i, a], EL[jj, bb]).cpu())
                        Y.append(0.0)
        print("loaded", os.path.basename(folder), len(Y), "pairs")
    X, Y = torch.stack(X), torch.tensor(Y)
    dev = _backbone()[1]
    head = _head().to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(neg_per_pos), device=dev))
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        total = 0.0
        for k in range(0, len(X), 256):
            idx = perm[k:k + 256]
            loss = lossf(head(X[idx].to(dev)).squeeze(1), Y[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {ep + 1}: loss {total / len(X):.4f}")
    torch.save(head.cpu().state_dict(), HEAD_PATH)
    print("saved", HEAD_PATH)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    if a.train:
        train_learned(a.data, a.epochs)
