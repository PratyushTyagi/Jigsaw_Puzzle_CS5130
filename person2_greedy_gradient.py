"""PERSON 2 - Greedy search (reading order and most-confident-first) + Gradient continuity (edge scorer)."""
import numpy as np

from common import (INF, PENALTY, all_sides, candidate_costs, empty_grid, energy, frame_ok,
                    neighbors, pair_cost)


# ------------------------------------------------------------------ edge scorer
def gradient_scorer(pieces):
    """Simplified Mahalanobis Gradient Compatibility (Gallagher 2012).

    For side q: G = edge row - row one pixel further in (how color changes toward the edge).
    Fit mean mu and covariance S of G along the side. The change ACROSS the seam is
    D = neighbor's edge row (reversed) - my edge row. If the picture continues smoothly, D looks like G,
    so cost = sum over the side of (D - mu)^T S^-1 (D - mu). build_cost_table averages both directions.
    """
    E0 = all_sides(pieces, "strip")[:, 0]          # (N*4) x 64 x 3 outermost row
    E1 = all_sides(pieces, "strip")[:, 1]          # one pixel further in
    G = E0 - E1
    E0_rev = E0[:, ::-1]
    n = len(E0)
    out = np.empty((n, n))
    for q in range(n):
        mu = G[q].mean(axis=0)
        S = np.cov(G[q].T) + np.eye(3)             # + I keeps flat regions from blowing up
        Sinv = np.linalg.inv(S)
        diff = (E0_rev - E0[q][None]) - mu         # n x 64 x 3
        out[q] = np.einsum("nlc,cd,nld->n", diff, Sinv, diff)
    N = len(pieces)
    return out.reshape(N, 4, N, 4)


# ------------------------------------------------------------------ search helpers
def _place_best(grid, y, x, C, types, used):
    """Put the cheapest legal (piece, r) in (y, x). If nothing is legal, place the cheapest unused piece
    anyway and report a violation. Returns (nodes_tried, violated)."""
    cand = candidate_costs(grid, y, x, C, types, used)
    tried = int((~used).sum()) * 4
    if np.isfinite(cand).any():
        pid, r = np.unravel_index(np.argmin(cand), cand.shape)
        violated = False
    else:                                       # fallback: ignore shape rules, use finite costs only
        R, W = len(grid), len(grid[0])
        best, pid, r = INF, None, 0
        for p in np.nonzero(~used)[0]:
            for rr in range(4):
                c = sum(min(pair_cost(C, (p, rr), d, q), PENALTY) for d, q in neighbors(grid, y, x))
                c += 0 if frame_ok(types, p, rr, R, W, y, x) else PENALTY
                if c < best:
                    best, pid, r = c, p, rr
        violated = True
    grid[y][x] = (int(pid), int(r))
    used[pid] = True
    return tried, violated


def _corner_seeds(puzzle):
    """(piece, r) pairs that fit cell (0, 0): corner pieces with flats facing up and left."""
    types = puzzle.types()
    return [(p, r) for p in range(puzzle.n) for r in range(4)
            if frame_ok(types, p, r, puzzle.R, puzzle.C, 0, 0)]


def _best_of_seeds(puzzle, C, fill):
    """Run `fill` once per corner seed; keep the lowest-energy grid."""
    types = puzzle.types()
    best, total_nodes = None, 0
    seeds = _corner_seeds(puzzle) or [(0, 0)]
    for seed in seeds:
        grid = empty_grid(puzzle.R, puzzle.C)
        used = np.zeros(puzzle.n, bool)
        grid[0][0] = seed
        used[seed[0]] = True
        nodes, viol = fill(grid, used)
        total_nodes += nodes + 1
        e = energy(grid, C, types)
        if best is None or e < best[0]:
            best = (e, grid, viol)
    return {"grid": best[1], "nodes": total_nodes, "timed_out": False, "violations": best[2]}


# ------------------------------------------------------------------ search 1: reading order
def greedy_reading(puzzle, C, time_limit=60, seed=0):
    """Fill cells left-to-right, top-to-bottom with the best-scoring piece. No backtracking.
    Tried once from each corner piece at (0, 0); the best result is kept."""
    types = puzzle.types()

    def fill(grid, used):
        nodes = viol = 0
        for y in range(puzzle.R):
            for x in range(puzzle.C):
                if grid[y][x] is None and not used.all():
                    t, v = _place_best(grid, y, x, C, types, used)
                    nodes += t
                    viol += v
        return nodes, viol

    return _best_of_seeds(puzzle, C, fill)


# ------------------------------------------------------------------ search 2: most confident first
def greedy_confident(puzzle, C, time_limit=60, seed=0):
    """Grow outward from (0, 0). At each step look at every empty cell next to placed pieces and fill
    the one we are most sure about: most placed neighbors first, then the biggest gap between its
    best and second-best candidate. (Simplified Pomeranz et al. 2011, kept inside the R x C frame.)"""
    types = puzzle.types()
    R, W = puzzle.R, puzzle.C

    def fill(grid, used):
        nodes = viol = 0
        while not used.all():
            best_key, best_cell = None, None
            for y in range(R):
                for x in range(W):
                    if grid[y][x] is not None:
                        continue
                    nb = sum(1 for _ in neighbors(grid, y, x))
                    if nb == 0:
                        continue
                    cand = np.sort(candidate_costs(grid, y, x, C, types, used).ravel())
                    nodes += int((~used).sum()) * 4
                    gap = (cand[1] - cand[0]) if np.isfinite(cand[1]) else (1e9 if np.isfinite(cand[0]) else -1)
                    key = (nb, gap)
                    if best_key is None or key > best_key:
                        best_key, best_cell = key, (y, x)
            if best_cell is None:
                break
            _, v = _place_best(grid, *best_cell, C, types, used)
            viol += v
        return nodes, viol

    return _best_of_seeds(puzzle, C, fill)
