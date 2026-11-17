"""PERSON 1 - Brute-force backtracking (search) + Color difference (edge scorer)."""
import numpy as np

from common import INF, Timeout, Timer, all_sides, candidate_costs, empty_grid


# ------------------------------------------------------------------ edge scorer
def color_scorer(pieces):
    """Sum of squared Lab differences along the two touching edges (outermost pixel row).

    Both sides are stored clockwise, so the neighbor's edge is reversed before comparing.
    Returns raw N x 4 x N x 4 costs; build_cost_table adds the tab/blank rule and normalization.
    """
    E = all_sides(pieces, "strip")[:, 0]            # (N*4) x 64 x 3, depth 0 = outermost row
    E_rev = E[:, ::-1]
    n = len(E)
    out = np.empty((n, n))
    for q in range(n):
        out[q] = ((E[q][None] - E_rev) ** 2).sum(axis=(1, 2))
    N = len(pieces)
    return out.reshape(N, 4, N, 4)


# ------------------------------------------------------------------ search
def brute_force(puzzle, C, time_limit=60, seed=0):
    """Naive baseline. Fill cells in reading order; try every unused piece in all 4 orientations
    in index order; back up only when the shape doesn't fit (frame rule or tab/blank rule).
    Explores ALL complete assemblies and keeps the one with the lowest total cost.
    Node expansion = one (piece, orientation) tried in a cell.
    """
    R, W, N = puzzle.R, puzzle.C, puzzle.n
    types = puzzle.types()
    grid = empty_grid(R, W)
    used = np.zeros(N, bool)
    cells = [(y, x) for y in range(R) for x in range(W)]
    timer = Timer(time_limit)
    best = {"cost": INF, "grid": None}
    nodes = 0

    def place(k, cost):
        nonlocal nodes
        if k == len(cells):
            if cost < best["cost"]:
                best["cost"], best["grid"] = cost, [row[:] for row in grid]
            return
        y, x = cells[k]
        cand = candidate_costs(grid, y, x, C, types, used)
        for pid in range(N):                      # index order, no score ordering
            if used[pid]:
                continue
            for r in range(4):
                nodes += 1
                if nodes % 2000 == 0 and timer.up():
                    raise Timeout
                if cand[pid, r] == INF:           # shape conflict -> back up
                    continue
                grid[y][x] = (pid, r)
                used[pid] = True
                place(k + 1, cost + cand[pid, r])
                grid[y][x] = None
                used[pid] = False

    timed_out = False
    try:
        place(0, 0.0)
    except Timeout:
        timed_out = True
    return {"grid": best["grid"] or empty_grid(R, W), "nodes": nodes, "timed_out": timed_out}
