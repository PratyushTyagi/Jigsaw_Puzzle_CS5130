"""PERSON 3 - Backtracking with constraint propagation (search) + Shape fit (edge scorer)."""
import numpy as np

from common import DIRS, INF, Timeout, Timer, all_sides, candidate_costs, empty_grid


# ------------------------------------------------------------------ edge scorer
def shape_scorer(pieces):
    """How well two side outlines interlock. Uses no color, so it still works on sky / water.

    Outlines are stored in each side's own frame (start corner (0,0), end corner (1,0), +y outward).
    To lay side b of piece j onto side a of piece i: reverse j's points and map (x, y) -> (1 - x, -y).
    Cost = mean squared distance between matching points + relative difference in side length.
    """
    P = all_sides(pieces, "contour")              # (N*4) x 64 x 2
    F = P[:, ::-1].copy()
    F[..., 0] = 1 - F[..., 0]
    F[..., 1] = -F[..., 1]
    L = all_sides(pieces, "side_len")             # (N*4,)
    n = len(P)
    out = np.empty((n, n))
    for q in range(n):
        out[q] = ((P[q][None] - F) ** 2).sum(axis=2).mean(axis=1) + np.abs(L[q] - L) / ((L[q] + L) / 2)
    N = len(pieces)
    return out.reshape(N, 4, N, 4)


# ------------------------------------------------------------------ search
def csp(puzzle, C, time_limit=60, seed=0):
    """Brute force + pruning, as branch and bound:
      * border restriction: candidate_costs already allows only pieces whose flats face out of the frame
      * forward checking: after a placement, every empty neighbor must still have a legal candidate
      * most-constrained cell first: fill the empty cell (next to placed ones) with the fewest legal candidates
      * value ordering: try candidates cheapest first, so a good full solution is found early
      * bound: skip a branch once its cost so far is no better than the best full solution (costs are >= 0)
    Returns the best complete assembly found (optimal if it finishes before the time limit).
    """
    R, W, N = puzzle.R, puzzle.C, puzzle.n
    types = puzzle.types()
    grid = empty_grid(R, W)
    used = np.zeros(N, bool)
    timer = Timer(time_limit)
    best = {"cost": INF, "grid": None}
    nodes = 0

    def frontier():
        """Empty cells next to a placed piece (or (0,0) if the grid is empty)."""
        cells = set()
        for y in range(R):
            for x in range(W):
                if grid[y][x] is None:
                    continue
                for dy, dx in DIRS.values():
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < R and 0 <= xx < W and grid[yy][xx] is None:
                        cells.add((yy, xx))
        return cells or {(0, 0)}

    def search(filled, cost):
        nonlocal nodes
        if filled == R * W:
            if cost < best["cost"]:
                best["cost"], best["grid"] = cost, [row[:] for row in grid]
            return
        # most constrained cell
        options = {c: candidate_costs(grid, *c, C, types, used) for c in frontier()}
        cell = min(options, key=lambda c: np.isfinite(options[c]).sum())
        cand = options[cell]
        y, x = cell
        order = np.argsort(cand, axis=None)
        nodes += int((~used).sum()) * 4
        for k in order:
            c = cand.flat[k]
            if c == INF or cost + c >= best["cost"]:
                break                              # sorted, so everything after is worse
            if timer.up():
                raise Timeout
            pid, r = divmod(int(k), 4)
            grid[y][x] = (pid, r)
            used[pid] = True
            # forward checking: each empty neighbor must keep at least one legal candidate
            ok = True
            for dy, dx in DIRS.values():
                yy, xx = y + dy, x + dx
                if 0 <= yy < R and 0 <= xx < W and grid[yy][xx] is None and filled + 1 < R * W:
                    if not np.isfinite(candidate_costs(grid, yy, xx, C, types, used)).any():
                        ok = False
                        break
            if ok:
                search(filled + 1, cost + c)
            grid[y][x] = None
            used[pid] = False

    timed_out = False
    try:
        search(0, 0.0)
    except Timeout:
        timed_out = True
    return {"grid": best["grid"] or empty_grid(R, W), "nodes": nodes, "timed_out": timed_out}
