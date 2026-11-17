"""Run experiments and score results.

    python evaluate.py --data dataset --exp E2 --out results/e2.csv
    python evaluate.py --data dataset --searches greedy_reading,csp --scorers color --sizes 3x3,4x4

Writes one CSV row per (puzzle, search, scorer): accuracy, node expansions, runtime, memory, timeout.
"""
import argparse
import csv
import os
import time
import tracemalloc

from common import BOTTOM, DIRS, RIGHT, build_cost_table, opposite, side_facing
from person1_brute_color import brute_force, color_scorer
from person2_greedy_gradient import gradient_scorer, greedy_confident, greedy_reading
from person3_csp_shape import csp, shape_scorer
from person4_anneal_learned import anneal, learned_scorer

SEARCHES = {"brute": brute_force, "greedy_reading": greedy_reading, "greedy_confident": greedy_confident,
            "csp": csp, "anneal": anneal}
SCORERS = {"color": color_scorer, "gradient": gradient_scorer, "shape": shape_scorer, "learned": learned_scorer}

BEST_SCORER = "color"     # update after E2 shows which scorer wins
ALL_SIZES = ["3x3", "4x4", "6x6", "10x10", "15x20"]
EXPERIMENTS = {
    "E1": dict(searches=list(SEARCHES), scorers=[BEST_SCORER], sizes=["3x3", "4x4", "6x6"]),
    "E2": dict(searches=["greedy_reading"], scorers=list(SCORERS), sizes=ALL_SIZES),
    "E3": dict(searches=list(SEARCHES), scorers=[BEST_SCORER], sizes=ALL_SIZES),
    "E4": dict(searches=list(SEARCHES), scorers=list(SCORERS), sizes=["3x3", "4x4", "6x6"]),
}


# ------------------------------------------------------------------ metrics
def angle_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def rotate_solution(grid, k):
    """Turn the whole assembled puzzle k quarter turns clockwise."""
    for _ in range(k):
        R, C = len(grid), len(grid[0])
        new = [[None] * R for _ in range(C)]
        for y in range(R):
            for x in range(C):
                p = grid[y][x]
                new[x][R - 1 - y] = None if p is None else (p[0], (p[1] + 1) % 4)
        grid = new
    return grid


def direct_accuracy(grid, puzzle, tol=10.0):
    """Fraction of pieces in the right cell with turn within `tol` degrees.
    A solution that is right but rotated as a whole still counts (best of the allowed global turns)."""
    best = 0
    for k in ([0, 1, 2, 3] if puzzle.R == puzzle.C else [0, 2]):
        g = rotate_solution(grid, k)
        ok = 0
        for y, row in enumerate(g):
            for x, p in enumerate(row):
                t = puzzle.gt.get(p[0]) if p else None
                if t and tuple(t["cell"]) == (y, x):
                    turn = (puzzle.pieces[p[0]].straighten_cw_deg + 90 * p[1]) % 360
                    if t["turn"] is None or angle_diff(turn, t["turn"]) <= tol:
                        ok += 1
        best = max(best, ok)
    return best / puzzle.meta.get("n_total", puzzle.R * puzzle.C)


def gt_grid(puzzle):
    """The correct solution as a grid."""
    g = [[None] * puzzle.C for _ in range(puzzle.R)]
    for pid, t in puzzle.gt.items():
        y, x = t["cell"]
        r = 0 if t["turn"] is None else round(((t["turn"] - puzzle.pieces[pid].straighten_cw_deg) % 360) / 90) % 4
        g[y][x] = (pid, r)
    return g


def touching_pairs(grid, with_sides=True):
    """Set of adjacent pairs in a grid, as {(piece, side), (piece, side)} (or {piece, piece})."""
    pairs = set()
    for y, row in enumerate(grid):
        for x, p in enumerate(row):
            for d in (RIGHT, BOTTOM):
                dy, dx = DIRS[d]
                if p is None or y + dy >= len(grid) or x + dx >= len(row) or grid[y + dy][x + dx] is None:
                    continue
                q = grid[y + dy][x + dx]
                if with_sides:
                    pairs.add(frozenset([(p[0], side_facing(p[1], d)), (q[0], side_facing(q[1], opposite(d)))]))
                else:
                    pairs.add(frozenset([p[0], q[0]]))
    return pairs


def neighbor_accuracy(grid, puzzle):
    """Fraction of true neighbor pairs that are neighbors in the solution with the same sides touching."""
    sides = all(t["turn"] is not None for t in puzzle.gt.values())
    truth = touching_pairs(gt_grid(puzzle), sides)
    n_true = puzzle.R * (puzzle.C - 1) + (puzzle.R - 1) * puzzle.C
    return len(truth & touching_pairs(grid, sides)) / n_true


# ------------------------------------------------------------------ running
def run_one(puzzle, C, search_name, time_limit, seed=0):
    tracemalloc.start()
    t0 = time.perf_counter()
    res = SEARCHES[search_name](puzzle, C, time_limit=time_limit, seed=seed)
    runtime = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return {"direct_acc": round(direct_accuracy(res["grid"], puzzle), 4),
            "neighbor_acc": round(neighbor_accuracy(res["grid"], puzzle), 4),
            "nodes": res["nodes"], "runtime_s": round(runtime, 3), "peak_mem_mb": round(peak, 2),
            "timed_out": res["timed_out"], "violations": res.get("violations", 0)}


def main():
    from generate import list_puzzles, load_puzzle
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--exp", choices=list(EXPERIMENTS))
    ap.add_argument("--searches", default="greedy_reading")
    ap.add_argument("--scorers", default="color")
    ap.add_argument("--sizes", default="3x3,4x4")
    ap.add_argument("--mode", default="oracle", choices=["oracle", "full"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="max puzzles per size (0 = all)")
    ap.add_argument("--time-limit", type=float, default=60)
    ap.add_argument("--out", default="results/results.csv")
    a = ap.parse_args()

    cfg = EXPERIMENTS[a.exp] if a.exp else dict(searches=a.searches.split(","), scorers=a.scorers.split(","),
                                                sizes=a.sizes.split(","))
    folders = []
    for size in cfg["sizes"]:
        fs = list_puzzles(a.data, [size], a.split)
        folders += fs[:a.limit] if a.limit else fs
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fields = ["puzzle_id", "mode", "R", "C", "search", "scorer", "direct_acc", "neighbor_acc", "nodes",
              "runtime_s", "table_s", "peak_mem_mb", "timed_out", "violations", "n_found"]
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for folder in folders:
            pz = load_puzzle(folder, a.mode)
            for sc in cfg["scorers"]:
                C, table_s = build_cost_table(pz.pieces, SCORERS[sc])
                for se in cfg["searches"]:
                    row = run_one(pz, C, se, a.time_limit)
                    row.update(puzzle_id=pz.meta["puzzle_id"], mode=a.mode, R=pz.R, C=pz.C, search=se,
                               scorer=sc, table_s=round(table_s, 3), n_found=pz.n)
                    w.writerow(row)
                    f.flush()
                    print(f"{pz.meta['puzzle_id']:30s} {sc:9s} {se:17s} direct={row['direct_acc']:.2f} "
                          f"neighbor={row['neighbor_acc']:.2f} nodes={row['nodes']} t={row['runtime_s']}s"
                          + (" TIMEOUT" if row["timed_out"] else ""))


if __name__ == "__main__":
    main()
