"""Solve one photo end to end and write the three outputs from the proposal.

    python solve_photo.py photo.jpg --rows 15 --cols 20 --search anneal --scorer color --out out/

Outputs: numbered.png (piece numbers on the photo), solution_map.png (number + turn per cell),
solution.json (one record per piece, 1-based), assembled.png (the pieces put together, for checking).
Leave out --rows/--cols to guess the grid size from the number of border pieces.
Real-photo labeling: run this once, open numbered.png, and write labels.csv (piece,row,col) from the box image.
"""
import argparse
import json
import os

import cv2
import numpy as np

from common import Puzzle, build_cost_table
from evaluate import SCORERS, SEARCHES
from extract import estimate_grid_size, extract_pieces
from generate import _rot90_points, read_rgb
from features import order_corners


def turn_of(piece, r):
    return int(round((piece.straighten_cw_deg + 90 * r) % 360)) % 360


def draw_numbers(photo, pieces):
    img = photo[:, :, ::-1].copy()
    for p in pieces:
        x, y = map(int, p.centroid_xy)
        txt = str(p.id + 1)
        cv2.putText(img, txt, (x - 10, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(img, txt, (x - 10, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def draw_map(grid, pieces, cell=70):
    R, C = len(grid), len(grid[0])
    img = np.full((R * cell + 1, C * cell + 1, 3), 255, np.uint8)
    for y in range(R):
        for x in range(C):
            cv2.rectangle(img, (x * cell, y * cell), ((x + 1) * cell, (y + 1) * cell), (0, 0, 0), 1)
            p = grid[y][x]
            if p is None:
                continue
            cv2.putText(img, str(p[0] + 1), (x * cell + 8, y * cell + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(img, f"{turn_of(pieces[p[0]], p[1])}", (x * cell + 8, y * cell + 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1)
    return img


def draw_assembled(grid, pieces, cell=64):
    R, C = len(grid), len(grid[0])
    img = np.zeros((R * cell, C * cell, 3), np.uint8)
    for y in range(R):
        for x in range(C):
            if grid[y][x] is None:
                continue
            pid, r = grid[y][x]
            p = pieces[pid]
            H, W = p.image.shape[:2]
            rot = np.rot90(p.image, k=-r)
            TL, TR, BR, BL = order_corners(_rot90_points(np.asarray(p.corners, float), H, W, r))
            sq = rot[int(TL[1]):int(BR[1]) + 1, int(TL[0]):int(BR[0]) + 1]
            if sq.size:
                img[y * cell:(y + 1) * cell, x * cell:(x + 1) * cell] = cv2.resize(sq, (cell, cell))
    return img[:, :, ::-1]


def write_outputs(out_dir, photo, puzzle, grid):
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "numbered.png"), draw_numbers(photo, puzzle.pieces))
    cv2.imwrite(os.path.join(out_dir, "solution_map.png"), draw_map(grid, puzzle.pieces))
    cv2.imwrite(os.path.join(out_dir, "assembled.png"), draw_assembled(grid, puzzle.pieces))
    records = [{"piece": p[0] + 1, "row": y + 1, "col": x + 1, "turn": turn_of(puzzle.pieces[p[0]], p[1])}
               for y, row in enumerate(grid) for x, p in enumerate(row) if p is not None]
    json.dump(sorted(records, key=lambda d: d["piece"]), open(os.path.join(out_dir, "solution.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--search", default="anneal", choices=list(SEARCHES))
    ap.add_argument("--scorer", default="color", choices=list(SCORERS))
    ap.add_argument("--time-limit", type=float, default=60)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    photo = read_rgb(a.photo)
    pieces, report = extract_pieces(photo)
    print(f"found {len(pieces)} pieces; flagged {report['flags']}; failed {len(report['failed'])}")
    R, C = (a.rows, a.cols) if a.rows and a.cols else (estimate_grid_size(pieces) or (None, None))
    if R is None:
        raise SystemExit("could not guess the grid size; pass --rows and --cols")
    puzzle = Puzzle(pieces, R, C, meta={"n_total": R * C})
    Ct, _ = build_cost_table(pieces, SCORERS[a.scorer])
    res = SEARCHES[a.search](puzzle, Ct, time_limit=a.time_limit)
    write_outputs(a.out, photo, puzzle, res["grid"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
