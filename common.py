"""Shared definitions. Everyone imports from here; nobody redefines these.

CONVENTIONS
  Sides of a straightened piece: TOP=0, RIGHT=1, BOTTOM=2, LEFT=3.
  A placement is (piece_id, r): the piece turned r * 90 degrees clockwise.
  After r clockwise turns, the ORIGINAL side facing direction d is side_facing(r, d) = (d - r) % 4.
  grid[row][col], (0, 0) is top-left. Everything internal is 0-based; outputs for humans are 1-based.
  Cost table C[i, a, j, b] = cost of side a of piece i touching side b of piece j (lower = better fit,
  inf = shapes can't fit). Normalized so the median finite cost is 1.
  Turn shown to the user = (piece.straighten_cw_deg + 90 * r) % 360.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np

FLAT, TAB, BLANK = 0, 1, -1
TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3
DIRS = {TOP: (-1, 0), RIGHT: (0, 1), BOTTOM: (1, 0), LEFT: (0, -1)}
STRIP_LEN, STRIP_DEPTH = 64, 3
INF = math.inf
PENALTY = 10.0          # cost charged for an impossible edge in local search (median real cost is 1)


@dataclass
class Piece:
    id: int
    image: np.ndarray          # straightened RGB crop (0 outside the mask)
    mask: np.ndarray           # bool
    corners: np.ndarray        # 4x2 (x, y): TL, TR, BR, BL
    side_type: list            # 4 values of FLAT / TAB / BLANK
    strip: np.ndarray          # 4 x 3 x 64 x 3 Lab colors along each side (depth 0 = outermost), clockwise
    contour: np.ndarray        # 4 x 64 x 2 side outline, normalized: start corner (0,0), end (1,0), +y outward
    side_len: np.ndarray       # 4 side lengths in pixels
    straighten_cw_deg: float = 0.0
    centroid_xy: tuple = (0.0, 0.0)
    flags: list = field(default_factory=list)


@dataclass
class Puzzle:
    pieces: list
    R: int
    C: int
    gt: dict = None            # piece_id -> {"cell": (row, col), "turn": degrees or None}
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.pieces)

    def types(self):
        """N x 4 array of side types."""
        return np.array([p.side_type for p in self.pieces], dtype=np.int8).reshape(-1, 4)


# ------------------------------------------------------------------ orientation and the frame
def side_facing(r, d):
    return (d - r) % 4


def opposite(d):
    return (d + 2) % 4


def empty_grid(R, C):
    return [[None] * C for _ in range(R)]


def on_border(R, C, y, x, d):
    dy, dx = DIRS[d]
    return not (0 <= y + dy < R and 0 <= x + dx < C)


def frame_ok(types, pid, r, R, C, y, x):
    """Flat sides face out of the frame, and only flat sides do."""
    return all(on_border(R, C, y, x, d) == (types[pid, side_facing(r, d)] == FLAT) for d in range(4))


def pair_cost(C, pa, d, pb):
    """Cost between placement pa and placement pb, where pb is in direction d from pa."""
    (i, ri), (j, rj) = pa, pb
    return C[i, side_facing(ri, d), j, side_facing(rj, opposite(d))]


def neighbors(grid, y, x):
    """Yield (direction, placement) for every placed neighbor of cell (y, x)."""
    R, Cc = len(grid), len(grid[0])
    for d, (dy, dx) in DIRS.items():
        yy, xx = y + dy, x + dx
        if 0 <= yy < R and 0 <= xx < Cc and grid[yy][xx] is not None:
            yield d, grid[yy][xx]


def candidate_costs(grid, y, x, C, types, used):
    """N x 4 array: cost of putting each (piece, r) in empty cell (y, x) given its placed neighbors.

    inf where the piece is already used, breaks the frame rule, or can't fit a neighbor's shape.
    This one helper gives every search the same feasibility rule. Finite entries = legal moves.
    """
    R, Cc = len(grid), len(grid[0])
    N = types.shape[0]
    out = np.zeros((N, 4))
    for r in range(4):
        for d in range(4):
            side = side_facing(r, d)
            flat = types[:, side] == FLAT
            out[flat != on_border(R, Cc, y, x, d), r] = INF
    for d, (j, rj) in neighbors(grid, y, x):
        sj = side_facing(rj, opposite(d))
        for r in range(4):
            out[:, r] += C[:, side_facing(r, d), j, sj]
    out[np.asarray(used, bool)] = INF
    return out


def total_cost(grid, C):
    """Sum over all adjacent placed pairs (inf if any pair can't fit)."""
    R, Cc = len(grid), len(grid[0])
    s = 0.0
    for y in range(R):
        for x in range(Cc):
            if grid[y][x] is None:
                continue
            if x + 1 < Cc and grid[y][x + 1] is not None:
                s += pair_cost(C, grid[y][x], RIGHT, grid[y][x + 1])
            if y + 1 < R and grid[y + 1][x] is not None:
                s += pair_cost(C, grid[y][x], BOTTOM, grid[y + 1][x])
    return s


def energy(grid, C, types):
    """Like total_cost but always finite: impossible edges, frame violations and empty cells cost PENALTY."""
    R, Cc = len(grid), len(grid[0])
    e = 0.0
    for y in range(R):
        for x in range(Cc):
            p = grid[y][x]
            if p is None:
                e += PENALTY
                continue
            if not frame_ok(types, p[0], p[1], R, Cc, y, x):
                e += PENALTY
            for d in (RIGHT, BOTTOM):
                dy, dx = DIRS[d]
                if y + dy < R and x + dx < Cc and grid[y + dy][x + dx] is not None:
                    c = pair_cost(C, p, d, grid[y + dy][x + dx])
                    e += c if c < INF else PENALTY
    return e


class Timer:
    def __init__(self, limit_s):
        self.t0, self.limit = time.perf_counter(), limit_s

    def up(self):
        return time.perf_counter() - self.t0 >= self.limit


class Timeout(Exception):
    pass


# ------------------------------------------------------------------ cost table
def all_sides(pieces, what):
    """Stack a per-side feature over all pieces into shape (N*4, ...), row index = 4*i + a."""
    arr = np.stack([getattr(p, what) for p in pieces])        # N x 4 x ...
    return arr.reshape(len(pieces) * 4, *arr.shape[2:])


def build_cost_table(pieces, scorer):
    """scorer(pieces) -> raw N x 4 x N x 4 costs. This adds the shape rule, symmetry and normalization."""
    t0 = time.perf_counter()
    N = len(pieces)
    t = np.array([p.side_type for p in pieces]).reshape(-1)    # N*4
    ok = ((t[:, None] == TAB) & (t[None, :] == BLANK)) | ((t[:, None] == BLANK) & (t[None, :] == TAB))
    ok = ok.reshape(N, 4, N, 4)
    ok[np.arange(N), :, np.arange(N), :] = False                 # a piece can't touch itself
    raw = np.asarray(scorer(pieces), dtype=float)
    Ct = np.where(ok, raw, INF)
    Ct = 0.5 * (Ct + Ct.transpose(2, 3, 0, 1))                   # same cost from both sides
    Ct[~np.isfinite(Ct)] = INF
    finite = Ct[np.isfinite(Ct)]
    if finite.size and np.median(finite) > 0:
        Ct = Ct / np.median(finite)
    return Ct, time.perf_counter() - t0
