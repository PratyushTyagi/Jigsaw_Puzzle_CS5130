
# Jigsaw Puzzle Solver: Team Guide (CS 5130)

Everything already runs end to end. Each person owns one file with one search strategy and one edge scorer.
The shared code (dataset, photo extraction, evaluation, outputs) is common and nobody needs to rewrite it.

## Setup (everyone, 5 minutes)

```
pip install -r requirements.txt
python smoke_test.py          # ~2 min, should end with "smoke test passed"
```

Person 4 also needs PyTorch for the learned scorer: `pip install -r requirements-learned.txt`.
Nobody else needs it. Run all commands from inside the project folder.

## Who owns what

| Person | File | Search strategy | Edge scorer |
|---|---|---|---|
| 1 | `person1_brute_color.py` | `brute_force` | `color_scorer` |
| 2 | `person2_greedy_gradient.py` | `greedy_reading`, `greedy_confident` | `gradient_scorer` |
| 3 | `person3_csp_shape.py` | `csp` | `shape_scorer` |
| 4 | `person4_anneal_learned.py` | `anneal` | `learned_scorer` |

Common files (shared, change only if you tell the group):

| File | What it does |
|---|---|
| `common.py` | Piece/Puzzle types, rotation rule, `candidate_costs`, `energy`, `build_cost_table` |
| `features.py` | Turns a cut-out piece into side types, side outlines and edge colors |
| `generate.py` | Makes synthetic puzzles; saves and loads puzzle folders |
| `extract.py` | Finds pieces in a photo, finds corners, straightens them |
| `evaluate.py` | Accuracy metrics and the experiment runner (E1 to E4) |
| `solve_photo.py` | Photo in, `numbered.png` + `solution_map.png` + `solution.json` out |

## The rules everyone shares (already coded in `common.py`)

- Sides of a straightened piece: `TOP=0, RIGHT=1, BOTTOM=2, LEFT=3`.
- A placement is `(piece_id, r)`: the piece turned `r × 90°` clockwise. After turning, the original side
  facing direction `d` is `side_facing(r, d) = (d - r) % 4`.
- Cost table `C[i, a, j, b]`: cost of side `a` of piece `i` touching side `b` of piece `j`. Lower is better,
  `inf` means the shapes can't fit (a tab must meet a blank, flat sides never touch anything).
  Normalized so a typical cost is 1.
- `candidate_costs(grid, y, x, C, types, used)` returns an N×4 array: the cost of putting every
  `(piece, r)` in cell `(y, x)`. `inf` = not allowed (used already, flat side not on the frame, or shape
  doesn't fit a neighbor). **Every search uses this, so "legal move" means the same thing everywhere.**
- **Node expansion** = one `(piece, r)` tried in a cell. For annealing, one cell changed by a move.
- Internally everything is 0-based. Anything a human reads (`solution.json`, the map, piece numbers) is 1-based.

## How a scorer plugs in

A scorer is a function `scorer(pieces) -> array of shape N×4×N×4` of raw costs. `build_cost_table` then
adds the tab/blank rule, makes it symmetric, and normalizes it. Each piece gives you, per side:

- `piece.strip[side]`: 3×64×3 Lab colors along the side, clockwise. Row 0 is the outermost pixel row.
- `piece.contour[side]`: 64×2 outline in the side's own frame (start corner at (0,0), end at (1,0), +y outward).
- `piece.side_len[side]`, `piece.side_type[side]`.

**When two sides touch, their point order is reversed** (both are stored clockwise). All three
scorers already do this; copy the pattern if you write a new one.

## How a search plugs in

A search is a function `solve(puzzle, C, time_limit=60, seed=0)` returning
`{"grid": R×C list of (piece_id, r) or None, "nodes": int, "timed_out": bool}`.
Check the time limit inside your loop and return the best you have when it runs out.

---

## Person 1: Brute force + Color difference

**`color_scorer`.** Sum of squared Lab color differences between the two touching edges
(outermost pixel row, neighbor reversed). This is the baseline every other scorer is compared to.

**`brute_force`.** Fill cells in reading order. At each cell try every unused piece, in index order, in all
4 orientations. Back up only when the shape doesn't fit. It explores every complete assembly and keeps the
one with the lowest total cost, so it's exact but explodes: expect it to finish 3×3 and time out from 4×4.
That timeout is a result, not a bug.

**Your jobs today**
1. Read your file and run it on 3×3 puzzles: `python evaluate.py --data dataset --searches brute --scorers color --sizes 3x3`.
2. Build the dataset (below). This blocks the others, so do it first.
3. Optional: try the color difference on 2 or 3 pixel rows instead of 1 and see if accuracy changes.

**Report:** node expansions vs. puzzle size; the size where brute force stops finishing.

## Person 2: Greedy (both orders) + Gradient continuity

**`gradient_scorer`.** Simplified version of Gallagher's MGC. For each side, it learns how colors change
when moving toward the edge (mean and covariance of "edge row minus next row in"). If the picture
continues across the seam, the change across the seam should look the same. Cost = Mahalanobis distance.

**`greedy_reading`.** Put the cheapest legal piece in each cell, left to right, top to bottom. No backtracking.
It's tried once from each corner piece at (0,0) and the best result is kept. If no legal piece exists
for a cell it places the cheapest one anyway and counts a `violation`.

**`greedy_confident`.** Grows outward. Each step it looks at every empty cell next to placed pieces and fills
the one it's most sure about: most placed neighbors first, then the biggest gap between best and second-best
candidate. Simplified from Pomeranz et al. (it stays inside the R×C frame).

**Your jobs today**
1. Run E2 as soon as the dataset exists (greedy is fast). `--exp E2` includes the learned scorer, which
   crashes until Person 4 has trained `learned_head.pt`, so start with the other three:
   `python evaluate.py --data dataset --searches greedy_reading --scorers color,gradient,shape --sizes 3x3,4x4,6x6 --out results/e2.csv`.
   Once the head exists, add the learned rows to the same file (it appends):
   `python evaluate.py --data dataset --searches greedy_reading --scorers learned --sizes 3x3,4x4,6x6 --out results/e2.csv`.
2. Tell the group which scorer wins, and set `BEST_SCORER` in `evaluate.py`.
3. Photograph the real puzzles (4–6 puzzles, ~30 photos, plain table, pieces face-up and not touching).

**Report:** E2 table (direct and neighbor accuracy per scorer); reading order vs. confident-first.

## Person 3: CSP backtracking + Shape fit

**`shape_scorer`.** Flips one side's outline onto the other (reverse points, map (x, y) → (1−x, −y)) and
measures the mean squared distance, plus the difference in side length. Uses no color, so it should help on
sky and water.

**`csp`.** Brute force plus pruning, as branch and bound:
- border restriction: only pieces whose flat sides face the frame (built into `candidate_costs`)
- forward checking: after a placement, every empty neighbor must still have a legal candidate
- most-constrained cell first: among empty cells next to placed pieces, fill the one with the fewest legal
  candidates (so the assembly stays connected)
- cheapest candidate first, so a good full solution is found early
- bound: drop a branch once its cost is no better than the best full solution (costs are ≥ 0)

It returns the best full solution found; if it finishes, that solution is optimal for the scorer.

**Your jobs today**
1. Compare `csp` vs `brute` node counts on 3×3 and 4×4.
2. Check photo extraction on synthetic photos: `python evaluate.py --data dataset --mode full --searches greedy_reading --sizes 4x4,6x6`. The `n_found` column shows how many pieces were found.
3. Once Person 2 has photos, run `solve_photo.py` on them and fix extraction problems (shadows, pieces colored like the table).

**Report:** node reduction from pruning; shape vs. color on low-texture images (`texture` in `meta.json`).

## Person 4: Simulated annealing + Learned classifier

**`anneal`.** Starts from `greedy_reading`'s solution. Moves: swap two pieces of the same kind (corner,
edge, interior) and re-turn both to their best orientation, or rotate one interior piece. A move is kept if
it lowers the energy, and sometimes when it doesn't, with probability `exp(-delta/T)`. T starts where about
80% of worsening moves are accepted (`start_accept`) and cools to 0.1% of that by the iteration budget or the
time limit, whichever comes first. Returns the lowest-energy grid seen.

**`learned_scorer`.** Crops a band along each side (turned to face right, and again to face left), embeds it
with a frozen ImageNet ResNet-18, and a small MLP predicts "do these two edges match?". Cost = −log P(match).
Train once on TRAIN-split puzzles (needs the dataset and `pip install -r requirements-learned.txt`;
GPU recommended):

```
python person4_anneal_learned.py --train --data dataset      # writes learned_head.pt next to the code
```

Tell Person 2 when `learned_head.pt` exists so the learned rows can be added to E2.

This part could not be tested where the code was written (no PyTorch there), so test it first on a few
3×3 puzzles: `python evaluate.py --data dataset --scorers learned --sizes 3x3 --limit 3`.

**Your jobs today**
1. Get the learned scorer training and running.
2. Own `evaluate.py`: run E1, E3 and E4 tomorrow and make the plots.

**Note from testing:** on 15×20 puzzles with a 40 s limit, annealing didn't beat its greedy start; lower
starting temperatures lowered the energy but broke up correct regions. That's worth reporting.
Try `start_accept` and `iters_per_piece` if you have time.

**Report:** annealing gain over greedy vs. size; learned vs. the hand-made scorers.

---

## Dataset (Person 1 runs this first)

Put source images in one folder (MIT and McGill benchmark images plus ~100 open-license photos, including
sky/water scenes), then:

```
# quick first batch so others can start (minutes)
python generate.py --images source_images --out dataset --sizes 3x3,4x4,6x6 --variants clean

# full set overnight (~1,400 puzzles with ~140 images)
python generate.py --images source_images --out dataset --sizes 3x3,4x4,6x6,10x10,15x20 --variants clean,effects
```

Re-running skips puzzles that already exist. The train/test split is by source image (30% test), so the
learned scorer never sees test images. `effects` adds blur, shadows and uneven lighting.

**Real photos.** Make one folder per photo containing `scene.png` and `meta.json` with
`{"R": rows, "C": cols, "split": "test"}`. The `"split": "test"` is required: `evaluate.py` only reads
test-split folders by default, so a folder without it is silently skipped.
Run `python solve_photo.py folder/scene.png --rows R --cols C --out folder/out` once (give each photo its
own `--out`, otherwise every run overwrites `out/`), open `folder/out/numbered.png`, and write
`labels.csv` with columns `piece,row,col,turn` (1-based, from the box image). `turn` is optional: the
clockwise degrees the piece must turn from how it lies in the photo to sit upright in the finished puzzle
(the same meaning as the turn printed on `solution_map.png`; within 10° counts). Without it, direct
accuracy only checks the cell and neighbor accuracy ignores which sides touch.
Then evaluate with `--mode full`.

## Running the experiments

```
python evaluate.py --data dataset --exp E2 --out results/e2.csv     # picks the best scorer; needs learned_head.pt
python evaluate.py --data dataset --exp E1 --out results/e1.csv
python evaluate.py --data dataset --exp E3 --out results/e3.csv
python evaluate.py --data dataset --exp E4 --out results/e4.csv
```

E2 and E4 include the learned scorer, so they crash until `learned_head.pt` exists. Before that, use the
manual E2 commands under Person 2.

Options: `--limit 20` (puzzles per size, for a quick first pass), `--time-limit 60`, `--mode full`
(pieces found from the photo instead of pre-cut). Each row has direct accuracy, neighbor accuracy, node
expansions, runtime, cost-table time, peak memory and timeout.

Accuracy counts a solution that is correct but rotated as a whole as correct. Direct accuracy needs the
right cell and a turn within 10°. Neighbor accuracy counts true neighbor pairs that touch with the right sides.

## Timeline

- **Today, first hour:** Person 1 makes the quick 3×3/4×4/6×6 batch. Everyone runs their own file on it.
- **Today:** Person 2 runs E2 and sets `BEST_SCORER`. Person 4 trains the learned scorer. Person 3 checks extraction. Person 2 takes photos.
- **Tonight:** full dataset builds.
- **Tomorrow:** E1, E3, E4, then real photos. If time is short, use `--limit` and drop the real 300-piece puzzles first.
