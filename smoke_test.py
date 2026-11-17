"""Quick check that everything runs. Takes ~2 minutes, needs no downloads.

    python smoke_test.py
"""
import os
import tempfile

import numpy as np
import skimage.data

from common import build_cost_table, total_cost
from evaluate import SCORERS, SEARCHES, direct_accuracy, gt_grid, neighbor_accuracy, run_one
from generate import generate_puzzle, load_puzzle, save_puzzle

tmp = tempfile.mkdtemp()
for name, img in [("astronaut", skimage.data.astronaut()), ("coffee", skimage.data.coffee())]:
    for R, C in [(3, 3), (4, 4)]:
        save_puzzle(os.path.join(tmp, f"{name}_{R}x{C}"), generate_puzzle(img, R, C, seed=R),
                    {"source": name, "split": "test"})

for folder in sorted(os.listdir(tmp)):
    pz = load_puzzle(os.path.join(tmp, folder), "oracle")
    truth = gt_grid(pz)
    assert direct_accuracy(truth, pz) == 1.0 and neighbor_accuracy(truth, pz) == 1.0
    print(f"\n{folder}")
    for sc in ["color", "gradient", "shape"]:          # "learned" needs torch + a trained head
        C, table_s = build_cost_table(pz.pieces, SCORERS[sc])
        assert np.isfinite(total_cost(truth, C)), "true solution must be shape-feasible"
        for se in SEARCHES:
            limit = 5 if se == "brute" else 30
            r = run_one(pz, C, se, limit)
            print(f"  {sc:9s} {se:17s} direct={r['direct_acc']:.2f} neighbor={r['neighbor_acc']:.2f} "
                  f"nodes={r['nodes']:>8} t={r['runtime_s']:6.2f}s" + (" TIMEOUT" if r["timed_out"] else ""))

pz = load_puzzle(os.path.join(tmp, "coffee_4x4"), "full")
print(f"\nfull mode (photo extraction): found {pz.n}/16 pieces, matched {len(pz.gt)} to ground truth")
C, _ = build_cost_table(pz.pieces, SCORERS["color"])
r = run_one(pz, C, "anneal", 30)
print(f"  color anneal from photo: direct={r['direct_acc']:.2f} neighbor={r['neighbor_acc']:.2f}")
print("\nsmoke test passed")
