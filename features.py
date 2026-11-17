"""compute_side_features: the one function that turns (image, mask, corners) into side features.

Both the synthetic oracle pieces and the pieces extracted from photos go through this, so the
scorers always see the same kind of input.

Per side s (TOP, RIGHT, BOTTOM, LEFT), going CLOCKWISE from corner s to corner (s+1) % 4:
  contour[s] : L x 2 points in the side's normalized frame:
               start corner -> (0, 0), end corner -> (1, 0), +y points OUTWARD from the piece,
               units = side length. A tab bulges to y > 0, a blank dips to y < 0.
  side_type  : TAB if max |y| deviation is outward and > 0.1, BLANK if inward and > 0.1, else FLAT.
  strip[s]   : K x L x 3 Lab colors (L channel 0-100) sampled at the same L contour points,
               offset 1, 2, 3 px INWARD along the local contour normal. k=0 is the outermost.

Matching rule every scorer must respect: side a of piece i touches side b of piece j with their
point orders REVERSED (both are stored clockwise). So compare strip_i[a][:, l] with strip_j[b][:, L-1-l].
For contours, map j's points into i's frame with (x, y) -> (1 - x, -y) and reverse their order.
"""
from __future__ import annotations

import cv2
import numpy as np

from common import BLANK, FLAT, STRIP_DEPTH, STRIP_LEN, TAB

TYPE_THRESHOLD = 0.10


def _outer_contour(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("empty mask")
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)  # N x 2 (x, y)
    # Make it clockwise as displayed (y down): positive shoelace sum.
    x, y = c[:, 0], c[:, 1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        c = c[::-1]
    return c


def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)


def _side_points(contour: np.ndarray, corners: np.ndarray) -> list:
    """Split the clockwise contour into 4 arcs between the contour points nearest each corner."""
    idx = [int(np.argmin(np.linalg.norm(contour - c, axis=1))) for c in corners]
    n = len(contour)
    arcs = []
    for s in range(4):
        a, b = idx[s], idx[(s + 1) % 4]
        if b < a:
            b += n
        seg = contour[np.arange(a, b + 1) % n].copy()
        seg[0], seg[-1] = corners[s], corners[(s + 1) % 4]  # pin the exact corners
        arcs.append(seg)
    return arcs


def compute_side_features(image: np.ndarray, mask: np.ndarray, corners: np.ndarray,
                          L: int = STRIP_LEN, K: int = STRIP_DEPTH) -> dict:
    """Return dict(side_type, contour, strip, side_len) for a straightened piece."""
    corners = np.asarray(corners, dtype=np.float64)
    contour = _outer_contour(mask)
    arcs = _side_points(contour, corners)

    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)

    side_type, contours, strips, side_len = [], [], [], []
    for s, arc in enumerate(arcs):
        pts = _resample(arc, L)
        A, B = corners[s], corners[(s + 1) % 4]
        length = float(np.linalg.norm(B - A))
        u = (B - A) / max(length, 1e-9)
        n_out = np.array([u[1], -u[0]])                 # outward normal for clockwise order (y down)
        local = np.stack([(pts - A) @ u, (pts - A) @ n_out], axis=1) / max(length, 1e-9)
        contours.append(local.astype(np.float32))
        side_len.append(length)

        dev = local[:, 1]
        k = int(np.argmax(np.abs(dev)))
        if abs(dev[k]) <= TYPE_THRESHOLD:
            side_type.append(FLAT)
        else:
            side_type.append(TAB if dev[k] > 0 else BLANK)

        # inward normal at each sample, from the local tangent of the resampled contour
        tan = np.gradient(pts, axis=0)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
        n_in = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
        rows = []
        for depth in range(1, K + 1):
            q = (pts + depth * n_in).astype(np.float32)
            samp = cv2.remap(lab, q[:, 0].reshape(1, -1), q[:, 1].reshape(1, -1),
                             interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            rows.append(samp[0])
        strips.append(np.stack(rows))                   # K x L x 3

    return {
        "side_type": side_type,
        "contour": np.stack(contours).astype(np.float32),
        "strip": np.stack(strips).astype(np.float32),
        "side_len": np.array(side_len, dtype=np.float32),
    }


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL for a roughly axis-aligned quadrilateral."""
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])  # y down: increasing angle = clockwise
    pts = pts[np.argsort(ang)]
    start = int(np.argmin(pts.sum(axis=1)))               # TL has the smallest x + y
    return np.roll(pts, -start, axis=0)
