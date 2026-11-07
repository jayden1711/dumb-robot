"""Scoring. Constants are fixed before any designer sees them (PLAN section 6)."""
from __future__ import annotations

import numpy as np

W_TIME = 0.30
W_ENERGY = 0.15
FLIP_PENALTY = 0.05
FLIP_CAP = 4


def score_episode(res, course) -> dict:
    p = float(np.clip(res.progress, 0.0, 1.0))
    if res.status == "exploded":
        return dict(score=0.0, valid=False, progress=0.0, done=False, time_s=res.time_s, energy_j=res.energy_j,
                    flips=res.flips, status=res.status)
    s = p
    if res.done:
        s += W_TIME * (1.0 - min(res.time_s / course.t_limit, 1.0))
        s += W_ENERGY * (1.0 - min(res.energy_j / course.e_ref, 1.0))
    s -= FLIP_PENALTY * min(res.flips, FLIP_CAP)
    return dict(score=float(s), valid=True, progress=p, done=bool(res.done), time_s=res.time_s, energy_j=res.energy_j,
                flips=res.flips, status=res.status)
