"""One proposal = one budget unit (PLAN section 7): validity gate, then the course set over fixed jitter seeds."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .assembly import assemble, load_script
from .brains import BRAINS
from .scoring import score_episode
from .sim import run_episode, summarize
from .validity import check_design
from .worlds import get_course

YAW_JITTER = 0.1  # rad, uniform +-


def jitter_yaw(seed: int) -> float:
    return float(np.random.default_rng(seed).uniform(-YAW_JITTER, YAW_JITTER))


def evaluate(build_fn, genome, courses=("bumps",), seeds=(0,), brain="A", record_trace=False) -> dict:
    t0 = time.time()
    rep = check_design(build_fn, genome)
    out = dict(valid=rep.ok, validity=rep.failures, validity_s=time.time() - t0, score=0.0, per_course={}, sim_s=0.0)
    if not rep.ok:
        out["wall_s"] = time.time() - t0
        return out
    scores = []
    for cname in courses:
        course = get_course(cname)
        eps = []
        for sd in seeds:
            asm = assemble(build_fn, genome, course, yaw=jitter_yaw(sd))
            res = run_episode(asm, BRAINS[brain](), record_trace=record_trace)
            sc = score_episode(res, course)
            out["sim_s"] += res.time_s
            eps.append(dict(seed=sd, score=sc, summary=summarize(res), result=res.to_dict(),
                            trace=res.trace if record_trace else None))
        mean = float(np.mean([e["score"]["score"] for e in eps]))
        out["per_course"][cname] = dict(mean_score=mean, episodes=eps)
        scores.append(mean)
    out["score"] = float(np.mean(scores))
    out["valid"] = all(e["score"]["valid"] for c in out["per_course"].values() for e in c["episodes"])
    out["wall_s"] = time.time() - t0
    return out


def evaluate_script(path, **kw) -> dict:
    genome, build_fn = load_script(path)
    return evaluate(build_fn, genome, **kw)


class Ledger:
    """Append-only budget ledger: one line per proposal, valid or not."""

    def __init__(self, path, budget: int):
        self.path, self.budget, self.used = Path(path), budget, 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def charge(self, arm: str, seed: int, out: dict) -> bool:
        """Record one proposal. Returns False (and records nothing) if the budget is exhausted."""
        if self.used >= self.budget:
            return False
        self.used += 1
        row = dict(arm=arm, seed=seed, proposal=self.used, valid=out["valid"],
                   why_invalid=[f["name"] for f in out["validity"]], score=out["score"], sim_s=out["sim_s"],
                   wall_s=out["wall_s"], validity_s=out["validity_s"])
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return True
